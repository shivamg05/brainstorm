"""
EEGNet-style model for continuous classification of ECoG signals.

Implements the EEGNet architecture described in EEG_NET_PAPER.md and adapts it to
streaming classification by predicting the label of the most recent timestep in a
sliding window. Uses PCA to reduce channels before windowing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from brainstorm.constants import N_CHANNELS
from brainstorm.ml.base import BaseModel


_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "eegnet.pt"
CHECKPOINT_DIR = _REPO_ROOT / "checkpoints"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "eegnet_best.pt"


class EEGNet(BaseModel):
    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        window_samples: int = 128,
        temporal_kernel: int = 64,
        F1: int = 8,
        D: int = 2,
        dropout: float = 0.25,
        pool1: int = 4,
        pool2: int = 8,
        pca_components: int | None = 64,
    ) -> None:
        super().__init__()
        self.raw_channels = n_channels
        self.pca_components = pca_components
        self.n_channels = pca_components if pca_components is not None else n_channels
        self.window_samples = window_samples
        self.temporal_kernel = temporal_kernel
        self.F1 = F1
        self.D = D
        self.F2 = F1 * D
        self.dropout_rate = dropout
        self.pool1 = pool1
        self.pool2 = pool2

        self.classes_: np.ndarray | None = None
        self._n_classes: int | None = None
        self._buffer: np.ndarray | None = None
        self._pca_components: np.ndarray | None = None
        self._pca_mean: np.ndarray | None = None

        self.temporal_conv = nn.Conv2d(
            1,
            F1,
            kernel_size=(1, temporal_kernel),
            padding=(0, temporal_kernel // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise_conv = nn.Conv2d(
            F1,
            F1 * D,
            kernel_size=(self.n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)

        self.sep_depthwise = nn.Conv2d(
            F1 * D,
            F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False,
        )
        self.sep_pointwise = nn.Conv2d(F1 * D, self.F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.F2)

        self.dropout = nn.Dropout(dropout)
        self._feature_dim = self._compute_feature_dim(self.window_samples)
        self.classifier = nn.Linear(self._feature_dim, 1)

    def _build_classifier(self, n_classes: int) -> None:
        self._n_classes = n_classes
        self._feature_dim = self._compute_feature_dim(self.window_samples)
        self.classifier = nn.Linear(self._feature_dim, n_classes)

    def _compute_feature_dim(self, window_samples: int) -> int:
        pooled = max(1, window_samples // self.pool1)
        pooled = max(1, pooled // self.pool2)
        return self.F2 * pooled

    def _project_pca(self, x: np.ndarray) -> np.ndarray:
        if self._pca_components is None or self._pca_mean is None:
            raise RuntimeError("PCA not initialized.")
        return (x - self._pca_mean) @ self._pca_components.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._n_classes is None:
            raise RuntimeError("Model not initialized. Call fit() first.")
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, kernel_size=(1, self.pool1))
        x = self.dropout(x)
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, kernel_size=(1, self.pool2))
        x = self.dropout(x)
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits

    def fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        stride: int = 1,
        class_weighted: bool = True,
        materialize_windows: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        verbose: bool = True,
        log_epoch_metrics: bool = True,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        eval_every: int | None = None,
        eval_max_samples: int | None = None,
        **kwargs,
    ) -> None:
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self._build_classifier(n_classes)

        X = np.asarray(X, dtype=np.float32)
        if self.pca_components is not None:
            pca = PCA(n_components=self.pca_components, random_state=42)
            X_proj = pca.fit_transform(X)
            self._pca_components = pca.components_.astype(np.float32)
            self._pca_mean = pca.mean_.astype(np.float32)
        else:
            X_proj = X
        y_indices = np.array([class_to_idx[label] for label in y], dtype=np.int64)

        if materialize_windows:
            windows = np.lib.stride_tricks.sliding_window_view(
                X_proj, self.window_samples, axis=0
            )
            windows = windows.reshape(-1, self.window_samples, self.n_channels)
            stride_step = max(1, stride)
            windows = windows[::stride_step]
            labels = y_indices[self.window_samples - 1 :: stride_step]
            X_tensor = torch.tensor(np.ascontiguousarray(windows), dtype=torch.float32)
            X_tensor = X_tensor.permute(0, 2, 1).unsqueeze(1)
            y_tensor = torch.tensor(labels, dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        else:
            dataset = _EEGNetWindowDataset(
                X_proj,
                y_indices,
                window_samples=self.window_samples,
                stride=stride,
            )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.to(device)
        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate / 10
        )
        if class_weighted:
            class_counts = np.bincount(y_indices, minlength=n_classes).astype(np.float32)
            class_counts = np.maximum(class_counts, 1.0)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * n_classes
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        best_val_acc = None
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        epoch_iterator = tqdm(range(epochs), desc="Training", disable=not verbose)
        for epoch in epoch_iterator:
            total_loss = 0.0
            n_batches = 0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                logits = self.forward(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.numel()

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            train_acc = correct / max(total, 1)
            best_loss = min(best_loss, avg_loss)
            epoch_iterator.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{train_acc:.3f}")
            if log_epoch_metrics:
                logger.info(
                    "Epoch {}/{} - loss={:.4f} acc={:.3f}",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    train_acc,
                )
            if (
                eval_every is not None
                and eval_every > 0
                and X_val is not None
                and y_val is not None
                and (epoch + 1) % eval_every == 0
            ):
                val_acc = self._evaluate_windowed_accuracy(
                    X_val, y_val, batch_size=batch_size, max_samples=eval_max_samples
                )
                if best_val_acc is None or val_acc > best_val_acc:
                    best_val_acc = val_acc
                    checkpoint = {
                        "config": {
                            "n_channels": self.raw_channels,
                            "window_samples": self.window_samples,
                            "temporal_kernel": self.temporal_kernel,
                            "F1": self.F1,
                            "D": self.D,
                            "dropout": self.dropout_rate,
                            "pool1": self.pool1,
                            "pool2": self.pool2,
                            "pca_components": self.pca_components,
                            "n_classes": self._n_classes,
                        },
                        "classes": self.classes_,
                        "state_dict": self.state_dict(),
                        "pca": {
                            "components": self._pca_components,
                            "mean": self._pca_mean,
                        },
                        "epoch": epoch + 1,
                        "val_bal_acc": val_acc,
                    }
                    torch.save(checkpoint, BEST_CHECKPOINT_PATH)
                logger.info(
                    "Epoch {}/{} - val balanced acc={:.3f}",
                    epoch + 1,
                    epochs,
                    val_acc,
                )

        self.to("cpu")
        self.eval()
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")

    def _evaluate_windowed_accuracy(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int,
        max_samples: int | None = None,
    ) -> float:
        if X_val.shape[0] != y_val.shape[0]:
            raise ValueError("X_val and y_val must have the same length")
        device = next(self.parameters()).device
        n_eval = X_val.shape[0] if max_samples is None else min(X_val.shape[0], max_samples)
        X_val = np.asarray(X_val[:n_eval], dtype=np.float32)
        y_val = np.asarray(y_val[:n_eval]).reshape(-1)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_indices = np.array([class_to_idx[label] for label in y_val], dtype=np.int64)

        if self.pca_components is not None:
            if self._pca_components is None or self._pca_mean is None:
                raise RuntimeError("PCA not initialized.")
            X_proj = (X_val - self._pca_mean) @ self._pca_components.T
        else:
            X_proj = X_val

        dataset = _EEGNetWindowDataset(
            X_proj,
            y_indices,
            window_samples=self.window_samples,
            stride=1,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        self.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                logits = self.forward(X_batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy().tolist())

        return float(balanced_accuracy_score(all_labels, all_preds))

    def predict(self, X: np.ndarray) -> int:
        if self.classes_ is None:
            raise RuntimeError("Model not trained. Call fit() first or load a trained model.")
        device = next(self.parameters()).device
        x = np.asarray(X, dtype=np.float32)
        if x.shape != (self.raw_channels,):
            raise ValueError(f"Expected input shape ({self.raw_channels},), got {x.shape}")

        if self.pca_components is not None:
            if self._pca_components is None or self._pca_mean is None:
                raise RuntimeError("PCA not initialized.")
            x_proj = (x - self._pca_mean) @ self._pca_components.T
        else:
            x_proj = x

        if self._buffer is None:
            pad = np.zeros((self.window_samples - 1, self.n_channels), dtype=np.float32)
            self._buffer = np.concatenate([pad, x_proj[None, :]], axis=0)
        else:
            self._buffer = np.concatenate([self._buffer[1:], x_proj[None, :]], axis=0)

        window = self._buffer
        x_tensor = (
            torch.tensor(window.T, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_tensor)
            predicted_idx = int(torch.argmax(logits, dim=1).item())
        return int(self.classes_[predicted_idx])

    def reset_state(self) -> None:
        self._buffer = None

    def save(self) -> Path:
        if self.classes_ is None:
            raise RuntimeError("Cannot save untrained model. Call fit() first.")
        checkpoint = {
            "config": {
                "n_channels": self.raw_channels,
                "window_samples": self.window_samples,
                "temporal_kernel": self.temporal_kernel,
                "F1": self.F1,
                "D": self.D,
                "dropout": self.dropout_rate,
                "pool1": self.pool1,
                "pool2": self.pool2,
                "pca_components": self.pca_components,
                "n_classes": self._n_classes,
            },
            "classes": self.classes_,
            "state_dict": self.state_dict(),
            "pca": {
                "components": self._pca_components,
                "mean": self._pca_mean,
            },
        }
        torch.save(checkpoint, MODEL_PATH)
        logger.debug(f"Model saved to {MODEL_PATH}")
        return MODEL_PATH

    @classmethod
    def load(cls) -> Self:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, weights_only=False)
        config = checkpoint["config"]
        model = cls(
            n_channels=config["n_channels"],
            window_samples=config["window_samples"],
            temporal_kernel=config["temporal_kernel"],
            F1=config["F1"],
            D=config["D"],
            dropout=config["dropout"],
            pool1=config["pool1"],
            pool2=config["pool2"],
            pca_components=config.get("pca_components"),
        )
        model._build_classifier(config["n_classes"])
        model.classes_ = checkpoint["classes"]
        model.load_state_dict(checkpoint["state_dict"])
        pca_state = checkpoint.get("pca", {})
        model._pca_components = pca_state.get("components")
        model._pca_mean = pca_state.get("mean")
        model.eval()
        logger.debug(f"Model loaded from {MODEL_PATH}")
        return model


class _EEGNetWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window_samples: int,
        stride: int,
    ) -> None:
        self.X = X
        self.y = y
        self.window_samples = window_samples
        self.stride = max(1, stride)
        self._indices = np.arange(window_samples - 1, len(X), self.stride)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = self._indices[idx]
        start = end - self.window_samples + 1
        window = self.X[start : end + 1]
        x = torch.tensor(window.T, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.y[end], dtype=torch.long)
        return x, y

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for i in range(len(self)):
            yield self[i]
