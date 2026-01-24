"""
CNN+GRU model with built-in preprocessing for streaming ECoG classification.

Preprocessing (causal):
1) Common average reference (CAR)
2) Bandpass -> power -> EMA -> log
3) Per-band, per-channel z-score using training stats
4) Map to 31x32 grid, pad to 32x32
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.signal import butter, lfilter, sosfilt
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from brainstorm.constants import GRID_HEIGHT, GRID_WIDTH, N_CHANNELS
from brainstorm.loading import load_channel_coordinates
from brainstorm.ml.base import BaseModel


_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "cnn_gru.pt"


@dataclass(frozen=True)
class NormalizerStats:
    mean: np.ndarray
    std: np.ndarray


def _design_bandpass_sos(
    bands_hz: Sequence[tuple[float, float]],
    fs_hz: float,
    order: int = 4,
) -> list[np.ndarray]:
    nyq = fs_hz / 2.0
    sos_bank: list[np.ndarray] = []
    for lo_hz, hi_hz in bands_hz:
        lo = lo_hz / nyq
        hi = hi_hz / nyq
        sos = butter(N=order, Wn=[lo, hi], btype="bandpass", output="sos")
        sos_bank.append(sos)
    return sos_bank


def _fit_normalizer(features: np.ndarray, eps: float = 1e-6) -> NormalizerStats:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + eps
    return NormalizerStats(mean=mean, std=std)


def _apply_normalizer(features: np.ndarray, stats: NormalizerStats) -> np.ndarray:
    return (features - stats.mean) / stats.std


def _build_channel_map(channels_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if channels_coords.shape != (N_CHANNELS, 2):
        raise ValueError(
            f"channels_coords must have shape ({N_CHANNELS}, 2), got {channels_coords.shape}"
        )
    y_offset = 2
    x = channels_coords[:, 0].astype(int)
    y = channels_coords[:, 1].astype(int) - y_offset
    valid = (x >= 0) & (x < GRID_WIDTH) & (y >= 0) & (y < GRID_HEIGHT)
    channel_idx = np.where(valid)[0]
    return channel_idx, x[valid], y[valid]


def _spatialize_features(
    features: np.ndarray,
    channel_idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError(f"features must have shape (T,B,C), got {features.shape}")
    t, b, _ = features.shape
    grid = np.zeros((t, b, GRID_HEIGHT, GRID_WIDTH), dtype=features.dtype)
    grid[:, :, y, x] = features[:, :, channel_idx]
    return grid


def _pad_to_32(spatial: np.ndarray) -> np.ndarray:
    if spatial.shape[2] == 32 and spatial.shape[3] == 32:
        return spatial
    h_pad = 32 - spatial.shape[2]
    w_pad = 32 - spatial.shape[3]
    return np.pad(spatial, ((0, 0), (0, 0), (0, h_pad), (0, w_pad)), mode="constant")


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.elu(x)


class CNNGRU(BaseModel):
    def __init__(
        self,
        bands_hz: Sequence[tuple[float, float]] | None = None,
        fs_hz: float = 1000.0,
        tau_s: float = 0.05,
        filter_order: int = 4,
        cnn_channels: tuple[int, int] = (16, 48),
        gru_hidden: int = 96,
        dropout: float = 0.3,
        pool_size: tuple[int, int] = (4, 4),
        use_mask: bool = True,
        high_gamma_band: tuple[float, float] = (70.0, 200.0),
        high_gamma_taus: Sequence[float] | None = (0.02, 0.2),
    ) -> None:
        super().__init__()
        self.fs_hz = fs_hz
        self.tau_s = tau_s
        self.filter_order = filter_order
        self.bands_hz = list(bands_hz) if bands_hz is not None else [
            (1.0, 4.0),
            (4.0, 8.0),
            (8.0, 13.0),
            (13.0, 35.0),
            (35.0, 70.0),
            (70.0, 200.0),
        ]
        self.high_gamma_band = high_gamma_band
        self.high_gamma_taus = list(high_gamma_taus) if high_gamma_taus else None
        self._unique_bands = list(dict.fromkeys(self.bands_hz))
        self._band_features: list[tuple[int, float]] = []
        band_idx_map = {band: i for i, band in enumerate(self._unique_bands)}
        for band in self.bands_hz:
            band_idx = band_idx_map[band]
            if self.high_gamma_taus and band == self.high_gamma_band:
                for tau in self.high_gamma_taus:
                    self._band_features.append((band_idx, float(tau)))
            else:
                self._band_features.append((band_idx, self.tau_s))
        self._feature_taus = [tau for _, tau in self._band_features]
        self._band_feature_indices: list[list[int]] = [
            [] for _ in range(len(self._unique_bands))
        ]
        for feature_idx, (band_idx, _) in enumerate(self._band_features):
            self._band_feature_indices[band_idx].append(feature_idx)
        self.n_bands = len(self._band_features)
        self.cnn_channels = cnn_channels
        self.gru_hidden = gru_hidden
        self.dropout_rate = dropout
        self.pool_size = pool_size
        self.use_mask = use_mask
        self.input_channels = self.n_bands + (1 if self.use_mask else 0)

        self.classes_: np.ndarray | None = None
        self._n_classes: int | None = None
        self._stats: NormalizerStats | None = None
        self._channel_idx: np.ndarray | None = None
        self._channel_x: np.ndarray | None = None
        self._channel_y: np.ndarray | None = None
        self._mask_grid: np.ndarray | None = None

        self._sos_bank: list[np.ndarray] | None = None
        self._sos_state: list[np.ndarray] | None = None
        self._ema_state: np.ndarray | None = None
        self._gru_state: torch.Tensor | None = None

        self.conv1 = nn.Conv2d(
            self.input_channels, cnn_channels[0], kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(cnn_channels[0])
        self.dwsep = DepthwiseSeparableConv2d(cnn_channels[0], cnn_channels[1])
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)
        pooled_dim = cnn_channels[1] * self.pool_size[0] * self.pool_size[1]
        self.gru = nn.GRU(
            input_size=pooled_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(gru_hidden, 1)

    def _build_classifier(self, n_classes: int) -> None:
        self._n_classes = n_classes
        self.classifier = nn.Linear(self.gru_hidden, n_classes)

    def _init_preprocessing(self) -> None:
        if self._sos_bank is None:
            self._sos_bank = _design_bandpass_sos(
                self._unique_bands, fs_hz=self.fs_hz, order=self.filter_order
            )
        if self._channel_idx is None:
            coords = load_channel_coordinates()
            channel_idx, x, y = _build_channel_map(coords)
            self._channel_idx = channel_idx
            self._channel_x = x
            self._channel_y = y
        if self._mask_grid is None and self.use_mask:
            mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
            mask[self._channel_y, self._channel_x] = 1.0
            self._mask_grid = _pad_to_32(mask[None, None, :, :])[0, 0]

    def _preprocess_batch(self, x: np.ndarray) -> tuple[np.ndarray, NormalizerStats]:
        x = x.astype(np.float32, copy=False)
        x = x - x.mean(axis=1, keepdims=True)
        sos_bank = self._sos_bank or _design_bandpass_sos(
            self._unique_bands, fs_hz=self.fs_hz, order=self.filter_order
        )
        n_samples, n_ch = x.shape
        features = np.empty((n_samples, self.n_bands, n_ch), dtype=np.float32)
        filter_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        for band_idx, sos in enumerate(sos_bank):
            y = sosfilt(sos, x, axis=0).astype(np.float32)
            p = y * y
            for feature_idx in self._band_feature_indices[band_idx]:
                tau = self._feature_taus[feature_idx]
                if tau not in filter_cache:
                    alpha = 1.0 - np.exp(-1.0 / (self.fs_hz * tau))
                    b = np.array([alpha], dtype=np.float32)
                    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float32)
                    filter_cache[tau] = (b, a)
                b, a = filter_cache[tau]
                env = lfilter(b, a, p, axis=0).astype(np.float32)
                features[:, feature_idx, :] = np.log(env + 1e-8)

        stats = _fit_normalizer(features)
        features = _apply_normalizer(features, stats)

        if self._channel_idx is None or self._channel_x is None or self._channel_y is None:
            coords = load_channel_coordinates()
            self._channel_idx, self._channel_x, self._channel_y = _build_channel_map(coords)
        spatial = _spatialize_features(
            features, self._channel_idx, self._channel_x, self._channel_y
        )
        spatial = _pad_to_32(spatial)
        if self.use_mask:
            if self._mask_grid is None:
                raise RuntimeError("Mask grid not initialized.")
            t = spatial.shape[0]
            mask = np.broadcast_to(self._mask_grid, (t, 1, 32, 32))
            spatial = np.concatenate([spatial, mask], axis=1)
        return spatial, stats

    def _preprocess_step(self, x: np.ndarray) -> np.ndarray:
        if self._sos_bank is None or self._stats is None:
            raise RuntimeError("Preprocessing not initialized. Train or load the model first.")
        if self._sos_state is None or self._ema_state is None:
            self._init_streaming_state()

        x = x.astype(np.float32, copy=False)
        x = x - x.mean()

        feats = np.empty((self.n_bands, x.shape[0]), dtype=np.float32)
        alpha_cache: dict[float, float] = {}

        for band_idx, sos in enumerate(self._sos_bank):
            zi = self._sos_state[band_idx]
            y, zf = sosfilt(sos, x[None, :], axis=0, zi=zi)
            self._sos_state[band_idx] = zf
            power = y[0] * y[0]
            for feature_idx in self._band_feature_indices[band_idx]:
                tau = self._feature_taus[feature_idx]
                if tau not in alpha_cache:
                    alpha_cache[tau] = 1.0 - np.exp(-1.0 / (self.fs_hz * tau))
                alpha = alpha_cache[tau]
                ema_prev = self._ema_state[feature_idx]
                ema = (1.0 - alpha) * ema_prev + alpha * power
                self._ema_state[feature_idx] = ema
                feats[feature_idx] = np.log(ema + 1e-8)

        feats = _apply_normalizer(feats[None, :, :], self._stats)[0]

        if self._channel_idx is None or self._channel_x is None or self._channel_y is None:
            raise RuntimeError("Channel map missing. Train or load the model first.")
        grid = np.zeros((self.n_bands, GRID_HEIGHT, GRID_WIDTH), dtype=feats.dtype)
        grid[:, self._channel_y, self._channel_x] = feats[:, self._channel_idx]
        grid = _pad_to_32(grid[None, :, :, :])[0]
        if self.use_mask:
            if self._mask_grid is None:
                raise RuntimeError("Mask grid not initialized.")
            grid = np.concatenate([grid, self._mask_grid[None, :, :]], axis=0)
        return grid

    def _init_streaming_state(self) -> None:
        if self._sos_bank is None:
            raise RuntimeError("Filters not initialized.")
        n_sections = self._sos_bank[0].shape[0]
        self._sos_state = [
            np.zeros((n_sections, 2, N_CHANNELS), dtype=np.float32)
            for _ in self._sos_bank
        ]
        self._ema_state = np.zeros((self.n_bands, N_CHANNELS), dtype=np.float32)
        self._gru_state = None

    def reset_state(self) -> None:
        self._sos_state = None
        self._ema_state = None
        self._gru_state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._n_classes is None:
            raise RuntimeError("Model not initialized. Call fit() first.")
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B,T,C,H,W), got {x.shape}")
        batch, seq_len, channels, height, width = x.shape
        x = x.reshape(batch * seq_len, channels, height, width)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.dwsep(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = x.reshape(batch, seq_len, -1)
        out, _ = self.gru(x)
        logits = self.classifier(out)
        return logits

    def fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 36,
        batch_size: int = 16,
        seq_len: int = 96,
        stride: int | None = None,
        chunk_len: int | None = None,
        chunks_per_epoch: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        stratify_sequences: bool = False,
        stratify_ratio: float = 0.45,
        loss_last_k: int = 1,
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

        self._init_preprocessing()
        spatial, stats = self._preprocess_batch(X)
        self._stats = stats

        y_indices = np.array([class_to_idx[label] for label in y])
        y_tensor = torch.tensor(y_indices, dtype=torch.long)
        x_tensor = torch.tensor(spatial, dtype=torch.float32)

        n_samples = x_tensor.shape[0]
        stride = stride or (seq_len * 3 // 4)
        if chunk_len is not None:
            if chunk_len <= seq_len:
                raise ValueError("chunk_len must be greater than seq_len")
            if chunk_len > n_samples:
                raise ValueError("chunk_len must be <= number of samples")

        nonzero = (y != 0).astype(np.int32)
        nonzero_classes = [c for c in self.classes_ if c != 0]
        nonzero_classes = np.array(nonzero_classes)

        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        class_counts = np.bincount(y_indices, minlength=n_classes).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)
        class_weights = class_counts.sum() / class_counts
        class_weights = class_weights / class_weights.mean()
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate / 10
        )

        best_loss = float("inf")
        epoch_iterator = tqdm(range(epochs), desc="Training", disable=not verbose)
        seq_offsets = torch.arange(seq_len, dtype=torch.long)

        def _build_starts_for_chunk(chunk_start: int, chunk_end: int) -> np.ndarray:
            starts_all = np.arange(chunk_start, chunk_end - seq_len, stride)
            if len(starts_all) == 0:
                return starts_all
            n_sequences = len(starts_all)
            if stratify_sequences:
                n_balanced = max(1, int(n_sequences * stratify_ratio))
                per_class = max(1, n_balanced // max(len(nonzero_classes), 1))
                k = max(1, min(loss_last_k, seq_len))

                balanced_starts = []
                for cls in nonzero_classes:
                    idx = np.where(y == cls)[0]
                    if idx.size == 0:
                        continue
                    starts = idx - (seq_len - k)
                    starts = np.clip(starts, chunk_start, chunk_end - seq_len - 1)
                    starts = starts - ((starts - chunk_start) % stride)
                    valid = (idx - starts) >= (seq_len - k)
                    valid &= (idx - starts) < seq_len
                    starts = starts[valid]
                    starts = starts[
                        (starts >= chunk_start)
                        & (starts <= (chunk_end - seq_len - 1))
                    ]
                    if starts.size == 0:
                        continue
                    starts = np.unique(starts)
                    take = min(per_class, starts.size)
                    balanced_starts.append(np.random.permutation(starts)[:take])
                if balanced_starts:
                    balanced_starts = np.concatenate(balanced_starts)
                else:
                    balanced_starts = np.array([], dtype=int)

                n_random = n_sequences - len(balanced_starts)
                if n_random > 0:
                    random_starts = np.random.permutation(starts_all)[:n_random]
                    starts = np.concatenate([balanced_starts, random_starts])
                else:
                    starts = balanced_starts[:n_sequences]
                return np.random.permutation(starts)

            return np.random.permutation(starts_all)

        for epoch in epoch_iterator:
            total_loss = 0.0
            n_batches = 0
            correct = 0
            total = 0

            if chunk_len is None:
                chunk_starts = [_build_starts_for_chunk(0, n_samples)]
            else:
                chunk_starts = []
                max_start = n_samples - chunk_len
                for _ in range(chunks_per_epoch):
                    start = int(np.random.randint(0, max_start + 1))
                    end = start + chunk_len
                    starts = _build_starts_for_chunk(start, end)
                    if len(starts) > 0:
                        chunk_starts.append(starts)

            for starts in chunk_starts:
                for batch_start in range(0, len(starts), batch_size):
                    batch_indices = starts[batch_start : batch_start + batch_size]
                    batch_indices_t = torch.as_tensor(batch_indices, dtype=torch.long)
                    idx = batch_indices_t[:, None] + seq_offsets[None, :]

                    X_batch = x_tensor[idx]
                    y_batch = y_tensor[idx]

                    optimizer.zero_grad()
                    logits = self.forward(X_batch)

                    k = max(1, min(loss_last_k, seq_len))
                    logits_k = logits[:, -k:, :].reshape(-1, n_classes)
                    y_k = y_batch[:, -k:].reshape(-1)
                    loss = criterion(logits_k, y_k)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

                    with torch.no_grad():
                        preds = torch.argmax(logits[:, -k:, :], dim=-1)
                        correct += (preds == y_batch[:, -k:]).sum().item()
                        total += y_batch[:, -k:].numel()

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            train_acc = correct / max(total, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
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
                val_acc = self._evaluate_streaming_accuracy(
                    X_val, y_val, max_samples=eval_max_samples
                )
                logger.info(
                    "Epoch {}/{} - val balanced acc={:.3f}",
                    epoch + 1,
                    epochs,
                    val_acc,
                )

        self.eval()
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")

    def _evaluate_streaming_accuracy(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_samples: int | None = None,
    ) -> float:
        if X_val.shape[0] != y_val.shape[0]:
            raise ValueError("X_val and y_val must have the same length")
        n_eval = X_val.shape[0] if max_samples is None else min(X_val.shape[0], max_samples)
        self.reset_state()
        preds = []
        for i in range(n_eval):
            preds.append(self.predict(X_val[i]))
        y_true = y_val[:n_eval]
        return float(balanced_accuracy_score(y_true, np.array(preds)))

    def predict(self, X: np.ndarray) -> int:
        if self.classes_ is None or self._stats is None:
            raise RuntimeError("Model not trained. Call fit() first or load a trained model.")
        if self._gru_state is None:
            self._gru_state = torch.zeros(1, 1, self.gru_hidden)
        grid = self._preprocess_step(X)
        x_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        x_tensor = x_tensor.unsqueeze(1)
        self.eval()
        with torch.no_grad():
            x = self.conv1(x_tensor[:, 0])
            x = self.bn1(x)
            x = F.elu(x)
            x = self.dropout(x)
            x = self.dwsep(x)
            x = self.dropout(x)
            x = self.pool(x)
            x = x.flatten(1)
            x = x.unsqueeze(1)
            out, self._gru_state = self.gru(x, self._gru_state)
            logits = self.classifier(out[:, -1, :])
            predicted_idx = int(torch.argmax(logits, dim=1).item())
        return int(self.classes_[predicted_idx])

    def save(self) -> Path:
        if self.classes_ is None or self._stats is None:
            raise RuntimeError("Cannot save untrained model. Call fit() first.")
        checkpoint = {
            "config": {
                "bands_hz": self.bands_hz,
                "fs_hz": self.fs_hz,
                "tau_s": self.tau_s,
                "filter_order": self.filter_order,
                "cnn_channels": self.cnn_channels,
                "gru_hidden": self.gru_hidden,
                "dropout": self.dropout_rate,
                "pool_size": self.pool_size,
                "use_mask": self.use_mask,
                "high_gamma_band": self.high_gamma_band,
                "high_gamma_taus": self.high_gamma_taus,
                "n_classes": self._n_classes,
            },
            "classes": self.classes_,
            "state_dict": self.state_dict(),
            "stats": {"mean": self._stats.mean, "std": self._stats.std},
            "channel_map": {
                "channel_idx": self._channel_idx,
                "x": self._channel_x,
                "y": self._channel_y,
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
            bands_hz=config["bands_hz"],
            fs_hz=config["fs_hz"],
            tau_s=config["tau_s"],
            filter_order=config["filter_order"],
            cnn_channels=tuple(config["cnn_channels"]),
            gru_hidden=config["gru_hidden"],
            dropout=config["dropout"],
            pool_size=tuple(config.get("pool_size", (4, 4))),
            use_mask=config.get("use_mask", True),
            high_gamma_band=tuple(config.get("high_gamma_band", (70.0, 200.0))),
            high_gamma_taus=config.get("high_gamma_taus"),
        )
        model._build_classifier(config["n_classes"])
        model.classes_ = checkpoint["classes"]
        model.load_state_dict(checkpoint["state_dict"])
        model._stats = NormalizerStats(
            mean=checkpoint["stats"]["mean"], std=checkpoint["stats"]["std"]
        )
        model._channel_idx = checkpoint["channel_map"]["channel_idx"]
        model._channel_x = checkpoint["channel_map"]["x"]
        model._channel_y = checkpoint["channel_map"]["y"]
        model._init_preprocessing()
        model.eval()
        logger.debug(f"Model loaded from {MODEL_PATH}")
        return model
