"""
EEG-TCNet with Spectral Preprocessing.

Extends EEG-TCNet with causal spectral feature extraction:
- High-gamma band power (70-150 Hz) - most informative for ECoG
- Other frequency bands (gamma, beta, alpha, etc.)
- Combines raw signal with spectral features
"""

from pathlib import Path
from typing import Self, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from brainstorm.constants import N_CHANNELS
from brainstorm.ml.base import BaseModel
from brainstorm.ml.preprocessing import ECoGPreprocessor


_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "model.pt"


class SpatialBlock(nn.Module):
    """
    Spatial feature extraction using depthwise separable convolutions.
    Adapted for variable input sizes (raw + spectral features).
    """

    def __init__(
        self,
        n_input_features: int,
        n_channels: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_bands = n_input_features // n_channels

        # Process each band with shared weights
        self.temporal_conv = nn.Conv1d(self.n_bands, F1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(F1)

        # Spatial convolution across channels
        self.spatial_conv = nn.Conv1d(
            F1, F1 * D, kernel_size=n_channels, groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(F1 * D)

        # Pointwise
        self.pointwise = nn.Conv1d(F1 * D, F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(F2)

        self.dropout = nn.Dropout(dropout)
        self.output_size = F2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_input_features) where n_input_features = n_channels * n_bands

        Returns:
            (batch, F2)
        """
        batch_size = x.shape[0]
        n_channels = x.shape[1] // self.n_bands

        # Reshape: (batch, n_bands, n_channels)
        x = x.view(batch_size, self.n_bands, n_channels)

        # Temporal conv across bands
        x = self.temporal_conv(x)
        x = self.bn1(x)

        # Spatial conv across channels
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Pointwise
        x = self.pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        # (batch, F2, 1) -> (batch, F2)
        return x.squeeze(-1)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with dilated causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        dilation: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn1(out)
        out = F.elu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn2(out)
        out = F.elu(out)
        out = self.dropout(out)

        return F.elu(out + residual)


class EEGTCNetSpectral(BaseModel):
    """
    EEG-TCNet with spectral preprocessing.

    Architecture:
        Raw ECoG (1024 channels)
        -> (OPTIONAL) PCA channel reduction (e.g., 1024 -> 64)   # CHANGE
        -> Spectral Feature Extraction (high-gamma, gamma, beta, etc.)
        -> Spatial Block (depthwise separable convs)
        -> TCN Blocks (dilated causal convs)
        -> Classifier
    """

    DEFAULT_BANDS = {
        "high_gamma": (70, 150),
        "gamma": (30, 70),
        "beta": (13, 30),
    }

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        # Spectral parameters
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        include_raw: bool = True,
        # PCA parameters                                                   # CHANGE
        n_pca_components: int = 64,                                       # CHANGE
        # Spatial block parameters
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        # TCN parameters
        tcn_channels: int = 16,
        tcn_kernel_size: int = 4,
        tcn_layers: int = 2,
        # General
        dropout: float = 0.3,
        context_window: int = 64,
    ):
        super().__init__()

        # CHANGE: keep track of raw channel count separately
        self.raw_n_channels = int(n_channels)                             # CHANGE
        self.n_channels = int(n_channels)                                 # (will become post-PCA after fit)

        self.bands = bands or self.DEFAULT_BANDS
        self.include_raw = include_raw
        self.n_pca_components = int(n_pca_components)                     # CHANGE

        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.tcn_channels = tcn_channels
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_layers = tcn_layers
        self.dropout_rate = dropout
        self.context_window = context_window

        self.classes_: np.ndarray | None = None
        self._n_classes: int | None = None

        # CHANGE: pass PCA config down to the preprocessor
        self.preprocessor = ECoGPreprocessor(
            use_spectral=True,
            bands=self.bands,
            include_raw=include_raw,
            n_pca_components=self.n_pca_components,                       # CHANGE
        )

        # CHANGE: DON'T build SpatialBlock yet (we don't know post-PCA channel count until fit())
        self.n_input_features: Optional[int] = None                       # CHANGE
        self.spatial_block: Optional[SpatialBlock] = None                 # CHANGE

        # TCN blocks (independent of input channel count)
        self.tcn_blocks = nn.ModuleList()
        in_ch = F2
        for i in range(tcn_layers):
            dilation = 2**i
            self.tcn_blocks.append(
                TCNBlock(
                    in_channels=in_ch,
                    out_channels=tcn_channels,
                    kernel_size=tcn_kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = tcn_channels

        self.classifier: nn.Linear | None = None

        # Feature buffer for streaming
        self._feature_buffer: torch.Tensor | None = None

    # CHANGE: helper to build spatial block once we know post-PCA channel count + feature size
    def _build_spatial_block(self) -> None:                               # CHANGE
        out_ch = self.preprocessor.n_out_channels                          # post-PCA channels
        n_bands = len(self.bands) + (1 if self.include_raw else 0)
        self.n_channels = int(out_ch)                                      # CHANGE: model now operates on post-PCA channels
        self.n_input_features = int(out_ch * n_bands)

        self.spatial_block = SpatialBlock(
            n_input_features=self.n_input_features,
            n_channels=int(out_ch),
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            dropout=self.dropout_rate,
        )

    def _build_classifier(self, n_classes: int) -> None:
        self._n_classes = n_classes
        self.classifier = nn.Linear(self.tcn_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            x: (batch, seq_len, n_input_features) - preprocessed features

        Returns:
            (batch, seq_len, n_classes)
        """
        if self.classifier is None:
            raise RuntimeError("Model not initialized (classifier missing).")
        if self.spatial_block is None or self.n_input_features is None:     # CHANGE
            raise RuntimeError("Spatial block not initialized. Call fit_model() first.")

        batch_size, seq_len, n_features = x.shape
        if n_features != self.n_input_features:                             # CHANGE: clearer error if wiring is wrong
            raise ValueError(
                f"Expected n_features={self.n_input_features}, got {n_features}. "
                "Did you change PCA/bands/include_raw without rebuilding the model?"
            )

        x = x.reshape(-1, n_features)
        x = self.spatial_block(x)  # (batch*seq_len, F2)

        x = x.reshape(batch_size, seq_len, -1).permute(0, 2, 1)

        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)

        x = x.permute(0, 2, 1)
        return self.classifier(x)

    def predict(self, X: np.ndarray) -> int:
        """
        Predict for a single sample (streaming mode).

        Args:
            X: Raw sample of shape (raw_n_channels,)

        Returns:
            Predicted label
        """
        if self.classes_ is None or self.classifier is None:
            raise RuntimeError("Model not trained")
        if self.spatial_block is None or self.n_input_features is None:     # CHANGE
            raise RuntimeError("Spatial block not initialized. Did you load/fit the model?")

        self.eval()
        with torch.no_grad():
            # Preprocess (includes PCA if enabled + spectral features)
            features = self.preprocessor.transform_sample(X)
            if features.shape[0] != self.n_input_features:                  # CHANGE
                raise ValueError(
                    f"Preprocessor returned {features.shape[0]} features, expected {self.n_input_features}."
                )
            features_t = torch.tensor(features, dtype=torch.float32)

            spatial_out = self.spatial_block(features_t.unsqueeze(0))  # (1, F2)

            if self._feature_buffer is None:
                self._feature_buffer = spatial_out.repeat(self.context_window, 1)
            else:
                self._feature_buffer = torch.cat([self._feature_buffer[1:], spatial_out], dim=0)

            x = self._feature_buffer.unsqueeze(0).permute(0, 2, 1)

            for tcn_block in self.tcn_blocks:
                x = tcn_block(x)

            x = x[:, :, -1]
            logits = self.classifier(x)
            pred_idx = int(torch.argmax(logits, dim=1).item())

        return int(self.classes_[pred_idx])

    def reset_buffer(self) -> None:
        self._feature_buffer = None
        self.preprocessor.reset_streaming()

    def fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        seq_len: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """Train the model."""

        logger.info("Fitting preprocessor (norm + optional PCA + spectral)...")  # CHANGE
        self.preprocessor.fit(X)

        # CHANGE: now that preprocessor is fitted, build spatial block using post-PCA dimensionality
        self._build_spatial_block()                                              # CHANGE
        assert self.spatial_block is not None and self.n_input_features is not None

        logger.info(
            f"Channels: raw={self.raw_n_channels} -> out={self.preprocessor.n_out_channels} "
            f"(PCA={'on' if self.preprocessor.pca is not None else 'off'})"
        )  # CHANGE

        logger.info("Extracting features...")
        X_features = self.preprocessor.transform(X)
        logger.info(f"Feature shape: {X_features.shape} (raw: {X.shape})")
        if X_features.shape[1] != self.n_input_features:                           # CHANGE
            raise RuntimeError(
                f"Feature dim mismatch: got {X_features.shape[1]} but model expects {self.n_input_features}"
            )

        # Setup classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        logger.info(f"Training with {n_classes} classes")

        self._build_classifier(n_classes)

        # Class weights
        class_counts = np.bincount([class_to_idx[label] for label in y])
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * n_classes
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # Convert to tensors
        X_tensor = torch.tensor(X_features, dtype=torch.float32)
        y_indices = np.array([class_to_idx[label] for label in y])
        y_tensor = torch.tensor(y_indices, dtype=torch.long)

        # Training setup
        n_samples = len(X)
        n_sequences = (n_samples - seq_len) // (seq_len // 2)

        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate / 10
        )

        best_loss = float("inf")
        epoch_iter = tqdm(range(epochs), desc="Training", disable=not verbose)

        for _epoch in epoch_iter:
            total_loss = 0.0
            n_batches = 0

            starts = np.random.permutation(n_samples - seq_len)[:n_sequences]

            for batch_start in range(0, len(starts), batch_size):
                batch_indices = starts[batch_start : batch_start + batch_size]

                X_batch = torch.stack([X_tensor[i : i + seq_len] for i in batch_indices])
                y_batch = torch.stack([y_tensor[i : i + seq_len] for i in batch_indices])

                optimizer.zero_grad()
                logits = self.forward(X_batch)
                loss = criterion(logits.reshape(-1, n_classes), y_batch.reshape(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss

        self.eval()
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")

    def save(self) -> Path:
        """Save model and preprocessor state."""
        if self.classes_ is None:
            raise RuntimeError("Cannot save untrained model")

        if self.spatial_block is None or self.n_input_features is None:          # CHANGE
            raise RuntimeError("Cannot save before spatial block is built (call fit_model first).")

        checkpoint = {
            "config": {
                # CHANGE: save both raw and post-PCA channel counts
                "raw_n_channels": self.raw_n_channels,                           # CHANGE
                "n_channels": self.n_channels,                                   # post-PCA (after fit)   # CHANGE
                "bands": self.bands,
                "include_raw": self.include_raw,
                "n_pca_components": self.n_pca_components,                       # CHANGE
                "F1": self.F1,
                "D": self.D,
                "F2": self.F2,
                "tcn_channels": self.tcn_channels,
                "tcn_kernel_size": self.tcn_kernel_size,
                "tcn_layers": self.tcn_layers,
                "dropout": self.dropout_rate,
                "context_window": self.context_window,
                "n_classes": self._n_classes,
            },
            "classes": self.classes_,
            "state_dict": self.state_dict(),
            # CHANGE: save full preprocessor state (includes PCA matrices + stats)
            "preprocessor_state": self.preprocessor.get_state(),                 # CHANGE
        }

        torch.save(checkpoint, MODEL_PATH)
        logger.debug(f"Model saved to {MODEL_PATH}")
        return MODEL_PATH

    @classmethod
    def load(cls) -> Self:
        """Load model from file."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        checkpoint = torch.load(MODEL_PATH, weights_only=False)
        config = checkpoint["config"]

        # CHANGE: initialize with raw_n_channels (preprocessor will handle PCA)
        model = cls(
            n_channels=config.get("raw_n_channels", config["n_channels"]),       # CHANGE
            bands=config["bands"],
            include_raw=config["include_raw"],
            n_pca_components=config.get("n_pca_components", 64),                 # CHANGE
            F1=config["F1"],
            D=config["D"],
            F2=config["F2"],
            tcn_channels=config["tcn_channels"],
            tcn_kernel_size=config["tcn_kernel_size"],
            tcn_layers=config["tcn_layers"],
            dropout=config["dropout"],
            context_window=config["context_window"],
        )

        # CHANGE: restore preprocessor fully (including PCA) BEFORE building SpatialBlock
        model.preprocessor.set_state(checkpoint["preprocessor_state"])            # CHANGE

        # CHANGE: now we can build spatial block with correct post-PCA dimensionality
        model._build_spatial_block()                                              # CHANGE

        model._build_classifier(config["n_classes"])
        model.classes_ = checkpoint["classes"]
        model.load_state_dict(checkpoint["state_dict"])

        model.eval()
        logger.debug(f"Model loaded from {MODEL_PATH}")
        return model
