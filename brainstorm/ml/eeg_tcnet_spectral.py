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
        self.spatial_conv = nn.Conv1d(F1, F1 * D, kernel_size=n_channels, groups=F1, bias=False)
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
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

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
        -> Spectral Feature Extraction (high-gamma, gamma, beta, etc.)
        -> Spatial Block (depthwise separable convs)
        -> TCN Blocks (dilated causal convs)
        -> Classifier
    """

    # Focus on most informative bands for ECoG
    DEFAULT_BANDS = {
        'high_gamma': (70, 150),  # Most important for ECoG
        'gamma': (30, 70),
        'beta': (13, 30),
    }

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        # Spectral parameters
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        include_raw: bool = True,
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

        self.n_channels = n_channels
        self.bands = bands or self.DEFAULT_BANDS
        self.include_raw = include_raw
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

        # Preprocessor (will be fitted during training)
        self.preprocessor = ECoGPreprocessor(
            use_spectral=True,
            bands=self.bands,
            include_raw=include_raw,
        )

        # Calculate input feature size
        n_bands = len(self.bands) + (1 if include_raw else 0)
        self.n_input_features = n_channels * n_bands

        # Spatial block
        self.spatial_block = SpatialBlock(
            n_input_features=self.n_input_features,
            n_channels=n_channels,
            F1=F1,
            D=D,
            F2=F2,
            dropout=dropout,
        )

        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        in_ch = F2
        for i in range(tcn_layers):
            dilation = 2 ** i
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

        # Classifier
        self.classifier: nn.Linear | None = None

        # Feature buffer for streaming
        self._feature_buffer: torch.Tensor | None = None

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
            raise RuntimeError("Model not initialized")

        batch_size, seq_len, n_features = x.shape

        # Apply spatial block to each timestep
        x = x.reshape(-1, n_features)
        x = self.spatial_block(x)  # (batch*seq_len, F2)

        # Reshape for TCN: (batch, F2, seq_len)
        x = x.reshape(batch_size, seq_len, -1).permute(0, 2, 1)

        # TCN blocks
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)

        # Classify: (batch, seq_len, n_classes)
        x = x.permute(0, 2, 1)
        return self.classifier(x)

    def predict(self, X: np.ndarray) -> int:
        """
        Predict for a single sample (streaming mode).

        Args:
            X: Raw sample of shape (n_channels,)

        Returns:
            Predicted label
        """
        if self.classes_ is None or self.classifier is None:
            raise RuntimeError("Model not trained")

        self.eval()
        with torch.no_grad():
            # Preprocess (extracts spectral features)
            features = self.preprocessor.transform_sample(X)
            features = torch.tensor(features, dtype=torch.float32)

            # Spatial features
            spatial_out = self.spatial_block(features.unsqueeze(0))  # (1, F2)

            # Update buffer
            if self._feature_buffer is None:
                self._feature_buffer = spatial_out.repeat(self.context_window, 1)
            else:
                self._feature_buffer = torch.cat([
                    self._feature_buffer[1:],
                    spatial_out
                ], dim=0)

            # TCN on buffer: (1, F2, context_window)
            x = self._feature_buffer.unsqueeze(0).permute(0, 2, 1)

            for tcn_block in self.tcn_blocks:
                x = tcn_block(x)

            # Classify last timestep
            x = x[:, :, -1]
            logits = self.classifier(x)
            pred_idx = int(torch.argmax(logits, dim=1).item())

        return int(self.classes_[pred_idx])

    def reset_buffer(self) -> None:
        """Reset streaming state."""
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

        # Fit preprocessor and transform training data
        logger.info("Extracting spectral features...")
        self.preprocessor.fit(X)
        X_features = self.preprocessor.transform(X)
        logger.info(f"Feature shape: {X_features.shape} (raw: {X.shape})")

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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate / 10
        )

        best_loss = float('inf')
        epoch_iter = tqdm(range(epochs), desc="Training", disable=not verbose)

        for epoch in epoch_iter:
            total_loss = 0.0
            n_batches = 0

            starts = np.random.permutation(n_samples - seq_len)[:n_sequences]

            for batch_start in range(0, len(starts), batch_size):
                batch_indices = starts[batch_start:batch_start + batch_size]

                X_batch = torch.stack([
                    X_tensor[i:i + seq_len] for i in batch_indices
                ])
                y_batch = torch.stack([
                    y_tensor[i:i + seq_len] for i in batch_indices
                ])

                optimizer.zero_grad()
                logits = self.forward(X_batch)
                loss = criterion(
                    logits.reshape(-1, n_classes),
                    y_batch.reshape(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
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

        checkpoint = {
            "config": {
                "n_channels": self.n_channels,
                "bands": self.bands,
                "include_raw": self.include_raw,
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
            "preprocessor": {
                "mean": self.preprocessor.mean,
                "std": self.preprocessor.std,
            },
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

        model = cls(
            n_channels=config["n_channels"],
            bands=config["bands"],
            include_raw=config["include_raw"],
            F1=config["F1"],
            D=config["D"],
            F2=config["F2"],
            tcn_channels=config["tcn_channels"],
            tcn_kernel_size=config["tcn_kernel_size"],
            tcn_layers=config["tcn_layers"],
            dropout=config["dropout"],
            context_window=config["context_window"],
        )

        model._build_classifier(config["n_classes"])
        model.classes_ = checkpoint["classes"]
        model.load_state_dict(checkpoint["state_dict"])

        # Restore preprocessor state
        model.preprocessor.mean = checkpoint["preprocessor"]["mean"]
        model.preprocessor.std = checkpoint["preprocessor"]["std"]
        model.preprocessor._n_raw_channels = config["n_channels"]

        model.eval()
        logger.debug(f"Model loaded from {MODEL_PATH}")
        return model
