"""
EEG-TCNet Model for continuous classification of ECoG signals.

This module implements a hybrid architecture combining:
1. EEGNet-style spatial feature extraction (depthwise separable convolutions)
2. TCN (Temporal Convolutional Network) for temporal pattern learning

Reference:
- EEG-TCNet: https://arxiv.org/abs/2006.00622
- EEGNet: https://arxiv.org/abs/1611.08024
- TCN: https://arxiv.org/abs/1803.01271
"""

from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from brainstorm.constants import N_CHANNELS
from brainstorm.ml.base import BaseModel


# Fixed model path within the repository
_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "model.pt"


class EEGNetBlock(nn.Module):
    """
    EEGNet-style spatial feature extraction block.

    Processes each time sample across all channels to extract spatial features.
    Uses depthwise separable convolutions for efficiency.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        F1: int = 8,           # Number of temporal filters
        D: int = 2,            # Depth multiplier for spatial filters
        F2: int = 16,          # Number of pointwise filters
        dropout: float = 0.3,
    ):
        super().__init__()

        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Temporal convolution: learns frequency filters
        # Input: (batch, 1, n_channels) -> (batch, F1, n_channels)
        self.temporal_conv = nn.Conv1d(1, F1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(F1)

        # Depthwise spatial convolution: learns spatial filters per temporal filter
        # Input: (batch, F1, n_channels) -> (batch, F1*D, 1)
        self.depthwise_conv = nn.Conv1d(
            F1, F1 * D, kernel_size=n_channels, groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(F1 * D)

        # Pointwise convolution: mixes features
        # Input: (batch, F1*D, 1) -> (batch, F2, 1)
        self.pointwise_conv = nn.Conv1d(F1 * D, F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(F2)

        self.dropout = nn.Dropout(dropout)
        self.output_size = F2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, n_channels)

        Returns:
            Output tensor of shape (batch, F2)
        """
        # Add channel dimension: (batch, n_channels) -> (batch, 1, n_channels)
        x = x.unsqueeze(1)

        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.bn1(x)

        # Depthwise spatial convolution
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Pointwise convolution
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Remove spatial dimension: (batch, F2, 1) -> (batch, F2)
        x = x.squeeze(-1)

        return x


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with dilated causal convolutions.

    Uses residual connections and handles sequences of spatial features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        dilation: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation

        # Causal padding: ensure output only depends on past inputs
        self.padding = (kernel_size - 1) * dilation

        # Dilated causal convolution
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

        # Residual connection (1x1 conv if dimensions don't match)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal convolution.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Output tensor of shape (batch, out_channels, seq_len)
        """
        residual = self.residual(x)

        # First conv block
        out = self.conv1(x)
        # Remove future padding (causal)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn1(out)
        out = F.elu(out)
        out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn2(out)
        out = F.elu(out)
        out = self.dropout(out)

        # Residual connection
        return F.elu(out + residual)


class EEGTCNet(BaseModel):
    """
    EEG-TCNet: Hybrid architecture combining EEGNet and TCN.

    Architecture:
        Input (n_channels)
        -> EEGNet block (spatial features)
        -> TCN blocks (temporal features)
        -> Linear classifier

    For streaming inference, the model maintains a buffer of past features
    and applies TCN causally.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        # EEGNet parameters
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        # TCN parameters
        tcn_channels: int = 16,
        tcn_kernel_size: int = 4,
        tcn_layers: int = 2,
        # General
        dropout: float = 0.3,
        # Context window for TCN (in samples)
        context_window: int = 64,
    ):
        super().__init__()

        self.n_channels = n_channels
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

        # Feature buffer for streaming inference
        self._feature_buffer: torch.Tensor | None = None

        # EEGNet block for spatial feature extraction
        self.eegnet = EEGNetBlock(
            n_channels=n_channels,
            F1=F1,
            D=D,
            F2=F2,
            dropout=dropout,
        )

        # TCN blocks with increasing dilation
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

        # Classifier (will be built after we know n_classes)
        self.classifier: nn.Linear | None = None

        # Normalization stats (computed during training)
        self.register_buffer('mean', torch.zeros(n_channels))
        self.register_buffer('std', torch.ones(n_channels))

    def _build_classifier(self, n_classes: int) -> None:
        """Build the classifier layer once n_classes is known."""
        self._n_classes = n_classes
        self.classifier = nn.Linear(self.tcn_channels, n_classes)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input features."""
        return (x - self.mean) / (self.std + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for batch training.

        Args:
            x: Input tensor of shape (batch, seq_len, n_channels)

        Returns:
            Logits tensor of shape (batch, seq_len, n_classes)
        """
        if self.classifier is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        batch_size, seq_len, n_channels = x.shape

        # Normalize
        x = self._normalize(x)

        # Apply EEGNet to each timestep
        # Reshape: (batch, seq_len, n_channels) -> (batch*seq_len, n_channels)
        x = x.reshape(-1, n_channels)
        x = self.eegnet(x)  # (batch*seq_len, F2)

        # Reshape for TCN: (batch*seq_len, F2) -> (batch, F2, seq_len)
        x = x.reshape(batch_size, seq_len, -1).permute(0, 2, 1)

        # Apply TCN blocks
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)

        # x shape: (batch, tcn_channels, seq_len)
        # Permute and classify: (batch, seq_len, tcn_channels)
        x = x.permute(0, 2, 1)
        logits = self.classifier(x)

        return logits

    def predict(self, X: np.ndarray) -> int:
        """
        Predict the label for a single sample (streaming inference).

        Maintains a buffer of past features for TCN context.

        Args:
            X: Feature array of shape (n_channels,) for a single timestep.

        Returns:
            Predicted label as an integer.
        """
        if self.classes_ is None or self.classifier is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        self.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)

            # Normalize
            x = self._normalize(x)

            # Extract spatial features with EEGNet
            features = self.eegnet(x.unsqueeze(0))  # (1, F2)

            # Update feature buffer
            if self._feature_buffer is None:
                self._feature_buffer = features.repeat(self.context_window, 1)
            else:
                self._feature_buffer = torch.cat([
                    self._feature_buffer[1:],
                    features
                ], dim=0)

            # Apply TCN on buffer
            # (context_window, F2) -> (1, F2, context_window)
            x = self._feature_buffer.unsqueeze(0).permute(0, 2, 1)

            for tcn_block in self.tcn_blocks:
                x = tcn_block(x)

            # Get prediction for last timestep
            x = x[:, :, -1]  # (1, tcn_channels)
            logits = self.classifier(x)
            predicted_idx = int(torch.argmax(logits, dim=1).item())

        return int(self.classes_[predicted_idx])

    def reset_buffer(self) -> None:
        """Reset the feature buffer (call between sequences)."""
        self._feature_buffer = None

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
        """
        Train the model.

        Args:
            X: Feature array of shape (n_samples, n_channels).
            y: Label array of shape (n_samples,).
            epochs: Number of training epochs.
            batch_size: Number of sequences per batch.
            seq_len: Length of each training sequence.
            learning_rate: Learning rate for Adam optimizer.
            weight_decay: L2 regularization weight.
            verbose: Whether to show progress.
        """
        # Compute normalization statistics
        self.mean = torch.tensor(X.mean(axis=0), dtype=torch.float32)
        self.std = torch.tensor(X.std(axis=0), dtype=torch.float32)

        # Determine classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        logger.info(f"Training EEG-TCNet with {n_classes} classes")

        # Build classifier
        self._build_classifier(n_classes)

        # Compute class weights for imbalanced data
        class_counts = np.bincount([class_to_idx[label] for label in y])
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * n_classes
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_indices = np.array([class_to_idx[label] for label in y])
        y_tensor = torch.tensor(y_indices, dtype=torch.long)

        # Create sequences for training
        n_samples = len(X)
        n_sequences = (n_samples - seq_len) // (seq_len // 2)  # 50% overlap

        # Setup training
        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate / 10
        )

        # Training loop
        best_loss = float('inf')
        epoch_iterator = tqdm(range(epochs), desc="Training", disable=not verbose)

        for epoch in epoch_iterator:
            total_loss = 0.0
            n_batches = 0

            # Shuffle sequence starting points
            starts = np.random.permutation(n_samples - seq_len)[:n_sequences]

            for batch_start in range(0, len(starts), batch_size):
                batch_indices = starts[batch_start:batch_start + batch_size]

                # Build batch of sequences
                X_batch = torch.stack([
                    X_tensor[i:i + seq_len] for i in batch_indices
                ])  # (batch, seq_len, n_channels)

                y_batch = torch.stack([
                    y_tensor[i:i + seq_len] for i in batch_indices
                ])  # (batch, seq_len)

                optimizer.zero_grad()
                logits = self.forward(X_batch)  # (batch, seq_len, n_classes)

                # Flatten for loss computation
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
            epoch_iterator.set_postfix(loss=f"{avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss

        self.eval()
        logger.info(f"Training complete. Best loss: {best_loss:.4f}")

    def save(self) -> Path:
        """Save the model."""
        if self.classes_ is None:
            raise RuntimeError("Cannot save untrained model.")

        checkpoint = {
            "config": {
                "n_channels": self.n_channels,
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
        }

        torch.save(checkpoint, MODEL_PATH)
        logger.debug(f"Model saved to {MODEL_PATH}")
        return MODEL_PATH

    @classmethod
    def load(cls) -> Self:
        """Load a model from file."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        checkpoint = torch.load(MODEL_PATH, weights_only=False)
        config = checkpoint["config"]

        model = cls(
            n_channels=config["n_channels"],
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
        model.eval()

        logger.debug(f"Model loaded from {MODEL_PATH}")
        return model
