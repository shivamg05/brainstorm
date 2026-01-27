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

from brainstorm.ml.base import BaseModel


# Fixed model path within the repository
_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "eeg_tcnet.pt"

def _get_device() -> torch.device:
    return torch.device("cpu")


def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Create GroupNorm with a safe group count for the channel size."""
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class EEGNetBlock(nn.Module):
    """
    EEGNet-style feature extraction block over time + space.

    Processes sequences of spatial grids to extract temporal and spatial features
    using depthwise separable convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        temporal_kernel: int = 32,
        F1: int = 8,           # Number of temporal filters
        D: int = 2,            # Depth multiplier for spatial filters
        F2: int = 16,          # Number of pointwise filters
        F2_bottleneck: int | None = 8,  # Optional channel bottleneck before TCN
        dropout: float = 0.3,
        use_mask: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.temporal_kernel = temporal_kernel
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.F2_bottleneck = F2_bottleneck
        self.use_mask = use_mask
        if self.use_mask and self.in_channels < 2:
            raise ValueError("use_mask requires input_channels >= 2")
        self.conv_in_channels = self.in_channels - 1 if self.use_mask else self.in_channels

        # Temporal convolution: learns frequency filters
        # Input: (batch, in_channels, time, H, W) -> (batch, F1, time, H, W)
        self.temporal_conv = nn.Conv3d(
            self.conv_in_channels,
            F1,
            kernel_size=(temporal_kernel, 1, 1),
            bias=False,
        )
        self.bn1 = _group_norm(F1)

        # Depthwise LOCAL spatial conv(s): learn local motifs without collapsing HxW immediately
        # Input: (batch, F1, time, H, W) -> (batch, F1*D, time, H, W)
        self.depthwise_conv1 = nn.Conv3d(
            F1,
            F1 * D,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=F1,
            bias=False,
        )
        self.bn2a = _group_norm(F1 * D)

        # Optional: a second local depthwise conv at the expanded channel count
        self.depthwise_conv2 = nn.Conv3d(
            F1 * D,
            F1 * D,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=F1 * D,
            bias=False,
        )
        self.bn2b = _group_norm(F1 * D)

        # Pointwise convolution: mixes features
        # Input: (batch, F1*D, time, 1, 1) -> (batch, F2, time, 1, 1)
        self.pointwise_conv = nn.Conv3d(F1 * D, F2, kernel_size=1, bias=False)
        self.bn3 = _group_norm(F2)

        # Optional bottleneck to reduce channel count before TCN
        if self.F2_bottleneck is not None:
            self.bottleneck = nn.Conv3d(F2, self.F2_bottleneck, kernel_size=1, bias=False)
            self.bn4 = _group_norm(self.F2_bottleneck)
            self.output_channels = self.F2_bottleneck
        else:
            self.bottleneck = None
            self.bn4 = None
            self.output_channels = F2

        self.dropout = nn.Dropout(dropout)
        self.output_size = F2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, time, H, W)

        Returns:
            Output tensor of shape (batch, F2, time, H, W)
        """
        # Causal padding on the temporal dimension
        if self.temporal_kernel > 1:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_kernel - 1, 0))

        if self.use_mask:
            mask = x[:, -1:, :, :, :]              # (B,1,T,H,W)
            x = x[:, :-1, :, :, :] * mask          # gate bands by electrode presence

        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Depthwise LOCAL spatial conv(s)
        x = self.depthwise_conv1(x)
        x = self.bn2a(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.depthwise_conv2(x)
        x = self.bn2b(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Pointwise convolution (mix channels)
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        if self.bottleneck is not None:
            x = self.bottleneck(x)
            x = self.bn4(x)  # type: ignore[arg-type]
            x = F.elu(x)
            x = self.dropout(x)

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
        self.bn1 = _group_norm(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            bias=False
        )
        self.bn2 = _group_norm(out_channels)

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
        Input (feature_dim = input_channels * height * width)
        -> EEGNet block (temporal + spatial features)
        -> TCN blocks (temporal features)
        -> Linear classifier

    For streaming inference, the model maintains a buffer of past grids
    and applies the model causally.
    """

    def __init__(
        self,
        input_channels: int,
        height: int,
        width: int,
        # EEGNet parameters
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        temporal_kernel: int = 32,
        F2_bottleneck: int | None = 8,
        # TCN parameters
        tcn_channels: int = 16,
        tcn_kernel_size: int = 4,
        tcn_layers: int = 2,
        # General
        dropout: float = 0.3,
        # Context window for TCN (in samples)
        context_window: int = 1000,
        use_mask: bool = True,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.temporal_kernel = temporal_kernel
        self.F2_bottleneck = F2_bottleneck
        self.tcn_channels = tcn_channels
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_layers = tcn_layers
        self.dropout_rate = dropout
        self.context_window = context_window
        self.input_size = input_channels * height * width
        self.use_mask = use_mask

        self.classes_: np.ndarray | None = None
        self._n_classes: int | None = None

        # Feature buffer for streaming inference
        self._feature_buffer: torch.Tensor | None = None

        # EEGNet block for spatial feature extraction
        if context_window < temporal_kernel:
            raise ValueError("context_window must be >= temporal_kernel")

        self.eegnet = EEGNetBlock(
            in_channels=input_channels,
            height=height,
            width=width,
            temporal_kernel=temporal_kernel,
            F1=F1,
            D=D,
            F2=F2,
            F2_bottleneck=F2_bottleneck,
            dropout=dropout,
            use_mask=use_mask,
        )

        # TCN blocks with increasing dilation
        self.tcn_blocks = nn.ModuleList()
        in_ch = self.eegnet.output_channels * height * width
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

    def _build_classifier(self, n_classes: int) -> None:
        """Build the classifier layer once n_classes is known."""
        self._n_classes = n_classes
        self.classifier = nn.Linear(self.tcn_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for batch training.

        Args:
            x: Input tensor of shape (batch, seq_len, feature_dim)

        Returns:
            Logits tensor of shape (batch, seq_len, n_classes)
        """
        if self.classifier is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        if x.ndim == 5:
            batch_size, seq_len, _, _, _ = x.shape
            x = x.permute(0, 2, 1, 3, 4)
        else:
            batch_size, seq_len, feature_dim = x.shape
            if feature_dim != self.input_size:
                raise ValueError(
                    f"Expected feature_dim {self.input_size}, got {feature_dim}"
                )
            x = x.reshape(
                batch_size,
                seq_len,
                self.input_channels,
                self.height,
                self.width,
            )
            x = x.permute(0, 2, 1, 3, 4)

        # EEGNet block: (batch, input_channels, time, H, W) -> (batch, C, time, H, W)
        x = self.eegnet(x)
        x = x.reshape(
            batch_size,
            self.eegnet.output_channels * self.height * self.width,
            seq_len,
        )

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
            X: Feature array of shape (feature_dim,) for a single timestep.

        Returns:
            Predicted label as an integer.
        """
        if self.classes_ is None or self.classifier is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=device)
            if x.numel() != self.input_size:
                raise ValueError(
                    f"Expected feature_dim {self.input_size}, got {x.numel()}"
                )
            x = x.reshape(self.input_channels, self.height, self.width)

            if self._feature_buffer is None:
                pad_len = self.context_window - 1
                pad = torch.zeros(
                    pad_len,
                    self.input_channels,
                    self.height,
                    self.width,
                    dtype=x.dtype,
                    device=x.device,
                )
                self._feature_buffer = torch.cat([pad, x.unsqueeze(0)], dim=0)
            else:
                self._feature_buffer = torch.cat(
                    [self._feature_buffer[1:], x.unsqueeze(0)], dim=0
                )

            x_seq = self._feature_buffer.unsqueeze(0).permute(0, 2, 1, 3, 4)
            x = self.eegnet(x_seq)
            x = x.reshape(
                1,
                self.eegnet.output_channels * self.height * self.width,
                self.context_window,
            )
            for tcn_block in self.tcn_blocks:
                x = tcn_block(x)

            x = x[:, :, -1]
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
        seq_len: int = 1000,
        stride: int | None = None,
        chunk_len: int | None = None,
        chunks_per_epoch: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        stratify_sequences: bool = False,
        stratify_ratio: float = 0.45,
        loss_last_k: int = 1,
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
            chunk_len: Length of each contiguous chunk sampled per epoch.
            chunks_per_epoch: Number of random chunks to sample per epoch.
            learning_rate: Learning rate for Adam optimizer.
            weight_decay: L2 regularization weight.
            verbose: Whether to show progress.
        """
        # Determine classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        logger.info(f"Training EEG-TCNet with {n_classes} classes")

        # Build classifier
        self._build_classifier(n_classes)

        device = _get_device()
        self.to(device)

        # Convert to tensors
        if X.shape[1] != self.input_size:
            raise ValueError(
                f"Expected feature_dim {self.input_size}, got {X.shape[1]}"
            )
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_indices = np.array([class_to_idx[label] for label in y])
        y_tensor = torch.tensor(y_indices, dtype=torch.long, device=device)

        # Create sequences for training
        n_samples = len(X)
        stride = stride or (seq_len)
        if chunk_len is not None:
            if chunk_len <= seq_len:
                raise ValueError("chunk_len must be greater than seq_len")
            if chunk_len > n_samples:
                raise ValueError("chunk_len must be <= number of samples")

        nonzero = (y != 0).astype(np.int32)
        nonzero_classes = [c for c in self.classes_ if c != 0]
        nonzero_classes = np.array(nonzero_classes)

        # Setup training
        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate / 10
        )

        # Training loop
        best_loss = float('inf')
        epoch_iterator = tqdm(range(epochs), desc="Training", disable=not verbose)
        seq_offsets = torch.arange(seq_len, dtype=torch.long, device=device)

        def _build_starts_for_chunk(chunk_start: int, chunk_end: int) -> np.ndarray:
            starts_all = np.arange(chunk_start, chunk_end - seq_len, stride)
            if len(starts_all) == 0:
                return starts_all
            n_sequences = len(starts_all)
            if stratify_sequences:
                n_balanced = max(1, int(n_sequences * stratify_ratio))
                per_class = max(1, n_balanced // max(len(nonzero_classes), 1))

                balanced_starts = []
                for cls in nonzero_classes:
                    idx = np.where(y == cls)[0]
                    if idx.size == 0:
                        continue
                    starts = idx - (seq_len // 2)
                    starts = np.clip(starts, chunk_start, chunk_end - seq_len - 1)
                    starts = (starts // stride) * stride
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
                    batch_indices = starts[batch_start:batch_start + batch_size]
                    batch_indices_t = torch.as_tensor(
                        batch_indices, dtype=torch.long, device=device
                    )
                    idx = batch_indices_t[:, None] + seq_offsets[None, :]

                    # Build batch of sequences (vectorized indexing)
                    X_batch = X_tensor[idx]  # (batch, seq_len, n_channels)
                    y_batch = y_tensor[idx]  # (batch, seq_len)

                    optimizer.zero_grad()
                    logits = self.forward(X_batch)  # (batch, seq_len, n_classes)

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
            epoch_iterator.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{train_acc:.3f}")

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
                "input_channels": self.input_channels,
                "height": self.height,
                "width": self.width,
                "F1": self.F1,
                "D": self.D,
                "F2": self.F2,
                "temporal_kernel": self.temporal_kernel,
                "F2_bottleneck": self.F2_bottleneck,
                "tcn_channels": self.tcn_channels,
                "tcn_kernel_size": self.tcn_kernel_size,
                "tcn_layers": self.tcn_layers,
                "dropout": self.dropout_rate,
                "context_window": self.context_window,
                "use_mask": self.use_mask,
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
            input_channels=config["input_channels"],
            height=config["height"],
            width=config["width"],
            F1=config["F1"],
            D=config["D"],
            F2=config["F2"],
            temporal_kernel=config["temporal_kernel"],
            F2_bottleneck=config.get("F2_bottleneck", None),
            tcn_channels=config["tcn_channels"],
            tcn_kernel_size=config["tcn_kernel_size"],
            tcn_layers=config["tcn_layers"],
            dropout=config["dropout"],
            context_window=config["context_window"],
            use_mask=config.get("use_mask", True),
        )

        model._build_classifier(config["n_classes"])
        model.classes_ = checkpoint["classes"]
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        logger.debug(f"Model loaded from {MODEL_PATH}")
        return model
