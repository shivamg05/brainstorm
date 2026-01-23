"""
MLP Model for continuous classification of ECoG signals.

This module provides a simple Multi-Layer Perceptron (MLP) model that predicts
discrete frequency labels from neural recordings at each timestep.
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


class MLP(BaseModel):
    """
    Multi-Layer Perceptron for continuous classification.

    A simple single hidden layer MLP that maps ECoG channel readings
    to stimulus frequency predictions.

    Architecture:
        Input (n_channels) -> Linear -> ReLU -> Dropout -> Linear -> Output (n_classes)

    The model is always saved to and loaded from a fixed location: `model.pt`
    in the repository root. This ensures consistent model management.

    Attributes:
        input_size: Number of input features (default: 1024 channels).
        hidden_size: Number of units in the hidden layer.
        n_classes: Number of output classes (stimulus frequencies + no-stimulus).
        classes_: Array of unique class labels learned during fit().

    Example:
        >>> model = MLP(hidden_size=256)
        >>> model.fit(train_features, train_labels)  # Trains, saves, and validates
        >>>
        >>> # Load for inference
        >>> model = MLP.load()
        >>> prediction = model.predict(sample)
    """

    def __init__(
        self,
        input_size: int = N_CHANNELS,
        hidden_size: int = 256,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize the MLP model.

        Args:
            input_size: Number of input features (ECoG channels). Default: 1024.
            hidden_size: Number of units in the hidden layer. Default: 256.
            dropout: Dropout rate for regularization. Default: 0.3.

        Note:
            The number of output classes is determined automatically during fit()
            based on the unique labels in the training data.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.classes_: np.ndarray | None = None

        # Layers will be initialized in _build_layers() after we know n_classes
        self.fc1: nn.Linear | None = None
        self.dropout: nn.Dropout | None = None
        self.fc2: nn.Linear | None = None

    def _build_layers(self, n_classes: int) -> None:
        """Build the network layers once n_classes is known."""
        self._n_classes = n_classes
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size) or (input_size,).

        Returns:
            Logits tensor of shape (batch_size, n_classes) or (n_classes,).
        """
        if self.fc1 is None or self.fc2 is None or self.dropout is None:
            raise RuntimeError(
                "Model layers not initialized. Call fit() first or load a trained model."
            )

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Train the model on the provided features and labels.

        This is called by the base class fit() method, which handles
        saving and validation.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Label array of shape (n_samples,) with integer class labels.
            epochs: Number of training epochs. Default: 50.
            batch_size: Mini-batch size for training. Default: 64.
            learning_rate: Learning rate for Adam optimizer. Default: 1e-3.
            verbose: Whether to show training progress. Default: True.
        """
        # Determine unique classes and create mapping
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        logger.info(f"Training MLP with {n_classes} classes: {self.classes_.tolist()}")

        # Build layers now that we know n_classes
        self._build_layers(n_classes)

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_indices = np.array([class_to_idx[label] for label in y])
        y_tensor = torch.tensor(y_indices, dtype=torch.long)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Setup training
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        avg_loss = 0.0
        epoch_iterator = tqdm(range(epochs), desc="Training", disable=not verbose)
        for epoch in epoch_iterator:
            total_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.forward(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            epoch_iterator.set_postfix(loss=f"{avg_loss:.4f}")

        self.eval()
        logger.info(f"Training complete. Final loss: {avg_loss:.4f}")

    def predict(self, X: np.ndarray) -> int:
        """
        Predict the label for a single sample.

        Args:
            X: Feature array of shape (n_features,) for a single timestep.

        Returns:
            Predicted label as an integer (original class value, not index).

        Raises:
            RuntimeError: If model is not trained or loaded.
        """
        if self.classes_ is None:
            raise RuntimeError(
                "Model not trained. Call fit() first or load a trained model."
            )

        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)  # Add batch dimension

            logits = self.forward(x_tensor)
            predicted_idx = int(torch.argmax(logits, dim=1).item())

        return int(self.classes_[predicted_idx])

    def save(self) -> Path:
        """
        Save the model weights and configuration to model.pt.

        Returns:
            Path to the saved model file.
        """
        if self.classes_ is None:
            raise RuntimeError("Cannot save untrained model. Call fit() first.")

        checkpoint = {
            "config": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "n_classes": self._n_classes,
                "dropout": self.dropout_rate,
            },
            "classes": self.classes_,
            "state_dict": self.state_dict(),
        }

        torch.save(checkpoint, MODEL_PATH)
        logger.debug(f"Model saved to {MODEL_PATH}")
        return MODEL_PATH

    @classmethod
    def load(cls) -> Self:
        """
        Load a model from model.pt.

        Returns:
            A new instance of MLP with loaded weights.

        Raises:
            FileNotFoundError: If model.pt does not exist.
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Train a model first using MLP.fit() which saves to this location."
            )

        checkpoint = torch.load(MODEL_PATH, weights_only=False)

        # Reconstruct the model
        config = checkpoint["config"]
        model = cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
        )

        # Rebuild layers with the saved n_classes
        model._build_layers(config["n_classes"])

        model.classes_ = checkpoint["classes"]
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        logger.debug(f"Model loaded from {MODEL_PATH}")
        return model
