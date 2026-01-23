"""
Utility functions for model validation and loading.

This module provides helpers to validate that models are correctly saved
and can be loaded programmatically for evaluation.
"""

import importlib
from pathlib import Path
from typing import Any

from loguru import logger


def validate_model_file(model_path: Path, max_size_mb: float = 25.0) -> float:
    """
    Validate that a model file exists and is accessible.

    Args:
        model_path: Path to the saved model file.
        max_size_mb: Maximum allowed file size in MB (default: 25.0).

    Raises:
        FileNotFoundError: If the model file doesn't exist.
        ValueError: If the path is not a file or file is too large.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not model_path.is_file():
        raise ValueError(f"Path is not a file: {model_path}")

    # Check file size
    file_size_bytes = model_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_mb > max_size_mb:
        raise ValueError(
            f"Model file is too large for upload to evaluation system: "
            f"{file_size_mb:.2f}MB (max: {max_size_mb}MB). "
            f"Please reduce model size by:\n"
            f"  - Using fewer parameters\n"
            f"  - Applying model compression\n"
            f"  - Quantizing weights\n"
            f"  - Pruning unnecessary layers"
        )

    logger.debug(f"✓ Model file exists: {model_path} ({file_size_mb:.2f}MB)")

    return file_size_mb


def import_model_class(import_string: str) -> type:
    """
    Dynamically import a model class from an import string.

    Args:
        import_string: Full import path (e.g., "brainstorm.ml.model.Model").

    Returns:
        The model class.

    Raises:
        ImportError: If the module or class cannot be imported.
        AttributeError: If the class doesn't exist in the module.

    Example:
        >>> ModelClass = import_model_class("brainstorm.ml.model.Model")
        >>> model = ModelClass.load()
    """
    try:
        module_path, class_name = import_string.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        logger.debug(f"✓ Successfully imported: {import_string}")
        return model_class
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Failed to import model class '{import_string}': {e}") from e


def validate_model_loadable(
    import_string: str, model_path: Path, max_size_mb: float = 25.0
) -> Any:
    """
    Validate that a model can be loaded programmatically.

    This performs a full check:
    1. The model file exists and is not too large
    2. The class can be imported
    3. The model can be loaded using the class's load() method

    Args:
        import_string: Full import path to the model class.
        model_path: Path to the saved model file.
        max_size_mb: Maximum allowed file size in MB (default: 25.0).

    Returns:
        The loaded model instance.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        ValueError: If model file is too large.
        ImportError: If the class can't be imported.
        Exception: If loading fails for any other reason.
    """
    # Check file exists and size
    file_size_mb = validate_model_file(model_path, max_size_mb=max_size_mb)

    # Import the class
    model_class = import_model_class(import_string)

    # Try to load the model
    try:
        model = model_class.load()
        logger.debug(
            f"✓ Model loaded successfully from {model_path} | {file_size_mb:.2f}MB"
        )
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {model_path} using {import_string}: {e}"
        ) from e
