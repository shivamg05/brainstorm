"""
Unit tests for submission requirements validation.

These tests verify that your submission meets all requirements:
1. model_metadata.json exists with required fields
2. uv.lock exists (dependency lock file)
3. Model file exists and is under 25MB
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def repo_root():
    """Get the repository root directory."""
    return Path(__file__).parent.parent


class TestSubmissionRequirements:
    """Test suite for submission validation."""

    def test_model_metadata_exists(self, repo_root):
        """Test that model_metadata.json exists."""
        metadata_path = repo_root / "model_metadata.json"
        assert metadata_path.exists(), (
            "model_metadata.json not found. "
            "Train a model using model.fit() to generate this file."
        )

    def test_model_metadata_is_valid_json(self, repo_root):
        """Test that model_metadata.json is valid JSON."""
        metadata_path = repo_root / "model_metadata.json"
        if not metadata_path.exists():
            pytest.skip("model_metadata.json not found")

        try:
            with open(metadata_path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"model_metadata.json is not valid JSON: {e}")

    def test_model_metadata_has_required_fields(self, repo_root):
        """Test that model_metadata.json contains required fields."""
        metadata_path = repo_root / "model_metadata.json"
        if not metadata_path.exists():
            pytest.skip("model_metadata.json not found")

        with open(metadata_path) as f:
            metadata = json.load(f)

        required_fields = ["model_path", "import_string"]
        missing_fields = [f for f in required_fields if f not in metadata]

        assert not missing_fields, (
            f"model_metadata.json missing required fields: {missing_fields}. "
            f"Required fields are: {required_fields}"
        )

    def test_model_file_exists(self, repo_root):
        """Test that the model file specified in metadata exists."""
        metadata_path = repo_root / "model_metadata.json"
        if not metadata_path.exists():
            pytest.skip("model_metadata.json not found")

        with open(metadata_path) as f:
            metadata = json.load(f)

        if "model_path" not in metadata:
            pytest.skip("model_path not in metadata")

        model_path = repo_root / metadata["model_path"]
        assert model_path.exists(), (
            f"Model file not found: {metadata['model_path']}. "
            f"Make sure your model.save() method saves to the correct location."
        )

    def test_model_file_size_under_25mb(self, repo_root):
        """Test that the model file is under 25MB."""
        metadata_path = repo_root / "model_metadata.json"
        if not metadata_path.exists():
            pytest.skip("model_metadata.json not found")

        with open(metadata_path) as f:
            metadata = json.load(f)

        if "model_path" not in metadata:
            pytest.skip("model_path not in metadata")

        model_path = repo_root / metadata["model_path"]
        if not model_path.exists():
            pytest.skip("Model file not found")

        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        assert model_size_mb <= 25, (
            f"Model file too large: {model_size_mb:.2f}MB (max 25MB). "
            f"Try model compression, pruning, quantization, or a smaller architecture."
        )

        print(f"\n✅ Model size: {model_size_mb:.2f}MB")

    def test_uv_lock_exists(self, repo_root):
        """Test that uv.lock exists."""
        lock_path = repo_root / "uv.lock"
        assert lock_path.exists(), (
            "uv.lock not found. Run 'uv sync' to generate the lock file, "
            "then commit it to your repository."
        )
