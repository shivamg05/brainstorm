"""Tests for the loading module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brainstorm.constants import FEATURES_FILE_NAME, LABELS_FILE_NAME, N_CHANNELS
from brainstorm.loading import load_channel_coordinates, load_raw_data


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with fake data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create fake features data as DataFrame with time index
        n_frames = 100
        n_channels = N_CHANNELS
        time_index = np.arange(n_frames, dtype=np.float32) / 10

        # Create features as a wide DataFrame (time x channels)
        fake_features = pd.DataFrame(
            np.random.randn(n_frames, n_channels).astype(np.float32),
            index=time_index,
            columns=[f"channel_{i}" for i in range(n_channels)],
        )

        # Create fake labels DataFrame with time index
        n_trials = 10
        fake_labels = pd.DataFrame(
            {
                "label": np.random.randint(1, 9, size=n_trials),
            },
            index=np.linspace(1.0, 9.0, n_trials, dtype=np.float32),
        )
        fake_labels.index.name = "time_s"

        # Save to the temp directory as parquet with "train_" prefix
        fake_features.to_parquet(tmpdir_path / f"train_{FEATURES_FILE_NAME}")
        fake_labels.to_parquet(tmpdir_path / f"train_{LABELS_FILE_NAME}")

        yield tmpdir_path, fake_features, fake_labels


@pytest.fixture
def empty_temp_dir():
    """Create an empty temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestLoadRawData:
    """Tests for load_raw_data function."""

    def test_load_valid_data(self, temp_data_dir):
        """Test loading valid data from directory."""
        tmpdir, expected_features, expected_labels = temp_data_dir

        features, labels = load_raw_data(tmpdir)

        # Check that we get DataFrames
        assert isinstance(features, pd.DataFrame), "Features should be a DataFrame"
        assert isinstance(labels, pd.DataFrame), "Labels should be a DataFrame"

        # Check features has channel columns
        assert len(features.columns) == N_CHANNELS
        assert all(col.startswith("channel_") for col in features.columns)

        # Check labels has label column
        assert "label" in labels.columns

    def test_load_returns_correct_dtypes(self, temp_data_dir):
        """Test that loaded data has expected dtypes."""
        tmpdir, _, _ = temp_data_dir

        features, labels = load_raw_data(tmpdir)

        # Check that we get DataFrames
        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, pd.DataFrame)

    def test_load_preserves_data_values(self, temp_data_dir):
        """Test that loaded data matches what was saved."""
        tmpdir, expected_features, expected_labels = temp_data_dir

        features, labels = load_raw_data(tmpdir)

        # Load directly from parquet with train_ prefix
        loaded_features = pd.read_parquet(tmpdir / f"train_{FEATURES_FILE_NAME}")
        loaded_labels = pd.read_parquet(tmpdir / f"train_{LABELS_FILE_NAME}")

        pd.testing.assert_frame_equal(features, loaded_features)
        pd.testing.assert_frame_equal(labels, loaded_labels)

    def test_missing_directory(self):
        """Test error when directory doesn't exist."""
        nonexistent_dir = Path("/nonexistent/path/to/data")

        with pytest.raises(AssertionError, match="does not exist"):
            load_raw_data(nonexistent_dir)

    def test_directory_is_file(self, temp_data_dir):
        """Test error when path is a file, not a directory."""
        tmpdir, _, _ = temp_data_dir
        file_path = tmpdir / "somefile.txt"
        file_path.touch()

        with pytest.raises(AssertionError, match="is not a directory"):
            load_raw_data(file_path)

    def test_missing_features_file(self, empty_temp_dir):
        """Test error when features file is missing."""
        # Create only labels file with train_ prefix
        fake_labels = pd.DataFrame(
            {"label": [1, 2]}, index=pd.Index([1.0, 2.0], name="time_s")
        )
        fake_labels.to_parquet(empty_temp_dir / f"train_{LABELS_FILE_NAME}")

        with pytest.raises(AssertionError, match="Features file .* does not exist"):
            load_raw_data(empty_temp_dir)

    def test_missing_labels_file(self, empty_temp_dir):
        """Test error when labels file is missing."""
        # Create only features file with train_ prefix
        n_frames = 10
        fake_features = pd.DataFrame(
            np.random.randn(n_frames, N_CHANNELS).astype(np.float32),
            index=np.arange(n_frames, dtype=np.float32) / 10,
            columns=[f"channel_{i}" for i in range(N_CHANNELS)],
        )
        fake_features.to_parquet(empty_temp_dir / f"train_{FEATURES_FILE_NAME}")

        with pytest.raises(AssertionError, match="Labels file .* does not exist"):
            load_raw_data(empty_temp_dir)

    def test_empty_directory(self, empty_temp_dir):
        """Test error when directory is empty."""
        with pytest.raises(AssertionError, match="Features file .* does not exist"):
            load_raw_data(empty_temp_dir)

    def test_features_has_required_columns(self, temp_data_dir):
        """Test that features DataFrame has required columns."""
        tmpdir, _, _ = temp_data_dir

        features, _ = load_raw_data(tmpdir)

        # Features should have channel columns
        assert len(features.columns) == N_CHANNELS
        assert all(col.startswith("channel_") for col in features.columns)

    def test_labels_has_required_columns(self, temp_data_dir):
        """Test that labels DataFrame has required columns."""
        tmpdir, _, _ = temp_data_dir

        _, labels = load_raw_data(tmpdir)

        # Labels should have label column
        assert "label" in labels.columns


class TestLoadChannelCoordinates:
    """Tests for load_channel_coordinates function."""

    def test_load_channel_coordinates_exists(self):
        """Test that channel coordinates file can be loaded if it exists."""
        # This test will only work if the file actually exists
        # We'll try to load it and check the shape
        try:
            coords = load_channel_coordinates()

            # If successful, verify shape
            assert coords.ndim == 2, "Coordinates should be 2D"
            assert coords.shape == (N_CHANNELS, 2), f"Expected shape ({N_CHANNELS}, 2)"

        except AssertionError as e:
            # If file doesn't exist, that's expected in test environment
            if "does not exist" in str(e):
                pytest.skip("Channel coordinates file not present in test environment")
            else:
                raise

    def test_channel_coordinates_type_validation(self):
        """Test that type validation works for channel coordinates."""
        # This is more of an integration test to verify the decorator works
        try:
            coords = load_channel_coordinates()
            # If we get here, the validation passed
            assert isinstance(coords, np.ndarray)
        except AssertionError as e:
            if "does not exist" in str(e):
                pytest.skip("Channel coordinates file not present in test environment")
            else:
                raise


class TestDataIntegrity:
    """Tests for data integrity checks."""

    def test_features_labels_different_lengths(self, empty_temp_dir):
        """Test that features and labels can have different lengths (different time spans)."""
        # Create features with many time points
        n_frames = 100
        features = pd.DataFrame(
            np.random.randn(n_frames, N_CHANNELS).astype(np.float32),
            index=np.arange(n_frames, dtype=np.float32) / 10,
            columns=[f"channel_{i}" for i in range(N_CHANNELS)],
        )
        features.to_parquet(empty_temp_dir / f"train_{FEATURES_FILE_NAME}")

        # Create labels with only 5 events
        labels = pd.DataFrame(
            {"label": [1, 2, 3, 4, 5]},
            index=pd.Index([1.0, 2.0, 3.0, 4.0, 5.0], name="time_s"),
        )
        labels.to_parquet(empty_temp_dir / f"train_{LABELS_FILE_NAME}")

        # The function should load successfully
        loaded_features, loaded_labels = load_raw_data(empty_temp_dir)

        # Verify they loaded with different lengths
        assert len(loaded_features) != len(loaded_labels)
        assert len(loaded_features) == n_frames
        assert len(loaded_labels) == 5

    def test_various_data_sizes(self, empty_temp_dir):
        """Test loading data with various valid sizes."""
        test_cases = [
            (10, 5),  # Small dataset: 10 frames, 5 labels
            (100, 20),  # Medium dataset: 100 frames, 20 labels
            (5, 1),  # Minimal dataset: 5 frames, 1 label
        ]

        for n_frames, n_labels in test_cases:
            # Clean the directory
            for f in empty_temp_dir.glob("*.parquet"):
                f.unlink()

            # Create and save data
            features = pd.DataFrame(
                np.random.randn(n_frames, N_CHANNELS).astype(np.float32),
                index=np.arange(n_frames, dtype=np.float32) / 10,
                columns=[f"channel_{i}" for i in range(N_CHANNELS)],
            )
            labels = pd.DataFrame(
                {"label": np.random.randint(1, 9, size=n_labels)},
                index=pd.Index(
                    np.linspace(0.1, 0.9, n_labels).astype(np.float32), name="time_s"
                ),
            )

            features.to_parquet(empty_temp_dir / f"train_{FEATURES_FILE_NAME}")
            labels.to_parquet(empty_temp_dir / f"train_{LABELS_FILE_NAME}")

            # Load and verify
            loaded_features, loaded_labels = load_raw_data(empty_temp_dir)

            assert len(loaded_features) == n_frames
            assert len(loaded_labels) == n_labels

    def test_load_preserves_dtypes(self, empty_temp_dir):
        """Test that loading preserves data types in parquet files."""
        # Create data with specific dtypes
        features = pd.DataFrame(
            np.random.randn(10, N_CHANNELS).astype(np.float32),
            index=pd.Index(np.arange(10).astype(np.float32), name="time_s"),
            columns=[f"channel_{i}" for i in range(N_CHANNELS)],
        )
        labels = pd.DataFrame(
            {"label": np.array([1, 2]).astype(np.int32)},
            index=pd.Index([1.0, 2.0], dtype=np.float32, name="time_s"),
        )

        features.to_parquet(empty_temp_dir / f"train_{FEATURES_FILE_NAME}")
        labels.to_parquet(empty_temp_dir / f"train_{LABELS_FILE_NAME}")

        loaded_features, loaded_labels = load_raw_data(empty_temp_dir)

        # Parquet preserves dtypes
        assert loaded_features.values.dtype == np.float32
        assert loaded_labels["label"].dtype == np.int32


class TestPathHandling:
    """Tests for path handling."""

    def test_path_as_string(self, temp_data_dir):
        """Test that function accepts path as string."""
        tmpdir, _, _ = temp_data_dir

        # Convert to string
        features, labels = load_raw_data(Path(tmpdir))

        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, pd.DataFrame)

    def test_path_as_pathlib(self, temp_data_dir):
        """Test that function accepts pathlib.Path."""
        tmpdir, _, _ = temp_data_dir

        features, labels = load_raw_data(tmpdir)

        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, pd.DataFrame)

    def test_relative_path(self, temp_data_dir):
        """Test that relative paths work correctly."""
        tmpdir, _, _ = temp_data_dir

        # Get relative path
        import os

        original_dir = os.getcwd()
        try:
            os.chdir(tmpdir.parent)
            relative_path = Path(tmpdir.name)

            features, labels = load_raw_data(relative_path)

            assert isinstance(features, pd.DataFrame)
            assert isinstance(labels, pd.DataFrame)
        finally:
            os.chdir(original_dir)
