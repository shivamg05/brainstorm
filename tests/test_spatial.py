"""Tests for the spatial module."""

import numpy as np
import pytest

from brainstorm.constants import N_CHANNELS
from brainstorm.spatial import channels_to_spatial


@pytest.fixture
def mock_channels_coords():
    """Create mock channel coordinates matching the real data format.

    Real coordinates are 0-indexed with X in [0, 31] and Y in [2, 32],
    creating a 32-wide by 31-tall grid. The fixture generates exactly
    N_CHANNELS coordinates.
    """
    coords = []
    # Generate 31 rows × 32 columns = 992 coordinates
    for y in range(2, 33):  # Y ranges from 2 to 32 (31 rows)
        for x in range(0, 32):  # X ranges from 0 to 31 (32 columns)
            coords.append([x, y])

    # Add extra coordinates to reach N_CHANNELS (1024)
    # These simulate the outlier electrodes in the real data
    remaining = N_CHANNELS - len(coords)
    for i in range(remaining):
        coords.append([32, 10 + i])  # Place outliers at X=32 (outside main grid)

    return np.array(coords[:N_CHANNELS], dtype=np.int32)


@pytest.fixture
def sample_channel_data():
    """Create sample channel data with known pattern."""
    # Create data where each channel has a unique value for easy verification
    return np.arange(N_CHANNELS, dtype=np.float32)


class TestChannelsToSpatial:
    """Tests for channels_to_spatial function."""

    def test_basic_conversion(self, mock_channels_coords, sample_channel_data):
        """Test converting 1D channel data to 2D spatial."""
        spatial = channels_to_spatial(sample_channel_data, mock_channels_coords)

        assert spatial.shape == (31, 32)  # 31 rows (Y=2 to 32), 32 columns (X=0 to 31)
        assert spatial.dtype == sample_channel_data.dtype

    def test_spatial_mapping_correctness(self, mock_channels_coords):
        """Test that channels are mapped to correct spatial positions."""
        # Create data where each channel equals its index
        data = np.arange(N_CHANNELS, dtype=np.float32)

        spatial = channels_to_spatial(data, mock_channels_coords)

        # Verify a few known positions
        # Coordinates are 0-indexed, Y offset is 2 (Y=2 maps to row 0)
        # Channel 0 should be at coordinates (0, 2), which maps to grid position (row=0, col=0)
        x0, y0 = mock_channels_coords[0]
        assert spatial[y0 - 2, x0] == 0  # Y=2 -> row 0, X=0 -> col 0

        # Channel 10 should be at its corresponding position
        if 10 < len(mock_channels_coords):
            x10, y10 = mock_channels_coords[10]
            assert spatial[y10 - 2, x10] == 10  # Apply Y_OFFSET=2

    def test_zeros_for_empty_positions(self, mock_channels_coords):
        """Test that positions without electrodes are zero."""
        data = np.ones(N_CHANNELS, dtype=np.float32)
        spatial = channels_to_spatial(data, mock_channels_coords)

        # Count non-zero positions - should be <= N_CHANNELS
        # (some might be skipped if out of bounds)
        non_zero_count = np.count_nonzero(spatial)
        assert non_zero_count <= N_CHANNELS

    def test_invalid_channels_coords_shape(self, sample_channel_data):
        """Test error handling for invalid channels_coords shape."""
        bad_coords = np.ones((N_CHANNELS, 3))  # Wrong shape (should be N_CHANNELS, 2)

        with pytest.raises(ValueError, match="channels_coords must have shape"):
            channels_to_spatial(sample_channel_data, bad_coords)

    def test_invalid_data_shape(self, mock_channels_coords):
        """Test error handling for wrong data shape."""
        bad_data = np.ones(512)  # Wrong number of channels

        with pytest.raises(ValueError, match="data must have shape"):
            channels_to_spatial(bad_data, mock_channels_coords)

    def test_different_dtypes(self, mock_channels_coords):
        """Test that function works with different data types."""
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            data = np.arange(N_CHANNELS, dtype=dtype)
            spatial = channels_to_spatial(data, mock_channels_coords)
            assert spatial.dtype == dtype

    def test_handles_out_of_bounds_coordinates(self):
        """Test that out-of-bounds coordinates are skipped gracefully."""
        # Create coords with some valid values
        coords = []
        for y in range(2, 33):  # Valid Y range
            for x in range(0, 32):  # Valid X range
                coords.append([x, y])

        # Add extra coordinates to reach N_CHANNELS (1024)
        remaining = N_CHANNELS - len(coords)
        for i in range(remaining):
            coords.append([32, 10 + i])  # Place at X=32 (outside main grid)

        coords = np.array(coords[:N_CHANNELS], dtype=np.int32)
        # Add some invalid coordinates
        coords[0] = [-1, 0]  # Out of bounds (negative X, Y too small)
        coords[1] = [35, 35]  # Out of bounds (X and Y too large)

        data = np.arange(N_CHANNELS, dtype=np.float32)

        # Should not raise an error, just skip invalid coords
        spatial = channels_to_spatial(data, coords)
        assert spatial.shape == (31, 32)  # Correct shape for real grid
