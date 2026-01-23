"""Functions for converting between channel and spatial representations of neural data."""

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, validate_call

from brainstorm.constants import N_CHANNELS, GRID_WIDTH, GRID_HEIGHT


@validate_call(config=ConfigDict(arbitrary_types_allowed=True), validate_return=True)
def channels_to_spatial(
    data: NDArray,
    channels_coords: NDArray,
) -> NDArray:
    """Convert 1D channel data to 2D spatial representation.

    Takes a 1D array with N_CHANNELS values and arranges them into a 2D spatial grid
    based on the physical coordinates of each channel.

    Args:
        data: Array of shape (N_CHANNELS,) containing values for each channel
        channels_coords: Array of shape (N_CHANNELS, 2) containing (x, y) coordinates
            for each channel. Coordinates are 0-indexed, with the main electrode array
            spanning X=[0,31] and Y=[2,32].

    Returns:
        Array of shape (H, W) representing the spatial layout, where H and W are
        determined by GRID_HEIGHT and GRID_WIDTH. Y coordinates are mapped such that
        Y=2 maps to row 0, Y=3 to row 1, etc. Positions without electrodes are zero.

    Example:
        >>> data.shape
        (1024,)
        >>> spatial = channels_to_spatial(data, coords)
        >>> spatial.shape
        (31, 32)
    """
    # Validate inputs
    if channels_coords.shape != (N_CHANNELS, 2):
        raise ValueError(
            f"channels_coords must have shape ({N_CHANNELS}, 2), "
            f"got {channels_coords.shape}"
        )

    if data.shape != (N_CHANNELS,):
        raise ValueError(f"data must have shape ({N_CHANNELS},), got {data.shape}")
    # Create output array with spatial dimensions
    spatial_data = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=data.dtype)

    # Map each channel to its spatial position
    # Y coordinates in the data range from 2-32 (with some outliers)
    # We map Y=2 to row 0, Y=3 to row 1, etc.
    Y_OFFSET = 2  # Minimum Y coordinate in the main electrode array

    for channel_idx in range(N_CHANNELS):
        x, y = channels_coords[channel_idx]
        # Apply Y offset to map Y=2 to row 0
        y_idx = int(y) - Y_OFFSET
        x_idx = int(x)

        # Skip coordinates outside the grid bounds
        if x_idx < 0 or x_idx >= GRID_WIDTH or y_idx < 0 or y_idx >= GRID_HEIGHT:
            continue

        spatial_data[y_idx, x_idx] = data[channel_idx]

    return spatial_data
