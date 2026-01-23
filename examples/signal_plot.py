#!/usr/bin/env python3
"""
Example script: Visualize Electrode Array and Signal Variance

This script demonstrates how to:
    1. Load channel coordinates
    2. Compute signal statistics (variance) per channel
    3. Visualize the electrode array layout
    4. Convert channel data to a 2D spatial grid

Usage:
    python examples/signal_plot.py

Make sure you've downloaded the data first:
    from brainstorm.download import download_train_validation_data
    download_train_validation_data()
"""

from pathlib import Path

import matplotlib.pyplot as plt

from brainstorm.loading import load_raw_data, load_channel_coordinates
from brainstorm.plotting import dot_plot
from brainstorm.spatial import channels_to_spatial


# Path to downloaded data
DATA_PATH = Path("./data")

# Load training data
train_features, train_labels = load_raw_data(DATA_PATH, step="train")

# Load electrode coordinates
channel_coords = load_channel_coordinates()

# Compute variance of the signal for each channel
channel_variances = train_features.var(axis=0)
channel_variances_spatial = channels_to_spatial(channel_variances.values, channel_coords)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# Left: Dot plot showing electrode positions colored by variance
dot_plot(channel_variances.values, channel_coords, marker_size=90, ax=axes[0])
axes[0].set(
    aspect="equal", 
    title="Array Electrode Locations", 
    xlim=[-8, 40], 
    ylim=[-8, 40]
)

# Right: 2D spatial grid representation
axes[1].imshow(channel_variances_spatial, cmap="inferno")
axes[1].set(aspect="equal", title="2D Spatial Grid", xlim=[-8, 40], ylim=[-8, 40])
axes[1].axis("off")

# Save to docs for documentation
fig.savefig("docs/signal_plot.png")

plt.show()
