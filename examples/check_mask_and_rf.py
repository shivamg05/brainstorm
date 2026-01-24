#!/usr/bin/env python3
"""
Quick sanity checks for electrode mask and receptive field.

Usage:
    uv run python examples/check_mask_and_rf.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rich import print as rprint

from brainstorm.constants import GRID_HEIGHT, GRID_WIDTH
from brainstorm.loading import load_channel_coordinates
from brainstorm.preprocessing import build_electrode_mask


@dataclass(frozen=True)
class RFConfig:
    temporal_kernel: int = 32
    tcn_kernel_size: int = 4
    tcn_layers: int = 3


def estimate_tcn_rf_samples(kernel_size: int, layers: int) -> int:
    """Approximate TCN receptive field in samples."""
    dilations = [2**i for i in range(layers)]
    return 1 + 2 * (kernel_size - 1) * sum(dilations)


def main() -> None:
    coords = load_channel_coordinates()
    mask = build_electrode_mask(coords)

    y_offset = 2
    x = coords[:, 0].astype(int)
    y = coords[:, 1].astype(int) - y_offset
    valid = (x >= 0) & (x < GRID_WIDTH) & (y >= 0) & (y < GRID_HEIGHT)
    expected_valid = int(valid.sum())

    mask_ones = int(mask.sum())
    rprint("[bold cyan]Mask check[/]")
    rprint(f"- mask ones: {mask_ones}")
    rprint(f"- expected valid coords: {expected_valid}")
    rprint(f"- grid shape: {mask.shape}")
    rprint(f"- coords min/max x: {x.min()} / {x.max()}")
    rprint(f"- coords min/max y (offset): {y.min()} / {y.max()}")
    if mask_ones != expected_valid:
        rprint("[bold yellow]Warning:[/] mask ones != expected valid coords")

    cfg = RFConfig()
    tcn_rf = estimate_tcn_rf_samples(cfg.tcn_kernel_size, cfg.tcn_layers)
    total_rf = cfg.temporal_kernel + tcn_rf - 1
    rprint("\n[bold cyan]Receptive field (approx)[/]")
    rprint(f"- EEG temporal kernel: {cfg.temporal_kernel} samples")
    rprint(f"- TCN RF: {tcn_rf} samples")
    rprint(f"- total RF: ~{total_rf} samples (~{total_rf} ms)")


if __name__ == "__main__":
    main()
