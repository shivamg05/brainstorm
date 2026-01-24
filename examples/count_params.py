#!/usr/bin/env python3
"""
Count model parameters and estimated size.

Usage:
    uv run python examples/count_params.py
"""

from __future__ import annotations

from rich import print as rprint

from brainstorm.ml.eeg_tcnet import EEGTCNet


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    model = EEGTCNet(
        input_channels=9,
        height=16,
        width=16,
        F1=16,
        D=2,
        F2=32,
        temporal_kernel=32,
        F2_bottleneck=8,
        tcn_channels=32,
        tcn_layers=3,
        dropout=0.4,
        context_window=128,
        use_mask=False,
    )

    total_params = count_params(model)
    size_mb = total_params * 4 / (1024 * 1024)
    rprint("[bold cyan]EEGTCNet parameter count[/]")
    rprint(f"- total params: {total_params:,}")
    rprint(f"- approx size (float32): {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
