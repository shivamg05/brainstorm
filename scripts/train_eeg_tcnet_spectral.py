#!/usr/bin/env python3
"""
Training script for EEG-TCNet with Spectral Preprocessing.

This script trains the EEG-TCNet Spectral model on ECoG data for continuous
classification. Designed to run on GPU clusters (e.g., OSCAR/CCV).

Usage:
    python scripts/train_eeg_tcnet_spectral.py [--epochs N] [--batch-size N]
"""

import argparse
from pathlib import Path

import torch
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.ml.eeg_tcnet_spectral import EEGTCNetSpectral


# Default paths
DATA_PATH = Path("./data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EEG-TCNet with Spectral Preprocessing"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (default: 30)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=64,
        help="Sequence length for TCN (default: 64)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--data-path", type=Path, default=DATA_PATH,
        help="Path to data directory (default: ./data)"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip automatic data download"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rprint(f"\n[bold cyan]Device:[/] {device}")
    if device == "cuda":
        rprint(f"[bold cyan]GPU:[/] {torch.cuda.get_device_name(0)}")
        rprint(f"[bold cyan]CUDA Version:[/] {torch.version.cuda}")

    # Download data if needed
    if not args.no_download:
        if not args.data_path.exists() or not any(args.data_path.glob("*.parquet")):
            rprint("\n[bold yellow]Downloading data from Hugging Face...[/]\n")
            download_train_validation_data()
            rprint("[bold green]Data downloaded successfully![/]\n")

    # Load data
    rprint(f"\n[bold cyan]Loading data from:[/] {args.data_path}\n")
    train_features, train_labels = load_raw_data(args.data_path, step="train")
    validation_features, validation_labels = load_raw_data(args.data_path, step="validation")

    # Display dataset info
    console = Console()
    table = Table(
        title="Dataset Overview", show_header=True, header_style="bold magenta"
    )
    table.add_column("Split", style="cyan", width=10)
    table.add_column("Features Shape", style="green")
    table.add_column("Labels Shape", style="green")
    table.add_column("Unique Labels", style="blue")

    table.add_row(
        "Train",
        str(train_features.shape),
        str(train_labels.shape),
        str(sorted(train_labels["label"].unique().tolist())),
    )
    table.add_row(
        "Validation",
        str(validation_features.shape),
        str(validation_labels.shape),
        str(sorted(validation_labels["label"].unique().tolist())),
    )
    console.print(table)

    # Create model
    rprint("\n[bold green]Creating EEG-TCNet Spectral model...[/]\n")
    model = EEGTCNetSpectral(
        n_channels=train_features.shape[1],
        context_window=args.seq_len,
    )

    # Display model config
    config_table = Table(title="Model Configuration", show_header=True, header_style="bold magenta")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Frequency Bands", str(list(model.bands.keys())))
    config_table.add_row("Include Raw Signal", str(model.include_raw))
    config_table.add_row("TCN Layers", str(model.tcn_layers))
    config_table.add_row("TCN Channels", str(model.tcn_channels))
    config_table.add_row("Context Window", str(model.context_window))
    config_table.add_row("Epochs", str(args.epochs))
    config_table.add_row("Batch Size", str(args.batch_size))
    config_table.add_row("Learning Rate", str(args.lr))
    console.print(config_table)

    # Train model
    rprint("\n[bold green]Training model...[/]\n")
    model.fit(
        X=train_features.values,
        y=train_labels["label"].values,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        verbose=True,
    )

    # Evaluate
    rprint("\n[bold green]Evaluating model on validation set...[/]\n")
    evaluator = ModelEvaluator(
        test_features=validation_features,
        test_labels=validation_labels[["label"]],
    )

    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)

    rprint(f"\n[bold green]Model saved to:[/] {model.save()}\n")
    rprint("[bold green]Training complete![/]\n")


if __name__ == "__main__":
    main()
