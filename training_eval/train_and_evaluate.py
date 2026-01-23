#!/usr/bin/env python3
"""
Evaluation Script

This script evaluates a pre-trained model (trained on OSCAR cluster):
    1. Downloading ECoG data from Hugging Face (if not already downloaded)
    2. Loading validation data
    3. Loading the pre-trained model
    4. Running inference and computing evaluation metrics
    5. Displaying results

Usage:
    python training_eval/train_and_evaluate.py

Assumes the model is already trained and saved to model.pt in the repository root.
"""

from pathlib import Path
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.ml.eeg_tcnet_spectral import EEGTCNetSpectral


# =============================================================================
# Configuration
# =============================================================================

# Path to the formatted data directory
DATA_PATH = Path("./data")

# Path to the trained model
MODEL_PATH = Path("./model.pt")


def main() -> None:
    rprint("\n[bold green]Evaluating Pre-trained Model...[/]\n")

    # Check if model exists
    if not MODEL_PATH.exists():
        rprint(f"\n[bold red]Error: Model not found at {MODEL_PATH}[/]")
        rprint("[bold yellow]Please ensure the model is trained and saved first.[/]\n")
        return

    # Download data if not already present
    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("\n[bold yellow]Downloading data from Hugging Face...[/]\n")
        download_train_validation_data()
        rprint("[bold green]✓ Data downloaded successfully![/]\n")

    rprint(f"\n[bold cyan]Loading validation data from:[/] {DATA_PATH}\n")
    validation_features, validation_labels = load_raw_data(DATA_PATH, step="validation")

    # Create a nice table to display dataset information
    console = Console()
    table = Table(
        title="Dataset Overview", show_header=True, header_style="bold magenta"
    )

    table.add_column("Split", style="cyan", width=10)
    table.add_column("Features Shape", style="green")
    table.add_column("Labels Shape", style="green")
    table.add_column("Time Range (s)", style="yellow")
    table.add_column("Unique Labels", style="blue")

    # Add validation data row
    table.add_row(
        "Validation",
        str(validation_features.shape),
        str(validation_labels.shape),
        f"{validation_features.index[0]:.2f} → {validation_features.index[-1]:.2f}",
        str(sorted(validation_labels["label"].unique().tolist())),
    )

    console.print(table)
    print()

    # Load pre-trained model
    rprint("\n[bold green]Loading pre-trained model...[/]\n")
    try:
        model = EEGTCNetSpectral.load()
        rprint(f"[bold green]✓ Model loaded successfully from {MODEL_PATH}[/]\n")
    except Exception as e:
        rprint(f"\n[bold red]Error loading model: {e}[/]\n")
        return

    # Display model configuration
    config_table = Table(
        title="Model Configuration", show_header=True, header_style="bold magenta"
    )
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Frequency Bands", str(list(model.bands.keys())))
    config_table.add_row("Include Raw Signal", str(model.include_raw))
    config_table.add_row("Spatial Block F1", str(model.F1))
    config_table.add_row("Spatial Block D", str(model.D))
    config_table.add_row("Spatial Block F2", str(model.F2))
    config_table.add_row("TCN Layers", str(model.tcn_layers))
    config_table.add_row("TCN Channels", str(model.tcn_channels))
    config_table.add_row("Context Window", str(model.context_window))
    config_table.add_row("Dropout", str(model.dropout_rate))
    config_table.add_row("Number of Classes", str(len(model.classes_) if model.classes_ is not None else "N/A"))
    console.print(config_table)
    print()

    # Evaluate model
    rprint("\n[bold green]Evaluating model on validation set...[/]\n")
    # NOTE: we use validation_features and labels because the test set is held out
    # and not accessible for local evaluation.
    evaluator = ModelEvaluator(
        test_features=validation_features,
        test_labels=validation_labels[["label"]],  # type: ignore[union-attr]
    )

    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)

    rprint("\n[bold green]Evaluation complete![/]\n")


if __name__ == "__main__":
    main()
