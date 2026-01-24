#!/usr/bin/env python3
"""
Train and evaluate script

This script demonstrates a local train+evaluate workflow for the EEG-TCNet
spectral model. It will:
    1. Download formatted ECoG data (if missing)
    2. Load training and validation splits
    3. Train an EEG-TCNet (spectral) model while printing loss progress
    4. Save the trained model and metadata
    5. Run evaluation on the validation set and print a summary

Usage:
    python training_eval/train_and_evaluate.py

The trained model is saved to `model.pt` in the repository root and metadata is
written to `model_metadata.json` so the evaluator can load it.
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

# Training hyperparameters (small defaults so script runs quickly)
EPOCHS = 30
BATCH_SIZE = 32
SEQ_LEN = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Path to the trained model (used for informational messages)
MODEL_PATH = Path("./model.pt")


def main() -> None:
    rprint("\n[bold green]Train and evaluate EEG-TCNet (spectral)...[/]\n")

    # Download data if not already present
    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("\n[bold yellow]Downloading data from Hugging Face...[/]\n")
        download_train_validation_data()
        rprint("[bold green]✓ Data downloaded successfully![/]\n")

    rprint(f"\n[bold cyan]Loading data from:[/] {DATA_PATH}\n")
    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
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

    # Add training and validation rows
    table.add_row(
        "Train",
        str(train_features.shape),
        str(train_labels.shape),
        f"{train_features.index[0]:.2f} → {train_features.index[-1]:.2f}",
        str(sorted(train_labels["label"].unique().tolist())),
    )
    table.add_row(
        "Validation",
        str(validation_features.shape),
        str(validation_labels.shape),
        f"{validation_features.index[0]:.2f} → {validation_features.index[-1]:.2f}",
        str(sorted(validation_labels["label"].unique().tolist())),
    )

    console.print(table)
    print()

    # Train model
    rprint("\n[bold green]Training EEG-TCNet (spectral)...[/]\n")
    model = EEGTCNetSpectral()

    # If a model file exists, inform the user it will be overwritten by this run
    if MODEL_PATH.exists():
        rprint(f"[bold yellow]Note: existing model at {MODEL_PATH} will be overwritten by training.[/]")

    # Call the high-level fit() which will run fit_model(), save the model, and write metadata
    model.fit(
        X=train_features.values,
        y=train_labels["label"].values,  # type: ignore[union-attr]
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        verbose=True,
    )

    rprint("\n[bold green]Training finished. Proceeding to evaluation...[/]\n")

    # Evaluate model on the validation set. ModelEvaluator will load the model using metadata written by BaseModel.fit()
    rprint("\n[bold green]Evaluating model on validation set...[/]\n")
    evaluator = ModelEvaluator(
        test_features=validation_features,
        test_labels=validation_labels[["label"]],  # type: ignore[union-attr]
    )

    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)

    rprint("\n[bold green]Train+Evaluation complete![/]\n")


if __name__ == "__main__":
    main()
