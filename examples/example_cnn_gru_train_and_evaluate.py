#!/usr/bin/env python3
"""
Example script: Train and Evaluate CNN+GRU with built-in preprocessing.

Usage:
    uv run python examples/example_cnn_gru_train_and_evaluate.py
"""

from pathlib import Path

from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.loading import load_raw_data
from brainstorm.ml.cnn_gru import CNNGRU


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("./data")

EPOCHS = 28
BATCH_SIZE = 16
SEQ_LEN = 96
STRIDE = SEQ_LEN
CHUNK_LEN = 8000
CHUNKS_PER_EPOCH = 6
LEARNING_RATE = 3e-4
STRATIFY_SEQUENCES = True
STRATIFY_RATIO = 0.45
LOSS_LAST_K = SEQ_LEN
VAL_EVERY = 7
VAL_MAX_SAMPLES = None


def _print_dataset_table(train_features, train_labels, val_features, val_labels) -> None:
    console = Console()
    table = Table(title="Dataset Overview", show_header=True, header_style="bold magenta")
    table.add_column("Split", style="cyan", width=10)
    table.add_column("Features Shape", style="green")
    table.add_column("Labels Shape", style="green")
    table.add_column("Time Range (s)", style="yellow")
    table.add_column("Unique Labels", style="blue")

    table.add_row(
        "Train",
        str(train_features.shape),
        str(train_labels.shape),
        f"{train_features.index[0]:.2f} → {train_features.index[-1]:.2f}",
        str(sorted(train_labels["label"].unique().tolist())),
    )
    table.add_row(
        "Val",
        str(val_features.shape),
        str(val_labels.shape),
        f"{val_features.index[0]:.2f} → {val_features.index[-1]:.2f}",
        str(sorted(val_labels["label"].unique().tolist())),
    )
    console.print(table)
    print()


def main() -> None:
    rprint("\n[bold green]Evaluating CNN+GRU...[/]\n")

    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("\n[bold yellow]Downloading data from Hugging Face...[/]\n")
        download_train_validation_data()
        rprint("[bold green]✓ Data downloaded successfully![/]\n")

    rprint(f"\n[bold cyan]Loading data from:[/] {DATA_PATH}\n")
    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    val_features, val_labels = load_raw_data(DATA_PATH, step="validation")
    _print_dataset_table(train_features, train_labels, val_features, val_labels)

    rprint("\n[bold green]Training model...[/]\n")
    model = CNNGRU()
    model.fit(
        X=train_features.to_numpy(),
        y=train_labels["label"].to_numpy(),  # type: ignore[union-attr]
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        chunk_len=CHUNK_LEN,
        chunks_per_epoch=CHUNKS_PER_EPOCH,
        learning_rate=LEARNING_RATE,
        stratify_sequences=STRATIFY_SEQUENCES,
        stratify_ratio=STRATIFY_RATIO,
        loss_last_k=LOSS_LAST_K,
        X_val=val_features.to_numpy(),
        y_val=val_labels["label"].to_numpy(),  # type: ignore[union-attr]
        eval_every=VAL_EVERY,
        eval_max_samples=VAL_MAX_SAMPLES,
        log_epoch_metrics=True,
        verbose=True,
    )

    rprint("\n[bold green]Evaluating model on validation set...[/]\n")
    evaluator = ModelEvaluator(
        test_features=val_features,
        test_labels=val_labels[["label"]],  # type: ignore[union-attr]
    )
    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)
    rprint("\n[bold green]Evaluation complete![/]\n")


if __name__ == "__main__":
    main()
