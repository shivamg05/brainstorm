#!/usr/bin/env python3
"""
Example script: Evaluate a saved CNN+GRU model (no training).

Usage:
    uv run python examples/example_cnn_gru_evaluate_only.py
"""

from pathlib import Path

from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.loading import load_raw_data


DATA_PATH = Path("./data")


def _print_dataset_table(val_features, val_labels) -> None:
    console = Console()
    table = Table(title="Validation Overview", show_header=True, header_style="bold magenta")
    table.add_column("Split", style="cyan", width=10)
    table.add_column("Features Shape", style="green")
    table.add_column("Labels Shape", style="green")
    table.add_column("Time Range (s)", style="yellow")
    table.add_column("Unique Labels", style="blue")

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
    rprint("\n[bold green]Evaluating CNN+GRU (saved model)...[/]\n")

    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("\n[bold yellow]Downloading data from Hugging Face...[/]\n")
        download_train_validation_data()
        rprint("[bold green]✓ Data downloaded successfully![/]\n")

    rprint(f"\n[bold cyan]Loading data from:[/] {DATA_PATH}\n")
    val_features, val_labels = load_raw_data(DATA_PATH, step="validation")
    _print_dataset_table(val_features, val_labels)

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
