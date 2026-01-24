#!/usr/bin/env python3
"""
Example script: Preprocess -> Train -> Evaluate (EEG-TCNet)

Usage:
    uv run python examples/example_eeg_tcnet_train_and_evaluate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.loading import load_channel_coordinates, load_raw_data
from brainstorm.ml.eeg_tcnet import EEGTCNet
from brainstorm.preprocessing import design_bandpass_sos, preprocess_spatial_features


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("./data")

EPOCHS = 36
BATCH_SIZE = 16
SEQ_LEN = 96
STRIDE = SEQ_LEN
CHUNK_LEN = 8000
CHUNKS_PER_EPOCH = 10
LEARNING_RATE = 1e-3
STRATIFY_SEQUENCES = True
STRATIFY_RATIO = 0.45
LOSS_LAST_K = 1

FS_HZ = 1000
TAU_S = 0.05
BANDS_HZ = [
    (1, 4),
    (4, 8),
    (8, 12),
    (13, 30),
    (30, 55),
    (65, 100),
    (100, 150),
    (150, 250),
]
POOL_KERNEL = 2  # 31x32 -> 15x16


def _print_dataset_table(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    val_features: pd.DataFrame,
    val_labels: pd.DataFrame,
) -> None:
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
    rprint("\n[bold green]Evaluating EEG-TCNet...[/]\n")

    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("\n[bold yellow]Downloading data from Hugging Face...[/]\n")
        download_train_validation_data()
        rprint("[bold green]✓ Data downloaded successfully![/]\n")

    rprint(f"\n[bold cyan]Loading data from:[/] {DATA_PATH}\n")
    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    val_features, val_labels = load_raw_data(DATA_PATH, step="validation")
    _print_dataset_table(train_features, train_labels, val_features, val_labels)

    coords = load_channel_coordinates()
    sos_bank = design_bandpass_sos(BANDS_HZ, fs_hz=FS_HZ)

    train_spatial, stats = preprocess_spatial_features(
        train_features.to_numpy(dtype=np.float32),
        coords,
        sos_bank,
        fs_hz=FS_HZ,
        tau_s=TAU_S,
        pool_kernel=POOL_KERNEL,
        append_mask=True,
    )
    val_spatial, _ = preprocess_spatial_features(
        val_features.to_numpy(dtype=np.float32),
        coords,
        sos_bank,
        fs_hz=FS_HZ,
        tau_s=TAU_S,
        stats=stats,
        pool_kernel=POOL_KERNEL,
        append_mask=True,
    )

    X_train = train_spatial.reshape(train_spatial.shape[0], -1)
    X_val = val_spatial.reshape(val_spatial.shape[0], -1)
    y_train = train_labels["label"].to_numpy()

    rprint("\n[bold green]Training model...[/]\n")
    model = EEGTCNet(
        input_channels=train_spatial.shape[1],
        height=train_spatial.shape[2],
        width=train_spatial.shape[3],
        F1=16,
        D=2,
        F2=32,
        temporal_kernel=32,
        F2_bottleneck=8,
        tcn_channels=32,
        tcn_layers=3,
        dropout=0.4,
        context_window=SEQ_LEN,
        use_mask=True,
    )
    model.fit(
        X=X_train,
        y=y_train,  # type: ignore[union-attr]
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
        verbose=True,
    )

    rprint("\n[bold green]Evaluating model on validation set...[/]\n")
    val_df = pd.DataFrame(X_val, index=val_features.index)
    evaluator = ModelEvaluator(
        test_features=val_df,
        test_labels=val_labels[["label"]],  # type: ignore[union-attr]
    )
    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)
    rprint("\n[bold green]Evaluation complete![/]\n")


if __name__ == "__main__":
    main()
