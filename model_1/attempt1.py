from pathlib import Path
from rich import print as rprint
from rich.console import Console
from rich.table import Table
import numpy as np
from scipy.signal import butter, sosfilt

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.ml.mlp import MLP
from brainstorm.ml.logistic_regression import LogisticRegression

# Path to the formatted data directory
DATA_PATH = Path("./data")

# Training parameters for MLP
EPOCHS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# Training parameters for Logistic Regression
MAX_ITER = 20
USE_PCA = True

MODEL_TO_USE = "mlp"

def main() -> None:
    rprint("\n[bold green]Evaluating model...[/]\n")

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

    # Add training data row
    table.add_row(
        "Train",
        str(train_features.shape),
        str(train_labels.shape),
        f"{train_features.index[0]:.2f} → {train_features.index[-1]:.2f}",
        str(sorted(train_labels["label"].unique().tolist())),
    )

    # Add test data row
    table.add_row(
        "Test",
        str(validation_features.shape),
        str(validation_labels.shape),
        f"{validation_features.index[0]:.2f} → {validation_features.index[-1]:.2f}",
        str(sorted(validation_labels["label"].unique().tolist())),
    )

    console.print(table)
    print()

    # Define train data and labels
    X_train = train_features.to_numpy()              # (n_samples, 1024)
    y_train = train_labels["label"].to_numpy()       # (n_samples,)

    #

    # TODO remove average across all channels to reduce noise

    X_train_car = X_train - X_train.mean(axis=1, keepdims=True).astype(np.float32) # remove time invariant component 
    # common average reference
    # note this is only for one time step at a time

    # TODO compute bandpass power vals

    fs = 1000 # sampling rate

    # define bandwidths
    bands = [
    (1, 4),
    (4, 8),
    (8, 12),
    (13, 30),
    (30, 55),
    (65, 100),   # skip 55–65 to avoid 60 Hz noise
    (100, 150),
    (150, 250),
    ]

    nyq = fs / 2 # this is the nyquist frequency, highest frequency we can reliably record at a given sampling rate

    sos_bank = [] # store one sos (second order sections) filter per band

    for (lo_hz, hi_hz) in bands:
        # Convert Hz cutoffs to normalized cutoffs in (0, 1),
        # where 1 corresponds to Nyquist.
        lo = lo_hz / nyq
        hi = hi_hz / nyq

        # Design a 4th-order Butterworth bandpass filter and return it as SOS.
        # "N=4" means the overall filter is 4th order.
        # Internally, SOS will represent it as multiple 2nd-order sections.
        sos = butter(
            N=4,
            Wn=[lo, hi],
            btype="bandpass",
            output="sos"      # key: stable second-order sections form
        )

        sos_bank.append(sos)

    n_samples, n_ch = X_train_car.shape # define the dimensions of each matrix
    n_bands = len(bands) # define the num of channels for our bandpass filters tensor

    # create a tensor to hold the power vals at each time at each bandwidth
    power_feat = np.empty((n_samples, n_bands, n_ch), dtype=np.float32)

    # EMA smoothing constant (choose 20–50ms to start)
    # Don't fully understand why we use this - CK
    tau_s = 0.05  # 50 ms
    alpha = 1.0 - np.exp(-1.0 / (fs * tau_s))
    for b, sos in enumerate(sos_bank):
        # y has same shape as X_train_car. It is the application of our filters over the data
        y = sosfilt(sos, X_train_car, axis=0).astype(np.float32)   # (n_samples, 1024)

        # define power as each voltage value squared
        p = y * y                                                   # (n_samples, 1024)

        # define another array the same shape as the power array above (n_samples, 1024)
        env = np.empty_like(p)
        # 0th value is the same
        env[0] = p[0]
        # this loop is used to smooth the raw power data
        for t in range(1, n_samples):
            # for each power value at time t, we will recorded as some weighted some of the instantaneous and past value
            # 1 step back
            env[t] = (1.0 - alpha) * env[t-1] + alpha * p[t]

        # normalize data with a logarithm
        # the 1e-8 prevents errors if the power at a given time is equal to 0
        power_feat[:, b, :] = np.log(env + 1e-8)


    X_train_features = power_feat.reshape(n_samples, n_bands * n_ch) 
    # this is now a tensor, with 8 channels of n_time_samples rows by 1024 cols


    # TODO define model
    if MODEL_TO_USE.lower() == "mlp":
        model = MLP(
            input_size=X_train_features.shape[1],
            hidden_size=HIDDEN_SIZE,
        )

        # fit() calls fit_model(), saves the model, validates it, and saves metadata
        model.fit(
            X=X_train_features,
            y=train_labels["label"].values,  # type: ignore[union-attr]
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            verbose=True,
        )

    elif MODEL_TO_USE == "logreg":
        model = LogisticRegression(
            input_size=X_train_features.shape[1],
            max_iter=MAX_ITER,
            use_pca=USE_PCA,
        )

        model.fit(
            X=X_train_features,
            y=train_labels["label"].values,  # type: ignore[union-attr]
            verbose=True,
        )

    rprint("\n[bold green]Evaluating model on test set...[/]\n")
    # NOTE we use validation_features and labels because the test set is held out and not accessible for local evaluation.
    evaluator = ModelEvaluator(
        test_features=validation_features,
        test_labels=validation_labels[["label"]],  # type: ignore[union-attr]
    )

    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)

    rprint("\n[bold green]Evaluation complete![/]\n")

if __name__ == "__main__":
    main()
