import numpy as np
from dataclasses import dataclass
from sklearn.metrics import balanced_accuracy_score

MAX_LAG_SAMPLES = 500
LAG_EXP_FACTOR = 6.0
MAX_SIZE_MB = 5.0
SIZE_EXP_FACTOR = 4.0


@dataclass
class MetricsResults:
    total_score: float
    accuracy_score: float
    lag_score: float
    size_score: float
    accuracy: float
    avg_lag_samples: float
    model_size_bytes: int


def compute_lag_metric(
    y_true: np.ndarray, y_pred: np.ndarray, max_lag_samples: int = MAX_LAG_SAMPLES
) -> float:
    """
    Compute average lag when predictions catch up to label transitions.

    When labels transition from 0 to X, we look for when predictions next become X.
    If it happens within max_lag_samples, we record the time difference (lag).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        max_lag_samples: Maximum lag to consider (500 samples = 500ms at 1kHz)

    Returns:
        Average lag in samples (or max_lag_samples if no valid lags found)
    """
    lags = []

    # Find transitions from 0 to X in y_true
    for i in range(1, len(y_true)):
        if y_true[i - 1] == 0 and y_true[i] != 0:
            target_value = y_true[i]
            # Look for when y_pred next equals target_value
            for j in range(i, min(i + max_lag_samples + 1, len(y_pred))):
                if y_pred[j] == target_value:
                    lag = j - i
                    lags.append(lag)
                    break

    if len(lags) == 0:
        return float(max_lag_samples)  # Worst case if no predictions matched

    return float(np.mean(lags))


def normalize_exponential_score(value: float, max_value: float, factor: float) -> float:
    """
    Normalize a value exponentially to a 0-1 range and return a
    score between 1 (v=0) and 0 (v=max_value).
    """

    normalized_value = value / max_value
    return np.exp(-factor * normalized_value)


def compute_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_size_bytes: int,
    accuracy_score_factor: float = 50.0,
    lag_score_factor: float = 25.0,
) -> MetricsResults:
    """
    Compute the final score as a weighted combination of metrics.

    The score is:
        - 50% Balanced Accuracy (normalized to 0-100)
        - 25% Lag (faster predictions are better, max 500ms)
        - 25% Model Size (smaller is better, with 5MB target)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_size_bytes: Model size in bytes
        accuracy_score_factor: Weight for accuracy (default 50)
        lag_score_factor: Weight for lag metric (default 25)

    Returns:
        MetricsResults containing total_score, accuracy_score, lag_score, size_score, accuracy, avg_lag_samples

    Scoring Details:
        - Balanced Accuracy: Linearly scaled from 0-100%
        - Lag: Exponential decay from max points at 0ms to ~0 at 500ms
        - Model Size: Exponential decay from max points at 0MB to ~0 at 5MB+
    """

    # Accuracy component (accuracy_score_factor points max, default 50)
    accuracy = balanced_accuracy_score(y_true, y_pred)
    accuracy_score = accuracy * accuracy_score_factor

    # Lag component (lag_score_factor points max, default 25)
    # Exponential decay: max points at 0ms lag, ~0 points at 500ms
    avg_lag_samples = compute_lag_metric(
        y_true, y_pred, max_lag_samples=int(MAX_LAG_SAMPLES)
    )

    # we use 6 so that at 100ms we get ~33% of the points, at 500ms we get ~0% of the points
    lag_score = (
        normalize_exponential_score(avg_lag_samples, MAX_LAG_SAMPLES, LAG_EXP_FACTOR)
        * lag_score_factor
    )

    # Model size component (remaining points, default 25)
    # Exponential decay: max points at 0MB, ~0 points at 5MB
    model_size_factor = 100.0 - accuracy_score_factor - lag_score_factor
    size_mb = model_size_bytes / (1024 * 1024)

    # we use 4 so that at 1MB we get ~50% of the points, at 5MB we get ~0% of the points
    size_score = (
        normalize_exponential_score(size_mb, MAX_SIZE_MB, SIZE_EXP_FACTOR)
        * model_size_factor
    )

    total_score = accuracy_score + lag_score + size_score
    return MetricsResults(
        total_score=total_score,
        accuracy_score=accuracy_score,
        lag_score=lag_score,
        size_score=size_score,
        accuracy=accuracy,
        avg_lag_samples=avg_lag_samples,
        model_size_bytes=model_size_bytes,
    )
