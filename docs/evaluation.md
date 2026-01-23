# Evaluation

This guide explains how to evaluate your model locally and understand the scoring criteria.

## Local Evaluation

Before submitting, test your model on the validation set:

```python
from pathlib import Path
from brainstorm.loading import load_raw_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.ml.logistic_regression import LogisticRegression

DATA_PATH = Path("./data")
train_features, train_labels = load_raw_data(DATA_PATH, step="train")
validation_features, validation_labels = load_raw_data(DATA_PATH, step="validation")

# Train your model
model = LogisticRegression(input_size=train_features.shape[1], max_iter=20, use_pca=True)
model.fit(X=train_features.values, y=train_labels["label"].values)

# Evaluate on validation set
evaluator = ModelEvaluator(
    test_features=validation_features,
    test_labels=validation_labels[["label"]],
)
evaluator.evaluate()
```

Use your **validation set** for local testing—the test set is held out for remote evaluation only.

See `examples/example_local_train_and_evaluate.py` for a complete example.

## Scoring Criteria

Each submission receives a score from **0 to 100**, computed from three weighted factors:

### 1. Balanced Accuracy (50%)

The primary metric. Since the dataset contains periods of silence (class imbalance), we use balanced accuracy—performance across all classes, weighted by their frequency.

**Why balanced?** This prevents models from gaming the score by always predicting the most common class.

### 2. Prediction Lag (25%)

The time delay between stimulus onset and your model's correct classification.

**Scoring is non-linear.** Reducing lag in the low-latency range yields significantly more points:

| Latency | Approximate Score (out of 25) |
|---------|-------------------------------|
| 10ms    | ~22 pts |
| 50ms    | ~14 pts |
| 100ms   | ~8 pts |
| 500ms   | ~0 pts |

### 3. Model Size (25%)

The size of your saved model file on disk.

**Scoring is non-linear** to reward ultra-compact models suitable for edge devices:

| Size | Approximate Score (out of 25) |
|------|-------------------------------|
| 0.5MB | ~17 pts |
| 1MB   | ~11 pts |
| 5MB   | ~0.5 pts |
| 25MB  | ~0 pts (but still valid) |

**Hard limit:** Models larger than 25MB are rejected.

### Scoring Curves

![Scoring Rulers](scoring.png)

## Detailed Formulas

**Accuracy Score:**
```
Accuracy Score = balanced_accuracy × 50
```

**Lag Score:**
```
Lag Score = exp(-6 × lag_ms / 500) × 25
```

**Size Score:**
```
Size Score = exp(-4 × size_mb / 5) × 25
```

**Total:**
```
Total = Accuracy Score + Lag Score + Size Score
```

## Optimization Strategy

1. **Focus on accuracy first** — It's 50% of your score and scales linearly
2. **Then optimize latency** — Big gains for improvements under 100ms
3. **Finally compress your model** — Models under 1MB get the best size scores

The non-linear scoring means small/fast models are rewarded disproportionately at the low end.

## Validation vs Test Set

- **Validation set**: Available to you for local testing and hyperparameter tuning
- **Test set**: Held out for remote evaluation only

Your local validation scores should approximate your test set performance.
