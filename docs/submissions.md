# Submissions

This guide explains how to submit your trained model for evaluation on the held-out test set.

## How It Works

1. Train and tune your model using the **validation set** locally
2. **Commit and push** your model to your GitHub repository
3. GitHub Actions **validates** your submission (basic checks)
4. We **periodically evaluate** all valid submissions on the held-out test set
5. Results are posted on **Slack** with an updated leaderboard

## Submission Requirements

Your repository must contain:

1. **Your trained model file** (e.g., `model.pkl`, `model.pt`)
   - Saved within the repository
   - ≤25MB in size

2. **`model_metadata.json`**
   - Created automatically when you call `model.fit()`
   - Contains the path and import string to load your model

## Step-by-Step

### 1. Train Your Model

```python
from pathlib import Path
from brainstorm.loading import load_raw_data
from brainstorm.ml.logistic_regression import LogisticRegression

DATA_PATH = Path("./data")
train_features, train_labels = load_raw_data(DATA_PATH, step="train")

# Train - this saves model.pkl and model_metadata.json automatically
model = LogisticRegression(input_size=train_features.shape[1])
model.fit(X=train_features.values, y=train_labels["label"].values)
```

### 2. Test Locally (Recommended)

```python
from brainstorm.evaluation import ModelEvaluator

validation_features, validation_labels = load_raw_data(DATA_PATH, step="validation")

evaluator = ModelEvaluator(
    test_features=validation_features,
    test_labels=validation_labels[["label"]],
)
evaluator.evaluate()
```

### 3. Commit and Push

```bash
git add model.pkl model_metadata.json
git commit -m "Submit trained model"
git push origin main
```

### 4. Check Validation Status

1. Go to your repository on GitHub
2. Click the **"Actions"** tab
3. A green checkmark ✅ means your submission is valid

**Run validation locally first:**
```bash
uv run pytest tests/test_submission_requirements.py -v
```

### 5. Wait for Test Evaluation

Test set evaluation happens **periodically**, not on every push. Results are posted to **Slack**.

**Strategy:**
- Commit whenever your validation score improves
- Keep iterating locally—don't wait for test results
- We always evaluate your most recent valid commit

## What You Can Modify

✅ **Allowed:**
- Create custom model classes inheriting from `BaseModel`
- Use any architecture (RNNs, CNNs, Transformers, ensembles, etc.)
- Use any libraries (PyTorch, scikit-learn, JAX, etc.)
- Apply preprocessing, feature engineering, data augmentation
- Use pre-trained models and transfer learning
- Use AI coding tools (Cursor, Claude, GitHub Copilot, etc.)

The only requirements: your model must be **causal** (no future data) and follow the `BaseModel` interface.

## What You Cannot Modify

⚠️ Do **not** modify these evaluation files:

- `brainstorm/evaluation.py`
- `brainstorm/ml/metrics.py`
- `brainstorm/ml/base.py`

Any modifications will invalidate your submission.

## Troubleshooting

### Model too large (>25MB)

- Use smaller architectures
- Apply pruning or quantization
- Remove optimizer state from checkpoints
- Try knowledge distillation

### "Model metadata not found"

- Make sure you called `model.fit()` (not `fit_model()`)
- Check that `model_metadata.json` exists in the repo root

### "Model file MUST be saved within the repository"

- Your `save()` method is saving outside the repo
- Use paths relative to repo root: `Path(__file__).parent.parent / "model.pt"`

### Validation fails on GitHub Actions

Check the Actions logs for the specific error. Common issues:
- Missing `model_metadata.json` — run `model.fit()`
- Missing `uv.lock` — run `uv sync`
- Model file not found — check the path in metadata

### Submission errors during test evaluation

If your submission fails during periodic evaluation, we'll notify you on Slack. Test in a fresh environment:

```bash
git clone <your-repo> fresh-test
cd fresh-test
make install
python -c "from brainstorm.ml.your_model import YourModel; m = YourModel.load()"
```

## Multiple Submissions

**Yes, submit as often as you want!** Each push triggers quick validation (~1-2 minutes). We evaluate your most recent valid commit during each periodic evaluation.

**Best practice:** Commit and push whenever your validation score improves.
