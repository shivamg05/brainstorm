# Defining a Model

This guide explains how to create a custom model for continuous classification of ECoG signals. All models must inherit from `BaseModel` and implement a standardized interface.

## Quick Start

```python
from brainstorm.ml.mlp import MLP

# Train (automatically saves to model.pt and validates)
model = MLP(hidden_size=256)
model.fit(X_train, y_train, epochs=50)

# Load for inference
model = MLP.load()
prediction = model.predict(X_test[0])
```

## The BaseModel Interface

Your custom model must implement four methods:

### 1. `fit_model(X, y, **kwargs) -> None`

Train your model on the provided data.

- `X`: Feature array of shape `(n_samples, 1024)`
- `y`: Label array of shape `(n_samples,)` — integer labels (0 = no stimulus, >0 = frequency in Hz)

**Important:** Don't call this directly. Use `fit()` instead—it handles saving and validation.

### 2. `predict(X) -> int`

Predict the label for a single timestep.

- `X`: Feature array of shape `(1024,)` for one timestep
- Returns: Integer label (frequency in Hz, or 0)

This is called once per timestep during evaluation—make it efficient.

### 3. `save() -> Path`

Save your trained model to disk. Returns the path to the saved file.

**Requirements:**
- Save within the repository directory (files outside can't be uploaded)
- Keep under 25MB (smaller models score better)
- Include everything needed for inference: weights, config, preprocessors

### 4. `load() -> Self` (classmethod)

Load a trained model from disk. Returns a new instance ready for inference.

## The `fit()` Method

Always use `fit()` to train your model. It automatically:

1. Trains your model via `fit_model()`
2. Saves the model via `save()`
3. Validates the model can be loaded
4. Checks the file is ≤25MB
5. Saves `model_metadata.json` for the evaluation system

```python
# ✅ Correct
model.fit(X_train, y_train, epochs=50)

# ❌ Wrong - skips validation
model.fit_model(X_train, y_train)
```

## Example Models

### PyTorch MLP

```python
from brainstorm.ml.mlp import MLP

model = MLP(hidden_size=256, dropout=0.3)
model.fit(X_train, y_train, epochs=50, batch_size=64, learning_rate=1e-3)

# Saves to: model.pt
```

### Scikit-learn Logistic Regression

```python
from brainstorm.ml.logistic_regression import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    use_pca=True,
    n_components=100
)
model.fit(X_train, y_train)

# Saves to: model.pkl
```

## Best Practices

### Save Within the Repository

Your model file **must** be inside the repository for remote evaluation.

```python
# ✅ Correct: relative to repo root
_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "my_model.pt"

# ❌ Wrong: outside repository
MODEL_PATH = Path.home() / "models" / "my_model.pt"
```

### Save Everything Needed for Inference

Include all state required to reconstruct the model:

```python
checkpoint = {
    "config": {"hidden_size": self.hidden_size, ...},
    "classes": self.classes_,
    "state_dict": self.state_dict(),
    "scaler": self._scaler,  # Don't forget preprocessors!
}
```

### Efficient Prediction

The `predict()` method is called thousands of times during evaluation:

```python
def predict(self, X: np.ndarray) -> int:
    self.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32)
        logits = self.forward(x_tensor)
        return int(torch.argmax(logits).item())
```

### 4. Handle Preprocessing Consistently

If you transform features during training (scaling, PCA, etc.), apply the same transformations during inference:

```python
def predict(self, X: np.ndarray) -> int:
    # Apply same preprocessing pipeline
    if self._scaler is not None:
        X = self._scaler.transform(X.reshape(1, -1))
    if self._pca is not None:
        X = self._pca.transform(X)
    
    return self._model.predict(X)[0]
```

### 5. Validate Before Submission

The `fit()` method automatically validates your model. If validation fails, you'll see an error immediately:

Common validation failures:

**1. Model saved outside repository:**
```
RuntimeError: Model file MUST be saved within the repository for remote evaluation!
Model path: /Users/you/models/model.pt
Repository root: /Users/you/repo
```
**Fix:** Use paths relative to repository root (see "Save Within Repository" above).

**2. Model file too large:**
```
RuntimeError: Model file exceeds 25MB size limit
```
**Fix:** Use compression, reduce model size, or apply quantization.

**3. Model cannot be loaded:**
```
RuntimeError: Model validation failed! The saved model cannot be loaded.
```
**Fix:** Ensure `save()` includes all necessary state and `load()` correctly reconstructs it.

**Other common issues:**
- ❌ Missing dependencies in `save()` checkpoint
- ❌ Import path errors in `load()`
- ❌ State mismatch between saved and loaded model

## Integration with Evaluation

Your custom model integrates seamlessly with the evaluation system through the `BaseModel` interface.

### How Evaluation Works

When you train your model using `fit()`:
1. Your `fit_model()` implementation trains the model
2. `save()` is called to persist the model
3. The system validates the model can be loaded via `load()`
4. Metadata is saved to `model_metadata.json` for remote evaluation

During evaluation (both local and remote):
1. `ModelEvaluator` loads your model using the metadata
2. `predict()` is called once per timestep in the test data
3. Predictions are scored on accuracy, lag, and model size

### Critical: Do Not Modify Base Files

⚠️ **IMPORTANT**: Do NOT modify `brainstorm/ml/base.py` or any evaluation-related files.

The following files contain evaluation logic and must remain unchanged:
- `brainstorm/ml/base.py` - BaseModel interface
- `brainstorm/evaluation.py` - ModelEvaluator class
- `brainstorm/ml/metrics.py` - Scoring functions

Any modifications to these files will invalidate your submission. The remote evaluation system requires the original implementation to ensure fair scoring.

**You should only:**
- Create new model classes that inherit from `BaseModel`
- Implement the required methods (`fit_model`, `predict`, `save`, `load`)
- Add your own helper functions and utilities in separate files

## Testing Your Model

Test locally before submitting:

```python
from brainstorm.evaluation import ModelEvaluator

evaluator = ModelEvaluator(
    test_features=validation_features,
    test_labels=validation_labels[["label"]],
)
results = evaluator.evaluate()
```

See `examples/example_local_train_and_evaluate.py` for a complete example.

## What Not to Modify

⚠️ Do **not** modify these files—they're used for evaluation:

- `brainstorm/ml/base.py`
- `brainstorm/evaluation.py`
- `brainstorm/ml/metrics.py`

Create new model files instead of editing these.
