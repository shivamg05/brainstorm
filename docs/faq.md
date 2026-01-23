# Frequently Asked Questions (FAQ)

## Submissions

### How does the submission system work?

1. Train locally using train/validation data
2. Commit your model and `model_metadata.json`
3. Push to GitHub → triggers validation
4. Periodically, we evaluate all valid submissions on the test set
5. Results posted to Slack

See the [Submissions Guide](submissions.md) for details.

### Where do I see my score?

Scores are posted to **Slack** after periodic test set evaluation. The green checkmark in GitHub Actions only means your submission passed validation (basic checks), not that it's been scored.

Use local validation to estimate your performance before submitting.

### Why did my submission fail?

Check the GitHub Actions logs for the specific error. Common issues:
- Missing `model_metadata.json` — train with `model.fit()` (not `fit_model()`)
- Model file not committed — check `git status`
- Model too large — must be ≤25MB

### How many times can I submit?

No limit. We evaluate your most recent valid commit during each periodic evaluation.

## Libraries and Tools

Make sure to use `uv` to manage your dependencies!

### Can I use external libraries?

**Yes!** Use any Python libraries: PyTorch, TensorFlow, JAX, scikit-learn, scipy, etc. Just include them as dependencies.

### Can I use ensemble models?

**Yes!** Ensembles, multi-model pipelines, or any architecture is allowed. The only requirement is that your model must be **causal**—no using future data.

### Can I use AI coding tools?

**Yes!** Cursor, Claude, GitHub Copilot, and any other AI coding assistants are encouraged.

### Can I use pre-trained models?

**Yes**, as long as:
- The final model (including pre-trained components) fits within 25MB
- Your model respects the causal/streaming constraint
- All dependencies are available during evaluation

## Model Development

### What preprocessing is allowed?

Anything, as long as:
1. Same preprocessing during training and inference
2. Preprocessing state is saved in your model file
3. Preprocessing is causal (no future data)

### Can I use the spatial electrode layout?

**Yes!** Electrode coordinates are provided. See [Dataset Guide](dataset.md#spatial-layout-of-electrodes) for how to use them.

### How do I optimize for the composite score?

- **50%** Accuracy — focus here first
- **25%** Latency — non-linear, improvements under 100ms matter most
- **25%** Size — non-linear, models under 1MB score best

See [Evaluation Guide](evaluation.md#optimization-strategy) for detailed strategy.

### My model is larger than 25MB

Options:
- Smaller architecture (fewer layers, smaller hidden dimensions)
- Pruning or quantization
- Remove optimizer state from checkpoint
- Knowledge distillation to a smaller model

### Can I modify the evaluation code?

You can read it locally, but **do not commit changes** to:
- `brainstorm/evaluation.py`
- `brainstorm/ml/metrics.py`
- `brainstorm/ml/base.py`

Any modifications invalidate your submission.

## Data

### Where is the test set?

The test set is held out and not provided to participants. It's only used during remote evaluation when you submit. Use your **validation set** for local testing.

### Can I use external datasets for pretraining?

Yes.
