# BrainStorm 2026 - Track 1

Welcome to the BrainStorm 2026 Brain-Computer Interface (BCI) Hackathon! Build accurate, fast, and lightweight neural decoders for real-time auditory stimulus classification from ECoG recordings.

## Quick Start

```bash
# 1. Install dependencies
make install
source .venv/bin/activate

# 2. Download data
uv run python -c "from brainstorm.download import download_train_validation_data; download_train_validation_data()"

# 3. Train and evaluate
uv run python examples/example_local_train_and_evaluate.py
```

## Documentation

### Getting Started

1. **[Overview](docs/overview.md)** - Understand the problem, constraints, and scoring
2. **[Installation](docs/installation.md)** - Set up your environment
3. **[Dataset](docs/dataset.md)** - Learn about the ECoG data format

### Building Your Solution

4. **[Defining a Model](docs/defining_a_model.md)** - Create custom models
5. **[Evaluation](docs/evaluation.md)** - Test locally and understand scoring
6. **[Submissions](docs/submissions.md)** - Submit for test set evaluation
7. **[FAQ](docs/faq.md)** - Common questions

## Project Structure

```
brainstorm2026-track1/
├── brainstorm/           # Core library
│   ├── ml/              # Model implementations
│   ├── loading.py       # Data loading
│   ├── spatial.py       # Spatial utilities
│   └── plotting.py      # Visualization
├── docs/                # Documentation
├── examples/            # Example scripts
└── tests/               # Test suite
```

## Rules

✅ **Allowed:** Any Python libraries, ensemble models, AI coding tools, pre-trained models

❌ **Not Allowed:** Non-causal models, modifying evaluation code, models >25MB

See the [FAQ](docs/faq.md) for detailed rules.

---

Good luck, and happy hacking! 🧠⚡
