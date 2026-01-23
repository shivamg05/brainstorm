# Installation

## Requirements

- **Python**: 3.10 or higher (tested with 3.12)
- **Hardware**: 
  - The dataset is small enough to train models in minutes on a standard laptop
  - Mac Silicon or GPU recommended for faster training but NOT required
  - CPU-only machines work fine

**⚠️ This project uses [uv](https://docs.astral.sh/uv/) for dependency management.** UV ensures fast, reproducible Python environments.

## Step 1: Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more options: https://docs.astral.sh/uv/

## Step 2: Install the Project

```bash
# Clone your repository
git clone <repo-url>
cd <your-repo-folder>

# Install everything
make install
```

This will:
- Check if UV is installed
- Create a virtual environment
- Install the package with dependencies
- Setup git hooks to prevent committing files >25MB
- Create `uv.lock`

Then activate the virtual environment:

```bash
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

## Step 3: Verify Installation

```bash
# Download the dataset
uv run python -c "from brainstorm.download import download_train_validation_data; download_train_validation_data()"

# Run the example
uv run python examples/example_local_train_and_evaluate.py
```

## Alternative: Manual Setup

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Sync all dependencies
uv sync

# Setup git hooks
make setup-hooks
```

## Git Hooks

The pre-commit hook prevents accidentally committing large files (>25MB). To bypass (not recommended for model files):

```bash
git commit --no-verify
```
