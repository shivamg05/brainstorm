# Makefile for Brainstorm project development

.PHONY: help check-uv install install-package install-dev setup-hooks sync format lint type-check test test-cov check-all clean

# Default target
help:
	@echo "Available commands:"
	@echo "  install        - Complete setup: check UV, create venv, install package with dev dependencies, setup git hooks"
	@echo "  install-package - Install the package only (no venv creation)"
	@echo "  install-dev    - Install with development dependencies"
	@echo "  sync           - Sync/reinstall all dependencies (fixes missing packages)"
	@echo "  setup-hooks    - Setup git pre-commit hooks to prevent large file commits"
	@echo "  format         - Format code with ruff"
	@echo "  lint           - Lint code with ruff"
	@echo "  type-check     - Run mypy type checking (requires install-dev)"
	@echo "  test           - Run tests with pytest"
	@echo "  test-cov       - Run tests with coverage"
	@echo "  check-all      - Run all checks (format, lint, type-check, test)"
	@echo "  clean          - Clean up __pycache__ and .pyc files"

# Check if UV is installed
check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "❌ Error: UV is not installed!"; \
		echo ""; \
		echo "UV is required to manage dependencies for this project."; \
		echo "Please install UV using one of these methods:"; \
		echo ""; \
		echo "  macOS/Linux:"; \
		echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo ""; \
		echo "  Homebrew:"; \
		echo "    brew install uv"; \
		echo ""; \
		echo "  Windows:"; \
		echo "    powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""; \
		echo ""; \
		echo "For more information, visit: https://docs.astral.sh/uv/"; \
		echo ""; \
		exit 1; \
	}
	@echo "✓ UV is installed (version: $$(uv --version))"

# Setup git hooks to prevent committing large files
setup-hooks:
	@echo "Setting up git hooks..."
	@chmod +x .git-hooks/pre-commit
	@git config core.hooksPath .git-hooks
	@echo "✓ Git hooks configured successfully"
	@echo "  Pre-commit hook will prevent files >25MB from being committed"

# Complete installation: check UV, create venv, install package, setup hooks
install: check-uv
	@echo "Starting complete installation..."
	@echo ""
	@echo "Step 1/4: Creating virtual environment with UV..."
	@if [ ! -d ".venv" ]; then \
		uv venv; \
		echo "✓ Virtual environment created"; \
	else \
		echo "✓ Virtual environment already exists"; \
	fi
	@echo ""
	@echo "Step 2/4: Installing package with development dependencies..."
	@uv pip install -e ".[dev]"
	@echo "✓ Package installed"
	@echo ""
	@echo "Step 3/4: Setting up git hooks..."
	@$(MAKE) setup-hooks
	@echo ""
	@echo "Step 4/4: Syncing dependencies..."
	@uv sync
	@echo "✓ Dependencies synced"
	@echo ""
	@echo "✅ Installation complete!"
	@echo ""
	@echo "To activate the virtual environment:"
	@echo "  source .venv/bin/activate    # macOS/Linux"
	@echo "  .venv\\Scripts\\activate        # Windows"
	@echo ""
	@echo "You can now run 'make check-all' to verify everything is working."

# Basic package installation (no venv creation)
install-package:
	@$(MAKE) check-uv
	@uv pip install -e .

# Install with development dependencies (no venv creation)
install-dev:
	@$(MAKE) check-uv
	@uv pip install -e ".[dev]"

# Sync dependencies (reinstall everything, useful if packages are missing)
sync:
	@$(MAKE) check-uv
	@echo "Syncing dependencies..."
	@uv sync
	@echo "✓ Dependencies synced"

# Code formatting
format:
	uv run --active ruff format .

# Linting
lint:
	uv run --active ruff check . --fix

# Type checking (requires development dependencies)
type-check:
	@uv run --active python -c "import mypy" 2>/dev/null || { \
		echo "❌ Error: mypy is not installed!"; \
		echo ""; \
		echo "Type checking requires development dependencies."; \
		echo "Please run: make install-dev"; \
		echo ""; \
		exit 1; \
	}
	@uv run --active mypy brainstorm/

# Testing
test:
	uv run --active pytest

test-cov:
	uv run --active pytest --cov=brainstorm --cov-report=html --cov-report=term

# Run all checks
check-all: format lint type-check test

# Clean up Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
