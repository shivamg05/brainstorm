# Repository Guidelines

## Project Structure & Module Organization
- `brainstorm/` contains the core library; `brainstorm/ml/` holds model implementations, with utilities in modules like `loading.py`, `spatial.py`, and `plotting.py`.
- `tests/` contains pytest suites (unit and integration-style checks).
- `examples/` provides runnable scripts for local training/evaluation.
- `docs/` stores reference documentation; start with `docs/overview.md`.
- `data/` is for local datasets/artifacts (do not commit large files).

## Build, Test, and Development Commands
- `make install` sets up the UV-managed venv, installs dev deps, and configures git hooks.
- `make install-dev` installs development dependencies into the active environment.
- `make format` runs `ruff format .` to auto-format code.
- `make lint` runs `ruff check . --fix` for linting and autofixes.
- `make type-check` runs `mypy brainstorm/` (requires dev deps).
- `make test` runs `pytest`.
- `make test-cov` runs coverage: `pytest --cov=brainstorm`.

## Coding Style & Naming Conventions
- Python, 4-space indentation; prefer explicit type hints in `brainstorm/` (mypy is strict).
- Use `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_CASE` for constants.
- Format and lint with Ruff (`make format`, `make lint`) before pushing.

## Testing Guidelines
- Framework: `pytest` with tests under `tests/`.
- Naming: files `test_*.py`, functions `test_*` (see `pyproject.toml`).
- Add tests for new behaviors; include edge cases and I/O validation where possible.

## Commit & Pull Request Guidelines
- Git history is minimal; use short, imperative commit subjects (e.g., "Add spatial metric helper").
- PRs should include a clear summary, testing notes (commands run), and links to relevant issues.
- Include screenshots or plots for visualization changes (e.g., updates in `brainstorm/plotting.py`).

## Data & Submission Constraints
- Do not commit large binaries; pre-commit hooks block files >25MB (`make install` sets them up).
- Submission rules prohibit non-causal models and modifying evaluation code; see `docs/faq.md`.
