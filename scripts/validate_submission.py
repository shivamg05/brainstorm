"""
Validate submission requirements for BrainStorm 2026 Track 1.

This script runs the submission validation tests and provides user-friendly output.
You can run this locally before pushing to check if your submission is valid.

Usage:
    python scripts/validate_submission.py

Or with uv:
    uv run python scripts/validate_submission.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run submission validation tests."""
    repo_root = Path(__file__).parent.parent
    test_file = repo_root / "tests" / "test_submission_requirements.py"

    print("=" * 70)
    print("Running Submission Validation Tests")
    print("=" * 70)
    print()

    # Run pytest with verbose output
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "--color=yes",
        ],
        cwd=repo_root,
    )

    print()
    print("=" * 70)

    if result.returncode == 0:
        print("✅ All submission requirements passed!")
        print("=" * 70)
        print()
        print("Your submission is valid and ready to be pushed.")
        print("When you push, your submission will be queued for test set evaluation.")
        print()
        return 0
    else:
        print("❌ Validation failed")
        print("=" * 70)
        print()
        print("Fix the issues above and try again.")
        print("See the test output for specific requirements that failed.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
