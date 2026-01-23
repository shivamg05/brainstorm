# Git Hooks

This directory contains git hooks to help maintain repository quality and prevent common issues.

## Pre-commit Hook: File Size Validation

The `pre-commit` hook prevents committing files larger than 25MB, which is the limit for the evaluation system.

### Installation

To enable this hook, run these commands from the repository root:

```bash
chmod +x .git-hooks/pre-commit
git config core.hooksPath .git-hooks
```

### What it does

- Automatically checks all files being committed
- Rejects the commit if any file exceeds 25MB
- Provides helpful error messages with suggestions for reducing file size

### Bypassing the hook

If you absolutely need to commit a large file (not recommended for model files):

```bash
git commit --no-verify
```

**Note:** Model files that exceed 25MB will be rejected by the training pipeline anyway, so it's better to address size issues early.

## Troubleshooting

If you need to remove a large file that was already committed:

```bash
# Remove file from git history (use with caution)
git filter-branch --tree-filter 'rm -f path/to/large/file' HEAD
```

Or use the more modern approach:

```bash
git filter-repo --path path/to/large/file --invert-paths
```

