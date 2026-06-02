# Contributing

Thanks for your interest in improving this project! 🎉

## Ways to contribute

- 🐛 **Bug reports** — open an [Issue](https://github.com/pedro799ab-sketch/rtdetr/issues/new/choose) with a minimal reproduction.
- ✨ **New analyses** — add a script under `tools/` plus a CSV of results and a matching plot.
- 📝 **Docs** — clarifications, typo fixes, methodology improvements all welcome.

## Development setup

```bash
git clone https://github.com/pedro799ab-sketch/rtdetr.git
cd rtdetr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Coding guidelines

- Python ≥ 3.10.
- Match the surrounding style — this is a research codebase, readability over cleverness.
- Don't commit:
  - Model weights (`*.pth`, `*.pt`, `*.onnx`, ...).
  - Datasets (`dataset/coco/`).
  - Anything > 100 MB (GitHub hard limit).
- Do commit:
  - The script that produced a result.
  - The summary CSV (small).
  - The generated plot (PNG).

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add (K, batch_size) Pareto sweep
fix: correct CPU utilization normalization on Apple Silicon
docs: clarify power formula in README
```

## Pull requests

1. Fork → branch → commit → push → open a PR against `main`.
2. CI must pass (lint + smoke import).
3. Describe **what** changed and **why** in the PR body.
