#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -c 'import torch, transformers; print("torch", torch.__version__); print("transformers", transformers.__version__)'
python -m compileall -q .

for script in scripts/*.sh; do
  bash -n "$script"
done

python -m unittest discover -s tests -v
