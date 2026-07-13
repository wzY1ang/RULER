#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python - <<'PY'
from pathlib import Path

import accelerate
import torch
import transformers
from packaging.version import Version

if Version(accelerate.__version__) < Version("1.6.0"):
    raise RuntimeError("accelerate>=1.6 is required with transformers 4.52.4")

for path in Path(".").rglob("*"):
    if path.is_file() and path.suffix in {".md", ".py", ".sh", ".txt"}:
        if b"\0" in path.read_bytes():
            raise RuntimeError(f"Null byte found in source file: {path}")

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
PY
python -m compileall -q .

for script in scripts/*.sh; do
  bash -n "$script"
done

python -m unittest discover -s tests -v
