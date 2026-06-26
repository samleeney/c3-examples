#!/bin/bash
set -euo pipefail

python3 --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 search.py --candidates 32 --seeds 4 --rounds 2 --workers 8
