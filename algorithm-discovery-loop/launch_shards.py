#!/usr/bin/env python3
"""Launch an algorithm-discovery loop across several C3 GPU jobs."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


JOB_RE = re.compile(r"job_[0-9]+_[a-z0-9]+")


def shard_script(shard: int, args: argparse.Namespace) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=algo-loop-s{shard}
#SBATCH --gres=gpu:l40:1
#SBATCH --time=00:05:00
#C3 GPU l40

set -euo pipefail

python3 --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 search.py \\
  --candidates {args.candidates} \\
  --seeds {args.seeds} \\
  --rounds {args.rounds} \\
  --workers {args.workers} \\
  --shard {shard} \\
  --shards {args.shards}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit several C3 shard jobs for one discovery loop.")
    parser.add_argument("--shards", type=int, default=3, help="Number of C3 GPU jobs to launch. Free trial users can burst to 3; Team can burst higher.")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.shards < 1:
        raise SystemExit("--shards must be at least 1")
    if args.shards > args.candidates:
        raise SystemExit("--shards must be less than or equal to --candidates")
    if args.shards != 3:
        print("Note: the default demo uses 3 shards because that fits the free trial burst limit.", file=sys.stderr)
    if args.shards > 50:
        print("Note: Team plans can burst to 50 GPUs; larger runs need a custom limit.", file=sys.stderr)

    script_dir = Path(".c3-shards")
    script_dir.mkdir(exist_ok=True)

    processes: list[tuple[int, Path, subprocess.Popen[str]]] = []
    for shard in range(args.shards):
        path = script_dir / f"shard-{shard}.sbatch"
        path.write_text(shard_script(shard, args), encoding="utf-8")
        path.chmod(0o755)
        print(f"Submitting shard {shard + 1}/{args.shards}...")
        proc = subprocess.Popen(
            ["c3", "deploy", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard, path, proc))

    job_ids: list[str] = []
    for shard, path, proc in processes:
        assert proc.stdout is not None
        output = proc.stdout.read()
        code = proc.wait()
        if code != 0:
            print(output)
            raise SystemExit(f"shard {shard} failed to submit with exit code {code}")
        matches = JOB_RE.findall(output)
        if not matches:
            print(output)
            raise SystemExit(f"could not find job id in deploy output for {path}")
        job_id = matches[-1]
        job_ids.append(job_id)
        print(f"Shard {shard + 1}: {job_id}")

    Path("shard-jobs.txt").write_text("\n".join(job_ids) + "\n", encoding="utf-8")
    print("\nSubmitted shard jobs:")
    for job_id in job_ids:
        print(f"  {job_id}")
    print("\nWhen they finish:")
    print("  while read job; do c3 pull \"$job\"; done < shard-jobs.txt")
    print("  python3 merge.py job_*/artifacts")
    print("  cat merged-results/leaderboard.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
