#!/usr/bin/env python3
"""Merge artifact leaderboards from several C3 shard jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from search import leaderboard_from_rows


def artifact_dir(path: Path) -> Path:
    if (path / "evaluations.jsonl").exists():
        return path
    if (path / "artifacts" / "evaluations.jsonl").exists():
        return path / "artifacts"
    raise FileNotFoundError(f"no evaluations.jsonl found under {path}")


def read_rows(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge C3 shard artifacts into one leaderboard.")
    parser.add_argument("paths", nargs="+", help="Artifact directories or pulled job directories.")
    parser.add_argument("--out", default="merged-results", help="Output directory.")
    args = parser.parse_args()

    all_rows: list[dict[str, float | int | str]] = []
    source_dirs = [artifact_dir(Path(p)) for p in args.paths]
    for source in source_dirs:
        all_rows.extend(read_rows(source / "evaluations.jsonl"))

    leaderboard = leaderboard_from_rows(all_rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.json").write_text(
        json.dumps(
            {
                "sources": [str(path) for path in source_dirs],
                "evaluations": len(all_rows),
                "top_10": leaderboard[:10],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Merged Algorithm Discovery Leaderboard",
        "",
        f"Shard artifacts: {len(source_dirs)}",
        f"Evaluations: {len(all_rows)}",
        "",
        "| Rank | Candidate | Shard | Round | Parent | Mean reward | Median best value | Evaluations |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(leaderboard[:10], start=1):
        lines.append(
            f"| {rank} | {row['candidate']} | {row['shard']} | {row['round']} | {row['parent'] or '-'} | "
            f"{row['mean_reward']:.6f} | {row['median_best_value']:.9f} | {row['evaluations']} |"
        )
    (out / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'leaderboard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
