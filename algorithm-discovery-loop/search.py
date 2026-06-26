#!/usr/bin/env python3
"""Tiny algorithm-discovery loop for C3.

The script runs a small generate-evaluate-select loop: propose optimizer
settings, evaluate them across a task/seed grid, mutate the winners, and write
artifacts an agent could use for the next step.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import statistics
import time
from pathlib import Path


TASKS = ("sphere", "rastrigin", "rosenbrock")


def task_value(task: str, x: float, y: float) -> float:
    if task == "sphere":
        return x * x + y * y
    if task == "rastrigin":
        return 20 + x * x + y * y - 10 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))
    if task == "rosenbrock":
        return (1 - x) ** 2 + 100 * (y - x * x) ** 2
    raise ValueError(f"unknown task: {task}")


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_candidate(
    candidate_id: int,
    round_id: int,
    shard: int,
    shards: int,
    parent: dict[str, float | int | str] | None = None,
) -> dict[str, float | int | str]:
    rng = random.Random(10_000 + round_id * 1_000 + candidate_id)
    if parent is None:
        step = rng.uniform(0.05, 0.8)
        cooling = rng.uniform(0.001, 0.06)
        momentum = rng.uniform(0.0, 0.85)
        explore = rng.uniform(0.0, 0.12)
        parent_name = ""
    else:
        step = clip(float(parent["step"]) + rng.gauss(0.0, 0.08), 0.05, 0.8)
        cooling = clip(float(parent["cooling"]) + rng.gauss(0.0, 0.008), 0.001, 0.06)
        momentum = clip(float(parent["momentum"]) + rng.gauss(0.0, 0.08), 0.0, 0.85)
        explore = clip(float(parent["explore"]) + rng.gauss(0.0, 0.02), 0.0, 0.12)
        parent_name = str(parent["name"])

    return {
        "id": candidate_id,
        "round": round_id,
        "shard": shard,
        "shards": shards,
        "name": f"r{round_id}-policy-{candidate_id:02d}",
        "parent": parent_name,
        "step": round(step, 6),
        "cooling": round(cooling, 6),
        "momentum": round(momentum, 6),
        "explore": round(explore, 6),
    }


def evaluate_one(args: tuple[dict[str, float | int | str], str, int, int]) -> dict[str, float | int | str]:
    candidate, task, seed, steps = args
    rng = random.Random(seed * 100_000 + int(candidate["id"]))
    x = rng.uniform(-4.0, 4.0)
    y = rng.uniform(-4.0, 4.0)
    best = task_value(task, x, y)
    vx = 0.0
    vy = 0.0

    for t in range(steps):
        scale = float(candidate["step"]) / (1.0 + float(candidate["cooling"]) * t)
        vx = float(candidate["momentum"]) * vx + rng.gauss(0.0, scale)
        vy = float(candidate["momentum"]) * vy + rng.gauss(0.0, scale)
        nx = max(-5.0, min(5.0, x + vx))
        ny = max(-5.0, min(5.0, y + vy))
        value = task_value(task, nx, ny)

        if value < best or rng.random() < float(candidate["explore"]):
            x, y = nx, ny
            best = min(best, value)

    reward = -math.log10(best + 1e-9)
    return {
        "candidate_id": int(candidate["id"]),
        "candidate": str(candidate["name"]),
        "round": int(candidate["round"]),
        "shard": int(candidate["shard"]),
        "shards": int(candidate["shards"]),
        "parent": str(candidate["parent"]),
        "step": float(candidate["step"]),
        "cooling": float(candidate["cooling"]),
        "momentum": float(candidate["momentum"]),
        "explore": float(candidate["explore"]),
        "task": task,
        "seed": seed,
        "best_value": round(best, 9),
        "reward": round(reward, 6),
    }


def leaderboard_from_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    by_candidate: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate"]), []).append(row)

    leaderboard = []
    for name, candidate_rows in by_candidate.items():
        first = candidate_rows[0]
        rewards = [float(row["reward"]) for row in candidate_rows]
        values = [float(row["best_value"]) for row in candidate_rows]
        leaderboard.append(
            {
                "candidate": name,
                "round": int(first["round"]),
                "shard": int(first["shard"]),
                "parent": str(first["parent"]),
                "step": float(first["step"]),
                "cooling": float(first["cooling"]),
                "momentum": float(first["momentum"]),
                "explore": float(first["explore"]),
                "mean_reward": round(statistics.mean(rewards), 6),
                "median_best_value": round(statistics.median(values), 9),
                "evaluations": len(candidate_rows),
            }
        )
    leaderboard.sort(key=lambda row: float(row["mean_reward"]), reverse=True)
    return leaderboard


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, float | int | str]],
    candidates_by_round: list[list[dict[str, float | int | str]]],
    elapsed: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("evaluations.jsonl", "candidates.json", "leaderboard.json", "leaderboard.md"):
        path = output_dir / filename
        if path.exists():
            path.unlink()

    with (output_dir / "evaluations.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    (output_dir / "candidates.json").write_text(
        json.dumps({"rounds": candidates_by_round}, indent=2) + "\n",
        encoding="utf-8",
    )

    leaderboard = leaderboard_from_rows(rows)
    round_summaries = []
    for round_id, _ in enumerate(candidates_by_round):
        round_rows = [row for row in rows if int(row["round"]) == round_id]
        round_summaries.append(
            {
                "round": round_id,
                "evaluations": len(round_rows),
                "top_3": leaderboard_from_rows(round_rows)[:3],
            }
        )

    summary = {
        "elapsed_seconds": round(elapsed, 3),
        "tasks": list(TASKS),
        "evaluations": len(rows),
        "shard": int(rows[0]["shard"]),
        "shards": int(rows[0]["shards"]),
        "rounds": round_summaries,
        "winner": leaderboard[0],
        "top_5": leaderboard[:5],
        "next_step": "Use top_5 as parents for the next candidate batch.",
    }
    (output_dir / "leaderboard.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Algorithm Discovery Leaderboard",
        "",
        f"Rounds: {len(candidates_by_round)}",
        f"Evaluations: {len(rows)}",
        f"Elapsed seconds: {elapsed:.3f}",
        "",
        "| Rank | Candidate | Shard | Round | Parent | Mean reward | Median best value | Evaluations |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(leaderboard[:10], start=1):
        lines.append(
            f"| {rank} | {row['candidate']} | {row['shard']} | {row['round']} | {row['parent'] or '-'} | "
            f"{row['mean_reward']:.6f} | {row['median_best_value']:.9f} | {row['evaluations']} |"
        )
    (output_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny grid-parallel algorithm discovery loop.")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.candidates < 2:
        raise SystemExit("--candidates must be at least 2")
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")
    if args.shards < 1:
        raise SystemExit("--shards must be at least 1")
    if args.shard < 0 or args.shard >= args.shards:
        raise SystemExit("--shard must be in the range [0, --shards)")

    candidate_ids = [i for i in range(args.candidates) if i % args.shards == args.shard]
    if not candidate_ids:
        raise SystemExit("this shard has no candidates; reduce --shards or increase --candidates")

    start = time.perf_counter()
    all_rows: list[dict[str, float | int | str]] = []
    candidates_by_round: list[list[dict[str, float | int | str]]] = []
    parents: list[dict[str, float | int | str]] = []

    for round_id in range(args.rounds):
        if round_id == 0:
            candidates = [make_candidate(i, round_id, args.shard, args.shards) for i in candidate_ids]
        else:
            candidates = [
                make_candidate(candidate_id, round_id, args.shard, args.shards, parents[idx % len(parents)])
                for idx, candidate_id in enumerate(candidate_ids)
            ]
        candidates_by_round.append(candidates)
        jobs = [
            (candidate, task, seed, args.steps)
            for candidate in candidates
            for task in TASKS
            for seed in range(args.seeds)
        ]
        workers = max(1, min(args.workers, os.cpu_count() or 1, len(jobs)))

        print(
            f"Shard {args.shard + 1}/{args.shards}, round {round_id + 1}/{args.rounds}: "
            f"evaluating {len(jobs)} candidate-task-seed jobs with {workers} workers"
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            round_rows = list(pool.map(evaluate_one, jobs))
        all_rows.extend(round_rows)

        winners = leaderboard_from_rows(round_rows)[: max(2, args.candidates // 8)]
        winner_names = {str(row["candidate"]) for row in winners}
        parents = [candidate for candidate in candidates if str(candidate["name"]) in winner_names]

    elapsed = time.perf_counter() - start
    output_dir = Path(os.environ.get("C3_ARTIFACTS_DIR", "results"))
    write_outputs(output_dir, all_rows, candidates_by_round, elapsed)

    summary = json.loads((output_dir / "leaderboard.json").read_text(encoding="utf-8"))
    print(f"Winner: {summary['winner']['candidate']} mean_reward={summary['winner']['mean_reward']}")
    print(f"Wrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
