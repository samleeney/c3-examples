# Algorithm Discovery Loop

This is a tiny C3 example for automated research systems. It fans out optimizer
candidate evaluations across several C3 GPU jobs, mutates local winners, then
merges the returned artifact leaderboards.

The example is intentionally dependency-free so it can run quickly on C3.

Smoke-test it locally:

```bash
C3_ARTIFACTS_DIR=results bash run.sh
cat results/leaderboard.md
```

Run it across three C3 GPU jobs:

```bash
python3 launch_shards.py --shards 3
while read job; do c3 pull "$job"; done < shard-jobs.txt
python3 merge.py job_*/artifacts
cat merged-results/leaderboard.md
```

Files:

- `.c3` requests an L40 with a five-minute limit.
- `run.sh` prints the Python/GPU environment and starts a single-job run.
- `launch_shards.py` submits C3 jobs concurrently. The default is 3 shards for
  the free trial burst limit; Team can burst to 50 GPUs.
- `search.py` evaluates one shard and writes artifacts.
- `merge.py` combines pulled shard artifacts into one leaderboard.
