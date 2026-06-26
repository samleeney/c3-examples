# C3 Examples

Example jobs for the [C3 GPU compute platform](https://cthree.cloud).

## Quick Start

```bash
# Install C3 CLI
curl -fsSL https://cthree.cloud/install.sh | sh

# Login
c3 login

# Clone and run an example
git clone https://github.com/c3-research/c3-examples.git
cd c3-examples/jax-matmul
c3 deploy

# Check status
c3 squeue

# Download results
c3 pull
```

## Examples

### hello.sbatch

Minimal "hello world" example. Prints system info and runs `nvidia-smi` to verify GPU access.

```bash
c3 deploy hello.sbatch
```

### jax-matmul

JAX matrix multiplication benchmark comparing loop-based vs vectorized (`jax.lax.scan`) approaches. Demonstrates GPU performance and generates a benchmark plot.

Features:
- Uses UV lockfile for reproducible dependencies
- Loads matrices from data file
- Outputs benchmark plot to `results/`

```bash
cd jax-matmul
c3 deploy
```

### algorithm-discovery-loop

Tiny dependency-free automated research example. It fans out candidate
evaluations across multiple C3 GPU jobs, then merges the returned artifact
leaderboards.

```bash
cd algorithm-discovery-loop
python3 launch_shards.py --shards 3
```
