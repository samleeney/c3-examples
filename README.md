# C3 Examples

Example jobs for the [C3 GPU compute platform](https://github.com/samleeney/c3).

## Quick Start

```bash
# Install C3 CLI
curl -fsSL https://raw.githubusercontent.com/samleeney/c3/main/install.sh | sh

# Login
c3 login

# Clone and run an example
git clone https://github.com/samleeney/c3-examples
cd c3-examples/jax-matmul
c3 deploy job.sbatch

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
c3 deploy job.sbatch
```
