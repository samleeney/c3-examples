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

### jax-matmul
Simple JAX matrix multiplication benchmark. Tests GPU availability and performance.
