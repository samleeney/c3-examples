# C3 Examples

Example jobs for the [C3 GPU compute platform](https://cthree.cloud).

## Quick Start

### For agents (MCP)

C3 hosts an MCP server at `https://api.cthree.cloud/mcp`. Connect your coding agent to it and it can list hardware, deploy jobs, follow logs and collect results with nothing installed.

```bash
# Claude Code, browser login on first use (run /mcp and choose Authenticate)
claude mcp add --transport http c3 https://api.cthree.cloud/mcp

# Headless: create a key with `c3 apikey create <name>` or on the dashboard
claude mcp add --transport http c3 https://api.cthree.cloud/mcp \
  --header "Authorization: Bearer c3_key_..."
```

Then, from an example directory, ask:

> Deploy this directory to C3 on an l40, wait for it, and show me the results.

Setup for Cursor, Codex, VS Code, claude.ai and ChatGPT is in the [MCP docs](https://docs.cthree.cloud/mcp).

### For humans (CLI)

```bash
curl -fsSL https://cthree.cloud/install.sh | sh   # install
c3 login
git clone https://github.com/c3-research/c3-examples.git
cd c3-examples/jax-matmul
c3 deploy -f   # submit and follow logs
c3 pull        # download results
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
