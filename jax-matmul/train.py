#!/usr/bin/env python3
"""JAX GPU benchmark - matrix multiplication."""

import os
import time
import jax
import jax.numpy as jnp

def main():
    print("JAX Matrix Multiplication Benchmark")
    print("=" * 40)

    # Check GPU
    devices = jax.devices()
    print(f"Devices: {devices}")

    # Create random matrices
    key = jax.random.PRNGKey(0)
    size = 4096
    A = jax.random.normal(key, (size, size))
    B = jax.random.normal(key, (size, size))

    # Warmup
    _ = jnp.matmul(A, B).block_until_ready()

    # Benchmark
    start = time.perf_counter()
    C = None
    for _ in range(10):
        C = jnp.matmul(A, B).block_until_ready()
    elapsed = time.perf_counter() - start

    print(f"\nMatrix size: {size}x{size}")
    print(f"10 multiplications: {elapsed:.3f}s")
    print(f"Average: {elapsed/10*1000:.1f}ms per multiply")
    print(f"Result checksum: {float(jnp.sum(C)):.2e}")

    # Save result
    os.makedirs("results", exist_ok=True)
    with open("results/output.txt", "w") as f:
        f.write(f"Matrix size: {size}x{size}\n")
        f.write(f"Time: {elapsed:.3f}s\n")
        f.write(f"Checksum: {float(jnp.sum(C)):.2e}\n")

    print("\nSaved to results/output.txt")

if __name__ == "__main__":
    main()
