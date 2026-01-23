#!/usr/bin/env python3
"""JAX matrix multiplication benchmark on GPU.

Compares loop-based vs vectorized (jax.lax.scan) matrix multiplication
to demonstrate GPU performance characteristics.
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def load_matrices(filepath: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load two 10x10 matrices from data file.

    File format: 20 rows of 10 space-separated floats.
    First 10 rows = Matrix A, next 10 rows = Matrix B.
    """
    data = np.loadtxt(filepath)
    A = jnp.array(data[:10])
    B = jnp.array(data[10:])
    return A, B


def loop_multiply(A: jnp.ndarray, B: jnp.ndarray, n: int = 1000) -> jnp.ndarray:
    """Multiply matrices n times in a Python loop."""
    result = A
    for _ in range(n):
        result = jnp.matmul(result, B)
    return result


@jax.jit(static_argnums=(2,))
def scan_multiply(A: jnp.ndarray, B: jnp.ndarray, n: int = 1000) -> jnp.ndarray:
    """Multiply matrices n times using jax.lax.scan (vectorized)."""
    def body_fn(carry, _):
        return jnp.matmul(carry, B), None
    result, _ = jax.lax.scan(body_fn, A, None, length=n)
    return result


def main():
    print("JAX Matrix Multiplication Benchmark")
    print("=" * 50)

    # Check GPU availability
    devices = jax.devices()
    print(f"Available devices: {devices}")
    gpu_available = any(d.platform == 'gpu' for d in devices)
    print(f"GPU available: {gpu_available}")

    if not gpu_available:
        print("WARNING: No GPU detected. Running on CPU.")

    # Load data
    print("\nLoading matrices from data.txt...")
    A, B = load_matrices("data.txt")
    print(f"Matrix A shape: {A.shape}")
    print(f"Matrix B shape: {B.shape}")

    # Warmup JAX compilation
    print("\nWarming up JAX compilation...")
    _ = jnp.matmul(A, B).block_until_ready()
    _ = scan_multiply(A, B, 10).block_until_ready()

    # Loop multiplication benchmark
    print("\n--- Loop Multiplication (1000 iterations) ---")
    start = time.perf_counter()
    result_loop = loop_multiply(A, B, 1000)
    result_loop.block_until_ready()
    loop_time = time.perf_counter() - start
    print(f"Loop time: {loop_time:.4f}s")
    print(f"Result checksum: {float(jnp.sum(result_loop)):.6e}")

    # Scan multiplication benchmark (vectorized)
    print("\n--- Scan/Vectorized Multiplication (1000 iterations) ---")
    start = time.perf_counter()
    result_scan = scan_multiply(A, B, 1000)
    result_scan.block_until_ready()
    scan_time = time.perf_counter() - start
    print(f"Scan time: {scan_time:.4f}s")
    print(f"Result checksum: {float(jnp.sum(result_scan)):.6e}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"Loop method:   {loop_time:.4f}s")
    print(f"Scan method:   {scan_time:.4f}s")
    if scan_time > 0:
        print(f"Speedup:       {loop_time/scan_time:.2f}x")
    print("=" * 50)

    # Generate and save benchmark plot to results directory
    print("\nGenerating benchmark plot...")
    os.makedirs("results", exist_ok=True)

    methods = ["Loop", "Scan (Vectorized)"]
    times = [loop_time, scan_time]
    colors = ["#FF6B6B", "#4ECDC4"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, times, color=colors, edgecolor="black", linewidth=1.2)

    # Add value labels on bars
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{t:.4f}s", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("JAX Matrix Multiplication Benchmark\n(1000 iterations, 10x10 matrices)", fontsize=14)
    ax.set_ylim(0, max(times) * 1.2)

    # Add speedup annotation
    if scan_time > 0:
        speedup = loop_time / scan_time
        ax.annotate(f"Speedup: {speedup:.1f}x", xy=(0.5, 0.95), xycoords="axes fraction",
                    ha="center", fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    plot_path = "results/benchmark.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")

    print("\nJob completed successfully!")


if __name__ == "__main__":
    main()
