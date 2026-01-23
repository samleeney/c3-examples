#!/bin/bash
#SBATCH --job-name=hello-world
#SBATCH --gpus=1
#SBATCH --time=00:05:00

echo "Hello from C3!"
echo "Running on: $(hostname)"
nvidia-smi
echo "Job completed successfully!"
