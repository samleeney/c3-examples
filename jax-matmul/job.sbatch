#!/bin/bash
#SBATCH --job-name=jax-matmul
#SBATCH --gres=gpu:l40:1
#SBATCH --time=00:10:00

#C3 PYTHON use --project ./
#C3 OUTPUT ./results

python train.py
