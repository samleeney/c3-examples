#!/bin/bash
#SBATCH --job-name=jax-matmul
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

#C3 OUTPUT ./results

pip install jax[cuda12] numpy
python train.py
