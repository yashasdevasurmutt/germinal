#!/bin/bash
#SBATCH --job-name=refactoring_test
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G

#SBATCH -p preemptible
#SBATCH --gpus=1
#SBATCH --output=jobs/refactor-%j.out
#SBATCH --requeue

python -u run_germinal.py

echo "finished"