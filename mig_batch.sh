#!/bin/bash
#SBATCH --job-name=run-experiments
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681

module load 2023
module load foss/2023a
module load CUDA/12.1.1

source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python main.py
deactivate