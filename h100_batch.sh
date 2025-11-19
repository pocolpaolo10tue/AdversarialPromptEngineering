#!/bin/bash
#SBATCH --job-name=run-experiment
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_h100
#SBATCH --time=10:00:00

module load 2023
module load foss/2023a
module load CUDA/12.1.1

source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python main.py
deactivate
