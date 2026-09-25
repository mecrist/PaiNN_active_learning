#!/bin/bash -l
#SBATCH --job-name=aimspax_al_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=metano
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=slurm%j.out
#SBATCH --error=slurm%j.err
module load fhi-aims/260331
source activate my_env
cd /home/maria.crist/pax/
aims-PAX-al --model-settings model.yaml --aimsPAX-settings aimsprobe.yaml
