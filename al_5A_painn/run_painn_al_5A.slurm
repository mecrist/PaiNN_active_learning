#!/bin/bash -l
#SBATCH --job-name=painn_al_5A
#SBATCH --partition=etileno,metano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=painn_al_5A_%j.out
#SBATCH --error=painn_al_5A_%j.err

set -e

echo "=========================================================="
echo "Starting PaiNN Closed-Loop AL on $SLURMD_NODENAME (5 A Gap)"
echo "Job ID: $SLURM_JOB_ID | Partition: $SLURM_JOB_PARTITION"
echo "Allocated: 1 GPU + 32 CPU cores"
echo "Start time: $(date)"
echo "=========================================================="

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module use /home/maria.crist/fhi-aims.260331/modulefiles
module load fhi-aims/260331

ulimit -s unlimited
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ENV_PREFIX=/home/maria.crist/.conda/envs/nff
export LD_LIBRARY_PATH=$ENV_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/maria.crist/dft_mlip/sep_pax/al_5A_painn

nvidia-smi

$ENV_PREFIX/bin/python run_painn_al_5A.py

echo "=========================================================="
echo "PaiNN Active Learning (5 A Gap) completed at $(date)"
echo "=========================================================="
