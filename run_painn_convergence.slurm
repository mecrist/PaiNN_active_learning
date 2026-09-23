#!/bin/bash -l
#SBATCH --job-name=painn_conv
#SBATCH --partition=etileno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=painn_conv_%j.out
#SBATCH --error=painn_conv_%j.err

set -e

echo "=========================================================="
echo "Starting PaiNN Ensemble Convergence Training on $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID | Partition: $SLURM_JOB_PARTITION"
echo "Allocated: 1 GPU + 16 CPU cores"
echo "Start time: $(date)"
echo "=========================================================="

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge

ulimit -s unlimited
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

ENV_PREFIX=/home/maria.crist/.conda/envs/nff
export PATH=$ENV_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$ENV_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/maria.crist/dft_mlip/sep_pax

nvidia-smi

$ENV_PREFIX/bin/python run_painn_convergence.py

echo "=========================================================="
echo "PaiNN Convergence Training completed at $(date)"
echo "=========================================================="
