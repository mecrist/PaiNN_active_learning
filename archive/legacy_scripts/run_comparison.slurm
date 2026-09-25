#!/bin/bash -l
#SBATCH --job-name=mace_vs_painn
#SBATCH --partition=etileno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=comparison_%j.out
#SBATCH --error=comparison_%j.err

set -e

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

cd /home/maria.crist/dft_mlip/sep_pax

echo "=========================================================="
echo "Starting MACE vs. PaiNN 500-Step MD Benchmark on $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID | Partition: $SLURM_JOB_PARTITION"
echo "Allocated: 1 GPU + 32 CPU cores"
echo "Start time: $(date)"
echo "=========================================================="

nvidia-smi

$ENV_PREFIX/bin/python compare_mace_vs_painn.py

echo "=========================================================="
echo "Comparison completed successfully at $(date)"
echo "=========================================================="
