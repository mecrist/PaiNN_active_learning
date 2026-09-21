#!/bin/bash -l
#SBATCH --job-name=painn_al
#SBATCH --partition=metano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=painn_al_%j.out
#SBATCH --error=painn_al_%j.err

set -e

# ==============================================================================
# 1. ENVIRONMENT & MODULES
# ==============================================================================
echo "=========================================================="
echo "Starting PaiNN Active Learning on $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID | Partition: $SLURM_JOB_PARTITION"
echo "Allocated: 1 GPU + 32 CPU cores"
echo "Start time: $(date)"
echo "=========================================================="

# Load Intel MPI and FHI-aims environment for local 32-core execution
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module use /home/maria.crist/fhi-aims.260331/modulefiles
module load fhi-aims/260331

# Stack size and multi-threading limits
ulimit -s unlimited
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Conda Python environment for NFF/PaiNN
ENV_PREFIX=/home/maria.crist/.conda/envs/nff
export LD_LIBRARY_PATH=$ENV_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/maria.crist/dft_mlip/sep_pax

# Verify GPU
nvidia-smi

# ==============================================================================
# 2. RUN ACTIVE LEARNING PIPELINE
# ==============================================================================
$ENV_PREFIX/bin/python run_painn_active_learning.py

echo "=========================================================="
echo "PaiNN Active Learning completed successfully at $(date)"
echo "=========================================================="
