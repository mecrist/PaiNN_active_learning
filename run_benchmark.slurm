#!/bin/bash -l
#SBATCH --job-name=painn_bench
#SBATCH --partition=etileno,metano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

set -e

echo "=========================================================="
echo "Starting PaiNN Benchmark Suite on $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID | Partition: $SLURM_JOB_PARTITION"
echo "Allocated: 1 GPU + 32 CPU cores"
echo "Start time: $(date)"
echo "=========================================================="

# Modules and libraries
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module use /home/maria.crist/fhi-aims.260331/modulefiles
module load fhi-aims/260331

ulimit -s unlimited
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Conda Python environment
ENV_PREFIX=/home/maria.crist/.conda/envs/nff
export LD_LIBRARY_PATH=$ENV_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/maria.crist/dft_mlip/sep_pax

# GPU Info
nvidia-smi

# Execute Benchmark Suite
$ENV_PREFIX/bin/python benchmark_pipeline.py

echo "=========================================================="
echo "Benchmark Suite completed at $(date)"
echo "=========================================================="
