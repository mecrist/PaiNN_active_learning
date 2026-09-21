#!/bin/bash -l
#SBATCH --job-name=painn_bench
#SBATCH --partition=grafite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

set -e

echo "=========================================================="
echo "Starting PaiNN Benchmark Suite on $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID | Partition: $SLURM_JOB_PARTITION"
echo "Allocated: 2 CPU cores on compute node"
echo "Start time: $(date)"
echo "=========================================================="

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module use /home/maria.crist/fhi-aims.260331/modulefiles
module load fhi-aims/260331

ulimit -s unlimited
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

ENV_PREFIX=/home/maria.crist/.conda/envs/nff
export LD_LIBRARY_PATH=$ENV_PREFIX/lib:$LD_LIBRARY_PATH

cd /home/maria.crist/dft_mlip/sep_pax

$ENV_PREFIX/bin/python benchmark_pipeline.py

echo "=========================================================="
echo "Benchmark completed successfully at $(date)"
echo "=========================================================="
