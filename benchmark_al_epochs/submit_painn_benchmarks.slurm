#!/bin/bash -l
#SBATCH --job-name=painn_al_rep
#SBATCH --partition=etileno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=painn_al_ep%a_%j.out
#SBATCH --error=painn_al_ep%a_%j.err

set -e

# Default to array task ID if set, otherwise environment variable EPOCHS, or 2
EPOCHS_PER_CYCLE=${SLURM_ARRAY_TASK_ID:-${EPOCHS:-2}}

echo "=========================================================="
echo "Starting PaiNN AL Replay Benchmark on $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID | Array Task: $SLURM_ARRAY_TASK_ID"
echo "Config: $EPOCHS_PER_CYCLE Epochs per AL cycle + 50 Epoch Final Convergence"
echo "Allocated: 1 GPU + 16 CPU cores"
echo "Start time: $(date)"
echo "=========================================================="

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge

ENV_PREFIX=/home/maria.crist/.conda/envs/nff
export LD_LIBRARY_PATH=$ENV_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/maria.crist/dft_mlip/sep_pax/benchmark_al_epochs

nvidia-smi

$ENV_PREFIX/bin/python replay_painn.py \
    --epochs_per_cycle $EPOCHS_PER_CYCLE \
    --conv_epochs 50 \
    --conv_patience 15 \
    --eval_every 5 \
    --batch_size 4 \
    --lr 1e-4 \
    --device cuda:0

echo "=========================================================="
echo "PaiNN AL Replay Benchmark completed at $(date)"
echo "=========================================================="
