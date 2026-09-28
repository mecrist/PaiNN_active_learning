#!/bin/bash -l
#SBATCH --job-name=mace_al_rep
#SBATCH --partition=etileno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=mace_al_replay_%j.out
#SBATCH --error=mace_al_replay_%j.err

set -e

echo "=========================================================="
echo "Starting MACE AL Replay Benchmark Suite on $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID | Partition: $SLURM_JOB_PARTITION"
echo "Allocated: 1 GPU + 32 CPU cores"
echo "Testing: 2, 3, 5, 10 Epochs per Cycle + 50-Epoch Final Convergence"
echo "Start time: $(date)"
echo "=========================================================="

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge

ENV_PREFIX=/home/maria.crist/.conda/envs/my_env
export LD_LIBRARY_PATH=$ENV_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd /home/maria.crist/dft_mlip/sep_pax/benchmark_al_epochs

nvidia-smi

# Target epoch counts to test (matching completed PaiNN suites: 2, 3, 5)
EPOCHS_LIST=(2 3 5)

for EP in "${EPOCHS_LIST[@]}"; do
    echo "----------------------------------------------------------"
    echo ">>> RUNNING MACE REPLAY: $EP EPOCHS/CYCLE (+ 50 Ep Conv) <<<"
    echo "----------------------------------------------------------"
    $ENV_PREFIX/bin/python replay_mace.py \
        --epochs_per_cycle $EP \
        --conv_epochs 50 \
        --conv_patience 15 \
        --eval_every 5 \
        --batch_size 4 \
        --lr 1e-3 \
        --device cuda
done

echo "=========================================================="
echo "All MACE AL Replay Benchmarks completed at $(date)"
echo "=========================================================="
