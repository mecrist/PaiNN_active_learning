#!/bin/bash -l
#SBATCH --job-name=dual_conv
#SBATCH --partition=etileno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=dual_conv_%j.out
#SBATCH --error=dual_conv_%j.err

set -e

echo "=========================================================="
echo "Starting Dual MLIP Convergence (MACE + PaiNN) on $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID | Partition: $SLURM_JOB_PARTITION"
echo "Allocated: 2 GPUs + 64 CPU cores"
echo "Start time: $(date)"
echo "=========================================================="

nvidia-smi

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module use /home/maria.crist/fhi-aims.260331/modulefiles
module load fhi-aims/260331

ulimit -s unlimited

# Directory setup
BASE_DIR=/home/maria.crist/dft_mlip/sep_pax
MACE_DIR=$BASE_DIR/al_5A_mace

echo "----------------------------------------------------------"
echo "Launching MACE Convergence on GPU 0 (CUDA_VISIBLE_DEVICES=0)..."
echo "----------------------------------------------------------"
(
    cd $MACE_DIR
    export CUDA_VISIBLE_DEVICES=0
    export OMP_NUM_THREADS=32
    export MKL_NUM_THREADS=32
    ENV_MACE=/home/maria.crist/.conda/envs/my_env
    export PATH=$ENV_MACE/bin:$PATH
    export LD_LIBRARY_PATH=$ENV_MACE/lib:$LD_LIBRARY_PATH
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    $ENV_MACE/bin/aims-PAX-al --model-settings model.yaml --aimsPAX-settings aimsprobe.yaml > mace_convergence.log 2>&1
    echo "[MACE DONE] MACE Convergence finished at $(date)"
) &
PID_MACE=$!

echo "----------------------------------------------------------"
echo "Launching PaiNN Convergence on GPU 1 (CUDA_VISIBLE_DEVICES=1)..."
echo "----------------------------------------------------------"
(
    cd $BASE_DIR
    export CUDA_VISIBLE_DEVICES=1
    export OMP_NUM_THREADS=32
    export MKL_NUM_THREADS=32
    ENV_PAINN=/home/maria.crist/.conda/envs/nff
    export PATH=$ENV_PAINN/bin:$PATH
    export LD_LIBRARY_PATH=$ENV_PAINN/lib:$LD_LIBRARY_PATH
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    $ENV_PAINN/bin/python run_painn_convergence.py > painn_convergence.log 2>&1
    echo "[PaiNN DONE] PaiNN Convergence finished at $(date)"
) &
PID_PAINN=$!

echo "Both convergence jobs running concurrently in the background."
echo "Waiting for MACE (PID: $PID_MACE) and PaiNN (PID: $PID_PAINN)..."

# Wait for both processes to complete
wait $PID_MACE
STATUS_MACE=$?
wait $PID_PAINN
STATUS_PAINN=$?

echo "=========================================================="
echo "Dual Convergence Job Completed at $(date)"
echo "Exit Status: MACE=$STATUS_MACE | PaiNN=$STATUS_PAINN"
echo "=========================================================="

if [ $STATUS_MACE -ne 0 ] || [ $STATUS_PAINN -ne 0 ]; then
    echo "[ERROR] One or both convergence runs failed."
    exit 1
fi
