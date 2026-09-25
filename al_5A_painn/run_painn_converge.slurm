#!/bin/bash -l
#SBATCH --job-name=painn_conv
#SBATCH --partition=etileno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=painn_conv_%j.out
#SBATCH --error=painn_conv_%j.err

set -e

echo "=========================================================="
echo "Starting PaiNN Final Convergence Training on $SLURMD_NODENAME"
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

$ENV_PREFIX/bin/python -c "
import sys
from pathlib import Path
import torch

repo_root = Path('/home/maria.crist/dft_mlip/sep_pax')
sys.path.insert(0, str(repo_root / 'src'))

from aims_PAX.tools.model_tools.painn_calculator import PainnEnsembleCalculator
from aims_PAX.tools.model_tools.train_painn import run_final_convergence

base_dir = Path('.').resolve()
dataset_path = base_dir / 'al_dataset.xyz'
val_dataset_path = base_dir / 'val.xyz'
ckpt_dir = base_dir / 'checkpoints'
out_dir = ckpt_dir / 'converged_models'

ref_energies = {
    1: 1295.1619808355229,
    8: -4671.611073387086,
    14: -2659.5960319024257,
}

print(f'Loading models from cycle 50 checkpoint...')
calc = PainnEnsembleCalculator(
    model_paths=[
        ckpt_dir / 'cycle_0050' / 'model_0',
        ckpt_dir / 'cycle_0050' / 'model_1',
        ckpt_dir / 'cycle_0050' / 'model_2',
    ],
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

print(f'Starting final convergence routine (max 200 epochs, early stopping patience 30)...')
run_final_convergence(
    calculator=calc,
    dataset_path=dataset_path,
    val_dataset_path=val_dataset_path,
    ref_energies=ref_energies,
    output_dir=out_dir,
    n_epochs=200,
    patience=30,
    lr=1e-4,
    batch_size=4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)
print('PaiNN Final Convergence Finished Successfully!')
"

echo "=========================================================="
echo "PaiNN Final Convergence Training completed at $(date)"
echo "=========================================================="
