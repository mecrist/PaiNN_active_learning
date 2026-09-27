#!/bin/bash
# ==============================================================================
# Master launcher for Active Learning Epoch Variations Benchmark
# Launches SLURM jobs for PaiNN and MACE testing 2, 3, 5, and 10 epochs per cycle
# followed by 50-epoch post-AL final convergence sessions.
# Complies with etileno QoS policy (DefCpuPerGPU=32, MaxSubmitPU=2).
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "================================================================================"
echo "LAUNCHING ACTIVE LEARNING EPOCH VARIATION BENCHMARKS (2, 3, 5, 10 EPOCHS)"
echo "Zero DFT calculations required; replaying 71 unified AL structures on GPUs."
echo "================================================================================"

echo "Submitting PaiNN Replay Suite Job (epochs: 2, 3, 5, 10)..."
PAINN_JOB=$(sbatch --parsable submit_painn_benchmarks.slurm)
echo "  --> PaiNN Job ID: $PAINN_JOB"

echo "Submitting MACE Replay Suite Job (epochs: 2, 3, 5, 10)..."
MACE_JOB=$(sbatch --parsable submit_mace_benchmarks.slurm)
echo "  --> MACE Job ID:  $MACE_JOB"

echo "================================================================================"
echo "All benchmark jobs queued successfully!"
echo "Check queue status with: squeue -u maria.crist"
echo "================================================================================"
