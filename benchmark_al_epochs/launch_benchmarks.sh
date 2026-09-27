#!/bin/bash
# ==============================================================================
# Master launcher for Active Learning Epoch Variations Benchmark
# Launches SLURM jobs for PaiNN and MACE testing 2, 3, 5, and 10 epochs per cycle
# followed by 50-epoch post-AL final convergence sessions.
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "================================================================================"
echo "LAUNCHING ACTIVE LEARNING EPOCH VARIATION BENCHMARKS (2, 3, 5, 10 EPOCHS)"
echo "Zero DFT calculations required; replaying 71 unified AL structures on GPUs."
echo "================================================================================"

# Array jobs for 2, 3, 5, 10
# PaiNN Array Job:
echo "Submitting PaiNN Replay Job Array (epochs: 2, 3, 5, 10)..."
PAINN_JOB=$(sbatch --parsable --array=2,3,5,10 submit_painn_benchmarks.slurm)
echo "  --> PaiNN Array Job ID: $PAINN_JOB"

# MACE Array Job:
echo "Submitting MACE Replay Job Array (epochs: 2, 3, 5, 10)..."
MACE_JOB=$(sbatch --parsable --array=2,3,5,10 submit_mace_benchmarks.slurm)
echo "  --> MACE Array Job ID:  $MACE_JOB"

echo "================================================================================"
echo "All benchmark jobs queued successfully!"
echo "Check queue status with: squeue -u maria.crist"
echo "================================================================================"
