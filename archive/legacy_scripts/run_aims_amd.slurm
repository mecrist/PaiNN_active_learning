#!/bin/bash
#SBATCH --job-name=amor_Slab
#SBATCH --partition=long            # serial | medium | long | bigmem | fast  (sinfo to list)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32         # match to what you actually need/have available
#SBATCH --cpus-per-task=1           # pure-MPI build: 1 thread per rank
#SBATCH --output=aims-%j.out
#SBATCH --error=aims-%j.err
#SBATCH --constraint=EPYC           # REQUIRED: this binary is compiled for EPYC
                                     # (-march=znver2) and will not run on Opteron nodes
# #SBATCH --mem-per-cpu=2048M       # cluster default is 2048M/cpu; uncomment to change

set -e

# ---- point this at wherever you copied the build to ------------------------
AIMS_DIR="$HOME/fhi-aims-amd-build"          # <-- edit if you put it elsewhere
AIMS_BIN="$AIMS_DIR/aims.x"

# ---- environment -------------------------------------------------------------
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load gnu9/9.4.0 openmpi4/4.1.1 scalapack/2.1.0 \
            amdblis/4.1-aocc-4.1.0-t54n2jv amdlibflame/4.1-aocc-4.1.0-2t4dqzl

# Pure-MPI: one thread per rank
export OMP_NUM_THREADS=1

# FHI-aims REQUIRES an unlimited stack or it aborts ("stacksize too low").
ulimit -s unlimited

# ---- run -------------------------------------------------------------------
# Run from the directory that contains your 'control.in' and 'geometry.in'.
# Submit with:   sbatch run_aims_amd.slurm
cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID on $SLURM_NNODES node(s), $SLURM_NTASKS MPI ranks"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Binary: $AIMS_BIN"

# This OpenMPI build has no Slurm PMI/PMIx support, so mpirun (not srun) must
# do the launching itself. --oversubscribe avoids a spurious "not enough
# slots" error some Slurm/OpenMPI combinations raise even when the ntasks
# below match the real allocation.
mpirun -np "$SLURM_NTASKS" --oversubscribe "$AIMS_BIN" > aims.out 2>&1

echo "Done. Output written to aims.out"
