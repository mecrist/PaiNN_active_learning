# PaiNN-AL: Active Learning Pipeline Based on aims-PAX

An adaptation of the [aims-PAX](https://github.com/tohenkes/aims-PAX/tree/main) for active learning pipeline for Machine Learning Interatomic Potentials (MLIPs) using **PaiNN** (Polarizable Atom Interaction Neural Network) models.

This code follows the same organization as the original aims-PAX repository, with additional codes written separately for easier debugging. Example of performance comparison on a system of silica-water interface is provided for the two models; as well as the influence of number of training epochs during active learning.

Main functionalities implemented:
- **Single-Node HPC Efficiency (1 GPU + 32 CPU Cores)**: Runs MD propagation and neural network inference on GPU while dispatching DFT single points directly to the allocated 32 CPU cores via local MPI. Bypasses cluster queues completely.
- **Zero Data Loss DFT Archiving**: Every triggered FHI-aims calculation is permanently preserved in its own directory (`dft_calculations/step_{step}_traj_{idx}/`) containing `geometry.in`, `control.in`, and the full `aims.out`.
- **Reference Energy ($E_0$) Baseline Correction**: Automatically shifts raw all-electron DFT energies to zero-centered reference energies for PaiNN training, preventing gradient explosion.
- **LAMMPS-Compatible Trajectory Dumps**:
  - **Unified Trajectory Dump** (`traj_traj{i}.lammpstrj`): Recorded every 1,000 steps (`id type mol xu yu zu`), with water vs. slab molecule tracking.
  - **Stress Dump** (`traj_stress_traj{i}.lammpstrj`): Recorded every 5,000 steps (`id type mol xu yu zu c_stress[1] c_stress[2] c_stress[3]`) for surface tension and virial analysis.
- **Walltime-Aware Checkpointing & Resume**: Tracks trajectory positions, velocities, step counts, and model weights in `al_checkpoint.json`. Traps Slurm SIGTERM signals before the walltime to allow seamless 1-command resumption.
