#!/usr/bin/env python
"""
Active Learning Pipeline for Silica/Water using a 6-member PaiNN Committee.

Features:
  - 6 fine-tuned PaiNN models (3 from mine, 3 from merged ep500).
  - 9 starting geometries simultaneously explored at 300K via ASE Langevin dynamics.
  - Uncertainty metric: maximum atomic force standard deviation (U = max_i sigma_i).
  - Initial uncertainty threshold: U_thresh = 25.0 meV/Angstrom (with dynamic relaxation).
  - Single-node execution: 1 GPU + 32 CPU cores on partition 'metano' (bypasses nanotubo queue).
  - Local MPI execution of FHI-aims DFT (mpirun -np 32).
  - PERMANENT DFT storage: every single calculation directory is preserved.
  - Safe 1-epoch fine-tuning after each DFT addition.
  - LAMMPS custom trajectory dumps:
      * Unified dump: traj_traj{idx}.lammpstrj (every 1000 steps)
      * Stress dump:  traj_stress_traj{idx}.lammpstrj (every 5000 steps)
  - Full checkpointing & graceful resume before 72h walltime.
"""

import os
import sys
import json
import signal
from pathlib import Path
import numpy as np
import torch
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from painn_ensemble_calc import PainnEnsembleCalculator
from lammps_dumps import write_lammps_dump, identify_molecules
from dft_interface import run_aims_single_point, AimsCalculationError
from retrain_engine import retrain_ensemble, run_final_convergence


class AimsPaxThresholdManager:
    """
    Rolling-window adaptive threshold engine following aims-PAX uncertainty protocol.
    Dynamically tightens the threshold as model uncertainty improves.
    """
    def __init__(self, initial_threshold=25.0, c_x=0.0, max_history=400, freeze_dataset_size=150, min_history=10):
        self.initial_threshold = initial_threshold
        self.threshold = initial_threshold
        self.c_x = c_x
        self.max_history = max_history
        self.freeze_dataset_size = freeze_dataset_size
        self.min_history = min_history
        self.uncertainty_history = []
        self.frozen = False

    def update(self, current_uncertainty, current_dataset_size):
        self.uncertainty_history.append(float(current_uncertainty))

        if current_dataset_size >= self.freeze_dataset_size and not self.frozen:
            print(f"[THRESHOLD] Freezing threshold at {self.threshold:.2f} meV/A (dataset size: {current_dataset_size})")
            self.frozen = True
            return self.threshold

        if not self.frozen and len(self.uncertainty_history) >= self.min_history:
            recent = self.uncertainty_history[-self.max_history:]
            avg_u = float(np.mean(recent))
            self.threshold = avg_u * (1.0 + self.c_x)

        return self.threshold

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
BASE_DIR = Path("/home/maria.crist/dft_mlip/sep_pax")
GEOMETRIES_DIR = BASE_DIR / "geometries"
TRAJECTORIES_DIR = BASE_DIR / "trajectories"
DFT_CALCS_DIR = BASE_DIR / "dft_calculations"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
DATASET_PATH = BASE_DIR / "al_dataset.xyz"
CHECKPOINT_FILE = BASE_DIR / "al_checkpoint.json"

CONTROL_IN = BASE_DIR / "control.in"
SPECIES_DIR = Path("/home/maria.crist/fhi-aims.260331/species_defaults/defaults_2020/light")
AIMS_BIN = "/home/maria.crist/fhi-aims.260331/bin/aims.x"

# The 6 fine-tuned PaiNN models
INITIAL_MODEL_DIRS = [
    Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_0"),
    Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_1"),
    Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_2"),
    Path("/home/maria.crist/dft_mlip/my_dataset/merged/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_0"),
    Path("/home/maria.crist/dft_mlip/my_dataset/merged/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_1"),
    Path("/home/maria.crist/dft_mlip/my_dataset/merged/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_2"),
]

# Elemental reference energies (E0) in eV
REF_ENERGIES = {
    1: 1295.1619808355229,   # H
    8: -4671.611073387086,   # O
    14: -2659.5960319024257  # Si
}

# Active Learning & MD Hyperparameters
TEMPERATURE_K = 300.0
MD_TIMESTEP_FS = 0.5
LANGEVIN_FRICTION = 0.002  # 1/fs
SKIP_STEP_MLFF = 25        # evaluate uncertainty every 25 steps
MAX_MD_STEPS = 10000       # max steps per trajectory
MAX_DFT_POINTS = 100       # maximum new DFT single points
INITIAL_THRESHOLD = 25.0   # meV/Angstrom
MIN_THRESHOLD = 20.0       # meV/Angstrom
THRESHOLD_RELAX_FACTOR = 1.02  # 2% dynamic relaxation per added point
FINE_TUNE_EPOCHS = 1       # 1 epoch default retraining
DUMP_INTERVAL_UNIFIED = 1000   # steps
DUMP_INTERVAL_STRESS = 5000    # steps
N_MPI_CORES = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 9 Geometries to explore
GEOMETRY_NAMES = [
    "geometry_10A_alpha.in",
    "geometry_10A_amor.in",
    "geometry_10A_beta.in",
    "geometry_20A_alpha.in",
    "geometry_20A_amor.in",
    "geometry_20A_beta.in",
    "geometry_alpha_5.in",
    "geometry_amor_5.in",
    "geometry_beta_5.in",
]


class ALPipeline:
    def __init__(self):
        self.device = DEVICE
        print(f"[INIT] Active Learning Pipeline on device: {self.device}")
        TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
        DFT_CALCS_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

        self.threshold_mgr = AimsPaxThresholdManager(
            initial_threshold=INITIAL_THRESHOLD,
            c_x=0.0,
            freeze_dataset_size=150,
        )
        self.threshold = INITIAL_THRESHOLD
        self.total_dft_evals = 0
        self.cycle_count = 0
        self.current_steps = {i: 0 for i in range(len(GEOMETRY_NAMES))}
        self.last_checkpoints = {}

        # Initialize dataset if missing
        if not DATASET_PATH.exists():
            seed_source = Path("/home/maria.crist/pax/train.xyz")
            if seed_source.exists():
                print(f"[INIT] Seeding active learning dataset from {seed_source}")
                initial_frames = read(str(seed_source), index=":")
                for fr in initial_frames:
                    if "REF_energy" not in fr.info:
                        fr.info["REF_energy"] = fr.get_potential_energy()
                    if "REF_forces" not in fr.arrays:
                        fr.arrays["REF_forces"] = fr.get_forces()
                write(str(DATASET_PATH), initial_frames, format="extxyz")
            else:
                DATASET_PATH.touch()

        # Load PaiNN committee calculator
        model_dirs = self._get_latest_model_dirs()
        self.calculator = PainnEnsembleCalculator(
            model_dirs=model_dirs,
            ref_energies=REF_ENERGIES,
            cutoff=6.0,
            device=self.device,
        )

        # Persistent Adam optimizers across AL cycles
        self.optimizers = {
            i: torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-4)
            for i, m in enumerate(self.calculator.models)
        }

        # Setup 9 trajectories
        self.trajectories = []
        self.md_drivers = []
        self.mol_ids = []

        self._setup_trajectories()

        # Check for resume checkpoint
        if CHECKPOINT_FILE.exists():
            self._load_checkpoint()

        # Register graceful exit handlers
        signal.signal(signal.SIGINT, self._handle_exit_signal)
        signal.signal(signal.SIGTERM, self._handle_exit_signal)

    def _get_latest_model_dirs(self):
        """Finds latest cycle checkpoints or defaults to INITIAL_MODEL_DIRS."""
        cycle_dirs = sorted(CHECKPOINTS_DIR.glob("cycle_*"))
        if cycle_dirs:
            latest = cycle_dirs[-1]
            subdirs = sorted([d for d in latest.iterdir() if d.is_dir() and "model_" in d.name])
            if len(subdirs) == len(INITIAL_MODEL_DIRS):
                print(f"[INIT] Resuming PaiNN models from checkpoint: {latest}")
                return [str(d) for d in subdirs]
        return [str(d) for d in INITIAL_MODEL_DIRS]

    def _setup_trajectories(self):
        """Initializes Atoms objects and ASE Langevin dynamics for all 9 geometries."""
        print(f"[INIT] Loading {len(GEOMETRY_NAMES)} initial geometries at {TEMPERATURE_K}K...")
        for idx, g_name in enumerate(GEOMETRY_NAMES):
            g_path = GEOMETRIES_DIR / g_name
            if not g_path.exists():
                raise FileNotFoundError(f"Geometry file not found: {g_path}")

            atoms = read(str(g_path), format="aims")
            atoms.calc = self.calculator

            # Initialize thermal velocities
            MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
            Stationary(atoms)

            # ASE Langevin dynamics driver
            dyn = Langevin(
                atoms,
                timestep=MD_TIMESTEP_FS * units.fs,
                temperature_K=TEMPERATURE_K,
                friction=LANGEVIN_FRICTION / units.fs,
            )

            mols = identify_molecules(atoms)
            self.trajectories.append(atoms)
            self.md_drivers.append(dyn)
            self.mol_ids.append(mols)

    def _save_checkpoint(self):
        """Saves pipeline state to JSON for instant resumption."""
        state = {
            "cycle_count": self.cycle_count,
            "total_dft_evals": self.total_dft_evals,
            "threshold": self.threshold,
            "current_steps": self.current_steps,
            "positions": [atoms.get_positions().tolist() for atoms in self.trajectories],
            "velocities": [atoms.get_velocities().tolist() for atoms in self.trajectories],
        }
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[CHECKPOINT] Saved state to {CHECKPOINT_FILE}")

    def _load_checkpoint(self):
        """Restores pipeline state from JSON."""
        print(f"[RESUME] Found existing checkpoint: {CHECKPOINT_FILE}. Loading state...")
        with open(CHECKPOINT_FILE) as f:
            state = json.load(f)

        self.cycle_count = state.get("cycle_count", 0)
        self.total_dft_evals = state.get("total_dft_evals", 0)
        self.threshold = state.get("threshold", INITIAL_THRESHOLD)
        self.current_steps = {int(k): v for k, v in state.get("current_steps", {}).items()}

        pos_list = state.get("positions", [])
        vel_list = state.get("velocities", [])
        for i in range(len(self.trajectories)):
            if i < len(pos_list):
                self.trajectories[i].set_positions(np.array(pos_list[i]))
            if i < len(vel_list):
                self.trajectories[i].set_velocities(np.array(vel_list[i]))
        print(f"[RESUME SUCCESS] Resumed at Cycle {self.cycle_count} | Total DFT evals: {self.total_dft_evals} | Threshold: {self.threshold:.2f} meV/A")

    def _handle_exit_signal(self, signum, frame):
        print(f"\n[INTERRUPT] Received signal {signum}. Saving state before exiting...")
        self._save_checkpoint()
        sys.exit(0)

    def run(self):
        print("\n" + "=" * 70)
        print("STARTING PAINN COMMITTEE ACTIVE LEARNING SIMULATION")
        print(f"Trajectories: {len(self.trajectories)} | Temp: {TEMPERATURE_K}K | Step: {MD_TIMESTEP_FS} fs")
        print(f"Initial Threshold: {self.threshold:.2f} meV/A | Max DFT Budget: {MAX_DFT_POINTS}")
        print("=" * 70 + "\n")

        all_done = False
        while not all_done and self.total_dft_evals < MAX_DFT_POINTS:
            all_done = True

            for traj_idx in range(len(self.trajectories)):
                atoms = self.trajectories[traj_idx]
                dyn = self.md_drivers[traj_idx]
                mol_id = self.mol_ids[traj_idx]
                step = self.current_steps[traj_idx]

                if step >= MAX_MD_STEPS:
                    continue
                all_done = False

                # Save safe coordinates before propagation
                safe_pos = atoms.get_positions().copy()
                safe_vel = atoms.get_velocities().copy()

                # 1. Propagate MD for SKIP_STEP_MLFF steps
                dyn.run(SKIP_STEP_MLFF)
                step += SKIP_STEP_MLFF
                self.current_steps[traj_idx] = step

                # Trajectory dumps with dynamic molecule identification
                if step % DUMP_INTERVAL_UNIFIED == 0:
                    dump_file = TRAJECTORIES_DIR / f"traj_traj{traj_idx}.lammpstrj"
                    mol_ids = identify_molecules(atoms)
                    write_lammps_dump(
                        filename=dump_file,
                        step=step,
                        atoms=atoms,
                        mol_ids=mol_ids,
                        stress=None,
                        append=True,
                    )

                if step % DUMP_INTERVAL_STRESS == 0:
                    calc_res = atoms.calc.results
                    stress_diag = calc_res.get("atomic_stress", np.zeros((len(atoms), 3)))
                    stress_file = TRAJECTORIES_DIR / f"traj_stress_traj{traj_idx}.lammpstrj"
                    mol_ids = identify_molecules(atoms)
                    write_lammps_dump(
                        filename=stress_file,
                        step=step,
                        atoms=atoms,
                        mol_ids=mol_ids,
                        stress=stress_diag,
                        append=True,
                    )

                # 2. Evaluate Uncertainty at Current Step
                calc_res = atoms.calc.results
                uncertainty = float(calc_res.get("max_atomic_sd", 0.0))
                mean_energy = float(calc_res.get("energy", 0.0))

                # Update rolling adaptive threshold
                current_ds_size = len(read(str(DATASET_PATH), index=":"))
                self.threshold = self.threshold_mgr.update(uncertainty, current_ds_size)

                print(
                    f"[{GEOMETRY_NAMES[traj_idx]} | Step {step:05d}] "
                    f"Energy: {mean_energy:.2f} eV | U: {uncertainty:.2f} meV/A "
                    f"(Thresh: {self.threshold:.2f})"
                )

                # 3. Check Uncertainty Trigger
                if uncertainty > self.threshold:
                    print(
                        f"\n>>> [ACTIVE LEARNING TRIGGER] Trajectory {traj_idx} ({GEOMETRY_NAMES[traj_idx]}) "
                        f"crossed uncertainty threshold: {uncertainty:.2f} > {self.threshold:.2f} meV/A!"
                    )

                    # A. Execute Local FHI-aims DFT (32 cores) with failure capture & rollback
                    try:
                        e_dft, f_dft, calc_dir = run_aims_single_point(
                            atoms=atoms,
                            dft_root_dir=DFT_CALCS_DIR,
                            step=step,
                            traj_idx=traj_idx,
                            control_in_path=CONTROL_IN,
                            species_dir=SPECIES_DIR,
                            aims_bin=AIMS_BIN,
                            n_cores=N_MPI_CORES,
                        )
                    except AimsCalculationError as err:
                        print(f"\n[DFT FAILED] Calculation failed at step {step} for traj {traj_idx}: {err}")
                        print("Rolling back trajectory to last safe checkpoint...")
                        atoms.set_positions(safe_pos)
                        atoms.set_velocities(safe_vel)
                        continue

                    self.total_dft_evals += 1
                    self.cycle_count += 1

                    # B. Append labeled frame to active learning dataset
                    labeled_atoms = atoms.copy()
                    labeled_atoms.info["REF_energy"] = e_dft
                    labeled_atoms.arrays["REF_forces"] = f_dft
                    write(str(DATASET_PATH), labeled_atoms, format="extxyz", append=True)

                    # C. Retrain the 6 PaiNN committee members with persistent optimizers
                    retrain_ensemble(
                        calculator=self.calculator,
                        dataset_path=DATASET_PATH,
                        ref_energies=REF_ENERGIES,
                        checkpoint_dir=CHECKPOINTS_DIR,
                        cycle=self.cycle_count,
                        n_epochs=FINE_TUNE_EPOCHS,
                        batch_size=4,
                        lr=1e-4,
                        device=self.device,
                        optimizers=self.optimizers,
                    )

                    # D. Update safe checkpoint
                    self.last_checkpoints[traj_idx] = (atoms.get_positions().copy(), atoms.get_velocities().copy())

                    # E. Save pipeline state
                    self._save_checkpoint()

                    if self.total_dft_evals >= MAX_DFT_POINTS:
                        print(f"\n[FINISHED] Reached maximum DFT budget ({MAX_DFT_POINTS}). Active learning loop complete.")
                        break

            if self.total_dft_evals >= MAX_DFT_POINTS:
                break

        print("\n[MD EXPLORATION COMPLETE] Running post-active learning convergence...")
        run_final_convergence(
            calculator=self.calculator,
            dataset_path=DATASET_PATH,
            ref_energies=REF_ENERGIES,
            output_dir=CHECKPOINTS_DIR / "converged_models",
            n_epochs=50,
            device=self.device,
        )
        self._save_checkpoint()


if __name__ == "__main__":
    pipeline = ALPipeline()
    pipeline.run()
