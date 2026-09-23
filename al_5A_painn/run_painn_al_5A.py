#!/usr/bin/env python
"""
================================================================================
   PAINN CLOSED-LOOP ACTIVE LEARNING PIPELINE: 5 ANGSTROM WATER GAP (3 SYSTEMS)
================================================================================
Drives closed-loop active learning on the 3 silica-water interfaces with 5 A gap:
  1. geometry_alpha_5.in  (Alpha-quartz + 5 A water layer)
  2. geometry_amor_5.in   (Amorphous silica + 5 A water layer)
  3. geometry_beta_5.in   (Beta-cristobalite + 5 A water layer)

Full Closed Loop:
  - 3 fine-tuned PaiNN models (mine/model_2_ep500)
  - MD at 300 K (ASE Langevin, dt=0.5 fs)
  - Uncertainty metric: U = max_i sigma_i (force standard deviation)
  - Threshold: U_thresh = 25.0 meV/A with dynamic relaxation
  - On trigger (U > U_thresh):
      1. Halts MD & archives structure in dft_calculations/
      2. Executes local 32-core FHI-aims single point (mpirun -np 32 aims.x)
      3. Shifts raw DFT energy with atomic baseline E0
      4. Appends to al_dataset.xyz
      5. Retrains 3 PaiNN committee members for 1 epoch
      6. Reloads weights and resumes MD
  - Dumps: unified LAMMPS dump every 1,000 steps; virial stress dump every 5,000 steps
  - State tracking: al_checkpoint.json
================================================================================
"""

import os
import sys
import json
import time
import shutil
import signal
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ase import Atoms, units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from painn_ensemble_calc import PainnEnsembleCalculator, KCAL_TO_EV
from lammps_dumps import write_lammps_dump, identify_molecules
from dft_interface import run_aims_single_point, AimsCalculationError
from retrain_engine import retrain_ensemble, run_final_convergence


class AimsPaxThresholdManager:
    """
    Rolling-window adaptive threshold engine following aims-PAX uncertainty protocol.
    Dynamically adjusts the threshold as model uncertainty evolves.
    """
    def __init__(self, initial_threshold=float("inf"), c_x=0.0, max_history=400, freeze_dataset_size=540, min_history=10):
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
            print(f"[THRESHOLD] Freezing threshold at {self.threshold:.4f} eV/A (dataset size: {current_dataset_size})")
            self.frozen = True
            return self.threshold

        if not self.frozen and len(self.uncertainty_history) > self.min_history:
            recent = self.uncertainty_history[-self.max_history:]
            avg_u = float(np.mean(recent))
            self.threshold = avg_u * (1.0 + self.c_x)

        return self.threshold

# Paths
BASE_DIR = Path("/home/maria.crist/dft_mlip/sep_pax/al_5A_painn")
GEOMETRIES_DIR = BASE_DIR / "geometries"
CONTROL_IN = BASE_DIR / "control.in"
SPECIES_DIR = Path("/home/maria.crist/fhi-aims.260331/species_defaults/defaults_2020/light")
AIMS_BIN = "/home/maria.crist/fhi-aims.260331/bin/aims.x"

# Models: 3 fine-tuned PaiNN models (mine dataset, epoch 500)
INITIAL_MODEL_DIRS = [
    Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_0"),
    Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_1"),
    Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_2"),
]
INITIAL_TRAIN_DATASET = Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/train.xyz")
INITIAL_VAL_DATASET = Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/val.xyz")

# Elemental Reference Energies (E0 in eV)
REF_ENERGIES = {
    1: 1295.1619808355229,
    8: -4671.611073387086,
    14: -2659.5960319024257,
}

# Simulation parameters
TEMPERATURE_K = 300.0
TIMESTEP_FS = 0.5
LANGEVIN_FRICTION = 0.002 / units.fs
MAX_MD_STEPS = 10000
SKIP_STEP_MLFF = 25
DUMP_EVERY_D1 = 1000
DUMP_EVERY_D2 = 5000
INITIAL_U_THRESH = float("inf")
VALID_RATIO = 0.1
MAX_AL_CYCLES = 50
DESIRED_ACC_FORCE_MAE = None  # Match aimsprobe.yaml (desired_acc=0.0): run full 10k steps unless max_train_set_size is reached


class PainnActiveLearningManager5A:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.dft_dir = self.base_dir / "dft_calculations"
        self.ckpt_dir = self.base_dir / "checkpoints"
        self.traj_dir = self.base_dir / "trajectories"
        self.dataset_file = self.base_dir / "al_dataset.xyz"
        self.val_dataset_file = self.base_dir / "val.xyz"
        self.state_file = self.base_dir / "al_checkpoint.json"

        self.failed_dir = self.base_dir / "failed_calculations"
        for d in [self.dft_dir, self.ckpt_dir, self.traj_dir, self.failed_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.threshold_mgr = AimsPaxThresholdManager(
            initial_threshold=INITIAL_U_THRESH,
            c_x=0.0,
            freeze_dataset_size=540,
        )
        self.u_thresh = INITIAL_U_THRESH
        self.cycle = 0
        self.step = 0
        self.train_points_added = 0
        self.val_points_added = 0
        self.active_traj_idx = 0
        self.accuracy_reached = False
        self.last_checkpoints = {}

        self.setup_signal_handlers()
        self.initialize_dataset()
        self.setup_models()
        self.load_geometries()
        self.load_checkpoint()

    def setup_signal_handlers(self):
        def handle_termination(signum, frame):
            print(f"\n[AL-Manager] Caught signal {signum}. Saving state before exit...")
            self.save_checkpoint()
            sys.exit(0)
        signal.signal(signal.SIGTERM, handle_termination)
        signal.signal(signal.SIGINT, handle_termination)

    def initialize_dataset(self):
        if not self.dataset_file.exists():
            print(f"[AL-Manager] Initializing al_dataset.xyz from {INITIAL_TRAIN_DATASET}...")
            shutil.copyfile(INITIAL_TRAIN_DATASET, self.dataset_file)
        if not self.val_dataset_file.exists() and INITIAL_VAL_DATASET.exists():
            print(f"[AL-Manager] Initializing val.xyz from {INITIAL_VAL_DATASET}...")
            shutil.copyfile(INITIAL_VAL_DATASET, self.val_dataset_file)
        frames = read(str(self.dataset_file), index=":")
        print(f"[AL-Manager] Active learning dataset contains {len(frames)} frames.")

    def setup_models(self):
        self.current_model_dirs = []
        cycle_0_dir = self.ckpt_dir / "cycle_00"
        for i, m_dir in enumerate(INITIAL_MODEL_DIRS):
            target_dir = cycle_0_dir / f"model_{i}"
            if not target_dir.exists():
                shutil.copytree(m_dir, target_dir)
            self.current_model_dirs.append(target_dir)

        print(f"[AL-Manager] Loading 3 PaiNN models to {self.device}...")
        self.calc = PainnEnsembleCalculator(
            model_dirs=self.current_model_dirs,
            ref_energies=REF_ENERGIES,
            cutoff=6.0,
            device=self.device,
        )
        # Persistent Adam optimizers
        self.optimizers = {
            i: torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-4)
            for i, m in enumerate(self.calc.models)
        }

    def load_geometries(self):
        self.geometry_files = sorted(list(GEOMETRIES_DIR.glob("*.in")))
        print(f"[AL-Manager] Loaded {len(self.geometry_files)} 5 Angstrom gap geometries:")
        self.trajectories = []
        for i, g_file in enumerate(self.geometry_files):
            atoms = read(str(g_file), format="aims")
            atoms.calc = self.calc
            print(f"  [{i}] {g_file.name} ({len(atoms)} atoms)")
            self.trajectories.append({
                "name": g_file.stem,
                "atoms": atoms,
                "step": 0,
                "d1_path": self.traj_dir / f"traj_{g_file.stem}.lammpstrj",
                "d2_path": self.traj_dir / f"traj_stress_{g_file.stem}.lammpstrj",
            })

    def save_checkpoint(self):
        traj_states = []
        for t in self.trajectories:
            traj_states.append({
                "step": t["step"],
                "positions": t["atoms"].get_positions().tolist(),
                "velocities": t["atoms"].get_velocities().tolist(),
            })

        state = {
            "cycle": self.cycle,
            "step": self.step,
            "train_points_added": self.train_points_added,
            "val_points_added": self.val_points_added,
            "active_traj_idx": self.active_traj_idx,
            "u_thresh": self.u_thresh,
            "uncertainty_history": self.threshold_mgr.uncertainty_history,
            "threshold_frozen": self.threshold_mgr.frozen,
            "current_model_dirs": [str(d) for d in self.current_model_dirs],
            "trajectories": traj_states,
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[AL-Manager] State saved to {self.state_file.name} (Cycle: {self.cycle}, Step: {self.step})")

    def load_checkpoint(self):
        if not self.state_file.exists():
            print("[AL-Manager] No previous checkpoint found. Starting fresh run at 300 K.")
            for t in self.trajectories:
                MaxwellBoltzmannDistribution(t["atoms"], temperature_K=TEMPERATURE_K, rng=np.random.RandomState(42))
                Stationary(t["atoms"])
            return

        print(f"[AL-Manager] Resuming from checkpoint {self.state_file}...")
        with open(self.state_file, "r") as f:
            state = json.load(f)

        self.cycle = state.get("cycle", 0)
        self.step = state.get("step", 0)
        self.train_points_added = state.get("train_points_added", 0)
        self.val_points_added = state.get("val_points_added", 0)
        self.active_traj_idx = state.get("active_traj_idx", 0)
        self.u_thresh = state.get("u_thresh", INITIAL_U_THRESH)
        self.threshold_mgr.threshold = self.u_thresh
        self.threshold_mgr.uncertainty_history = state.get("uncertainty_history", [])
        self.threshold_mgr.frozen = state.get("threshold_frozen", False)

        saved_dirs = [Path(d) for d in state.get("current_model_dirs", [])]
        if saved_dirs and all(d.exists() for d in saved_dirs):
            self.current_model_dirs = saved_dirs
            self.calc = PainnEnsembleCalculator(
                model_dirs=self.current_model_dirs,
                ref_energies=REF_ENERGIES,
                cutoff=6.0,
                device=self.device,
            )
            self.optimizers = {
                i: torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-4)
                for i, m in enumerate(self.calc.models)
            }

        for i, t_state in enumerate(state.get("trajectories", [])):
            if i < len(self.trajectories):
                self.trajectories[i]["step"] = t_state["step"]
                self.trajectories[i]["atoms"].set_positions(np.array(t_state["positions"]))
                self.trajectories[i]["atoms"].set_velocities(np.array(t_state["velocities"]))
                self.trajectories[i]["atoms"].calc = self.calc

    def handle_uncertainty_trigger(self, traj_idx: int, atoms: Atoms, u_value: float):
        self.cycle += 1
        print("\n" + "=" * 80)
        print(f">>> [TRIGGER] Cycle {self.cycle} | Trajectory {traj_idx} ({self.trajectories[traj_idx]['name']})")
        print(f">>> Uncertainty: {u_value:.4f} eV/A > Threshold: {self.u_thresh:.4f} eV/A")
        print("=" * 80)

        # 1. Run FHI-aims single point (32 cores) with failure capture
        try:
            e_dft, f_dft, calc_dir = run_aims_single_point(
                atoms=atoms,
                dft_root_dir=self.dft_dir,
                step=self.step,
                traj_idx=traj_idx,
                control_in_path=CONTROL_IN,
                species_dir=SPECIES_DIR,
                aims_bin=AIMS_BIN,
                n_cores=32,
            )
        except AimsCalculationError as err:
            self.cycle -= 1
            print("\n" + "!" * 80)
            print(f">>> [CALCULATION FAILED] FHI-aims failed on Trajectory {traj_idx} ({self.trajectories[traj_idx]['name']}) at step {self.step}!")
            print(f">>> Preserving failed calculation in '{self.failed_dir.name}/' for investigation.")
            print("!" * 80 + "\n")

            failed_target = self.failed_dir / f"step_{self.step:06d}_traj_{traj_idx}_{self.trajectories[traj_idx]['name']}"
            if err.calc_dir.exists():
                if failed_target.exists():
                    shutil.rmtree(failed_target)
                shutil.move(str(err.calc_dir), str(failed_target))

            log_entry = (
                f"\n{'=' * 80}\n"
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Trajectory: {traj_idx} ({self.trajectories[traj_idx]['name']}) | MD Step: {self.step}\n"
                f"Uncertainty: {u_value:.4f} eV/A | Threshold: {self.u_thresh:.4f} eV/A\n"
                f"Preserved Folder: {failed_target}\n"
                f"Error Message: {err}\n"
                f"--- aims.out Error Snippet ---\n"
                f"{err.log_snippet}\n"
                f"{'=' * 80}\n"
            )
            with open(self.failed_dir / "failed_calculations.log", "a") as f:
                f.write(log_entry)

            # Rollback trajectory to last safe checkpoint
            if traj_idx in self.last_checkpoints:
                safe_pos, safe_vel = self.last_checkpoints[traj_idx]
                atoms.set_positions(safe_pos)
                atoms.set_velocities(safe_vel)
                print(f"[AL-Manager] Rolled back {self.trajectories[traj_idx]['name']} to last safe checkpoint.")
            return

        # 2. Append labeled frame to dataset (with 10% validation quota per aims-PAX)
        labeled_atoms = atoms.copy()
        labeled_atoms.info["REF_energy"] = e_dft
        labeled_atoms.arrays["REF_forces"] = f_dft

        total_points_added = self.train_points_added + self.val_points_added + 1
        if self.val_points_added < VALID_RATIO * total_points_added:
            write(str(self.val_dataset_file), labeled_atoms, format="extxyz", append=True)
            self.val_points_added += 1
            print(f"[AL-Manager] Added point to validation set (Total Val: {self.val_points_added}). Skipping retrain per aims-PAX quota.")
            self.last_checkpoints[traj_idx] = (atoms.get_positions().copy(), atoms.get_velocities().copy())
            self.save_checkpoint()
            return

        write(str(self.dataset_file), labeled_atoms, format="extxyz", append=True)
        self.train_points_added += 1

        # Update last known safe checkpoint
        self.last_checkpoints[traj_idx] = (atoms.get_positions().copy(), atoms.get_velocities().copy())

        # 3. Retrain 3 PaiNN models (1 epoch) with persistent optimizers
        print(f"[AL-Manager] Retraining 3 PaiNN models for 1 epoch (Cycle {self.cycle})...")
        updated_paths = retrain_ensemble(
            calculator=self.calc,
            dataset_path=self.dataset_file,
            ref_energies=REF_ENERGIES,
            checkpoint_dir=self.ckpt_dir,
            cycle=self.cycle,
            n_epochs=1,
            batch_size=4,
            lr=1e-4,
            device=self.device,
            val_dataset_path=self.val_dataset_file,
            optimizers=self.optimizers,
        )
        self.current_model_dirs = [Path(p) for p in updated_paths]

        # 4. Ensure updated calculator is bound to all trajectories
        for t in self.trajectories:
            t["atoms"].calc = self.calc

        # 5. Evaluate validation set & check desired_acc stopping criterion
        val_f_mae, val_e_mae = self.evaluate_validation()
        if val_f_mae is not None:
            print(f"[AL-Manager] Cycle {self.cycle} Validation -> Force MAE: {val_f_mae:.2f} meV/A | Energy MAE: {val_e_mae:.2f} meV/atom")
            if DESIRED_ACC_FORCE_MAE is not None and val_f_mae <= DESIRED_ACC_FORCE_MAE:
                print("\n" + "*" * 80)
                print(f">>> [TARGET REACHED] Validation Force MAE ({val_f_mae:.2f} meV/A) <= desired_acc ({DESIRED_ACC_FORCE_MAE:.2f} meV/A)!")
                print(f">>> Active Learning converged successfully on accuracy criterion.")
                print("*" * 80 + "\n")
                self.accuracy_reached = True

        self.save_checkpoint()

    def evaluate_validation(self):
        """
        Evaluates the committee on the held-out validation set to compute Force MAE (meV/A)
        and Energy MAE (meV/atom).
        """
        val_path = self.val_dataset_file if self.val_dataset_file.exists() else INITIAL_VAL_DATASET
        if not val_path.exists():
            return None, None

        val_frames = read(str(val_path), index=":")
        f_maes = []
        e_maes = []
        for atoms in val_frames:
            if "REF_forces" in atoms.arrays:
                f_ref = atoms.arrays["REF_forces"]
            elif "forces" in atoms.arrays:
                f_ref = atoms.arrays["forces"]
            else:
                continue

            if "REF_energy" in atoms.info:
                e_ref = atoms.info["REF_energy"]
            elif "energy" in atoms.info:
                e_ref = atoms.info["energy"]
            else:
                e_ref = None

            atoms.calc = self.calc
            self.calc.calculate(atoms)
            f_pred = self.calc.results["forces"]
            f_mae = np.mean(np.abs(f_pred - f_ref)) * 1000.0  # meV/A
            f_maes.append(f_mae)

            if e_ref is not None:
                e_pred = self.calc.results["energy"]
                e_mae = abs(e_pred - e_ref) / len(atoms) * 1000.0  # meV/atom
                e_maes.append(e_mae)

        mean_f_mae = float(np.mean(f_maes)) if f_maes else None
        mean_e_mae = float(np.mean(e_maes)) if e_maes else None
        return mean_f_mae, mean_e_mae

    def run(self):
        print("\n==========================================================")
        print(f"STARTING PAINN ACTIVE LEARNING RUN ON 5 A GEOMETRIES")
        print(f"Max steps: {MAX_MD_STEPS} | T: {TEMPERATURE_K} K | Device: {self.device}")
        if DESIRED_ACC_FORCE_MAE is not None:
            print(f"Desired accuracy limit: {DESIRED_ACC_FORCE_MAE:.2f} meV/A Force MAE")
        print("==========================================================\n")

        dyn_drivers = []
        for i, t in enumerate(self.trajectories):
            dyn = Langevin(
                t["atoms"],
                timestep=TIMESTEP_FS * units.fs,
                temperature_K=TEMPERATURE_K,
                friction=LANGEVIN_FRICTION,
                rng=np.random.RandomState(100 + i),
            )
            dyn_drivers.append(dyn)

        while any(t["step"] < MAX_MD_STEPS for t in self.trajectories) and self.cycle < MAX_AL_CYCLES and not self.accuracy_reached:
            for i, t in enumerate(self.trajectories):
                if t["step"] >= MAX_MD_STEPS:
                    continue

                self.active_traj_idx = i
                dyn = dyn_drivers[i]
                atoms = t["atoms"]

                # Save safe state before stepping
                safe_pos = atoms.get_positions().copy()
                safe_vel = atoms.get_velocities().copy()
                self.last_checkpoints[i] = (safe_pos, safe_vel)

                dyn.run(SKIP_STEP_MLFF)
                t["step"] += SKIP_STEP_MLFF
                self.step = max(tr["step"] for tr in self.trajectories)

                res = atoms.calc.results
                u = float(res.get("max_atomic_sd", 0.0))

                # Update rolling adaptive threshold
                ds_size = len(read(str(self.dataset_file), index=":"))
                self.u_thresh = self.threshold_mgr.update(u, ds_size)

                # Dumps with dynamic molecule identification
                if t["step"] % DUMP_EVERY_D1 == 0:
                    mol_ids = identify_molecules(atoms)
                    write_lammps_dump(t["d1_path"], step=t["step"], atoms=atoms, mol_ids=mol_ids, stress=None, append=True)
                if t["step"] % DUMP_EVERY_D2 == 0:
                    mol_ids = identify_molecules(atoms)
                    stress = res.get("atomic_stress", None)
                    write_lammps_dump(t["d2_path"], step=t["step"], atoms=atoms, mol_ids=mol_ids, stress=stress, append=True)

                if t["step"] % 100 == 0:
                    temp = atoms.get_temperature()
                    thresh_str = f"{self.u_thresh:.4f} eV/A" if np.isfinite(self.u_thresh) else "inf"
                    print(f"Step {t['step']:05d} | {t['name']} | T: {temp:.1f} K | U: {u:.4f} eV/A | Thresh: {thresh_str}")

                # Trigger condition
                if u > self.u_thresh:
                    self.handle_uncertainty_trigger(traj_idx=i, atoms=atoms, u_value=u)
                    if self.accuracy_reached:
                        break

            if self.accuracy_reached:
                break

            if self.step % 500 == 0:
                self.save_checkpoint()

        print("\n==========================================================")
        if self.accuracy_reached:
            print("Active Learning HALTED: Desired accuracy target achieved!")
        else:
            print(f"PaiNN Active Learning Completed Successfully!")
        print(f"Total Cycles: {self.cycle} | Final Steps: {self.step}")
        print("==========================================================")

        # Post-AL convergence routine
        if not self.accuracy_reached:
            run_final_convergence(
                calculator=self.calc,
                dataset_path=self.dataset_file,
                val_dataset_path=self.val_dataset_file,
                ref_energies=REF_ENERGIES,
                output_dir=self.ckpt_dir / "converged_models",
                n_epochs=50,
                device=self.device,
            )


if __name__ == "__main__":
    manager = PainnActiveLearningManager5A()
    manager.run()
