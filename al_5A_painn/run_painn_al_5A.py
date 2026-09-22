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
from lammps_dumps import write_lammps_dump
from dft_interface import run_aims_single_point
from retrain_engine import retrain_ensemble

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
DUMP_EVERY_D1 = 1000
DUMP_EVERY_D2 = 5000
INITIAL_U_THRESH = 25.0
MIN_U_THRESH = 20.0
THRESH_RELAX_FACTOR = 1.02
MAX_AL_CYCLES = 50
DESIRED_ACC_FORCE_MAE = 180.0  # meV/A (target Force MAE on validation set to stop early; set None to disable)


class PainnActiveLearningManager5A:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.dft_dir = self.base_dir / "dft_calculations"
        self.ckpt_dir = self.base_dir / "checkpoints"
        self.traj_dir = self.base_dir / "trajectories"
        self.dataset_file = self.base_dir / "al_dataset.xyz"
        self.state_file = self.base_dir / "al_checkpoint.json"

        for d in [self.dft_dir, self.ckpt_dir, self.traj_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.u_thresh = INITIAL_U_THRESH
        self.cycle = 0
        self.step = 0
        self.active_traj_idx = 0
        self.accuracy_reached = False

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
            "active_traj_idx": self.active_traj_idx,
            "u_thresh": self.u_thresh,
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
        self.active_traj_idx = state.get("active_traj_idx", 0)
        self.u_thresh = state.get("u_thresh", INITIAL_U_THRESH)

        saved_dirs = [Path(d) for d in state.get("current_model_dirs", [])]
        if saved_dirs and all(d.exists() for d in saved_dirs):
            self.current_model_dirs = saved_dirs
            self.calc = PainnEnsembleCalculator(
                model_dirs=self.current_model_dirs,
                ref_energies=REF_ENERGIES,
                cutoff=6.0,
                device=self.device,
            )

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
        print(f">>> Uncertainty: {u_value:.2f} meV/A > Threshold: {self.u_thresh:.2f} meV/A")
        print("=" * 80)

        # 1. Run FHI-aims single point (32 cores)
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

        # 2. Append labeled frame to dataset
        labeled_atoms = atoms.copy()
        labeled_atoms.info["REF_energy"] = e_dft
        labeled_atoms.arrays["REF_forces"] = f_dft
        write(str(self.dataset_file), labeled_atoms, format="extxyz", append=True)

        # 3. Dynamic relaxation of threshold
        self.u_thresh = max(MIN_U_THRESH, self.u_thresh * THRESH_RELAX_FACTOR)
        print(f"[AL-Manager] Relaxed uncertainty threshold to {self.u_thresh:.2f} meV/A")

        # 4. Retrain 3 PaiNN models (1 epoch)
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
        )
        self.current_model_dirs = [Path(p) for p in updated_paths]

        # 5. Ensure updated calculator is bound to all trajectories
        for t in self.trajectories:
            t["atoms"].calc = self.calc

        # 6. Evaluate validation set & check desired_acc stopping criterion
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
        if not INITIAL_VAL_DATASET.exists():
            return None, None

        val_frames = read(str(INITIAL_VAL_DATASET), index=":")
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

        while self.step < MAX_MD_STEPS and self.cycle < MAX_AL_CYCLES and not self.accuracy_reached:
            self.step += 1

            for i, t in enumerate(self.trajectories):
                self.active_traj_idx = i
                dyn = dyn_drivers[i]
                atoms = t["atoms"]

                dyn.run(1)
                t["step"] += 1

                res = atoms.calc.results
                u = res.get("max_atomic_sd", 0.0)

                # Dumps
                if t["step"] % DUMP_EVERY_D1 == 0:
                    write_lammps_dump(t["d1_path"], step=t["step"], atoms=atoms, stress=None, append=True)
                if t["step"] % DUMP_EVERY_D2 == 0:
                    stress = res.get("atomic_stress", None)
                    write_lammps_dump(t["d2_path"], step=t["step"], atoms=atoms, stress=stress, append=True)

                if self.step % 100 == 0 and i == 0:
                    temp = atoms.get_temperature()
                    print(f"Step {self.step:05d} | {t['name']} | T: {temp:.1f} K | U: {u:.2f} meV/A | Thresh: {self.u_thresh:.2f} meV/A")

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


if __name__ == "__main__":
    manager = PainnActiveLearningManager5A()
    manager.run()
