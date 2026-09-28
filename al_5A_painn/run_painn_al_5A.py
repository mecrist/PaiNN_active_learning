#!/usr/bin/env python
"""
PaiNN closed-loop active learning pipeline for 5 Angstrom water gap silica systems.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aims_PAX.tools.model_tools.setup_painn import PainnEnsembleCalculator, KCAL_TO_EV
from aims_PAX.tools.utilities.lammps_dumps import write_lammps_dump, identify_molecules
from aims_PAX.tools.utilities.dft_interface import run_aims_single_point, AimsCalculationError
from aims_PAX.tools.model_tools.train_painn import retrain_ensemble, run_final_convergence


class AimsPaxThresholdManager:
    """
    Rolling-window adaptive threshold engine following aims-PAX uncertainty protocol.
    Dynamically adjusts the threshold as model uncertainty evolves with margin and floor.
    """
    def __init__(self, initial_threshold=2.50, c_x=0.25, min_threshold=1.80, max_history=400, freeze_dataset_size=540, min_history=10):
        self.initial_threshold = initial_threshold
        self.threshold = initial_threshold
        self.c_x = c_x
        self.min_threshold = min_threshold
        self.max_history = max_history
        self.freeze_dataset_size = freeze_dataset_size
        self.min_history = min_history
        self.uncertainty_history = []
        self.frozen = False

    def update(self, current_uncertainty, current_dataset_size):
        if np.isfinite(current_uncertainty) and current_uncertainty > 0:
            self.uncertainty_history.append(float(current_uncertainty))

        if current_dataset_size >= self.freeze_dataset_size and not self.frozen:
            print(f"[THRESHOLD] Freezing threshold at {self.threshold:.4f} eV/A (dataset size: {current_dataset_size})")
            self.frozen = True
            return self.threshold

        if not self.frozen and len(self.uncertainty_history) > self.min_history:
            recent = self.uncertainty_history[-self.max_history:]
            avg_u = float(np.mean(recent))
            calculated_thresh = avg_u * (1.0 + self.c_x)
            self.threshold = max(calculated_thresh, self.min_threshold)

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
INITIAL_U_THRESH = 2.50
MIN_U_THRESH = 1.80
C_X_RATIO = 0.25
TRIGGER_COOLDOWN_STEPS = 100
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
            c_x=C_X_RATIO,
            min_threshold=MIN_U_THRESH,
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
        self.last_safe_checkpoints = {}
        self.cooldown_counters = {}
        self.consecutive_failures = {}

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
        # Check if resuming from checkpoint to load latest model weights
        saved_dirs = None
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    st = json.load(f)
                cand_dirs = [Path(d) for d in st.get("current_model_dirs", [])]
                if cand_dirs and all(d.exists() for d in cand_dirs):
                    saved_dirs = cand_dirs
            except Exception:
                saved_dirs = None

        if saved_dirs:
            self.current_model_dirs = saved_dirs
            print(f"[AL-Manager] Resuming with {len(self.current_model_dirs)} models from checkpoint ({self.current_model_dirs[0].parent.name})...")
        else:
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

            # Check initial forces to prevent explosive thermal shocks (e.g. Beta-cristobalite)
            try:
                init_forces = atoms.get_forces()
                max_f = float(np.max(np.abs(init_forces)))
                if max_f > 4.0:
                    print(f"  --> [STABILITY] High initial force ({max_f:.2f} eV/A > 4.0 eV/A) in {g_file.name}. Pre-relaxing with FIRE (max 25 steps)...")
                    from ase.optimize import FIRE
                    opt = FIRE(atoms, logfile=None)
                    opt.run(fmax=3.0, steps=25)
                    new_max_f = float(np.max(np.abs(atoms.get_forces())))
                    print(f"  --> [STABILITY] Pre-relaxation finished: max force reduced to {new_max_f:.2f} eV/A.")
            except Exception as e:
                print(f"  --> [WARNING] Pre-relaxation skipped for {g_file.name}: {e}")

            self.trajectories.append({
                "name": g_file.stem,
                "atoms": atoms,
                "step": 0,
                "d1_path": self.traj_dir / f"traj_{g_file.stem}.lammpstrj",
                "d2_path": self.traj_dir / f"traj_stress_{g_file.stem}.lammpstrj",
            })
            self.cooldown_counters[i] = 0
            self.consecutive_failures[i] = 0
            self.last_safe_checkpoints[i] = {
                "positions": atoms.get_positions().copy(),
                "velocities": np.zeros_like(atoms.get_positions()),
                "step": 0,
            }

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
        self.u_thresh = max(state.get("u_thresh", INITIAL_U_THRESH), MIN_U_THRESH)
        self.threshold_mgr.threshold = self.u_thresh
        self.threshold_mgr.uncertainty_history = state.get("uncertainty_history", [])
        self.threshold_mgr.frozen = state.get("threshold_frozen", False)


        for i, t_state in enumerate(state.get("trajectories", [])):
            if i < len(self.trajectories):
                pos = np.array(t_state["positions"])
                vel = np.array(t_state["velocities"])
                # Guard against corrupted / NaN trajectory state in checkpoint
                if not np.isfinite(pos).all() or not np.isfinite(vel).all():
                    print(f"[AL-Manager] WARNING: Found NaN in checkpoint for trajectory {i} ({self.trajectories[i]['name']})! Resetting to fresh initial geometry at {TEMPERATURE_K} K.")
                    init_atoms = read(str(self.geometry_files[i]), format="aims")
                    self.trajectories[i]["atoms"].set_positions(init_atoms.get_positions())
                    MaxwellBoltzmannDistribution(self.trajectories[i]["atoms"], temperature_K=TEMPERATURE_K, rng=np.random.RandomState(42 + i))
                    Stationary(self.trajectories[i]["atoms"])
                    self.trajectories[i]["step"] = 0
                else:
                    self.trajectories[i]["step"] = t_state["step"]
                    self.trajectories[i]["atoms"].set_positions(pos)
                    self.trajectories[i]["atoms"].set_velocities(vel)
                self.trajectories[i]["atoms"].calc = self.calc
                self.last_safe_checkpoints[i] = {
                    "positions": self.trajectories[i]["atoms"].get_positions().copy(),
                    "velocities": self.trajectories[i]["atoms"].get_velocities().copy(),
                    "step": self.trajectories[i]["step"],
                }

    def handle_uncertainty_trigger(self, traj_idx: int, atoms: Atoms, u_value: float, atom_std: np.ndarray = None):
        self.cycle += 1
        trigger_info = ""
        if atom_std is not None:
            max_idx = int(np.argmax(atom_std))
            elem = atoms.get_chemical_symbols()[max_idx]
            z_pos = atoms.get_positions()[max_idx, 2]
            trigger_info = f" | Trigger Atom: #{max_idx} ({elem}) at z={z_pos:.2f} Å"

        print("\n" + "=" * 80)
        print(f">>> [TRIGGER] Cycle {self.cycle} | Trajectory {traj_idx} ({self.trajectories[traj_idx]['name']})")
        print(f">>> Uncertainty: {u_value:.4f} eV/A > Threshold: {self.u_thresh:.4f} eV/A{trigger_info}")
        print("=" * 80)

        # 1. Run FHI-aims single point (32 cores) with failure capture
        try:
            e_dft, f_dft, calc_dir, *_ = run_aims_single_point(
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

            # Rollback trajectory to last safe checkpoint and RE-THERMALIZE velocities
            if traj_idx in self.last_safe_checkpoints:
                safe_data = self.last_safe_checkpoints[traj_idx]
                atoms.set_positions(safe_data["positions"].copy())
                # Re-thermalize with fresh random seed so MD explores a different non-divergent path
                seed = int(time.time() * 1000) % 100000 + traj_idx
                MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random.RandomState(seed))
                Stationary(atoms)
                atoms.calc = self.calc
                self.trajectories[traj_idx]["step"] = safe_data["step"]
                self.consecutive_failures[traj_idx] = self.consecutive_failures.get(traj_idx, 0) + 1
                self.cooldown_counters[traj_idx] = TRIGGER_COOLDOWN_STEPS

                # If repeated failures occur at this region, apply small perturbation (0.02 A) to escape basin
                if self.consecutive_failures[traj_idx] >= 2:
                    print(f"[AL-Manager] Multiple DFT failures ({self.consecutive_failures[traj_idx]}) on {self.trajectories[traj_idx]['name']}. Applying subtle thermal displacement (0.02 A) to escape basin...")
                    noise = np.random.normal(0.0, 0.02, size=atoms.get_positions().shape)
                    atoms.set_positions(atoms.get_positions() + noise)

                print(f"[AL-Manager] Rolled back {self.trajectories[traj_idx]['name']} to safe step {safe_data['step']} with re-thermalized velocities.")
            return False

        # 2. Append labeled frame to dataset (with 10% validation quota per aims-PAX)
        self.consecutive_failures[traj_idx] = 0
        self.last_safe_checkpoints[traj_idx] = {
            "positions": atoms.get_positions().copy(),
            "velocities": atoms.get_velocities().copy(),
            "step": self.trajectories[traj_idx]["step"],
        }
        self.cooldown_counters[traj_idx] = TRIGGER_COOLDOWN_STEPS

        labeled_atoms = atoms.copy()
        labeled_atoms.info["REF_energy"] = e_dft
        labeled_atoms.arrays["REF_forces"] = f_dft
        if atom_std is not None:
            labeled_atoms.arrays["force_std"] = atom_std
            max_idx = int(np.argmax(atom_std))
            labeled_atoms.info["trigger_atom_idx"] = max_idx
            labeled_atoms.info["trigger_element"] = atoms.get_chemical_symbols()[max_idx]
            labeled_atoms.info["trigger_z"] = float(atoms.get_positions()[max_idx, 2])

        total_points_added = self.train_points_added + self.val_points_added + 1
        if self.val_points_added < VALID_RATIO * total_points_added:
            write(str(self.val_dataset_file), labeled_atoms, format="extxyz", append=True)
            self.val_points_added += 1
            print(f"[AL-Manager] Added point to validation set (Total Val: {self.val_points_added}). Skipping retrain per aims-PAX quota.")
            self.last_checkpoints[traj_idx] = (atoms.get_positions().copy(), atoms.get_velocities().copy())
            self.save_checkpoint()
            return True

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

                dyn.run(SKIP_STEP_MLFF)
                t["step"] += SKIP_STEP_MLFF
                self.step = max(tr["step"] for tr in self.trajectories)

                pos = atoms.get_positions()
                vel = atoms.get_velocities()

                # Guard 1: Detect NaN/Inf coordinates and recover immediately
                if not np.isfinite(pos).all() or not np.isfinite(vel).all():
                    print(f"\n[AL-Manager] WARNING: NaN/Inf coordinates detected in {t['name']} at step {t['step']}! Recovering from last safe checkpoint...")
                    safe_data = self.last_safe_checkpoints[i]
                    atoms.set_positions(safe_data["positions"].copy())
                    seed = int(time.time() * 1000) % 100000 + i
                    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random.RandomState(seed))
                    Stationary(atoms)
                    t["step"] = safe_data["step"]
                    self.cooldown_counters[i] = TRIGGER_COOLDOWN_STEPS
                    continue

                # Guard 2: Detect thermal runaway (T > 1000 K) and re-thermalize
                temp = atoms.get_temperature()
                if temp > 1000.0:
                    print(f"\n[AL-Manager] WARNING: Thermal runaway detected in {t['name']} (T = {temp:.1f} K > 1000 K)! Re-thermalizing velocities to {TEMPERATURE_K} K...")
                    seed = int(time.time() * 1000) % 100000 + i
                    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random.RandomState(seed))
                    Stationary(atoms)

                res = atoms.calc.results
                u = float(res.get("max_atomic_sd", 0.0))

                # Update rolling adaptive threshold
                ds_size = len(read(str(self.dataset_file), index=":"))
                self.u_thresh = self.threshold_mgr.update(u, ds_size)

                # Update safe checkpoint if trajectory is healthy and stable
                if temp < 550.0 and u < self.u_thresh * 0.85:
                    self.last_safe_checkpoints[i] = {
                        "positions": pos.copy(),
                        "velocities": vel.copy(),
                        "step": t["step"],
                    }

                # Decrement cooldown counter if active
                if self.cooldown_counters[i] > 0:
                    self.cooldown_counters[i] -= SKIP_STEP_MLFF

                # Dumps with dynamic molecule identification
                if t["step"] % DUMP_EVERY_D1 == 0:
                    mol_ids = identify_molecules(atoms)
                    write_lammps_dump(t["d1_path"], step=t["step"], atoms=atoms, mol_ids=mol_ids, stress=None, append=True)
                if t["step"] % DUMP_EVERY_D2 == 0:
                    mol_ids = identify_molecules(atoms)
                    stress = res.get("atomic_stress", None)
                    write_lammps_dump(t["d2_path"], step=t["step"], atoms=atoms, mol_ids=mol_ids, stress=stress, append=True)

                if t["step"] % 100 == 0:
                    thresh_str = f"{self.u_thresh:.4f} eV/A" if np.isfinite(self.u_thresh) else "inf"
                    cd_str = f" | CD: {self.cooldown_counters[i]}" if self.cooldown_counters[i] > 0 else ""
                    print(f"Step {t['step']:05d} | {t['name']} | T: {temp:.1f} K | U: {u:.4f} eV/A | Thresh: {thresh_str}{cd_str}")

                # Trigger condition (checked only if cooldown is expired)
                if u > self.u_thresh:
                    if self.cooldown_counters[i] > 0:
                        if t["step"] % 100 == 0:
                            print(f"[COOLDOWN] Trajectory {i} ({t['name']}) U={u:.4f} > {self.u_thresh:.4f}, but in cooldown ({self.cooldown_counters[i]} steps left).")
                    else:
                        atom_std = res.get("std_per_atom", None)
                        self.handle_uncertainty_trigger(traj_idx=i, atoms=atoms, u_value=u, atom_std=atom_std)
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
