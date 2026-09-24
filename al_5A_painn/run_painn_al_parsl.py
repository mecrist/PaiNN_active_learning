#!/usr/bin/env python
"""
================================================================================
   PAINN CLOSED-LOOP ACTIVE LEARNING PIPELINE WITH PARSL: 5 ANGSTROM WATER GAP
================================================================================
Drives closed-loop active learning on the 3 silica-water interfaces with 5 A gap:
  1. geometry_alpha_5.in  (Alpha-quartz + 5 A water layer)
  2. geometry_amor_5.in   (Amorphous silica + 5 A water layer)
  3. geometry_beta_5.in   (Beta-cristobalite + 5 A water layer)

Key PARSL Asynchronous Architecture:
  - Non-blocking Trajectory Propagation: When Trajectory i triggers an out-of-
    distribution uncertainty state (U > U_thresh), it dispatches FHI-aims to
    Parsl as a background task and enters 'waiting' status.
  - Concurrent GPU MD: Other non-triggering trajectories continue propagating
    Langevin dynamics on the GPU without idling for 15-20 minutes.
  - Multi-threaded DFT Pool: Parsl ThreadPoolExecutor manages local 32-core MPI
    FHI-aims executions safely in the background.
  - Seamless Committee Fine-Tuning: Upon DFT completion, the new structure is
    assimilated into the dataset, 3 PaiNN committee members undergo 1 epoch of
    fine-tuning, all trajectory calculators are updated, and the trajectory resumes.
  - Dynamic Adaptive Thresholding: U_thresh = mean(U[-400:]) * (1 + c_x) with
    lower floor protection (min_threshold = 1.80 eV/A) and 100-step cooldowns.
  - Robust Fault-Tolerance: Catches SCF convergence failures, rolls back to
    safe checkpoints with stochastic velocity re-thermalization, avoiding
    infinite rollback loops.
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

import parsl
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from parsl.app.app import python_app

from painn_ensemble_calc import PainnEnsembleCalculator, KCAL_TO_EV
from lammps_dumps import write_lammps_dump, identify_molecules
from dft_interface import run_aims_single_point, AimsCalculationError
from retrain_engine import retrain_ensemble, run_final_convergence


# ==============================================================================
# Global Paths & Constants
# ==============================================================================
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
DESIRED_ACC_FORCE_MAE = None  # None: run full 10k steps unless max_train_set_size is reached


# ==============================================================================
# Adaptive Threshold Engine
# ==============================================================================
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


# ==============================================================================
# Parsl Initialization & Python App
# ==============================================================================
def initialize_parsl(calc_dir: Path, max_workers: int = 1):
    """Initializes Parsl with a local ThreadPoolExecutor."""
    parsl_info_dir = calc_dir / "parsl_info"
    parsl_info_dir.mkdir(parents=True, exist_ok=True)
    config = Config(
        executors=[
            ThreadPoolExecutor(
                label="dft_pool",
                max_threads=max_workers,
            )
        ],
        run_dir=str(parsl_info_dir / "run_dir"),
        initialize_logging=False,
        retries=0,
    )
    try:
        parsl.load(config)
        print(f"[PARSL] Initialized Parsl ThreadPoolExecutor (max_workers={max_workers}).")
    except Exception as e:
        print(f"[PARSL] Parsl DFK already loaded or re-used: {e}")


@python_app(executors=["dft_pool"])
def run_aims_parsl_task(
    atoms_dict: dict,
    dft_root_dir: str,
    step: int,
    traj_idx: int,
    control_in_path: str,
    species_dir: str,
    aims_bin: str,
    n_cores: int = 32,
):
    """
    Parsl application that executes an FHI-aims single-point DFT calculation.
    Runs asynchronously in the Parsl background worker thread.
    """
    import os
    import sys
    from pathlib import Path
    import numpy as np
    from ase import Atoms
    from dft_interface import run_aims_single_point, AimsCalculationError

    atoms = Atoms(
        positions=atoms_dict["positions"],
        numbers=atoms_dict["numbers"],
        cell=atoms_dict["cell"],
        pbc=atoms_dict["pbc"],
    )
    if "velocities" in atoms_dict and atoms_dict["velocities"] is not None:
        atoms.set_velocities(atoms_dict["velocities"])

    calc_dir = Path(dft_root_dir) / f"step_{step:06d}_traj_{traj_idx}"

    try:
        e_dft, f_dft, c_dir, hirshfeld_charges, free_vols = run_aims_single_point(
            atoms=atoms,
            dft_root_dir=dft_root_dir,
            step=step,
            traj_idx=traj_idx,
            control_in_path=control_in_path,
            species_dir=species_dir,
            aims_bin=aims_bin,
            n_cores=n_cores,
        )
        return {
            "success": True,
            "e_dft": float(e_dft),
            "f_dft": f_dft.tolist(),
            "calc_dir": str(c_dir),
            "hirshfeld_charges": hirshfeld_charges.tolist() if hirshfeld_charges is not None else None,
            "free_volumes": free_vols.tolist() if free_vols is not None else None,
            "error": "",
            "log_snippet": "",
        }
    except AimsCalculationError as err:
        return {
            "success": False,
            "e_dft": None,
            "f_dft": None,
            "calc_dir": str(err.calc_dir),
            "hirshfeld_charges": None,
            "free_volumes": None,
            "error": str(err),
            "log_snippet": err.log_snippet,
        }
    except Exception as err:
        return {
            "success": False,
            "e_dft": None,
            "f_dft": None,
            "calc_dir": str(calc_dir),
            "hirshfeld_charges": None,
            "free_volumes": None,
            "error": str(err),
            "log_snippet": "",
        }


# ==============================================================================
# Main Parsl Active Learning Manager
# ==============================================================================
class PainnActiveLearningManager5AParsl:
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
        self.accuracy_reached = False

        initialize_parsl(calc_dir=self.base_dir, max_workers=1)

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
        self.dataset_size = len(frames)
        print(f"[AL-Manager] Active learning dataset contains {self.dataset_size} frames.")

    def setup_models(self):
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
        self.optimizers = {
            i: torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-4)
            for i, m in enumerate(self.calc.models)
        }
        from torch_ema import ExponentialMovingAverage
        self.emas = {
            i: ExponentialMovingAverage(m.parameters(), decay=0.99)
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

            # Check initial forces to prevent explosive thermal shocks
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
                "status": "running",
                "future": None,
                "pending_point": None,
                "cooldown": 0,
                "consecutive_failures": 0,
                "last_safe_checkpoint": {
                    "positions": atoms.get_positions().copy(),
                    "velocities": np.zeros_like(atoms.get_positions()),
                    "step": 0,
                },
                "d1_path": self.traj_dir / f"traj_{g_file.stem}.lammpstrj",
                "d2_path": self.traj_dir / f"traj_stress_{g_file.stem}.lammpstrj",
            })

    def save_checkpoint(self):
        traj_states = []
        for t in self.trajectories:
            traj_states.append({
                "step": t["step"],
                "status": t["status"],
                "positions": t["atoms"].get_positions().tolist(),
                "velocities": t["atoms"].get_velocities().tolist(),
            })

        state = {
            "cycle": self.cycle,
            "step": self.step,
            "train_points_added": self.train_points_added,
            "val_points_added": self.val_points_added,
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
        self.u_thresh = max(state.get("u_thresh", INITIAL_U_THRESH), MIN_U_THRESH)
        self.threshold_mgr.threshold = self.u_thresh
        self.threshold_mgr.uncertainty_history = state.get("uncertainty_history", [])
        self.threshold_mgr.frozen = state.get("threshold_frozen", False)

        for i, t_state in enumerate(state.get("trajectories", [])):
            if i < len(self.trajectories):
                pos = np.array(t_state["positions"])
                vel = np.array(t_state["velocities"])
                t = self.trajectories[i]
                if not np.isfinite(pos).all() or not np.isfinite(vel).all():
                    print(f"[AL-Manager] WARNING: Found NaN in checkpoint for trajectory {i} ({t['name']})! Resetting to fresh initial geometry at {TEMPERATURE_K} K.")
                    init_atoms = read(str(self.geometry_files[i]), format="aims")
                    t["atoms"].set_positions(init_atoms.get_positions())
                    MaxwellBoltzmannDistribution(t["atoms"], temperature_K=TEMPERATURE_K, rng=np.random.RandomState(42 + i))
                    Stationary(t["atoms"])
                    t["step"] = 0
                else:
                    t["step"] = t_state["step"]
                    t["atoms"].set_positions(pos)
                    t["atoms"].set_velocities(vel)
                t["atoms"].calc = self.calc
                t["last_safe_checkpoint"] = {
                    "positions": t["atoms"].get_positions().copy(),
                    "velocities": t["atoms"].get_velocities().copy(),
                    "step": t["step"],
                }
                # Reset status to running so Parsl dispatches fresh calculations if needed
                t["status"] = "running"
                t["future"] = None
                t["pending_point"] = None

    def dispatch_parsl_trigger(self, traj_idx: int, atoms: Atoms, u_value: float, res: dict):
        """Submits an FHI-aims single-point calculation to the background Parsl pool."""
        t = self.trajectories[traj_idx]
        atom_std = res.get("std_per_atom", None)
        trigger_info = ""
        if atom_std is not None:
            max_idx = int(np.argmax(atom_std))
            elem = atoms.get_chemical_symbols()[max_idx]
            z_pos = atoms.get_positions()[max_idx, 2]
            trigger_info = f" | Trigger Atom: #{max_idx} ({elem}) at z={z_pos:.2f} Å"

        print("\n" + "=" * 80)
        print(f">>> [PARSL ASYNC TRIGGER] Trajectory {traj_idx} ({t['name']}) at step {t['step']}")
        print(f">>> Uncertainty: {u_value:.4f} eV/A > Threshold: {self.u_thresh:.4f} eV/A{trigger_info}")
        print(f">>> Dispatched FHI-aims to Parsl background pool. Other trajectories continue running MD!")
        print("=" * 80 + "\n")

        t["status"] = "waiting"
        t["pending_point"] = {
            "atoms": atoms.copy(),
            "u": u_value,
            "atom_std": atom_std,
            "step": t["step"],
        }

        atoms_dict = {
            "positions": atoms.get_positions().tolist(),
            "numbers": atoms.get_atomic_numbers().tolist(),
            "cell": atoms.get_cell().tolist(),
            "pbc": atoms.get_pbc().tolist(),
            "velocities": atoms.get_velocities().tolist() if atoms.get_velocities() is not None else None,
        }

        t["future"] = run_aims_parsl_task(
            atoms_dict=atoms_dict,
            dft_root_dir=str(self.dft_dir),
            step=t["step"],
            traj_idx=traj_idx,
            control_in_path=str(CONTROL_IN),
            species_dir=str(SPECIES_DIR),
            aims_bin=AIMS_BIN,
            n_cores=32,
        )

    def handle_completed_dft(self, traj_idx: int, result: dict, pending: dict):
        """Processes a completed Parsl DFT calculation, fine-tunes models, and resumes."""
        t = self.trajectories[traj_idx]
        atoms = pending["atoms"]
        u_value = pending["u"]
        atom_std = pending["atom_std"]
        step = pending["step"]

        if not result["success"]:
            print("\n" + "!" * 80)
            print(f">>> [CALCULATION FAILED] FHI-aims failed on Trajectory {traj_idx} ({t['name']}) at step {step}!")
            print(f">>> Preserving failed calculation in '{self.failed_dir.name}/' for investigation.")
            print("!" * 80 + "\n")

            calc_dir = Path(result["calc_dir"])
            failed_target = self.failed_dir / f"step_{step:06d}_traj_{traj_idx}_{t['name']}"
            if calc_dir.exists():
                if failed_target.exists():
                    shutil.rmtree(failed_target)
                shutil.move(str(calc_dir), str(failed_target))

            log_entry = (
                f"\n{'=' * 80}\n"
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Trajectory: {traj_idx} ({t['name']}) | MD Step: {step}\n"
                f"Uncertainty: {u_value:.4f} eV/A | Threshold: {self.u_thresh:.4f} eV/A\n"
                f"Preserved Folder: {failed_target}\n"
                f"Error Message: {result.get('error', 'Unknown')}\n"
                f"--- aims.out Error Snippet ---\n"
                f"{result.get('log_snippet', '')}\n"
                f"{'=' * 80}\n"
            )
            with open(self.failed_dir / "failed_calculations.log", "a") as f:
                f.write(log_entry)

            # Rollback trajectory to last safe checkpoint and RE-THERMALIZE velocities
            safe_data = t["last_safe_checkpoint"]
            t["atoms"].set_positions(safe_data["positions"].copy())
            seed = int(time.time() * 1000) % 100000 + traj_idx
            MaxwellBoltzmannDistribution(t["atoms"], temperature_K=TEMPERATURE_K, rng=np.random.RandomState(seed))
            Stationary(t["atoms"])
            t["atoms"].calc = self.calc
            t["step"] = safe_data["step"]
            t["consecutive_failures"] += 1
            t["cooldown"] = TRIGGER_COOLDOWN_STEPS

            if t["consecutive_failures"] >= 2:
                print(f"[AL-Manager] Multiple DFT failures ({t['consecutive_failures']}) on {t['name']}. Applying subtle thermal displacement (0.02 A) to escape basin...")
                noise = np.random.normal(0.0, 0.02, size=t["atoms"].get_positions().shape)
                t["atoms"].set_positions(t["atoms"].get_positions() + noise)

            print(f"[AL-Manager] Rolled back {t['name']} to safe step {safe_data['step']} with re-thermalized velocities.")
            return

        # Success case
        self.cycle += 1
        e_dft = result["e_dft"]
        f_dft = np.array(result["f_dft"])
        calc_dir = Path(result["calc_dir"])
        print(f"\n[DFT] Reference calculation finished successfully: {calc_dir}")
        print(f"[DFT] Energy: {e_dft:.4f} eV | Max Force: {np.max(np.abs(f_dft)):.4f} eV/A")

        t["consecutive_failures"] = 0
        t["last_safe_checkpoint"] = {
            "positions": atoms.get_positions().copy(),
            "velocities": atoms.get_velocities().copy(),
            "step": step,
        }
        t["cooldown"] = TRIGGER_COOLDOWN_STEPS

        labeled_atoms = atoms.copy()
        labeled_atoms.info["REF_energy"] = e_dft
        labeled_atoms.arrays["REF_forces"] = f_dft
        if atom_std is not None:
            labeled_atoms.arrays["force_std"] = atom_std
            max_idx = int(np.argmax(atom_std))
            labeled_atoms.info["trigger_atom_idx"] = max_idx
            labeled_atoms.info["trigger_element"] = atoms.get_chemical_symbols()[max_idx]
            labeled_atoms.info["trigger_z"] = float(atoms.get_positions()[max_idx, 2])

        h_charges = result.get("hirshfeld_charges", None)
        free_vols = result.get("free_volumes", None)
        if h_charges is not None:
            labeled_atoms.arrays["hirshfeld_charges"] = np.array(h_charges, dtype=np.float64)
            q_arr = np.array(h_charges)
            symbols = atoms.get_chemical_symbols()
            q_str = []
            for elem in sorted(set(symbols)):
                elem_q = [q_arr[j] for j, s in enumerate(symbols) if s == elem]
                q_str.append(f"{elem}: {np.mean(elem_q):+.3f} e (min: {np.min(elem_q):+.3f}, max: {np.max(elem_q):+.3f})")
            print(f"[DFT HIRSHFELD] Mean Charges -> " + " | ".join(q_str))
        if free_vols is not None:
            labeled_atoms.arrays["free_volumes"] = np.array(free_vols, dtype=np.float64)

        total_points_added = self.train_points_added + self.val_points_added + 1
        if self.val_points_added < VALID_RATIO * total_points_added:
            write(str(self.val_dataset_file), labeled_atoms, format="extxyz", append=True)
            self.val_points_added += 1
            print(f"[AL-Manager] Added point to validation set (Total Val: {self.val_points_added}). Skipping retrain per aims-PAX quota.")
            self.save_checkpoint()
            return

        write(str(self.dataset_file), labeled_atoms, format="extxyz", append=True)
        self.train_points_added += 1
        self.dataset_size += 1

        # Online fine-tuning: retrain 3 PaiNN committee members for 1 epoch with EMA smoothing
        print(f"[AL-Manager] Retraining 3 PaiNN models for 1 epoch (Cycle {self.cycle}) with EMA weight smoothing (decay=0.99)...")
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
            emas=self.emas,
            ema_decay=0.99,
        )
        self.current_model_dirs = [Path(p) for p in updated_paths]

        # Bind updated calculator to all trajectory instances
        for tr in self.trajectories:
            tr["atoms"].calc = self.calc

        # Validation error evaluation
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
            f_mae = np.mean(np.abs(f_pred - f_ref)) * 1000.0
            f_maes.append(f_mae)

            if e_ref is not None:
                e_pred = self.calc.results["energy"]
                e_mae = abs(e_pred - e_ref) / len(atoms) * 1000.0
                e_maes.append(e_mae)

        mean_f_mae = float(np.mean(f_maes)) if f_maes else None
        mean_e_mae = float(np.mean(e_maes)) if e_maes else None
        return mean_f_mae, mean_e_mae

    def run(self):
        print("\n==========================================================")
        print(f"STARTING PAINN PARSL ASYNCHRONOUS AL RUN ON 5 A GEOMETRIES")
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
            # ------------------------------------------------------------------
            # Phase 1: Poll Parsl background futures for completed DFT calculations
            # ------------------------------------------------------------------
            for i, t in enumerate(self.trajectories):
                if t["status"] == "waiting" and t["future"] is not None:
                    if t["future"].done():
                        res = t["future"].result()
                        pending = t["pending_point"]
                        t["status"] = "running"
                        t["future"] = None
                        t["pending_point"] = None
                        self.handle_completed_dft(traj_idx=i, result=res, pending=pending)

            if self.accuracy_reached:
                break

            # ------------------------------------------------------------------
            # Phase 2: Check if all non-finished trajectories are currently waiting
            # ------------------------------------------------------------------
            unfinished = [t for t in self.trajectories if t["step"] < MAX_MD_STEPS]
            if len(unfinished) > 0 and all(t["status"] == "waiting" for t in unfinished):
                time.sleep(1.0)
                continue

            # ------------------------------------------------------------------
            # Phase 3: Propagate running trajectories on GPU
            # ------------------------------------------------------------------
            for i, t in enumerate(self.trajectories):
                if t["status"] != "running" or t["step"] >= MAX_MD_STEPS:
                    continue

                dyn = dyn_drivers[i]
                atoms = t["atoms"]

                dyn.run(SKIP_STEP_MLFF)
                t["step"] += SKIP_STEP_MLFF
                self.step = max(tr["step"] for tr in self.trajectories)

                pos = atoms.get_positions()
                vel = atoms.get_velocities()

                # Guard 1: NaN/Inf detection
                if not np.isfinite(pos).all() or not np.isfinite(vel).all():
                    print(f"\n[AL-Manager] WARNING: NaN/Inf coordinates detected in {t['name']} at step {t['step']}! Recovering from last safe checkpoint...")
                    safe_data = t["last_safe_checkpoint"]
                    atoms.set_positions(safe_data["positions"].copy())
                    seed = int(time.time() * 1000) % 100000 + i
                    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random.RandomState(seed))
                    Stationary(atoms)
                    t["step"] = safe_data["step"]
                    t["cooldown"] = TRIGGER_COOLDOWN_STEPS
                    continue

                # Guard 2: Thermal runaway detection (T > 1000 K)
                temp = atoms.get_temperature()
                if temp > 1000.0:
                    print(f"\n[AL-Manager] WARNING: Thermal runaway detected in {t['name']} (T = {temp:.1f} K > 1000 K)! Re-thermalizing velocities to {TEMPERATURE_K} K...")
                    seed = int(time.time() * 1000) % 100000 + i
                    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random.RandomState(seed))
                    Stationary(atoms)

                res = atoms.calc.results
                u = float(res.get("max_atomic_sd", 0.0))

                # Update rolling adaptive threshold
                self.u_thresh = self.threshold_mgr.update(u, self.dataset_size)

                # Update safe checkpoint if trajectory is healthy and stable
                if temp < 550.0 and u < self.u_thresh * 0.85:
                    t["last_safe_checkpoint"] = {
                        "positions": pos.copy(),
                        "velocities": vel.copy(),
                        "step": t["step"],
                    }

                # Decrement cooldown counter if active
                if t["cooldown"] > 0:
                    t["cooldown"] -= SKIP_STEP_MLFF

                # Periodic LAMMPS and stress dumps
                if t["step"] % DUMP_EVERY_D1 == 0:
                    mol_ids = identify_molecules(atoms)
                    write_lammps_dump(t["d1_path"], step=t["step"], atoms=atoms, mol_ids=mol_ids, stress=None, append=True)
                if t["step"] % DUMP_EVERY_D2 == 0:
                    mol_ids = identify_molecules(atoms)
                    stress = res.get("atomic_stress", None)
                    write_lammps_dump(t["d2_path"], step=t["step"], atoms=atoms, mol_ids=mol_ids, stress=stress, append=True)

                if t["step"] % 100 == 0:
                    thresh_str = f"{self.u_thresh:.4f} eV/A" if np.isfinite(self.u_thresh) else "inf"
                    cd_str = f" | CD: {t['cooldown']}" if t["cooldown"] > 0 else ""
                    print(f"Step {t['step']:05d} | {t['name']} | T: {temp:.1f} K | U: {u:.4f} eV/A | Thresh: {thresh_str}{cd_str}")

                # Uncertainty trigger check
                if u > self.u_thresh:
                    if t["cooldown"] > 0:
                        if t["step"] % 100 == 0:
                            print(f"[COOLDOWN] Trajectory {i} ({t['name']}) U={u:.4f} > {self.u_thresh:.4f}, but in cooldown ({t['cooldown']} steps left).")
                    else:
                        self.dispatch_parsl_trigger(traj_idx=i, atoms=atoms, u_value=u, res=res)

            # Checkpoint every 500 steps
            if self.step % 500 == 0:
                self.save_checkpoint()

        # Clean shutdown of Parsl
        try:
            parsl.dfk().cleanup()
        except Exception:
            pass

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
    manager = PainnActiveLearningManager5AParsl()
    manager.run()
