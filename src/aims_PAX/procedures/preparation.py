#!/usr/bin/env python
"""
Active Learning Preparation Layer.
Provides base classes and decoupled managers for:
  - ALConfiguration: Centralized configuration container.
  - ALStateManager: Active learning cycle, trajectory states, and checkpoint persistence.
  - ALEnsemble: PaiNN committee models, paths, and persistent optimizers.
  - ALCalculatorMLFF: Instantiates PainnEnsembleCalculator and attaches it to trajectories.
  - ALMD: Sets up Langevin molecular dynamics drivers.
  - PrepareALProcedure: Base procedure orchestrating initialization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import json
import shutil
import time
import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE
import ase.units as units

from aims_PAX.tools.model_tools.setup_painn import PainnEnsembleCalculator, setup_painn_ensemble
from aims_PAX.tools.uncertainty import RollingAdaptiveThresholdManager
from aims_PAX.tools.utilities.data_handling import load_dataset, save_labeled_point


@dataclass
class ALConfiguration:
    """Centralized configuration container for active learning."""
    base_dir: Path
    control_in: Path
    species_dir: Path
    aims_bin: str
    initial_train_dataset: Path
    initial_val_dataset: Path
    ref_energies: Dict[int, float]
    initial_model_dirs: List[Path]
    geometry_files: List[Path] = field(default_factory=list)

    temperature_K: float = 300.0
    timestep_fs: float = 0.5
    friction: float = 0.002 / units.fs
    skip_step_mlff: int = 25
    max_md_steps: int = 10000
    trigger_cooldown_steps: int = 100

    c_x: float = 0.0
    c_x_ratio: Optional[float] = None
    min_threshold: float = 0.05
    initial_threshold: float = float("inf")
    freeze_dataset_size: Optional[int] = 540
    max_al_cycles: Optional[int] = 50
    desired_acc_force_mae: Optional[float] = None  # in meV/A

    n_dft_cores: int = 32
    valid_ratio: float = 0.1
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    dump_every_d1: int = 1000
    dump_every_d2: int = 5000

    def __post_init__(self):
        if self.c_x_ratio is not None:
            self.c_x = self.c_x_ratio


class ALStateManager:
    """Manages active learning cycles, steps, trajectory statuses, and checkpoint persistence."""

    def __init__(self, state_file: Path, config: ALConfiguration):
        self.state_file = Path(state_file)
        self.config = config
        self.cycle: int = 0
        self.step: int = 0
        self.train_points_added: int = 0
        self.val_points_added: int = 0
        self.dataset_size: int = 0
        self.threshold_mgr = RollingAdaptiveThresholdManager(
            c_x=config.c_x,
            min_threshold=config.min_threshold,
            rolling_window=400,
            freeze_size=config.freeze_dataset_size,
            initial_threshold=config.initial_threshold,
        )

    def save_checkpoint(self, trajectories: List[Dict[str, Any]], current_model_dirs: List[Path]):
        """Persists state and trajectories to al_checkpoint.json."""
        traj_states = []
        for t in trajectories:
            traj_states.append({
                "step": t["step"],
                "status": t["status"],
                "positions": t["atoms"].get_positions().tolist(),
                "velocities": t["atoms"].get_velocities().tolist() if t["atoms"].get_velocities() is not None else None,
            })

        state = {
            "cycle": self.cycle,
            "step": self.step,
            "train_points_added": self.train_points_added,
            "val_points_added": self.val_points_added,
            "u_thresh": self.threshold_mgr.threshold,
            "uncertainty_history": self.threshold_mgr.uncertainty_history,
            "threshold_frozen": self.threshold_mgr.frozen,
            "current_model_dirs": [str(d) for d in current_model_dirs],
            "trajectories": traj_states,
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[ALStateManager] State saved to {self.state_file.name} (Cycle: {self.cycle}, Step: {self.step})")

    def load_checkpoint(self, trajectories: List[Dict[str, Any]], calc: PainnEnsembleCalculator):
        """Restores state and trajectories from al_checkpoint.json."""
        if not self.state_file.exists():
            print("[ALStateManager] No previous checkpoint found. Initializing fresh run.")
            for i, t in enumerate(trajectories):
                MaxwellBoltzmannDistribution(t["atoms"], temperature_K=self.config.temperature_K, rng=np.random.RandomState(42 + i))
                Stationary(t["atoms"])
            return

        print(f"[ALStateManager] Resuming from checkpoint {self.state_file}...")
        with open(self.state_file, "r") as f:
            state = json.load(f)

        self.cycle = state.get("cycle", 0)
        self.step = state.get("step", 0)
        self.train_points_added = state.get("train_points_added", 0)
        self.val_points_added = state.get("val_points_added", 0)

        u_thresh = max(state.get("u_thresh", self.config.initial_threshold), self.config.min_threshold)
        self.threshold_mgr.threshold = u_thresh
        self.threshold_mgr.uncertainty_history = state.get("uncertainty_history", [])
        self.threshold_mgr.frozen = state.get("threshold_frozen", False)

        for i, t_state in enumerate(state.get("trajectories", [])):
            if i < len(trajectories):
                t = trajectories[i]
                pos = np.array(t_state["positions"]) if t_state.get("positions") is not None else None
                vel = np.array(t_state["velocities"]) if t_state.get("velocities") is not None else None

                if pos is None or not np.isfinite(pos).all() or (vel is not None and not np.isfinite(vel).all()):
                    print(f"[ALStateManager] WARNING: NaN in checkpoint for trajectory {i} ({t['name']})! Resetting to initial geometry.")
                    init_atoms = read(str(self.config.geometry_files[i]), format="aims")
                    t["atoms"].set_positions(init_atoms.get_positions())
                    MaxwellBoltzmannDistribution(t["atoms"], temperature_K=self.config.temperature_K, rng=np.random.RandomState(42 + i))
                    Stationary(t["atoms"])
                    t["step"] = 0
                else:
                    t["step"] = t_state.get("step", 0)
                    t["atoms"].set_positions(pos)
                    if vel is not None:
                        t["atoms"].set_velocities(vel)
                    else:
                        MaxwellBoltzmannDistribution(t["atoms"], temperature_K=self.config.temperature_K, rng=np.random.RandomState(42 + i))

                t["atoms"].calc = calc
                t["last_safe_checkpoint"] = {
                    "positions": t["atoms"].get_positions().copy(),
                    "velocities": t["atoms"].get_velocities().copy() if t["atoms"].get_velocities() is not None else None,
                    "step": t["step"],
                }
                t["status"] = "running"
                t["future"] = None
                t["pending_point"] = None


class ALEnsemble:
    """Manages the PaiNN committee models, model directories, and optimizers."""

    def __init__(self, config: ALConfiguration, state_file: Path):
        self.config = config
        self.state_file = state_file
        self.current_model_dirs = self._determine_model_dirs()
        self.calc = setup_painn_ensemble(
            model_dirs=self.current_model_dirs,
            ref_energies=self.config.ref_energies,
            cutoff=6.0,
            device=self.config.device,
        )
        self.optimizers: Dict[int, torch.optim.Adam] = {}
        self.emas: Dict[int, Any] = {}

    def _determine_model_dirs(self) -> List[Path]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    st = json.load(f)
                cand_dirs = [Path(d) for d in st.get("current_model_dirs", [])]
                if cand_dirs and all(d.exists() for d in cand_dirs):
                    print(f"[ALEnsemble] Resuming with {len(cand_dirs)} models from checkpoint ({cand_dirs[0].parent.name}).")
                    return cand_dirs
            except Exception:
                pass

        target_dirs = []
        for i, src_dir in enumerate(self.config.initial_model_dirs):
            dest_dir = self.config.base_dir / "checkpoints" / "cycle_0000" / f"model_{i}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            best_model_file = dest_dir / "best_model"
            if not best_model_file.exists():
                src_best = src_dir / "best_model"
                if src_best.exists():
                    shutil.copyfile(src_best, best_model_file)
                else:
                    raise FileNotFoundError(f"Initial model not found at {src_best}")
            target_dirs.append(dest_dir)
        return target_dirs


class PrepareALProcedure:
    """Base procedure: sets up directory layout, initializes datasets, pre-relaxes geometries, and builds MD drivers."""

    def __init__(self, config: ALConfiguration):
        self.config = config
        self.base_dir = config.base_dir
        self.ckpt_dir = self.base_dir / "checkpoints"
        self.dft_dir = self.base_dir / "dft_calculations"
        self.failed_dir = self.base_dir / "failed_calculations"
        self.traj_dir = self.base_dir / "trajectories"
        self.state_file = self.base_dir / "al_checkpoint.json"
        self.dataset_file = self.base_dir / "al_dataset.xyz"
        self.val_dataset_file = self.base_dir / "val.xyz"

        for d in [self.ckpt_dir, self.dft_dir, self.failed_dir, self.traj_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.ensemble = ALEnsemble(self.config, self.state_file)
        self.state_manager = ALStateManager(self.state_file, self.config)
        self._initialize_datasets()
        self.trajectories = self._setup_trajectories()
        self.dyn_drivers = self._setup_md_drivers()

        # Restore state from checkpoint if available
        self.state_manager.load_checkpoint(self.trajectories, self.ensemble.calc)

    def _initialize_datasets(self):
        if not self.dataset_file.exists():
            shutil.copyfile(self.config.initial_train_dataset, self.dataset_file)
        if not self.val_dataset_file.exists() and self.config.initial_val_dataset.exists():
            shutil.copyfile(self.config.initial_val_dataset, self.val_dataset_file)

        frames = load_dataset(self.dataset_file)
        self.dataset_size = len(frames)
        self.state_manager.dataset_size = self.dataset_size
        print(f"[PrepareALProcedure] Active learning dataset contains {self.dataset_size} frames.")

    def _setup_trajectories(self) -> List[Dict[str, Any]]:
        if not self.config.geometry_files:
            geom_dir = self.config.base_dir / "geometries"
            if geom_dir.exists():
                self.config.geometry_files = sorted(geom_dir.glob("*.in"))

        trajectories = []
        is_fresh_run = not self.state_file.exists()
        for i, g_file in enumerate(self.config.geometry_files):
            atoms = read(str(g_file), format="aims")
            atoms.calc = self.ensemble.calc

            # Pre-relax high force initial geometries with FIRE only if starting fresh
            if is_fresh_run:
                f_init = atoms.get_forces()
                max_f = np.max(np.abs(f_init))
                if max_f > 4.0:
                    print(f"[PrepareALProcedure] High initial force ({max_f:.2f} > 4.0 eV/A) in {g_file.name}. Pre-relaxing with FIRE...")
                    dyn_fire = FIRE(atoms, logfile=None)
                    dyn_fire.run(fmax=2.0, steps=25)
                    f_after = atoms.get_forces()
                    print(f"  --> Pre-relaxation finished: max force reduced to {np.max(np.abs(f_after)):.2f} eV/A.")

            trajectories.append({
                "name": g_file.stem,
                "atoms": atoms,
                "step": 0,
                "status": "running",
                "future": None,
                "pending_point": None,
                "consecutive_failures": 0,
                "cooldown": 0,
                "last_safe_checkpoint": {
                    "positions": atoms.get_positions().copy(),
                    "velocities": atoms.get_velocities().copy() if atoms.get_velocities() is not None else None,
                    "step": 0,
                },
                "d1_path": self.traj_dir / f"traj_{g_file.stem}.lammpstrj",
                "d2_path": self.traj_dir / f"traj_stress_{g_file.stem}.lammpstrj",
            })
        return trajectories

    def _setup_md_drivers(self) -> List[Langevin]:
        drivers = []
        for i, t in enumerate(self.trajectories):
            dyn = Langevin(
                t["atoms"],
                timestep=self.config.timestep_fs * units.fs,
                temperature_K=self.config.temperature_K,
                friction=self.config.friction,
                trajectory=None,
                logfile=None,
            )
            drivers.append(dyn)
        return drivers

    def check_al_done(self) -> bool:
        """Check if active learning is done based on max steps, max cycles, or checkpoint status."""
        if self.state_manager.step >= self.config.max_md_steps:
            return True
        if self.config.max_al_cycles is not None and self.state_manager.cycle >= self.config.max_al_cycles:
            return True
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                return state.get("al_done", False)
            except Exception:
                pass
        return False

