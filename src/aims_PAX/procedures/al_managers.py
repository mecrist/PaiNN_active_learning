#!/usr/bin/env python
"""
Active Learning Manager Layer.
Decouples distinct concerns matching aims-PAX architecture:
  - ALRunningManager: Trajectory propagation, stability checks, and uncertainty monitoring.
  - ALDataManager: Dataset appending, validation quota, and atomic properties logging.
  - ALTrainingManager: Fine-tuning orchestrator, validation evaluation, and target accuracy check.
  - ALDFTReferenceManagerPARSL: Background Parsl execution, error logging, and rollback handling.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import shutil
import time
import numpy as np
from ase import Atoms
from ase.io import read, write

from aims_PAX.tools.model_tools.train_painn import retrain_painn_ensemble
from aims_PAX.tools.utilities.stability_guards import is_structure_finite, thermal_runaway_guard, rollback_trajectory
from aims_PAX.tools.utilities.lammps_dumps import identify_molecules, write_lammps_dump
from aims_PAX.tools.utilities.parsl_utils import run_aims_parsl_task
from aims_PAX.tools.utilities.data_handling import save_labeled_point, load_dataset


class ALRunningManager:
    """Coordinates GPU-based propagation of molecular dynamics trajectories."""

    def __init__(self, config, state_manager, ensemble, dyn_drivers, trajectories):
        self.config = config
        self.state_manager = state_manager
        self.ensemble = ensemble
        self.dyn_drivers = dyn_drivers
        self.trajectories = trajectories

    def step_trajectories(self, on_trigger_callback):
        """Propagates each running trajectory for skip_step_mlff steps on the GPU."""
        for i, t in enumerate(self.trajectories):
            if t["status"] != "running" or t["step"] >= self.config.max_md_steps:
                continue

            dyn = self.dyn_drivers[i]
            atoms = t["atoms"]

            dyn.run(self.config.skip_step_mlff)
            t["step"] += self.config.skip_step_mlff
            self.state_manager.step = max(tr["step"] for tr in self.trajectories)

            # Stability Guard 1: Detect NaN/Inf coordinates
            if not is_structure_finite(atoms):
                print(f"\n[ALRunningManager] WARNING: NaN/Inf coordinates in {t['name']} at step {t['step']}! Recovering from checkpoint...")
                safe_data = t["last_safe_checkpoint"]
                seed = int(time.time() * 1000) % 100000 + i
                rollback_trajectory(atoms, safe_data, target_temp_K=self.config.temperature_K, seed=seed)
                t["step"] = safe_data["step"]
                t["cooldown"] = self.config.trigger_cooldown_steps
                continue

            # Stability Guard 2: Detect thermal runaway (T > 1000 K)
            thermal_runaway_guard(
                atoms,
                traj_name=t["name"],
                max_temp_K=1000.0,
                target_temp_K=self.config.temperature_K,
                seed=int(time.time() * 1000) % 100000 + i,
            )

            res = atoms.calc.results
            u = float(res.get("max_atomic_sd", 0.0))
            temp = atoms.get_temperature()

            # Update adaptive threshold using in-memory dataset size counter
            current_thresh = self.state_manager.threshold_mgr.update(u, self.state_manager.dataset_size)

            # Update safe checkpoint if trajectory is healthy and stable
            if temp < 550.0 and u < current_thresh * 0.85:
                t["last_safe_checkpoint"] = {
                    "positions": atoms.get_positions().copy(),
                    "velocities": atoms.get_velocities().copy() if atoms.get_velocities() is not None else None,
                    "step": t["step"],
                }

            # Decrement cooldown counter if active
            if t["cooldown"] > 0:
                t["cooldown"] -= self.config.skip_step_mlff

            # Periodic trajectory dumps
            if t["step"] % self.config.dump_every_d1 == 0:
                mol_ids = identify_molecules(atoms)
                write_lammps_dump(t["d1_path"], step=t["step"], atoms=atoms, mol_ids=mol_ids, stress=None, append=True)

            if t["step"] % self.config.dump_every_d2 == 0:
                mol_ids = identify_molecules(atoms)
                stress = res.get("atomic_stress", None)
                write_lammps_dump(t["d2_path"], step=t["step"], atoms=atoms, mol_ids=mol_ids, stress=stress, append=True)

            if t["step"] % 100 == 0:
                cd_str = f" | CD: {t['cooldown']}" if t["cooldown"] > 0 else ""
                print(f"Step {t['step']:05d} | {t['name']} | T: {temp:.1f} K | U: {u:.4f} eV/A | Thresh: {current_thresh:.4f} eV/A{cd_str}")

            # Trigger condition
            if u > current_thresh:
                if t["cooldown"] > 0:
                    if t["step"] % 100 == 0:
                        print(f"[COOLDOWN] Trajectory {i} ({t['name']}) U={u:.4f} > {current_thresh:.4f}, but in cooldown ({t['cooldown']} steps left).")
                else:
                    on_trigger_callback(traj_idx=i, atoms=atoms, u_value=u, res=res)


class ALDataManager:
    """Manages appending labeled structures to training and validation sets."""

    def __init__(self, config, state_manager):
        self.config = config
        self.state_manager = state_manager
        self.dataset_file = config.base_dir / "al_dataset.xyz"
        self.val_dataset_file = config.base_dir / "val.xyz"

    def handle_received_point(self, atoms: Atoms, result: Dict[str, Any], pending: Dict[str, Any]) -> str:
        """Labels and saves the received DFT structure to dataset or validation pool."""
        e_dft = result["e_dft"]
        f_dft = np.array(result["f_dft"])
        atom_std = pending.get("atom_std", None)

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

        # Validation set quota (aims-PAX standard)
        total_points = self.state_manager.train_points_added + self.state_manager.val_points_added + 1
        if self.state_manager.val_points_added < self.config.valid_ratio * total_points:
            save_labeled_point(self.val_dataset_file, labeled_atoms, append=True)
            self.state_manager.val_points_added += 1
            print(f"[ALDataManager] Added point to validation set (Total Val: {self.state_manager.val_points_added}).")
            return "validation"
        else:
            save_labeled_point(self.dataset_file, labeled_atoms, append=True)
            self.state_manager.train_points_added += 1
            self.state_manager.dataset_size += 1
            print(f"[ALDataManager] Added point to training set (Total Train added: {self.state_manager.train_points_added}, Total dataset: {self.state_manager.dataset_size}).")
            return "training"


class ALTrainingManager:
    """Orchestrates PaiNN ensemble fine-tuning and validation metrics evaluation."""

    def __init__(self, config, state_manager, ensemble, data_manager):
        self.config = config
        self.state_manager = state_manager
        self.ensemble = ensemble
        self.data_manager = data_manager
        self.accuracy_reached = False

    def train_epoch(self):
        """Retrains PaiNN models for 1 epoch on the expanded dataset."""
        print(f"[ALTrainingManager] Retraining PaiNN models for 1 epoch (Cycle {self.state_manager.cycle}) with EMA smoothing...")
        updated_paths = retrain_painn_ensemble(
            calculator=self.ensemble.calc,
            dataset_path=self.data_manager.dataset_file,
            ref_energies=self.config.ref_energies,
            checkpoint_dir=self.config.base_dir / "checkpoints",
            cycle=self.state_manager.cycle,
            n_epochs=1,
            batch_size=4,
            lr=1e-4,
            device=self.config.device,
            val_dataset_path=self.data_manager.val_dataset_file,
            optimizers=self.ensemble.optimizers,
            emas=self.ensemble.emas,
            ema_decay=0.99,
        )
        self.ensemble.current_model_dirs = [Path(p) for p in updated_paths]

        # Evaluate validation metrics
        val_f_mae, val_e_mae = self.evaluate_validation()
        if val_f_mae is not None:
            print(f"[ALTrainingManager] Cycle {self.state_manager.cycle} Validation -> Force MAE: {val_f_mae:.2f} meV/A | Energy MAE: {val_e_mae:.2f} meV/atom")
            if self.config.desired_acc_force_mae is not None and val_f_mae <= self.config.desired_acc_force_mae:
                print("\n" + "*" * 80)
                print(f">>> [TARGET REACHED] Validation Force MAE ({val_f_mae:.2f} meV/A) <= desired_acc ({self.config.desired_acc_force_mae:.2f} meV/A)!")
                print(f">>> Active Learning converged successfully on accuracy criterion.")
                print("*" * 80 + "\n")
                self.accuracy_reached = True

    def evaluate_validation(self) -> Tuple[Optional[float], Optional[float]]:
        val_path = self.data_manager.val_dataset_file
        if not val_path.exists():
            return None, None

        val_frames = load_dataset(val_path)
        f_maes = []
        e_maes = []
        for atoms in val_frames:
            if "REF_forces" in atoms.arrays:
                f_ref = atoms.arrays["REF_forces"]
            elif "forces" in atoms.arrays:
                f_ref = atoms.arrays["forces"]
            else:
                continue

            e_ref = atoms.info.get("REF_energy", atoms.info.get("energy", None))

            atoms.calc = self.ensemble.calc
            self.ensemble.calc.calculate(atoms)
            f_pred = self.ensemble.calc.results["forces"]
            f_mae = np.mean(np.abs(f_pred - f_ref)) * 1000.0
            f_maes.append(f_mae)

            if e_ref is not None:
                e_pred = self.ensemble.calc.results["energy"]
                e_mae = abs(e_pred - e_ref) / len(atoms) * 1000.0
                e_maes.append(e_mae)

        mean_f = float(np.mean(f_maes)) if f_maes else None
        mean_e = float(np.mean(e_maes)) if e_maes else None
        return mean_f, mean_e


class ALDFTReferenceManagerPARSL:
    """Dispatches and manages asynchronous FHI-aims DFT calculations via Parsl."""

    def __init__(self, config, state_manager, ensemble):
        self.config = config
        self.state_manager = state_manager
        self.ensemble = ensemble
        self.failed_dir = config.base_dir / "failed_calculations"

    def dispatch_trigger(self, traj_idx: int, t: Dict[str, Any], atoms: Atoms, u_value: float, res: Dict[str, Any]):
        """Submits an FHI-aims single-point calculation to the background Parsl pool."""
        atom_std = res.get("std_per_atom", None)
        trigger_info = ""
        if atom_std is not None:
            max_idx = int(np.argmax(atom_std))
            elem = atoms.get_chemical_symbols()[max_idx]
            z_pos = atoms.get_positions()[max_idx, 2]
            trigger_info = f" | Trigger Atom: #{max_idx} ({elem}) at z={z_pos:.2f} Å"

        print("\n" + "=" * 80)
        print(f">>> [PARSL ASYNC TRIGGER] Trajectory {traj_idx} ({t['name']}) at step {t['step']}")
        print(f">>> Uncertainty: {u_value:.4f} eV/A > Threshold: {self.state_manager.threshold_mgr.threshold:.4f} eV/A{trigger_info}")
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
            dft_root_dir=str(self.config.base_dir / "dft_calculations"),
            step=t["step"],
            traj_idx=traj_idx,
            control_in_path=str(self.config.control_in),
            species_dir=str(self.config.species_dir),
            aims_bin=self.config.aims_bin,
            n_cores=self.config.n_dft_cores,
        )

    def handle_failed_dft(self, traj_idx: int, t: Dict[str, Any], result: Dict[str, Any], pending: Dict[str, Any]):
        """Preserves failed directory, records diagnostic log, and rolls back trajectory."""
        step = pending["step"]
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
            f"Uncertainty: {pending['u']:.4f} eV/A | Threshold: {self.state_manager.threshold_mgr.threshold:.4f} eV/A\n"
            f"Preserved Folder: {failed_target}\n"
            f"Error Message: {result.get('error', 'Unknown')}\n"
            f"--- aims.out Error Snippet ---\n"
            f"{result.get('log_snippet', '')}\n"
            f"{'=' * 80}\n"
        )
        with open(self.failed_dir / "failed_calculations.log", "a") as f:
            f.write(log_entry)

        # Rollback trajectory and re-thermalize
        t["consecutive_failures"] += 1
        noise = 0.02 if t["consecutive_failures"] >= 2 else 0.0
        if noise > 0.0:
            print(f"[ALReferenceManager] Multiple DFT failures ({t['consecutive_failures']}) on {t['name']}. Applying subtle thermal displacement (0.02 A)...")

        seed = int(time.time() * 1000) % 100000 + traj_idx
        rollback_trajectory(
            t["atoms"],
            t["last_safe_checkpoint"],
            target_temp_K=self.config.temperature_K,
            seed=seed,
            noise_std=noise,
        )
        t["atoms"].calc = self.ensemble.calc
        t["step"] = t["last_safe_checkpoint"]["step"]
        t["cooldown"] = self.config.trigger_cooldown_steps
        print(f"[ALReferenceManager] Rolled back {t['name']} to step {t['step']} with fresh velocities.")
