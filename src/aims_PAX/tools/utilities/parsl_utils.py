#!/usr/bin/env python
"""
Parsl Concurrency Setup and App Definitions.
Provides asynchronous execution of FHI-aims single-point DFT calculations in background threads.
"""

from pathlib import Path
from typing import Dict, Any
import parsl
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from parsl.app.app import python_app


def initialize_parsl(calc_dir: Path, max_workers: int = 1):
    """Initializes Parsl with a local ThreadPoolExecutor."""
    parsl_info_dir = Path(calc_dir) / "parsl_info"
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
) -> Dict[str, Any]:
    """
    Parsl application executing an FHI-aims single-point DFT calculation.
    Runs asynchronously in the Parsl background worker thread pool.
    """
    from pathlib import Path
    from ase import Atoms
    from aims_PAX.tools.utilities.dft_interface import run_aims_single_point, AimsCalculationError

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
