#!/usr/bin/env python
"""
FHI-aims DFT Single Point Runner.
Prepares calculation inputs, executes local MPI FHI-aims, parses outputs,
extracts Hirshfeld analysis, and permanently preserves calculation directories.
"""

import os
from pathlib import Path
import subprocess
import re
from typing import Tuple, Optional, Union
import numpy as np
from ase import Atoms
from ase.io import read, write


class AimsCalculationError(RuntimeError):
    """Raised when an FHI-aims single-point calculation or output parsing fails."""
    def __init__(self, message: str, calc_dir: Union[str, Path], log_snippet: str = ""):
        super().__init__(message)
        self.calc_dir = Path(calc_dir)
        self.log_snippet = log_snippet


def prepare_control_in(base_control_path: Union[str, Path], species_dir: Union[str, Path], chemical_symbols: list, output_path: Union[str, Path]):
    """
    Combines the base control.in with the required FHI-aims species defaults.
    """
    base_text = Path(base_control_path).read_text()
    species_needed = sorted(set(chemical_symbols))
    species_dir_path = Path(species_dir)

    species_blocks = []
    for spec in species_needed:
        matched_files = list(species_dir_path.glob(f"*_{spec}_*")) + list(
            species_dir_path.glob(f"*_{spec}")
        )
        if not matched_files:
            raise FileNotFoundError(
                f"Species default file for element '{spec}' not found in {species_dir}"
            )
        species_blocks.append(matched_files[0].read_text())

    full_control = (
        base_text
        + "\n# ========================================================\n"
        + "# SPECIES DEFAULTS (AUTOMATICALLY CONCATENATED)\n"
        + "# ========================================================\n"
        + "\n".join(species_blocks)
        + "\n"
    )
    Path(output_path).write_text(full_control)


def parse_hirshfeld_data(out_file: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parses Hirshfeld atomic charges and free atom volumes from aims.out.
    """
    charges = []
    free_vols = []
    active = False
    try:
        with open(out_file, "r") as f:
            for line in f:
                if not active:
                    if "Performing Hirshfeld analysis of fragment charges and moments" in line:
                        active = True
                else:
                    if "Hirshfeld charge" in line:
                        m = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                        if m:
                            charges.append(float(m[0]))
                    elif "Free atom volume" in line:
                        m = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                        if m:
                            free_vols.append(float(m[0]))
        charges_arr = np.array(charges, dtype=np.float64) if charges else None
        free_vols_arr = np.array(free_vols, dtype=np.float64) if free_vols else None
        return charges_arr, free_vols_arr
    except Exception:
        return None, None


def run_aims_single_point(
    atoms: Atoms,
    dft_root_dir: Union[str, Path],
    step: int,
    traj_idx: int,
    control_in_path: str = "/home/maria.crist/dft_mlip/sep_pax/al_5A_painn/control.in",
    species_dir: str = "/home/maria.crist/fhi-aims.260331/species_defaults/defaults_2020/light",
    aims_bin: str = "/home/maria.crist/fhi-aims.260331/bin/aims.x",
    n_cores: int = 32,
) -> Tuple[float, np.ndarray, Path, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Executes an FHI-aims single-point DFT calculation on the local node.
    Permanently preserves the calculation directory and all outputs.
    """
    calc_dir = Path(dft_root_dir) / f"step_{step:06d}_traj_{traj_idx}"
    calc_dir.mkdir(parents=True, exist_ok=True)

    geom_file = calc_dir / "geometry.in"
    control_file = calc_dir / "control.in"
    out_file = calc_dir / "aims.out"

    # 1. Write geometry.in
    write(geom_file, atoms, format="aims")

    # 2. Build and write control.in
    prepare_control_in(
        base_control_path=control_in_path,
        species_dir=species_dir,
        chemical_symbols=atoms.get_chemical_symbols(),
        output_path=control_file,
    )

    # 3. Environment configuration for Intel MPI & FHI-aims
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["AIMS_SPECIES_DEFAULTS"] = str(species_dir)

    # 4. Execute mpirun
    mpi_cmd = f"ulimit -s unlimited && mpirun -np {n_cores} {aims_bin} > aims.out 2>&1"
    print(f"\n[DFT] Running FHI-aims in: {calc_dir}")
    print(f"[DFT] Command: {mpi_cmd}")

    res = subprocess.run(
        mpi_cmd,
        shell=True,
        cwd=str(calc_dir),
        env=env,
        capture_output=False,
    )

    if res.returncode != 0 or not out_file.exists():
        snippet = ""
        if out_file.exists():
            lines = out_file.read_text().splitlines()
            snippet = "\n".join(lines[-35:])
        raise AimsCalculationError(
            f"FHI-aims calculation failed in {calc_dir} with exit code {res.returncode}.",
            calc_dir=calc_dir,
            log_snippet=snippet,
        )

    # 5. Parse output using ASE aims parser
    try:
        atoms_dft = read(out_file, format="aims-output")
        e_dft = float(atoms_dft.get_potential_energy())
        f_dft = np.array(atoms_dft.get_forces(), dtype=np.float64)
    except Exception as e:
        snippet = ""
        if out_file.exists():
            lines = out_file.read_text().splitlines()
            snippet = "\n".join(lines[-35:])
        raise AimsCalculationError(
            f"Failed to parse FHI-aims output in {calc_dir}: {str(e)}",
            calc_dir=calc_dir,
            log_snippet=snippet,
        )

    hirshfeld_charges, free_vols = parse_hirshfeld_data(out_file)
    max_f = float(np.max(np.abs(f_dft)))
    print(f"[DFT SUCCESS] Energy: {e_dft:.4f} eV | Max Force: {max_f:.4f} eV/A")
    if hirshfeld_charges is not None:
        print(f"[DFT HIRSHFELD] Parsed {len(hirshfeld_charges)} atomic charges (Net charge: {np.sum(hirshfeld_charges):+.4f} e)")
    print(f"[DFT SAVED] Calculation folder permanently preserved at: {calc_dir}\n")

    return e_dft, f_dft, calc_dir, hirshfeld_charges, free_vols
