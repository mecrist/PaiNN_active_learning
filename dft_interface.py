#!/usr/bin/env python
"""
FHI-aims DFT Single Point Runner.
Prepares calculation inputs, executes local 32-core MPI FHI-aims, parses outputs,
and permanently preserves all calculation directories without deletion.
"""

import os
from pathlib import Path
import subprocess
import numpy as np
from ase.io import read, write


def prepare_control_in(base_control_path, species_dir, chemical_symbols, output_path):
    """
    Combines the base control.in with the required FHI-aims species defaults.
    """
    base_text = Path(base_control_path).read_text()
    species_needed = sorted(set(chemical_symbols))
    species_dir_path = Path(species_dir)

    species_blocks = []
    for spec in species_needed:
        # Match species files like 01_H_default, 08_O_default, 14_Si_default
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


def run_aims_single_point(
    atoms,
    dft_root_dir,
    step,
    traj_idx,
    control_in_path="/home/maria.crist/dft_mlip/sep_pax/control.in",
    species_dir="/home/maria.crist/fhi-aims.260331/species_defaults/defaults_2020/light",
    aims_bin="/home/maria.crist/fhi-aims.260331/bin/aims.x",
    n_cores=32,
):
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
        raise RuntimeError(
            f"FHI-aims calculation failed in {calc_dir} with exit code {res.returncode}. "
            f"Check {out_file} for error details."
        )

    # 5. Parse output using ASE aims parser
    atoms_dft = read(out_file, format="aims-output")
    e_dft = float(atoms_dft.get_potential_energy())
    f_dft = np.array(atoms_dft.get_forces(), dtype=np.float64)

    max_f = float(np.max(np.abs(f_dft)))
    print(f"[DFT SUCCESS] Energy: {e_dft:.4f} eV | Max Force: {max_f:.4f} eV/A")
    print(f"[DFT SAVED] Calculation folder permanently preserved at: {calc_dir}\n")

    return e_dft, f_dft, calc_dir
