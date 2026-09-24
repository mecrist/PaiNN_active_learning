#!/usr/bin/env python
"""
LAMMPS Trajectory Dump Writer with Dynamic Molecule Identification.
Labels water molecules vs surface/bulk silica atoms and exports custom dumps for OVITO.
"""

from pathlib import Path
from typing import Optional, Union, List
import numpy as np
from ase import Atoms


def identify_molecules(atoms: Atoms, max_oh_bond: float = 1.3) -> np.ndarray:
    """
    Identifies water molecules in the system based on covalent bond connectivity.
    Assigns unique positive molecule IDs to water molecules, and 0 to solid slab atoms.
    """
    n_atoms = len(atoms)
    mol_ids = np.zeros(n_atoms, dtype=int)
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    h_indices = [i for i, s in enumerate(symbols) if s == "H"]
    o_indices = [i for i, s in enumerate(symbols) if s == "O"]

    mol_counter = 1
    o_bonded_h = {o: [] for o in o_indices}

    for h in h_indices:
        h_pos = pos[h]
        best_o = None
        min_dist = max_oh_bond
        for o in o_indices:
            d_vec = pos[o] - h_pos
            if any(pbc):
                d_vec -= np.round(d_vec @ np.linalg.inv(cell)) @ cell
            dist = np.linalg.norm(d_vec)
            if dist < min_dist:
                min_dist = dist
                best_o = o
        if best_o is not None:
            o_bonded_h[best_o].append(h)

    for o, hydrogens in o_bonded_h.items():
        if len(hydrogens) == 2:  # Intact water molecule
            mol_ids[o] = mol_counter
            for h in hydrogens:
                mol_ids[h] = mol_counter
            mol_counter += 1

    return mol_ids


def write_lammps_dump(
    filename: Union[str, Path],
    step: int,
    atoms: Atoms,
    mol_ids: Optional[np.ndarray] = None,
    stress: Optional[np.ndarray] = None,
    append: bool = True,
):
    """
    Writes or appends an atomic configuration in custom LAMMPS trajectory format.
    Includes atom ID, type, molecule ID, positions, forces, velocities, and optional stress.
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    symbols = atoms.get_chemical_symbols()
    type_map = {"H": 1, "O": 2, "Si": 3}
    types = [type_map.get(s, 4) for s in symbols]

    pos = atoms.get_positions()
    forces = atoms.calc.results.get("forces", np.zeros_like(pos)) if atoms.calc else np.zeros_like(pos)
    vel = atoms.get_velocities()
    if vel is None:
        vel = np.zeros_like(pos)

    if mol_ids is None:
        mol_ids = np.zeros(len(atoms), dtype=int)

    cell = atoms.get_cell()
    xlo, xhi = 0.0, cell[0, 0]
    ylo, yhi = 0.0, cell[1, 1]
    zlo, zhi = 0.0, cell[2, 2]

    has_stress = stress is not None and len(stress) == len(atoms)

    with open(filepath, mode) as f:
        f.write(f"ITEM: TIMESTEP\n{step}\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{len(atoms)}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"{xlo:.6f} {xhi:.6f}\n")
        f.write(f"{ylo:.6f} {yhi:.6f}\n")
        f.write(f"{zlo:.6f} {zhi:.6f}\n")

        if has_stress:
            f.write("ITEM: ATOMS id type mol x y z vx vy vz fx fy fz c_stress[1] c_stress[2] c_stress[3]\n")
            for i in range(len(atoms)):
                f.write(
                    f"{i+1} {types[i]} {mol_ids[i]} "
                    f"{pos[i, 0]:.6f} {pos[i, 1]:.6f} {pos[i, 2]:.6f} "
                    f"{vel[i, 0]:.6f} {vel[i, 1]:.6f} {vel[i, 2]:.6f} "
                    f"{forces[i, 0]:.6f} {forces[i, 1]:.6f} {forces[i, 2]:.6f} "
                    f"{stress[i, 0]:.6f} {stress[i, 1]:.6f} {stress[i, 2]:.6f}\n"
                )
        else:
            f.write("ITEM: ATOMS id type mol x y z vx vy vz fx fy fz\n")
            for i in range(len(atoms)):
                f.write(
                    f"{i+1} {types[i]} {mol_ids[i]} "
                    f"{pos[i, 0]:.6f} {pos[i, 1]:.6f} {pos[i, 2]:.6f} "
                    f"{vel[i, 0]:.6f} {vel[i, 1]:.6f} {vel[i, 2]:.6f} "
                    f"{forces[i, 0]:.6f} {forces[i, 1]:.6f} {forces[i, 2]:.6f}\n"
                )
