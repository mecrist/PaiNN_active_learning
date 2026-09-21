#!/usr/bin/env python
"""
Trajectory Dump Utilities supporting LAMMPS Custom Dump Format.
Provides:
  - Unified dump: id type mol xu yu zu
  - Stress dump:  id type mol xu yu zu c_stress[1] c_stress[2] c_stress[3]
  - Water and slab molecule identification
"""

from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

TYPE_MAP = {
    "H": 1,
    "O": 2,
    "Si": 3,
}


def identify_molecules(atoms):
    """
    Identifies water molecules in a silica/water system.
    Assigns:
      - Slab atoms: mol_id = 1
      - Water molecules: mol_id = 2, 3, ... (each water molecule gets its own ID)
    """
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    natoms = len(atoms)

    mol_ids = np.ones(natoms, dtype=int)  # default slab = 1

    o_indices = [i for i, s in enumerate(symbols) if s == "O"]
    h_indices = [i for i, s in enumerate(symbols) if s == "H"]

    if not o_indices or not h_indices:
        return mol_ids

    dists = cdist(pos[o_indices], pos[h_indices])
    closest_o = np.argmin(dists, axis=0)
    min_dists = np.min(dists, axis=0)

    # Group Hydrogens to Oxygen if within 1.3 Angstroms (covalent O-H distance)
    water_groups = {}
    for h_local_idx, o_local_idx in enumerate(closest_o):
        if min_dists[h_local_idx] < 1.3:
            o_idx = o_indices[o_local_idx]
            h_idx = h_indices[h_local_idx]
            water_groups.setdefault(o_idx, []).append(h_idx)

    cur_mol = 2
    for o_idx, h_list in water_groups.items():
        if len(h_list) == 2:  # valid H2O
            mol_ids[o_idx] = cur_mol
            for h_idx in h_list:
                mol_ids[h_idx] = cur_mol
            cur_mol += 1

    return mol_ids


def write_lammps_dump(
    filename,
    step,
    atoms,
    mol_ids=None,
    stress=None,
    append=True,
):
    """
    Writes a snapshot in LAMMPS custom dump format.
    
    If stress is None:
      Columns: id type mol xu yu zu
    If stress is provided:
      Columns: id type mol xu yu zu c_stress[1] c_stress[2] c_stress[3]
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and filepath.exists() else "w"

    if mol_ids is None:
        mol_ids = identify_molecules(atoms)

    pos = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    natoms = len(atoms)

    # Determine box bounds
    if atoms.cell is not None and atoms.cell.volume > 0:
        lengths = atoms.cell.lengths()
        xlo, xhi = 0.0, float(lengths[0])
        ylo, yhi = 0.0, float(lengths[1])
        zlo, zhi = 0.0, float(lengths[2])
    else:
        # Bounding box around positions with padding
        xlo, xhi = float(np.min(pos[:, 0]) - 2.0), float(np.max(pos[:, 0]) + 2.0)
        ylo, yhi = float(np.min(pos[:, 1]) - 2.0), float(np.max(pos[:, 1]) + 2.0)
        zlo, zhi = float(np.min(pos[:, 2]) - 2.0), float(np.max(pos[:, 2]) + 2.0)

    include_stress = stress is not None

    with open(filepath, mode) as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{step}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{natoms}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"{xlo:.8f} {xhi:.8f}\n")
        f.write(f"{ylo:.8f} {yhi:.8f}\n")
        f.write(f"{zlo:.8f} {zhi:.8f}\n")

        if include_stress:
            f.write("ITEM: ATOMS id type mol xu yu zu c_stress[1] c_stress[2] c_stress[3]\n")
            for i in range(natoms):
                atom_id = i + 1
                atom_type = TYPE_MAP.get(symbols[i], 1)
                mol = int(mol_ids[i])
                xu, yu, zu = pos[i]
                s1, s2, s3 = stress[i]
                f.write(
                    f"{atom_id} {atom_type} {mol} {xu:.6f} {yu:.6f} {zu:.6f} "
                    f"{s1:.6e} {s2:.6e} {s3:.6e}\n"
                )
        else:
            f.write("ITEM: ATOMS id type mol xu yu zu\n")
            for i in range(natoms):
                atom_id = i + 1
                atom_type = TYPE_MAP.get(symbols[i], 1)
                mol = int(mol_ids[i])
                xu, yu, zu = pos[i]
                f.write(f"{atom_id} {atom_type} {mol} {xu:.6f} {yu:.6f} {zu:.6f}\n")
