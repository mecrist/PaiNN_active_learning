#!/usr/bin/env python
"""
Recovery script to ingest converged FHI-aims calculations from failed_calculations/
into the active learning training and validation datasets.
"""

from pathlib import Path
import numpy as np
from ase.io import read, write
import json

from dft_interface import parse_hirshfeld_data


def recover():
    base_dir = Path("/home/maria.crist/dft_mlip/sep_pax/al_5A_painn")
    failed_dir = base_dir / "failed_calculations"
    train_file = base_dir / "al_dataset.xyz"
    val_file = base_dir / "val.xyz"
    ckpt_file = base_dir / "al_checkpoint.json"

    recovered = []
    for step_dir in sorted(failed_dir.glob("step_*")):
        out_file = step_dir / "aims.out"
        if not out_file.exists():
            continue

        try:
            atoms = read(out_file, format="aims-output")
            e_dft = float(atoms.get_potential_energy())
            f_dft = np.array(atoms.get_forces(), dtype=np.float64)
            q, v = parse_hirshfeld_data(out_file)

            labeled = atoms.copy()
            labeled.info["REF_energy"] = e_dft
            labeled.arrays["REF_forces"] = f_dft
            if q is not None:
                labeled.arrays["hirshfeld_charges"] = np.array(q, dtype=np.float64)
            if v is not None:
                labeled.arrays["free_volumes"] = np.array(v, dtype=np.float64)

            recovered.append((step_dir.name, labeled, e_dft, np.max(np.abs(f_dft))))
        except Exception:
            continue

    print(f"Found {len(recovered)} fully converged calculations in {failed_dir.name}/:")
    for name, at, e, mf in recovered:
        print(f"  - {name}: Atoms={len(at)}, E={e:.3f} eV, MaxForce={mf:.3f} eV/A")

    if not recovered:
        print("No recoverable structures found.")
        return

    # Check against existing train/val sets to prevent duplicates
    existing_energies = set()
    if train_file.exists():
        for at in read(str(train_file), index=":"):
            e = at.info.get("REF_energy", at.info.get("energy"))
            if e is not None:
                existing_energies.add(round(float(e), 4))
    if val_file.exists():
        for at in read(str(val_file), index=":"):
            e = at.info.get("REF_energy", at.info.get("energy"))
            if e is not None:
                existing_energies.add(round(float(e), 4))

    to_add = [item for item in recovered if round(item[2], 4) not in existing_energies]
    print(f"\nStructures to add after deduplication: {len(to_add)}")

    if not to_add:
        print("All recovered structures already present in datasets.")
        return

    # Split: 1 to validation, rest to training
    val_target = to_add[0]
    train_targets = to_add[1:]

    write(str(val_file), val_target[1], format="extxyz", append=True)
    print(f"Added {val_target[0]} to {val_file.name}")

    for name, at, _, _ in train_targets:
        write(str(train_file), at, format="extxyz", append=True)
        print(f"Added {name} to {train_file.name}")

    # Update checkpoint counts if checkpoint exists
    if ckpt_file.exists():
        with open(ckpt_file, "r") as f:
            st = json.load(f)
        st["train_points_added"] = st.get("train_points_added", 0) + len(train_targets)
        st["val_points_added"] = st.get("val_points_added", 0) + 1
        st["cycle"] = st.get("cycle", 0) + len(to_add)
        with open(ckpt_file, "w") as f:
            json.dump(st, f, indent=2)
        print(f"\nUpdated {ckpt_file.name}: Cycle now at {st['cycle']}, Train added: {st['train_points_added']}, Val added: {st['val_points_added']}")

    print("\nRecovery complete! You can now run retrain_engine.py or resume active learning.")


if __name__ == "__main__":
    recover()
