#!/usr/bin/env python3
"""
================================================================================
   DATASET PREPARATION FOR ACTIVE LEARNING EPOCHS BENCHMARK
================================================================================
Extracts and unifies:
1. Base training set (490 frames)
2. Base validation set (140 frames)
3. Combined Active Learning pool (71 frames: 45 PaiNN train + 5 PaiNN val + 19 MACE train + 2 MACE val)
4. Standardized 145-frame 5 Å interface evaluation test set
================================================================================
"""

import os
from pathlib import Path
from ase.io import read, write

DATA_DIR = Path("/home/maria.crist/dft_mlip/sep_pax/benchmark_al_epochs/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(">>> [1/4] Loading base initial datasets...")
base_train_path = Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/train.xyz")
base_val_path = Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/val.xyz")

base_train = read(str(base_train_path), index=":", format="extxyz")
base_val = read(str(base_val_path), index=":", format="extxyz")
print(f"    Loaded {len(base_train)} base train frames, {len(base_val)} base val frames.")

print(">>> [2/4] Extracting active learning queried frames...")
# PaiNN AL frames
p_train_path = Path("/home/maria.crist/dft_mlip/sep_pax/al_5A_painn/al_dataset.xyz")
p_val_path = Path("/home/maria.crist/dft_mlip/sep_pax/al_5A_painn/val.xyz")
p_train_all = read(str(p_train_path), index=":", format="extxyz")
p_val_all = read(str(p_val_path), index=":", format="extxyz")

painn_al_train = p_train_all[490:]
painn_al_val = p_val_all[140:]
print(f"    Extracted PaiNN AL: {len(painn_al_train)} train + {len(painn_al_val)} val = {len(painn_al_train) + len(painn_al_val)} frames.")

# MACE AL frames
m_train_path = Path("/home/maria.crist/dft_mlip/sep_pax/al_5A_mace/data/final/training/train_set_silica_water_mace_run-123.xyz")
m_val_path = Path("/home/maria.crist/dft_mlip/sep_pax/al_5A_mace/data/final/validation/valid_set_silica_water_mace_run-123.xyz")
m_train_all = read(str(m_train_path), index=":", format="extxyz")
m_val_all = read(str(m_val_path), index=":", format="extxyz")

mace_al_train = m_train_all[490:]
mace_al_val = m_val_all[140:]
print(f"    Extracted MACE AL:  {len(mace_al_train)} train + {len(mace_al_val)} val = {len(mace_al_train) + len(mace_al_val)} frames.")

# Tag each frame with provenance in info dict
for i, at in enumerate(painn_al_train):
    at.info["al_source"] = "painn_train"
    at.info["al_cycle"] = i + 1
for i, at in enumerate(painn_al_val):
    at.info["al_source"] = "painn_val"
for i, at in enumerate(mace_al_train):
    at.info["al_source"] = "mace_train"
    at.info["al_cycle"] = i + 1
for i, at in enumerate(mace_al_val):
    at.info["al_source"] = "mace_val"

# Interleave or combine into unified stream
combined_al_pool = painn_al_train + mace_al_train + painn_al_val + mace_al_val
print(f"    Combined unified AL pool size: {len(combined_al_pool)} frames.")

print(">>> [3/4] Standardizing property keys (REF_energy, REF_forces)...")
def standardize_atoms(atoms_list):
    for at in atoms_list:
        if "REF_energy" not in at.info:
            if "energy" in at.info:
                at.info["REF_energy"] = at.info["energy"]
        if "REF_forces" not in at.arrays:
            if "forces" in at.arrays:
                at.arrays["REF_forces"] = at.arrays["forces"]
    return atoms_list

base_train = standardize_atoms(base_train)
base_val = standardize_atoms(base_val)
combined_al_pool = standardize_atoms(combined_al_pool)
test_145 = standardize_atoms(p_val_all)

print(">>> [4/4] Writing datasets to disk...")
out_base_train = DATA_DIR / "base_train.extxyz"
out_base_val = DATA_DIR / "base_val.extxyz"
out_al_pool = DATA_DIR / "combined_al_pool.extxyz"
out_test_145 = DATA_DIR / "test_145_interface.extxyz"

write(str(out_base_train), base_train, format="extxyz")
write(str(out_base_val), base_val, format="extxyz")
write(str(out_al_pool), combined_al_pool, format="extxyz")
write(str(out_test_145), test_145, format="extxyz")

print("================================================================================")
print("DATASET PREPARATION COMPLETE:")
print(f"  - Base Train:      {out_base_train} ({len(base_train)} frames)")
print(f"  - Base Val:        {out_base_val} ({len(base_val)} frames)")
print(f"  - Combined AL:     {out_al_pool} ({len(combined_al_pool)} frames)")
print(f"  - Test Interface:  {out_test_145} ({len(test_145)} frames)")
print("================================================================================")
