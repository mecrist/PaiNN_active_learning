#!/usr/bin/env python
"""
MACE Committee Ensemble ASE Calculator
Evaluates an ensemble of MACE models and computes committee mean energy,
forces, and per-atom force standard deviation (uncertainty in meV/Angstrom).
"""

from typing import List, Union
from pathlib import Path
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from mace.calculators import MACECalculator


class MaceEnsembleCalculator(Calculator):
    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(
        self,
        model_paths: List[Union[str, Path]],
        device: str = "cuda",
        default_dtype: str = "float64",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_paths = [str(p) for p in model_paths]
        self.device = device
        self.default_dtype = default_dtype

        print(f"[MaceEnsembleCalculator] Loading {len(self.model_paths)} MACE models to {device}...")
        self.calcs = []
        for i, path in enumerate(self.model_paths):
            calc = MACECalculator(
                model_paths=path,
                device=device,
                default_dtype=default_dtype,
            )
            self.calcs.append(calc)
        print(f"[MaceEnsembleCalculator] Successfully loaded {len(self.calcs)} MACE models.")

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        energies = []
        forces_list = []

        for calc in self.calcs:
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            energies.append(atoms_copy.get_potential_energy())
            forces_list.append(atoms_copy.get_forces())

        energies = np.array(energies)
        forces_list = np.array(forces_list)  # (M, N, 3)

        mean_energy = float(np.mean(energies))
        mean_forces = np.mean(forces_list, axis=0)

        n_models = len(self.calcs)
        if n_models > 1:
            diff = forces_list - mean_forces[None, :, :]
            diff_norm_sq = np.sum(diff**2, axis=-1)
            var = np.sum(diff_norm_sq, axis=0) / (n_models - 1)
            atomic_stds = np.sqrt(var) * 1000.0  # meV/A
            max_atomic_sd = float(np.max(atomic_stds))
        else:
            atomic_stds = np.zeros(len(atoms))
            max_atomic_sd = 0.0

        self.results["energy"] = mean_energy
        self.results["free_energy"] = mean_energy
        self.results["forces"] = mean_forces
        self.results["atomic_stds"] = atomic_stds
        self.results["std_per_atom"] = atomic_stds
        self.results["max_atomic_sd"] = max_atomic_sd
        self.results["energies_committee"] = energies
