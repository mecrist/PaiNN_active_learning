#!/usr/bin/env python
"""
ASE-compatible Calculator for a 6-member PaiNN Committee Ensemble.
Handles forward passes, conversion between kcal/mol and eV, reference energy E0 addition,
atomic force disagreement standard deviation, and uncertainty metrics.
"""

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from nff.data.graphs import get_neighbor_list
from nff.train.builders.model import load_model

KCAL_TO_EV = 0.04336414


class PainnEnsembleCalculator(Calculator):
    """
    Calculator wrapping a committee of PaiNN models.
    
    Provides:
      - 'energy': Committee mean energy shifted to raw DFT absolute scale (+ sum(E0)).
      - 'forces': Committee mean force vector per atom (eV/Angstrom).
      - 'stress': Global stress tensor in Voigt 6-vector format (eV/Angstrom^3).
      - 'max_atomic_sd': Maximum atomic force standard deviation across committee (meV/Angstrom).
      - 'std_per_atom': Atomic force standard deviation for each atom (meV/Angstrom).
      - 'atomic_stress': Diagonal per-atom virial components (xu*Fx, yu*Fy, zu*Fz).
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model_dirs,
        ref_energies,
        cutoff=6.0,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.device = torch.device(device)
        self.ref_energies = {int(k): float(v) for k, v in ref_energies.items()}
        self.model_dirs = model_dirs
        self.models = []
        self.load_models(model_dirs)

    def load_models(self, model_dirs):
        """Loads or reloads models from disk into memory/GPU."""
        self.models = []
        print(f"[PainnEnsembleCalculator] Loading {len(model_dirs)} models to {self.device}...")
        for p in model_dirs:
            m = load_model(str(p))
            m.to(self.device)
            m.eval()
            self.models.append(m)
        print(f"[PainnEnsembleCalculator] Successfully loaded {len(self.models)} PaiNN models.")

    def _build_batch(self, atoms):
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=self.device)
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        nbrs = get_neighbor_list(pos.cpu(), cutoff=self.cutoff, undirected=False).to(self.device)
        offs = torch.zeros((nbrs.shape[0], 3), device=self.device)
        nxyz = torch.cat([z.view(-1, 1).float(), pos], dim=1)

        batch = {
            "nxyz": nxyz,
            "nbr_list": nbrs,
            "offsets": offs,
            "num_atoms": torch.tensor([len(atoms)], device=self.device),
        }
        return batch

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = ["energy", "forces"]
        super().calculate(atoms, properties, system_changes)

        batch = self._build_batch(self.atoms)

        energies_kcal = []
        forces_kcal_ang = []

        for m in self.models:
            out = m(batch)
            e = out["energy"].detach().cpu().item()
            # Gradient of energy with respect to positions -> Force = -grad
            g = out["energy_grad"].detach().cpu().numpy()
            energies_kcal.append(e)
            forces_kcal_ang.append(-g)

        energies_ev = np.array(energies_kcal) * KCAL_TO_EV
        forces_ev_ang = np.array(forces_kcal_ang) * KCAL_TO_EV  # shape: (M, N, 3)

        M = len(self.models)
        # Shift energy by atomic reference baseline (E0)
        e0_total = sum(self.ref_energies[int(at)] for at in self.atoms.get_atomic_numbers())
        mean_energy = float(np.mean(energies_ev)) + e0_total
        mean_forces = np.mean(forces_ev_ang, axis=0)  # shape: (N, 3)

        # Force disagreement: standard deviation across committee in meV/Angstrom
        diff = forces_ev_ang - mean_forces[None, :, :]  # (M, N, 3)
        diff_sq = np.sum(diff**2, axis=-1)  # (M, N)
        # Sample standard deviation across models
        if M > 1:
            var = np.sum(diff_sq, axis=0) / (M - 1)
        else:
            var = np.zeros(len(self.atoms))
        std_per_atom = np.sqrt(var) * 1000.0  # meV/Angstrom
        max_atomic_sd = float(np.max(std_per_atom))

        # Atomic stress / virial proxy (r_i (x) F_i)
        # c_stress: [c_stress[1], c_stress[2], c_stress[3]] = - [x*Fx, y*Fy, z*Fz]
        pos = self.atoms.get_positions()
        atomic_virial = - pos * mean_forces  # (N, 3) in eV

        # Voigt stress tensor (xx, yy, zz, yz, xz, xy)
        vol = abs(self.atoms.get_volume()) if self.atoms.cell and self.atoms.cell.volume > 0 else 1.0
        total_virial_diag = np.sum(atomic_virial, axis=0) / vol  # eV/Angstrom^3
        stress_voigt = np.array([total_virial_diag[0], total_virial_diag[1], total_virial_diag[2], 0.0, 0.0, 0.0])

        self.results = {
            "energy": mean_energy,
            "forces": mean_forces,
            "stress": stress_voigt,
            "max_atomic_sd": max_atomic_sd,
            "std_per_atom": std_per_atom,
            "atomic_stress": atomic_virial,
            "forces_comm": forces_ev_ang,
        }
