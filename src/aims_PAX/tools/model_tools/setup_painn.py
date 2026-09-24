#!/usr/bin/env python
"""
PaiNN Committee Model Setup and ASE Calculator.
Provides:
  - PainnEnsembleCalculator: ASE-compatible committee calculator.
  - setup_painn_ensemble: Constructs and loads committee models.
"""

from pathlib import Path
from typing import List, Dict, Union, Optional
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from nff.data.graphs import get_neighbor_list
from nff.train.builders.model import load_model

KCAL_TO_EV = 0.04336414


class PainnEnsembleCalculator(Calculator):
    """
    Calculator wrapping an ensemble committee of PaiNN models.

    Provides:
      - 'energy': Committee mean energy shifted to raw DFT absolute scale (+ sum(E0)).
      - 'forces': Committee mean force vector per atom (eV/Angstrom).
      - 'stress': Global stress tensor in Voigt 6-vector format (eV/Angstrom^3).
      - 'max_atomic_sd': Maximum atomic force standard deviation across committee following aims-PAX (eV/Angstrom).
      - 'std_per_atom': Atomic force standard deviation for each atom following aims-PAX (eV/Angstrom).
      - 'atomic_stress': Diagonal per-atom virial components (xu*Fx, yu*Fy, zu*Fz).
      - 'forces_comm': Committee member force predictions array of shape (M, N, 3) in eV/Angstrom.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model_dirs: List[Union[str, Path]],
        ref_energies: Dict[int, float],
        cutoff: float = 6.0,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.device = torch.device(device)
        self.ref_energies = {int(k): float(v) for k, v in ref_energies.items()}
        self.model_dirs = [Path(p) for p in model_dirs]
        self.models = []
        self.load_models(self.model_dirs)

    def load_models(self, model_dirs: List[Union[str, Path]]):
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
        pos = atoms.get_positions()
        z = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        if any(pbc):
            from ase.neighborlist import neighbor_list
            i, j, S = neighbor_list("ijS", atoms, cutoff=self.cutoff)
            if len(i) > 0:
                nbr_tensor = torch.tensor(np.stack([i, j], axis=1), dtype=torch.long, device=self.device)
                cart_offsets = np.dot(S, cell)
                offsets_tensor = torch.tensor(cart_offsets, dtype=torch.float32, device=self.device)
            else:
                nbr_tensor = torch.empty((0, 2), dtype=torch.long, device=self.device)
                offsets_tensor = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        else:
            pos_t = torch.tensor(pos, dtype=torch.float32)
            nbr_tensor = get_neighbor_list(pos_t, cutoff=self.cutoff, undirected=False).to(self.device)
            offsets_tensor = torch.zeros((nbr_tensor.shape[0], 3), device=self.device)

        pos_tensor = torch.tensor(pos, dtype=torch.float32, device=self.device)
        z_tensor = torch.tensor(z, dtype=torch.long, device=self.device)
        nxyz = torch.cat([z_tensor.view(-1, 1).float(), pos_tensor], dim=1)

        batch = {
            "nxyz": nxyz,
            "nbr_list": nbr_tensor,
            "offsets": offsets_tensor,
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
            g = out["energy_grad"].detach().cpu().numpy()
            energies_kcal.append(e)
            forces_kcal_ang.append(-g)

        energies_ev = np.array(energies_kcal) * KCAL_TO_EV
        forces_ev_ang = np.array(forces_kcal_ang) * KCAL_TO_EV  # (M, N, 3)

        e0_total = sum(self.ref_energies[int(at)] for at in self.atoms.get_atomic_numbers())
        mean_energy = float(np.mean(energies_ev)) + e0_total
        mean_forces = np.mean(forces_ev_ang, axis=0)  # (N, 3)

        # Force standard deviation across committee following aims-PAX exactly (in eV/A)
        pred_av = np.average(forces_ev_ang, axis=0, keepdims=True)
        diff_sq = (forces_ev_ang - pred_av) ** 2.0
        diff_sq_mean = np.mean(diff_sq, axis=(0, -1))
        std_per_atom = np.sqrt(diff_sq_mean)
        max_atomic_sd = float(np.max(std_per_atom))

        # Atomic stress proxy (uncalibrated diagnostic)
        pos_diag = self.atoms.get_positions(wrap=True) if any(self.atoms.get_pbc()) else self.atoms.get_positions()
        pos_rel = pos_diag - np.mean(pos_diag, axis=0)
        atomic_virial = - pos_rel * mean_forces

        vol = abs(self.atoms.get_volume()) if self.atoms.cell and self.atoms.cell.volume > 0 else 1.0
        total_virial_diag = np.sum(atomic_virial, axis=0) / vol
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


def setup_painn_ensemble(
    model_dirs: List[Union[str, Path]],
    ref_energies: Dict[int, float],
    cutoff: float = 6.0,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> PainnEnsembleCalculator:
    """Helper to instantiate and initialize a PainnEnsembleCalculator."""
    return PainnEnsembleCalculator(
        model_dirs=model_dirs,
        ref_energies=ref_energies,
        cutoff=cutoff,
        device=device,
    )
