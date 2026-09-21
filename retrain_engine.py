#!/usr/bin/env python
"""
PaiNN Committee Incremental Retraining Engine.
Fine-tunes the 6 PaiNN committee members on newly added DFT points for 1 epoch,
safeguarded with safe loss wrappers and finite-gradient guards against NaN divergence.
"""

import os
from pathlib import Path
import types
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from ase.io import read

from nff.data import Dataset, collate_dicts
from nff.data.graphs import get_neighbor_list
from nff.train import Trainer, hooks, loss, metrics
from nff.train.builders.model import load_model


def make_safe_loss_fn(loss_coef):
    """
    Substitutes a zero, ungraphed loss whenever raw loss is non-finite,
    effectively skipping the batch instead of propagating Inf/NaN gradients.
    """
    base_loss_fn = loss.build_mse_loss(loss_coef=loss_coef)

    def safe_loss_fn(batch, results):
        raw_loss = base_loss_fn(batch, results)
        if not torch.isfinite(raw_loss):
            return torch.zeros((), device=raw_loss.device, requires_grad=True)
        return raw_loss

    return safe_loss_fn


def grad_is_finite(self):
    """
    Replaces Trainer.grad_is_nan: also skips the optimizer step on Inf gradients,
    not just NaN ones.
    """
    for group in self.optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return True
    return False


def build_nff_dataset(atoms_list, ref_energies, cutoff=6.0):
    """
    Converts ASE Atoms list into an NFF Dataset with graph neighbor lists,
    shifted by reference atomic energies (E0) and scaled to kcal/mol.
    """
    nxyz_list = []
    energy_list = []
    energy_grad_list = []
    nbr_list = []
    offsets_list = []

    for atoms in atoms_list:
        z = atoms.get_atomic_numbers()
        pos = atoms.get_positions()
        nxyz = np.concatenate([z.reshape(-1, 1), pos], axis=1)

        # Reference energy shift: E_ref = E_dft - sum(E0)
        e_dft = atoms.info.get("REF_energy", atoms.get_potential_energy())
        e0_sum = sum(ref_energies[int(at)] for at in z)
        e_ref = e_dft - e0_sum

        # Force -> gradient: energy_grad = -Force
        forces = atoms.arrays.get("REF_forces", atoms.get_forces())
        energy_grad = -np.array(forces, dtype=np.float64)

        # Graph neighbor list
        pos_tensor = torch.tensor(pos, dtype=torch.float32)
        nbrs = get_neighbor_list(pos_tensor, cutoff=cutoff, undirected=False)
        offs = torch.zeros(nbrs.shape[0], 3).float()

        nxyz_list.append(nxyz)
        energy_list.append(float(e_ref))
        energy_grad_list.append(energy_grad)
        nbr_list.append(nbrs)
        offsets_list.append(offs)

    props = {
        "nxyz": nxyz_list,
        "energy": energy_list,
        "energy_grad": energy_grad_list,
        "nbr_list": nbr_list,
        "offsets": offsets_list,
    }

    # Initialize in eV, convert to kcal/mol for PaiNN
    ds = Dataset(props, units="eV", device="cpu")
    ds.to_units("kcal/mol")
    return ds


def retrain_ensemble(
    calculator,
    dataset_path,
    ref_energies,
    checkpoint_dir,
    cycle=1,
    n_epochs=1,
    batch_size=4,
    lr=1e-4,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
):
    """
    Retrains all 6 models in the ensemble for n_epochs (default 1 epoch)
    and saves checkpoints.
    """
    print(f"\n[RETRAIN] Starting online fine-tuning for cycle {cycle} ({n_epochs} epoch)...")
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Training dataset file not found: {dataset_file}")

    atoms_list = read(str(dataset_file), index=":", format="extxyz")
    print(f"[RETRAIN] Training pool size: {len(atoms_list)} structures.")

    nff_dataset = build_nff_dataset(atoms_list, ref_energies, cutoff=calculator.cutoff)

    train_loader = DataLoader(
        nff_dataset,
        batch_size=batch_size,
        collate_fn=collate_dicts,
        sampler=RandomSampler(nff_dataset),
    )

    cycle_ckpt_dir = Path(checkpoint_dir) / f"cycle_{cycle:04d}"
    cycle_ckpt_dir.mkdir(parents=True, exist_ok=True)

    updated_model_paths = []

    for idx, model in enumerate(calculator.models):
        m_dir = cycle_ckpt_dir / f"model_{idx}"
        m_dir.mkdir(parents=True, exist_ok=True)

        model.train()
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(trainable_params, lr=lr)
        loss_fn = make_safe_loss_fn(loss_coef={"energy_grad": 0.95, "energy": 0.05})

        train_hooks = [
            hooks.MaxEpochHook(n_epochs),
        ]

        trainer = Trainer(
            model_path=str(m_dir),
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            validation_loader=train_loader,  # lightweight placeholder
            checkpoint_interval=1,
            hooks=train_hooks,
        )
        trainer.grad_is_nan = types.MethodType(grad_is_finite, trainer)

        print(f"  --> Fine-tuning Model {idx}...")
        trainer.train(device=torch.device(device), n_epochs=n_epochs)

        best_path = m_dir / "best_model"
        if not best_path.exists():
            # If trainer did not save best_model yet, save model directly
            torch.save(model, str(best_path))

        updated_model_paths.append(str(m_dir))

    # Reload all 6 updated models into the calculator
    calculator.load_models(updated_model_paths)
    print(f"[RETRAIN COMPLETE] All 6 PaiNN models updated and reloaded.\n")
    return updated_model_paths
