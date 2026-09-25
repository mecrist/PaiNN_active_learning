#!/usr/bin/env python
"""
PaiNN Committee Incremental Retraining Engine.
Fine-tunes PaiNN committee members on newly added DFT points for 1 epoch,
safeguarded with safe loss wrappers and finite-gradient guards against NaN divergence.
"""

from pathlib import Path
from typing import List, Dict, Optional, Union
import types
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch_ema import ExponentialMovingAverage
from ase import Atoms
from ase.io import read

from nff.data import Dataset, collate_dicts
from nff.data.graphs import get_neighbor_list
from nff.train import Trainer, hooks, loss, metrics


def make_safe_loss_fn(loss_coef: Dict[str, float]):
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
    Skips the optimizer step on NaN or Inf gradients.
    """
    for group in self.optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return True
    return False


def build_nff_dataset(atoms_list: List[Atoms], ref_energies: Dict[int, float], cutoff: float = 6.0) -> Dataset:
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
        if "REF_energy" in atoms.info:
            e_dft = atoms.info["REF_energy"]
        elif "energy" in atoms.info:
            e_dft = atoms.info["energy"]
        else:
            e_dft = atoms.get_potential_energy()

        e0_sum = sum(ref_energies[int(at)] for at in z)
        e_ref = e_dft - e0_sum

        # Force -> gradient: energy_grad = -Force
        if "REF_forces" in atoms.arrays:
            forces = atoms.arrays["REF_forces"]
        elif "forces" in atoms.arrays:
            forces = atoms.arrays["forces"]
        else:
            forces = atoms.get_forces()

        energy_grad = -np.array(forces, dtype=np.float64)

        # Periodic-aware Graph neighbor list
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        if any(pbc):
            from ase.neighborlist import neighbor_list
            i, j, S = neighbor_list("ijS", atoms, cutoff=cutoff)
            if len(i) > 0:
                nbrs = torch.tensor(np.stack([i, j], axis=1), dtype=torch.long)
                cart_offsets = np.dot(S, cell)
                offs = torch.tensor(cart_offsets, dtype=torch.float32)
            else:
                nbrs = torch.empty((0, 2), dtype=torch.long)
                offs = torch.empty((0, 3), dtype=torch.float32)
        else:
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


class EMAHook(hooks.Hook):
    """
    Exponential Moving Average hook for NFF Trainer.
    Updates smoothed parameters after each batch step, and copies the averaged
    weights into the model at the end of training.
    """
    def __init__(self, ema: ExponentialMovingAverage):
        self.ema = ema

    def on_batch_end(self, trainer, train_batch, result, loss):
        self.ema.update()

    def on_train_ends(self, trainer):
        self.ema.copy_to(trainer._model.parameters())


def retrain_painn_ensemble(
    calculator,
    dataset_path: Union[str, Path],
    ref_energies: Dict[int, float],
    checkpoint_dir: Union[str, Path],
    cycle: int = 1,
    n_epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    val_dataset_path: Optional[Union[str, Path]] = None,
    optimizers: Optional[Dict[int, Adam]] = None,
    emas: Optional[Dict[int, ExponentialMovingAverage]] = None,
    ema_decay: float = 0.99,
) -> List[str]:
    """
    Retrains all PaiNN models in the ensemble for n_epochs (default 1 epoch)
    and saves checkpoints. Supports persistent optimizers and dedicated validation set.
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

    if val_dataset_path and Path(val_dataset_path).exists():
        val_atoms = read(str(val_dataset_path), index=":", format="extxyz")
        val_dataset = build_nff_dataset(val_atoms, ref_energies, cutoff=calculator.cutoff)
        validation_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_dicts,
            shuffle=False,
        )
    else:
        validation_loader = train_loader

    cycle_ckpt_dir = Path(checkpoint_dir) / f"cycle_{cycle:04d}"
    cycle_ckpt_dir.mkdir(parents=True, exist_ok=True)

    updated_model_paths = []

    for idx, model in enumerate(calculator.models):
        m_dir = cycle_ckpt_dir / f"model_{idx}"
        m_dir.mkdir(parents=True, exist_ok=True)

        model.train()
        if optimizers is not None and idx in optimizers:
            optimizer = optimizers[idx]
        else:
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = Adam(trainable_params, lr=lr)
            if optimizers is not None:
                optimizers[idx] = optimizer

        loss_fn = make_safe_loss_fn(loss_coef={"energy_grad": 0.95, "energy": 0.05})

        train_hooks = [
            hooks.MaxEpochHook(n_epochs),
        ]
        if emas is not None:
            if idx in emas:
                ema = emas[idx]
            else:
                ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
                emas[idx] = ema
            train_hooks.append(EMAHook(ema))

        trainer = Trainer(
            model_path=str(m_dir),
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            validation_loader=validation_loader,
            checkpoint_interval=1,
            hooks=train_hooks,
        )
        trainer.grad_is_nan = types.MethodType(grad_is_finite, trainer)

        print(f"  --> Fine-tuning Model {idx}...")
        trainer.train(device=torch.device(device), n_epochs=n_epochs)

        best_path = m_dir / "best_model"
        if not best_path.exists():
            torch.save(model, str(best_path))

        updated_model_paths.append(str(m_dir))

    calculator.load_models(updated_model_paths)
    print(f"[RETRAIN COMPLETE] All {len(calculator.models)} PaiNN models updated and reloaded.\n")
    return updated_model_paths


def run_final_convergence(
    calculator,
    dataset_path: Union[str, Path],
    ref_energies: Dict[int, float],
    output_dir: Union[str, Path],
    val_dataset_path: Optional[Union[str, Path]] = None,
    n_epochs: int = 200,
    patience: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    """
    Final convergence routine: trains the ensemble on the completed active learning
    dataset with Early Stopping (default max 200 epochs, patience 30) matching aims-PAX.
    """
    print(f"\n[FINAL CONVERGENCE] Training {len(calculator.models)} PaiNN models (max {n_epochs} epochs, patience {patience})...")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms_list = read(str(dataset_path), index=":", format="extxyz")
    nff_dataset = build_nff_dataset(atoms_list, ref_energies, cutoff=calculator.cutoff)

    train_loader = DataLoader(
        nff_dataset,
        batch_size=batch_size,
        collate_fn=collate_dicts,
        sampler=RandomSampler(nff_dataset),
    )

    if val_dataset_path and Path(val_dataset_path).exists():
        val_atoms = read(str(val_dataset_path), index=":", format="extxyz")
        val_dataset = build_nff_dataset(val_atoms, ref_energies, cutoff=calculator.cutoff)
        validation_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_dicts,
            shuffle=False,
        )
    else:
        validation_loader = train_loader

    for idx, model in enumerate(calculator.models):
        m_dir = out_dir / f"model_{idx}"
        m_dir.mkdir(parents=True, exist_ok=True)
        model.train()

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(trainable_params, lr=lr)
        loss_fn = make_safe_loss_fn(loss_coef={"energy_grad": 0.95, "energy": 0.05})
        ema = ExponentialMovingAverage(model.parameters(), decay=0.99)

        train_hooks = [
            hooks.MaxEpochHook(n_epochs),
            EMAHook(ema),
            hooks.EarlyStoppingHook(patience=patience),
        ]

        trainer = Trainer(
            model_path=str(m_dir),
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            validation_loader=validation_loader,
            checkpoint_interval=1,
            hooks=train_hooks,
        )
        trainer.grad_is_nan = types.MethodType(grad_is_finite, trainer)
        print(f"  --> Converging Model {idx} (max {n_epochs} epochs, early stopping patience {patience})...")
        trainer.train(device=torch.device(device), n_epochs=n_epochs)

        best_path = m_dir / "best_model"
        if not best_path.exists():
            torch.save(model, str(best_path))

    print("[FINAL CONVERGENCE] Complete. Converged models saved to:", out_dir)
