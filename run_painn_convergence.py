#!/usr/bin/env python3
"""
================================================================================
   PaiNN POST-ACTIVE LEARNING FINAL CONVERGENCE TRAINING SCRIPT
================================================================================
Trains the 3 PaiNN ensemble models on the accumulated post-AL dataset
(509 training structures, 142 validation structures) for up to 100 epochs
with ReduceLROnPlateau, matching the aims-PAX MACE convergence protocol.
================================================================================
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import torch
from ase.io import read

from painn_ensemble_calc import PainnEnsembleCalculator
from retrain_engine import run_final_convergence, build_nff_dataset
from torch.utils.data import DataLoader
from nff.data import collate_dicts

BASE_DIR = Path("/home/maria.crist/dft_mlip/sep_pax")
MACE_DATA_DIR = BASE_DIR / "al_5A_mace" / "data" / "final"
TRAIN_DATASET = MACE_DATA_DIR / "training" / "train_set_silica_water_mace_run-123.xyz"
VAL_DATASET = MACE_DATA_DIR / "validation" / "valid_set_silica_water_mace_run-123.xyz"

OUTPUT_DIR = BASE_DIR / "painn_converged_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference isolated atomic energies (PBE-TS, FHI-aims)
REF_ENERGIES = {
    1: -13.589886,
    8: -2041.365313,
    14: -7899.199478,
}

BASE_MODEL_DIRS = [
    "/home/maria.crist/dft_mlip/my_dataset/mine/model_0_ep500",
    "/home/maria.crist/dft_mlip/my_dataset/mine/model_1_ep500",
    "/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500",
]


def evaluate_models_on_val(calculator, val_dataset_path, device="cuda:0"):
    """Evaluates the ensemble on the validation dataset in meV/atom and meV/A."""
    val_atoms = read(str(val_dataset_path), index=":", format="extxyz")
    force_errors = []
    energy_errors = []

    print(f"\n[EVALUATION] Evaluating ensemble on {len(val_atoms)} validation structures...")
    for atoms in val_atoms:
        ref_e = atoms.info.get("REF_energy", atoms.info.get("energy"))
        ref_f = atoms.arrays.get("REF_forces", atoms.arrays.get("forces"))
        n_atoms = len(atoms)

        # Ensemble prediction
        calculator.calculate(atoms, properties=["energy", "forces"])
        pred_e = calculator.results["energy"]
        pred_f = calculator.results["forces"]

        energy_errors.append(abs(pred_e - ref_e) / n_atoms * 1000.0)  # meV/atom
        f_diff = np.abs(pred_f - ref_f).flatten()
        force_errors.extend(f_diff * 1000.0)  # meV/A

    f_mae = float(np.mean(force_errors))
    f_rmse = float(np.sqrt(np.mean(np.square(force_errors))))
    f_q95 = float(np.percentile(force_errors, 95))
    e_mae = float(np.mean(energy_errors))

    return {
        "force_mae_meV_A": round(f_mae, 2),
        "force_rmse_meV_A": round(f_rmse, 2),
        "force_q95_meV_A": round(f_q95, 2),
        "energy_mae_meV_atom": round(e_mae, 2),
    }


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"==========================================================")
    print(f"Starting PaiNN Ensemble Convergence Training on {device}")
    print(f"Training dataset: {TRAIN_DATASET}")
    print(f"Validation dataset: {VAL_DATASET}")
    print(f"==========================================================")

    # 1. Initialize calculator with baseline models
    calc = PainnEnsembleCalculator(
        model_paths=BASE_MODEL_DIRS,
        ref_energies=REF_ENERGIES,
        cutoff=6.0,
        device=device,
    )

    # Baseline evaluation before convergence
    baseline_metrics = evaluate_models_on_val(calc, VAL_DATASET, device=device)
    print("\n[BASELINE PRE-CONVERGENCE EVALUATION]")
    print(f"  Force MAE:   {baseline_metrics['force_mae_meV_A']} meV/A")
    print(f"  Force RMSE:  {baseline_metrics['force_rmse_meV_A']} meV/A")
    print(f"  Force Q95:   {baseline_metrics['force_q95_meV_A']} meV/A")
    print(f"  Energy MAE:  {baseline_metrics['energy_mae_meV_atom']} meV/atom")

    # 2. Run convergence training
    start_time = time.time()
    converged_model_paths = run_final_convergence(
        calculator=calc,
        dataset_path=TRAIN_DATASET,
        val_dataset_path=VAL_DATASET,
        ref_energies=REF_ENERGIES,
        output_dir=OUTPUT_DIR / "converged_models",
        n_epochs=100,
        batch_size=4,
        lr=1e-4,
        patience=15,
        device=device,
    )
    elapsed = time.time() - start_time
    print(f"\n[CONVERGENCE FINISHED] Elapsed time: {elapsed/60:.1f} min")

    # 3. Final evaluation after convergence
    converged_metrics = evaluate_models_on_val(calc, VAL_DATASET, device=device)
    print("\n[POST-CONVERGENCE FINAL EVALUATION]")
    print(f"  Force MAE:   {converged_metrics['force_mae_meV_A']} meV/A")
    print(f"  Force RMSE:  {converged_metrics['force_rmse_meV_A']} meV/A")
    print(f"  Force Q95:   {converged_metrics['force_q95_meV_A']} meV/A")
    print(f"  Energy MAE:  {converged_metrics['energy_mae_meV_atom']} meV/atom")

    results_summary = {
        "model": "PaiNN-3-Ensemble",
        "elapsed_seconds": round(elapsed, 1),
        "dataset_size_train": 509,
        "dataset_size_val": 142,
        "baseline_pre_al": baseline_metrics,
        "post_convergence": converged_metrics,
    }

    out_json = OUTPUT_DIR / "painn_convergence_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to: {out_json}")


if __name__ == "__main__":
    main()
