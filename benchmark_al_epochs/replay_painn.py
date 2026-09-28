import sys
import os
import time
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from ase.io import read, write

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aims_PAX.tools.model_tools.setup_painn import setup_painn_ensemble, PainnEnsembleCalculator
from aims_PAX.tools.model_tools.train_painn import retrain_painn_ensemble, run_final_convergence

REF_ENERGIES = {
    1: 1295.1619808355229,    # H (eV)
    8: -4671.611073387086,    # O (eV)
    14: -2659.5960319024257,  # Si (eV)
}
KCAL_TO_MEV = 43.36414


def evaluate_model_on_test(calculator: PainnEnsembleCalculator, test_frames: List) -> Dict[str, float]:
    """Evaluates the model or ensemble on the test set and calculates RMSE, MAE, and Q95."""
    e_dft_all = []
    e_pred_all = []
    natoms_all = []
    f_err_all = []

    for atoms in test_frames:
        at = atoms.copy()
        at.calc = calculator

        # Ground truth
        e_true = at.info.get("REF_energy", at.info.get("energy"))
        f_true = at.arrays.get("REF_forces", at.arrays.get("forces"))
        n_atoms = len(at)

        # Prediction
        e_pred = at.get_potential_energy()
        f_pred = at.get_forces()

        e_dft_all.append(e_true)
        e_pred_all.append(e_pred)
        natoms_all.append(n_atoms)

        # Force errors per atom
        f_diff = f_pred - f_true
        f_err = np.linalg.norm(f_diff, axis=1)  # (N,) in eV/Å
        f_err_all.extend(f_err)

    e_dft = np.array(e_dft_all)
    e_pred = np.array(e_pred_all)
    natoms = np.array(natoms_all)

    # Per-atom energy errors (meV/atom)
    e_err_per_atom_mev = (e_pred - e_dft) / natoms * 1000.0
    e_rmse = float(np.sqrt(np.mean(e_err_per_atom_mev ** 2)))
    e_mae = float(np.mean(np.abs(e_err_per_atom_mev)))

    # Force errors (meV/Å)
    f_err_mev = np.array(f_err_all) * 1000.0
    f_rmse = float(np.sqrt(np.mean(f_err_mev ** 2)))
    f_mae = float(np.mean(f_err_mev))
    f_q95 = float(np.percentile(f_err_mev, 95))
    f_q99 = float(np.percentile(f_err_mev, 99))

    return {
        "energy_rmse_mev_per_atom": e_rmse,
        "energy_mae_mev_per_atom": e_mae,
        "force_rmse_mev_per_A": f_rmse,
        "force_mae_mev_per_A": f_mae,
        "force_q95_mev_per_A": f_q95,
        "force_q99_mev_per_A": f_q99,
    }


def main():
    parser = argparse.ArgumentParser(description="PaiNN Active Learning Replay Benchmark")
    parser.add_argument("--epochs_per_cycle", type=int, required=True, help="Epochs to train per AL cycle (e.g. 2, 3, 5, 10)")
    parser.add_argument("--conv_epochs", type=int, default=50, help="Epochs for final post-AL convergence training")
    parser.add_argument("--conv_patience", type=int, default=15, help="Early stopping patience for final convergence")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints and metrics")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate on test set every N cycles")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--single_model", action="store_true", help="Train single model (model_0) instead of full 3-model committee")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    ep_tag = f"ep{args.epochs_per_cycle}"
    if args.output_dir is None:
        model_str = "single" if args.single_model else "ensemble"
        args.output_dir = str(BENCHMARK_DIR / f"results_painn_{model_str}_{ep_tag}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"STARTING PAINN AL REPLAY BENCHMARK: {args.epochs_per_cycle} EPOCHS/CYCLE")
    print(f"Device:           {args.device}")
    print(f"Output Directory: {out_dir}")
    print(f"Final Conv:       {args.conv_epochs} epochs (patience: {args.conv_patience})")
    print(f"Model Type:       {'Single Model (model_0)' if args.single_model else '3-Member Committee'}")
    print("=" * 80)

    base_ensemble_dir = Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble")
    model_indices = [0] if args.single_model else [0, 1, 2]
    active_model_dirs = []

    for idx in model_indices:
        src = base_ensemble_dir / f"model_{idx}"
        dst = ckpt_dir / f"model_{idx}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        active_model_dirs.append(dst)

    calc = setup_painn_ensemble(
        model_dirs=active_model_dirs,
        ref_energies=REF_ENERGIES,
        device=args.device,
    )

    data_dir = BENCHMARK_DIR / "data"
    base_train = read(str(data_dir / "base_train.extxyz"), index=":", format="extxyz")
    base_val = read(str(data_dir / "base_val.extxyz"), index=":", format="extxyz")
    al_pool = read(str(data_dir / "combined_al_pool.extxyz"), index=":", format="extxyz")
    test_145 = read(str(data_dir / "test_145_interface.extxyz"), index=":", format="extxyz")

    print(f"Loaded {len(base_train)} base train, {len(base_val)} base val, {len(al_pool)} AL frames, {len(test_145)} test frames.")

    current_train_file = out_dir / "current_train.extxyz"
    val_file = data_dir / "base_val.extxyz"
    current_train_pool = list(base_train)
    write(str(current_train_file), current_train_pool, format="extxyz")

    print("\n>>> Evaluating Pre-AL baseline accuracy on 145-frame test set...")
    initial_metrics = evaluate_model_on_test(calc, test_145)
    print(f"    Initial Energy RMSE: {initial_metrics['energy_rmse_mev_per_atom']:.2f} meV/atom")
    print(f"    Initial Force RMSE:  {initial_metrics['force_rmse_mev_per_A']:.2f} meV/Å (Q95: {initial_metrics['force_q95_mev_per_A']:.2f} meV/Å)")

    replay_log = {
        "config": vars(args),
        "initial_metrics": initial_metrics,
        "cycles": [],
        "final_convergence": None,
    }

    total_start_time = time.time()
    num_cycles = len(al_pool)

    for cycle_idx in range(1, num_cycles + 1):
        c_start = time.time()
        new_frame = al_pool[cycle_idx - 1]
        current_train_pool.append(new_frame)
        write(str(current_train_file), current_train_pool, format="extxyz")

        print(f"\n--- [Cycle {cycle_idx:02d}/{num_cycles}] Appended AL Point #{cycle_idx} (Total pool: {len(current_train_pool)} frames) ---")

        updated_paths = retrain_painn_ensemble(
            calculator=calc,
            dataset_path=current_train_file,
            ref_energies=REF_ENERGIES,
            checkpoint_dir=ckpt_dir,
            cycle=cycle_idx,
            n_epochs=args.epochs_per_cycle,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            val_dataset_path=val_file,
        )
        c_duration = time.time() - c_start

        cycle_info = {
            "cycle": cycle_idx,
            "dataset_size": len(current_train_pool),
            "epochs_trained": args.epochs_per_cycle,
            "train_wallclock_sec": c_duration,
            "test_eval": None,
        }

        if cycle_idx == 1 or cycle_idx % args.eval_every == 0 or cycle_idx == num_cycles:
            print(f"    >>> Running periodic test evaluation (Cycle {cycle_idx})...")
            t_eval = evaluate_model_on_test(calc, test_145)
            cycle_info["test_eval"] = t_eval
            print(f"    Cycle {cycle_idx} Test Energy RMSE: {t_eval['energy_rmse_mev_per_atom']:.2f} meV/atom | Force RMSE: {t_eval['force_rmse_mev_per_A']:.2f} meV/Å (Q95: {t_eval['force_q95_mev_per_A']:.2f})")

        replay_log["cycles"].append(cycle_info)

        with open(out_dir / "replay_metrics.json", "w") as f:
            json.dump(replay_log, f, indent=2)

    total_replay_time = time.time() - total_start_time
    print(f"\n================================================================================")
    print(f"REPLAY COMPLETE for {num_cycles} cycles in {total_replay_time/60.0:.1f} minutes.")
    print(f"================================================================================")

    if args.conv_epochs > 0:
        print(f"\n>>> Running Post-AL Final Convergence Session ({args.conv_epochs} epochs, patience {args.conv_patience})...")
        conv_out_dir = out_dir / "converged_models"
        conv_start = time.time()

        run_final_convergence(
            calculator=calc,
            dataset_path=current_train_file,
            ref_energies=REF_ENERGIES,
            output_dir=conv_out_dir,
            val_dataset_path=val_file,
            n_epochs=args.conv_epochs,
            patience=args.conv_patience,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        conv_duration = time.time() - conv_start

        print("\n>>> Final evaluation of fully converged model on 145-frame test set...")
        conv_metrics = evaluate_model_on_test(calc, test_145)
        conv_metrics["conv_wallclock_sec"] = conv_duration
        conv_metrics["conv_epochs_requested"] = args.conv_epochs

        replay_log["final_convergence"] = conv_metrics
        print(f"    CONVERGED Energy RMSE: {conv_metrics['energy_rmse_mev_per_atom']:.2f} meV/atom")
        print(f"    CONVERGED Force RMSE:  {conv_metrics['force_rmse_mev_per_A']:.2f} meV/Å (Q95: {conv_metrics['force_q95_mev_per_A']:.2f} meV/Å)")
        print(f"    Convergence time:      {conv_duration/60.0:.2f} minutes")

    with open(out_dir / "replay_metrics.json", "w") as f:
        json.dump(replay_log, f, indent=2)

    print(f"\nAll benchmark results saved to: {out_dir / 'replay_metrics.json'}")


if __name__ == "__main__":
    main()
