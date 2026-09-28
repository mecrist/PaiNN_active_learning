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

try:
    from mace.calculators import MACECalculator
    from mace import data, tools
    from mace.tools.torch_geometric.dataloader import DataLoader
except ImportError:
    pass


def evaluate_mace_on_test(calculator: Any, test_frames: List) -> Dict[str, float]:
    """Evaluates MACE on the test set and calculates RMSE, MAE, and Q95."""
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
    parser = argparse.ArgumentParser(description="MACE Active Learning Replay Benchmark")
    parser.add_argument("--epochs_per_cycle", type=int, required=True, help="Epochs to train per AL cycle (e.g. 2, 3, 5, 10)")
    parser.add_argument("--conv_epochs", type=int, default=50, help="Epochs for final post-AL convergence training")
    parser.add_argument("--conv_patience", type=int, default=15, help="Early stopping patience for final convergence")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints and metrics")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate on test set every N cycles")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    ep_tag = f"ep{args.epochs_per_cycle}"
    if args.output_dir is None:
        args.output_dir = str(BENCHMARK_DIR / f"results_mace_{ep_tag}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"STARTING MACE AL REPLAY BENCHMARK: {args.epochs_per_cycle} EPOCHS/CYCLE")
    print(f"Device:           {args.device}")
    print(f"Output Directory: {out_dir}")
    print(f"Final Conv:       {args.conv_epochs} epochs (patience: {args.conv_patience})")
    print("=" * 80)

    base_model_path = Path("/home/maria.crist/dft_mlip/sep_pax/al_5A_mace/model/silica_water_mace_run-777.model")
    current_model_path = ckpt_dir / "current_mace.model"
    shutil.copyfile(base_model_path, current_model_path)

    calc = MACECalculator(model_paths=str(current_model_path), device=args.device)

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
    initial_metrics = evaluate_mace_on_test(calc, test_145)
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

    # Load torch model object for fine-tuning
    try:
        model = torch.load(str(current_model_path), map_location=args.device, weights_only=False)
    except TypeError:
        model = torch.load(str(current_model_path), map_location=args.device)
    model.to(args.device)

    z_table = tools.AtomicNumberTable([1, 8, 14])
    keyspec = data.KeySpecification(info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"})
    model_dtype = next(model.parameters()).dtype

    for cycle_idx in range(1, num_cycles + 1):
        c_start = time.time()
        new_frame = al_pool[cycle_idx - 1]
        current_train_pool.append(new_frame)
        write(str(current_train_file), current_train_pool, format="extxyz")

        print(f"\n--- [Cycle {cycle_idx:02d}/{num_cycles}] Appended AL Point #{cycle_idx} (Total pool: {len(current_train_pool)} frames) ---")

        # Convert atoms to MACE AtomicData batches
        data_loader = DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    data.config_from_atoms(at, key_specification=keyspec),
                    z_table=z_table,
                    cutoff=5.0,
                )
                for at in current_train_pool
            ],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

        for ep in range(args.epochs_per_cycle):
            ep_loss = 0.0
            n_batches = 0
            for batch in data_loader:
                batch = batch.to(args.device)
                optimizer.zero_grad()
                out = model(batch.to_dict(), compute_force=True, training=True)

                # Loss: energy (weight 1.0) + force (weight 100.0)
                e_true = batch.energy.view(-1).to(model_dtype)
                e_pred = out["energy"].view(-1)
                f_true = batch.forces.to(model_dtype)
                f_pred = out["forces"]

                loss_e = torch.mean((e_pred - e_true) ** 2)
                loss_f = torch.mean((f_pred - f_true) ** 2)
                total_loss = loss_e + 100.0 * loss_f

                total_loss.backward()
                optimizer.step()

                ep_loss += total_loss.item()
                n_batches += 1

        c_duration = time.time() - c_start

        # Update calculator model
        torch.save(model, str(current_model_path))
        calc = MACECalculator(model_paths=str(current_model_path), device=args.device)

        cycle_info = {
            "cycle": cycle_idx,
            "dataset_size": len(current_train_pool),
            "epochs_trained": args.epochs_per_cycle,
            "train_wallclock_sec": c_duration,
            "avg_cycle_loss": ep_loss / max(1, n_batches),
            "test_eval": None,
        }

        if cycle_idx == 1 or cycle_idx % args.eval_every == 0 or cycle_idx == num_cycles:
            print(f"    >>> Running periodic test evaluation (Cycle {cycle_idx})...")
            t_eval = evaluate_mace_on_test(calc, test_145)
            cycle_info["test_eval"] = t_eval
            print(f"    Cycle {cycle_idx} Test Energy RMSE: {t_eval['energy_rmse_mev_per_atom']:.2f} meV/atom | Force RMSE: {t_eval['force_rmse_mev_per_A']:.2f} meV/Å (Q95: {t_eval['force_q95_mev_per_A']:.2f})")

        replay_log["cycles"].append(cycle_info)

        with open(out_dir / "replay_metrics.json", "w") as f:
            json.dump(replay_log, f, indent=2)

    total_replay_time = time.time() - total_start_time
    print(f"\n================================================================================")
    print(f"MACE REPLAY COMPLETE for {num_cycles} cycles in {total_replay_time/60.0:.1f} minutes.")
    print(f"================================================================================")

    if args.conv_epochs > 0:
        print(f"\n>>> Running Post-AL Final Convergence Session ({args.conv_epochs} epochs)...")
        conv_start = time.time()

        conv_loader = DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    data.config_from_atoms(at, key_specification=keyspec),
                    z_table=z_table,
                    cutoff=5.0,
                )
                for at in current_train_pool
            ],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.5, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        for conv_ep in range(1, args.conv_epochs + 1):
            model.train()
            ep_loss = 0.0
            n_batches = 0
            for batch in conv_loader:
                batch = batch.to(args.device)
                optimizer.zero_grad()
                out = model(batch.to_dict(), compute_force=True, training=True)

                e_true = batch.energy.view(-1).to(model_dtype)
                e_pred = out["energy"].view(-1)
                f_true = batch.forces.to(model_dtype)
                f_pred = out["forces"]

                loss_e = torch.mean((e_pred - e_true) ** 2)
                loss_f = torch.mean((f_pred - f_true) ** 2)
                total_loss = loss_e + 100.0 * loss_f

                total_loss.backward()
                optimizer.step()

                ep_loss += total_loss.item()
                n_batches += 1

            avg_loss = ep_loss / max(1, n_batches)
            scheduler.step(avg_loss)
            if conv_ep % 10 == 0 or conv_ep == args.conv_epochs:
                print(f"    Convergence Epoch {conv_ep:02d}/{args.conv_epochs:02d} -> Loss: {avg_loss:.4f}")

        conv_duration = time.time() - conv_start
        torch.save(model, str(ckpt_dir / "converged_mace.model"))
        calc = MACECalculator(model_paths=str(ckpt_dir / "converged_mace.model"), device=args.device)

        print("\n>>> Final evaluation of fully converged MACE on 145-frame test set...")
        conv_metrics = evaluate_mace_on_test(calc, test_145)
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
