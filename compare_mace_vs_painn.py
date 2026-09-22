#!/usr/bin/env python
"""
================================================================================
     HEAD-TO-HEAD COMPARATIVE BENCHMARK: MACE vs. PAINN (500 MD STEPS)
================================================================================
Compares a 3-member MACE ensemble vs. a 3-member PaiNN ensemble on the 3
silica-water geometries with a 5 Angstrom water layer:
  - geometry_alpha_5.in  (Alpha-quartz slab + 5 A water)
  - geometry_amor_5.in   (Amorphous silica slab + 5 A water)
  - geometry_beta_5.in   (Beta-cristobalite slab + 5 A water)

Protocol:
  - Exactly identical initial coordinates and velocities (same seed).
  - T = 300 K Langevin dynamics (dt = 0.5 fs, friction = 0.002 / fs).
  - Identical uncertainty formulation: U = max_i sigma_i (meV/A).
  - Trajectory duration: 500 steps (250 fs).
================================================================================
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from painn_ensemble_calc import PainnEnsembleCalculator, KCAL_TO_EV
from mace_ensemble_calc import MaceEnsembleCalculator

REF_ENERGIES = {
    1: 1295.1619808355229,
    8: -4671.611073387086,
    14: -2659.5960319024257,
}

BASE_DIR = Path("/home/maria.crist/dft_mlip/sep_pax")
GEOMETRIES_DIR = BASE_DIR / "geometries"
OUTPUT_DIR = BASE_DIR / "comparison_results"

# 3-Member PaiNN Ensemble (trained on mine data, 500 epochs)
PAINN_MODEL_DIRS = [
    Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_0"),
    Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_1"),
    Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_2"),
]

# 3-Member MACE Ensemble (trained on mine data, 500 epochs)
MACE_MODEL_PATHS = [
    Path("/home/maria.crist/dft_mlip/my_dataset/mine/mace_ensemble/model_0/silica_water_mace.model"),
    Path("/home/maria.crist/dft_mlip/my_dataset/mine/mace_ensemble/model_1/silica_water_mace_seed42.model"),
    Path("/home/maria.crist/dft_mlip/my_dataset/mine/mace_ensemble/model_2/silica_water_mace_seed123.model"),
]

GEOMETRY_FILES = [
    "geometry_alpha_5.in",
    "geometry_amor_5.in",
    "geometry_beta_5.in",
]

MD_STEPS = 500
TIMESTEP = 0.5 * units.fs
TEMPERATURE_K = 300.0
FRICTION = 0.002 / units.fs
RECORD_INTERVAL = 10
UNCERTAINTY_THRESH = 25.0  # meV/A


def run_single_model_md(model_name, calc, initial_atoms, initial_velocities, rng_seed, out_traj_path):
    atoms = initial_atoms.copy()
    atoms.calc = calc
    atoms.set_velocities(initial_velocities)

    dyn = Langevin(
        atoms,
        timestep=TIMESTEP,
        temperature_K=TEMPERATURE_K,
        friction=FRICTION,
        rng=np.random.RandomState(rng_seed),
    )

    history = []
    step_times = []

    # Clear trajectory file
    if out_traj_path.exists():
        out_traj_path.unlink()

    print(f"  [{model_name}] Running {MD_STEPS} steps...")

    def log_step():
        step = dyn.get_number_of_steps()
        if step % RECORD_INTERVAL != 0:
            return

        t_start = time.time()
        epot = atoms.get_potential_energy()
        temp = atoms.get_temperature()
        u_max = atoms.calc.results.get("max_atomic_sd", 0.0)
        stds = atoms.calc.results.get("std_per_atom", atoms.calc.results.get("atomic_stds", np.zeros(len(atoms))))
        u_mean = float(np.mean(stds))

        max_idx = int(np.argmax(stds))
        top_symbol = atoms.get_chemical_symbols()[max_idx]

        history.append({
            "step": step,
            "time_fs": step * 0.5,
            "temperature_K": round(float(temp), 2),
            "potential_energy_eV": round(float(epot), 4),
            "max_uncertainty_meV_A": round(float(u_max), 2),
            "mean_uncertainty_meV_A": round(float(u_mean), 2),
            "top_atom_idx": max_idx,
            "top_atom_symbol": top_symbol,
            "top_atom_sd": round(float(stds[max_idx]), 2),
            "triggered": bool(u_max > UNCERTAINTY_THRESH),
        })

        write(str(out_traj_path), atoms, append=True)

    dyn.attach(log_step, interval=RECORD_INTERVAL)

    t0 = time.time()
    dyn.run(MD_STEPS)
    total_time = time.time() - t0
    ms_per_step = (total_time / MD_STEPS) * 1000.0

    print(f"  [{model_name}] Finished in {total_time:.2f}s ({ms_per_step:.2f} ms/step)")

    return history, ms_per_step


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("MACE vs. PAINN COMPARATIVE 500-STEP ACTIVE LEARNING BENCHMARK")
    print(f"Device: {device}")
    print(f"Total Steps: {MD_STEPS} (250 fs) | Temperature: {TEMPERATURE_K} K")
    print("=" * 80)

    # Verify MACE models exist
    missing_mace = [p for p in MACE_MODEL_PATHS if not p.exists()]
    if missing_mace:
        print("\n[ERROR] The following MACE model(s) were not found:")
        for p in missing_mace:
            print(f"  - {p}")
        print("\nPlease run train_mace_ensemble_2gpu.slurm first to train the missing models.")
        return 1

    # Load Calculators
    print("\n1. Initializing PaiNN 3-member ensemble...")
    painn_calc = PainnEnsembleCalculator(
        model_dirs=PAINN_MODEL_DIRS,
        ref_energies=REF_ENERGIES,
        cutoff=6.0,
        device=device,
    )

    print("\n2. Initializing MACE 3-member ensemble...")
    mace_calc = MaceEnsembleCalculator(
        model_paths=MACE_MODEL_PATHS,
        device=device,
        default_dtype="float64",
    )

    overall_results = {}

    for geom_name in GEOMETRY_FILES:
        geom_path = GEOMETRIES_DIR / geom_name
        stem = geom_path.stem
        print(f"\n==========================================================")
        print(f"Benchmarking Geometry: {geom_name}")
        print(f"==========================================================")

        atoms_orig = read(str(geom_path), format="aims")

        # Assign identical initial velocities
        base_rng = np.random.RandomState(42)
        atoms_init = atoms_orig.copy()
        MaxwellBoltzmannDistribution(atoms_init, temperature_K=TEMPERATURE_K, rng=base_rng)
        Stationary(atoms_init)
        shared_velocities = atoms_init.get_velocities().copy()

        # A. Run PaiNN
        painn_traj = OUTPUT_DIR / f"painn_traj_{stem}.xyz"
        painn_hist, painn_speed = run_single_model_md(
            "PaiNN", painn_calc, atoms_orig, shared_velocities, rng_seed=123, out_traj_path=painn_traj
        )

        # B. Run MACE
        mace_traj = OUTPUT_DIR / f"mace_traj_{stem}.xyz"
        mace_hist, mace_speed = run_single_model_md(
            "MACE", mace_calc, atoms_orig, shared_velocities, rng_seed=123, out_traj_path=mace_traj
        )

        # Statistics
        p_u_max_all = [h["max_uncertainty_meV_A"] for h in painn_hist]
        m_u_max_all = [h["max_uncertainty_meV_A"] for h in mace_hist]
        p_triggers = sum(1 for h in painn_hist if h["triggered"])
        m_triggers = sum(1 for h in mace_hist if h["triggered"])

        geom_summary = {
            "natoms": len(atoms_orig),
            "painn": {
                "ms_per_step": round(painn_speed, 2),
                "mean_T_K": round(float(np.mean([h["temperature_K"] for h in painn_hist])), 1),
                "peak_uncertainty_meV_A": round(float(np.max(p_u_max_all)), 2),
                "mean_uncertainty_meV_A": round(float(np.mean(p_u_max_all)), 2),
                "threshold_triggers": p_triggers,
            },
            "mace": {
                "ms_per_step": round(mace_speed, 2),
                "mean_T_K": round(float(np.mean([h["temperature_K"] for h in mace_hist])), 1),
                "peak_uncertainty_meV_A": round(float(np.max(m_u_max_all)), 2),
                "mean_uncertainty_meV_A": round(float(np.mean(m_u_max_all)), 2),
                "threshold_triggers": m_triggers,
            },
            "painn_history": painn_hist,
            "mace_history": mace_hist,
        }

        overall_results[stem] = geom_summary

        print(f"\nSummary for {geom_name}:")
        print(f"  PaiNN: Peak U = {geom_summary['painn']['peak_uncertainty_meV_A']} meV/A | "
              f"Triggers: {p_triggers}/{len(painn_hist)} | Speed: {painn_speed:.1f} ms/step")
        print(f"  MACE:  Peak U = {geom_summary['mace']['peak_uncertainty_meV_A']} meV/A | "
              f"Triggers: {m_triggers}/{len(mace_hist)} | Speed: {mace_speed:.1f} ms/step")

    # Save outputs
    json_path = OUTPUT_DIR / "mace_vs_painn_summary.json"
    with open(json_path, "w") as f:
        json.dump(overall_results, f, indent=2)

    print(f"\n==========================================================")
    print(f"BENCHMARK COMPLETE! Results written to:")
    print(f"  - Summary: {json_path}")
    print(f"  - Trajectories: {OUTPUT_DIR}")
    print(f"==========================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
