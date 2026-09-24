#!/usr/bin/env python
"""
================================================================================
   PAINN CLOSED-LOOP ACTIVE LEARNING WITH PARSL (5 ANGSTROM WATER GAP)
================================================================================
Entry point for the closed-loop active learning workflow using aims_PAX procedures.
Configures ALConfiguration and executes ALProcedurePARSL.
================================================================================
"""

import sys
from pathlib import Path
from ase import units
import torch

# Ensure aims_PAX package is accessible
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aims_PAX.procedures import ALConfiguration, ALProcedurePARSL


def main():
    base_dir = Path(__file__).resolve().parent
    config = ALConfiguration(
        base_dir=base_dir,
        control_in=base_dir / "control.in",
        species_dir=Path("/home/maria.crist/fhi-aims.260331/species_defaults/defaults_2020/light"),
        aims_bin="/home/maria.crist/fhi-aims.260331/bin/aims.x",
        initial_train_dataset=Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/train.xyz"),
        initial_val_dataset=Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/val.xyz"),
        ref_energies={
            1: 1295.1619808355229,
            8: -4671.611073387086,
            14: -2659.5960319024257,
        },
        initial_model_dirs=[
            Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_0"),
            Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_1"),
            Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_2"),
        ],
        temperature_K=300.0,
        timestep_fs=0.5,
        friction=0.002 / units.fs,
        max_md_steps=10000,
        skip_step_mlff=25,
        dump_every_d1=1000,
        dump_every_d2=5000,
        initial_threshold=2.50,
        min_threshold=1.80,
        c_x_ratio=0.25,
        trigger_cooldown_steps=100,
        valid_ratio=0.1,
        max_al_cycles=50,
        desired_acc_force_mae=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    al = ALProcedurePARSL(config=config)

    if not al.check_al_done():
        al.run()

    al.converge()


if __name__ == "__main__":
    main()
