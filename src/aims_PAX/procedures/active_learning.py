#!/usr/bin/env python
"""
Active Learning Procedure with Parsl Concurrency.
Orchestrates asynchronous closed-loop active learning:
  - Propagates multiple trajectories on GPU.
  - Triggers asynchronous FHI-aims DFT calculations via background Parsl threads.
  - Dynamically retrains PaiNN committee models upon calculation completion.
  - Manages validation quotas, checkpoints, and final model convergence.
"""

from pathlib import Path
import time
import parsl
import numpy as np

from aims_PAX.procedures.preparation import PrepareALProcedure, ALConfiguration
from aims_PAX.procedures.al_managers import (
    ALRunningManager,
    ALDataManager,
    ALTrainingManager,
    ALDFTReferenceManagerPARSL,
)
from aims_PAX.tools.utilities.parsl_utils import initialize_parsl
from aims_PAX.tools.model_tools.train_painn import run_final_convergence


class ALProcedurePARSL(PrepareALProcedure):
    """
    Implementation of the asynchronous active learning procedure using Parsl.
    Integrates decoupled managers for running MD, data handling, retraining, and reference execution.
    """

    def __init__(self, config: ALConfiguration):
        super().__init__(config=config)

        # Initialize Parsl ThreadPoolExecutor
        initialize_parsl(calc_dir=self.base_dir, max_workers=1)

        # Instantiate decoupled managers
        self.data_manager = ALDataManager(
            config=self.config,
            state_manager=self.state_manager,
        )

        self.train_manager = ALTrainingManager(
            config=self.config,
            state_manager=self.state_manager,
            ensemble=self.ensemble,
            data_manager=self.data_manager,
        )

        self.reference_manager = ALDFTReferenceManagerPARSL(
            config=self.config,
            state_manager=self.state_manager,
            ensemble=self.ensemble,
        )

        self.run_manager = ALRunningManager(
            config=self.config,
            state_manager=self.state_manager,
            ensemble=self.ensemble,
            dyn_drivers=self.dyn_drivers,
            trajectories=self.trajectories,
        )

    def _on_trigger(self, traj_idx: int, atoms, u_value: float, res: dict):
        """Dispatches an uncertainty trigger to the Parsl reference manager."""
        t = self.trajectories[traj_idx]
        self.reference_manager.dispatch_trigger(
            traj_idx=traj_idx,
            t=t,
            atoms=atoms,
            u_value=u_value,
            res=res,
        )

    def run(self):
        """Executes the closed-loop asynchronous active learning loop."""
        print("\n==========================================================")
        print("STARTING PAINN PARSL ASYNCHRONOUS ACTIVE LEARNING RUN")
        print(f"Max steps: {self.config.max_md_steps} | T: {self.config.temperature_K} K | Device: {self.config.device}")
        if self.config.desired_acc_force_mae is not None:
            print(f"Desired accuracy limit: {self.config.desired_acc_force_mae:.2f} meV/A Force MAE")
        print("==========================================================\n")

        while self.state_manager.step < self.config.max_md_steps:
            # ------------------------------------------------------------------
            # Phase 1: Poll Parsl background futures for completed DFT calculations
            # ------------------------------------------------------------------
            for i, t in enumerate(self.trajectories):
                if t["status"] == "waiting" and t["future"] is not None:
                    if t["future"].done():
                        res = t["future"].result()
                        pending = t["pending_point"]
                        t["status"] = "running"
                        t["future"] = None
                        t["pending_point"] = None

                        if not res["success"]:
                            self.reference_manager.handle_failed_dft(
                                traj_idx=i,
                                t=t,
                                result=res,
                                pending=pending,
                            )
                        else:
                            self.state_manager.cycle += 1
                            t["consecutive_failures"] = 0
                            t["last_safe_checkpoint"] = {
                                "positions": pending["atoms"].get_positions().copy(),
                                "velocities": pending["atoms"].get_velocities().copy() if pending["atoms"].get_velocities() is not None else None,
                                "step": pending["step"],
                            }
                            t["cooldown"] = self.config.trigger_cooldown_steps

                            target_pool = self.data_manager.handle_received_point(
                                atoms=pending["atoms"],
                                result=res,
                                pending=pending,
                            )

                            if target_pool == "training":
                                self.train_manager.train_epoch()
                                # Update calculator reference for all trajectories
                                for tr in self.trajectories:
                                    tr["atoms"].calc = self.ensemble.calc

                            self.state_manager.save_checkpoint(
                                trajectories=self.trajectories,
                                current_model_dirs=self.ensemble.current_model_dirs,
                            )

            if self.train_manager.accuracy_reached:
                print("\n>>> Active Learning target accuracy reached! Halting loop.")
                break

            # ------------------------------------------------------------------
            # Phase 2: Check if all unfinished trajectories are currently waiting
            # ------------------------------------------------------------------
            unfinished = [t for t in self.trajectories if t["step"] < self.config.max_md_steps]
            if len(unfinished) > 0 and all(t["status"] == "waiting" for t in unfinished):
                time.sleep(1.0)
                continue

            # ------------------------------------------------------------------
            # Phase 3: Propagate running trajectories on GPU
            # ------------------------------------------------------------------
            self.run_manager.step_trajectories(on_trigger_callback=self._on_trigger)

            # Checkpoint every 500 steps
            if self.state_manager.step > 0 and self.state_manager.step % 500 == 0:
                self.state_manager.save_checkpoint(
                    trajectories=self.trajectories,
                    current_model_dirs=self.ensemble.current_model_dirs,
                )

        # ------------------------------------------------------------------
        # Shutdown and final convergence
        # ------------------------------------------------------------------
        try:
            parsl.dfk().cleanup()
        except Exception:
            pass

        print("\n==========================================================")
        print("PaiNN Active Learning Completed Successfully!")
        print(f"Total Cycles: {self.state_manager.cycle} | Final Steps: {self.state_manager.step}")
        print("==========================================================")

        if self.train_manager.accuracy_reached:
            self.converge()

    def converge(self, n_epochs: int = 50):
        """Runs post-AL final model convergence routine on the full dataset."""
        print("[ALProcedurePARSL] Running post-AL final model convergence routine...")
        run_final_convergence(
            calculator=self.ensemble.calc,
            dataset_path=self.data_manager.dataset_file,
            val_dataset_path=self.data_manager.val_dataset_file,
            ref_energies=self.config.ref_energies,
            output_dir=self.ckpt_dir / "converged_models",
            n_epochs=n_epochs,
            device=self.config.device,
        )

