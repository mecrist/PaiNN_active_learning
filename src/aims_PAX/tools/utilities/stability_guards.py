#!/usr/bin/env python
"""
Trajectory Stability Guards for Molecular Dynamics.
Safeguards active learning against NaN/Inf divergence, thermal runaway, and unphysical basins:
  - is_structure_finite: Checks coordinates and velocities for non-finite values.
  - thermal_runaway_guard: Detects kinetic overheating and re-thermalizes.
  - rollback_trajectory: Restores safe geometry and velocities with stochastic re-thermalization.
"""

import time
from typing import Optional, Dict, Any
import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


def is_structure_finite(atoms: Atoms) -> bool:
    """Returns True if all positions and velocities are finite, False if NaN or Inf."""
    pos = atoms.get_positions()
    if not np.isfinite(pos).all():
        return False
    vel = atoms.get_velocities()
    if vel is not None and not np.isfinite(vel).all():
        return False
    return True


def thermal_runaway_guard(
    atoms: Atoms,
    traj_name: str = "trajectory",
    max_temp_K: float = 1000.0,
    target_temp_K: float = 300.0,
    seed: Optional[int] = None,
) -> bool:
    """
    Checks if instantaneous temperature exceeds threshold. If so, re-thermalizes
    velocities to target_temp_K. Returns True if runaway was detected and handled.
    """
    temp = atoms.get_temperature()
    if temp > max_temp_K:
        print(f"\n[StabilityGuard] WARNING: Thermal runaway in {traj_name} (T = {temp:.1f} K > {max_temp_K:.1f} K)!")
        print(f"[StabilityGuard] Re-thermalizing velocities to {target_temp_K:.1f} K...")
        if seed is None:
            seed = int(time.time() * 1000) % 100000
        rng = np.random.RandomState(seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=target_temp_K, rng=rng)
        Stationary(atoms)
        return True
    return False


def rollback_trajectory(
    atoms: Atoms,
    safe_data: Dict[str, Any],
    target_temp_K: float = 300.0,
    seed: Optional[int] = None,
    noise_std: float = 0.0,
):
    """
    Restores positions from safe checkpoint, assigns fresh Maxwell-Boltzmann
    velocities at target_temp_K, zeroes center-of-mass momentum, and optionally applies
    subtle Gaussian displacement noise to escape unphysical local minima.
    """
    atoms.set_positions(safe_data["positions"].copy())

    if seed is None:
        seed = int(time.time() * 1000) % 100000
    rng = np.random.RandomState(seed)

    MaxwellBoltzmannDistribution(atoms, temperature_K=target_temp_K, rng=rng)
    Stationary(atoms)

    if noise_std > 0.0:
        noise = rng.normal(0.0, noise_std, size=atoms.get_positions().shape)
        atoms.set_positions(atoms.get_positions() + noise)
