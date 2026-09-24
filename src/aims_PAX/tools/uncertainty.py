#!/usr/bin/env python
"""
Uncertainty quantification and adaptive threshold management for active learning.
Implements:
  - HandleUncertainty: Computes atomic force standard deviations across committee predictions.
  - MolForceUncertainty: aims-PAX standard interface for committee disagreement.
  - RollingAdaptiveThresholdManager: Rolling history-based threshold adaptation.
"""

from typing import List, Optional
import numpy as np


class HandleUncertainty:
    """Computes committee force disagreement metrics."""

    def __init__(self, uncertainty_type: str = "max_atomic_sd"):
        self.uncertainty_type = uncertainty_type

    def ensemble_sd(self, ensemble_forces: np.ndarray) -> np.ndarray:
        """
        Calculates standard deviation of forces per atom.
        Args:
            ensemble_forces: (M, N, 3) array of force predictions from M committee members.
        Returns:
            (N,) array of per-atom force standard deviations.
        """
        # Average prediction over ensemble: (1, N, 3)
        pred_av = np.average(ensemble_forces, axis=0, keepdims=True)
        diff_sq = (ensemble_forces - pred_av) ** 2.0
        diff_sq_mean = np.mean(diff_sq, axis=(0, -1))
        sd = np.sqrt(diff_sq_mean)
        return sd

    def max_atomic_sd(self, ensemble_forces: np.ndarray) -> float:
        """Computes max atomic force standard deviation across committee."""
        sd_atomic = self.ensemble_sd(ensemble_forces)
        return float(np.max(sd_atomic))

    def mean_atomic_sd(self, ensemble_forces: np.ndarray) -> float:
        """Computes mean atomic force standard deviation across committee."""
        sd_atomic = self.ensemble_sd(ensemble_forces)
        return float(np.mean(sd_atomic))

    def __call__(self, ensemble_forces: np.ndarray) -> float:
        if self.uncertainty_type == "max_atomic_sd":
            return self.max_atomic_sd(ensemble_forces)
        elif self.uncertainty_type == "mean_atomic_sd":
            return self.mean_atomic_sd(ensemble_forces)
        elif self.uncertainty_type == "ensemble_sd":
            return self.ensemble_sd(ensemble_forces)
        else:
            raise ValueError(f"Uncertainty type '{self.uncertainty_type}' not recognized.")


class MolForceUncertainty(HandleUncertainty):
    """aims-PAX standard interface for maximum atomic force standard deviation."""
    def __init__(self, uncertainty_type: str = "max_atomic_sd"):
        super().__init__(uncertainty_type=uncertainty_type)


class RollingAdaptiveThresholdManager:
    """
    Adaptive threshold manager maintaining a rolling window of uncertainties.
    Calculates dynamic threshold tau = mean(history) * (1 + c_x) with hard floor min_threshold.
    """

    def __init__(
        self,
        c_x: float = 0.5,
        min_threshold: float = 0.05,
        rolling_window: int = 100,
        freeze_size: Optional[int] = None,
        initial_threshold: Optional[float] = None,
    ):
        self.c_x = c_x
        self.min_threshold = min_threshold
        self.rolling_window = rolling_window
        self.freeze_size = freeze_size
        self.uncertainty_history: List[float] = []
        self.threshold = initial_threshold if initial_threshold is not None else min_threshold
        self.frozen = False

    def update(self, u_value: float, current_dataset_size: Optional[int] = None) -> float:
        """Records an uncertainty sample and returns the updated threshold."""
        if np.isfinite(u_value) and u_value > 0.0:
            self.uncertainty_history.append(float(u_value))
            if len(self.uncertainty_history) > self.rolling_window:
                self.uncertainty_history.pop(0)

        if self.freeze_size is not None and current_dataset_size is not None:
            if current_dataset_size >= self.freeze_size:
                self.frozen = True

        if not self.frozen and len(self.uncertainty_history) > 0:
            avg_u = float(np.mean(self.uncertainty_history))
            calculated_thresh = avg_u * (1.0 + self.c_x)
            self.threshold = max(calculated_thresh, self.min_threshold)

        return self.threshold
