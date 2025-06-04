# AI_Scenario_Sim/engine/drift_vol_aggregator.py
# ----------------------------------------------
"""
Combine baseline μ/σ with event-tree deltas.
Returns:
    yearly_mu  : (years, n_assets)
    yearly_cov : (years, n_assets, n_assets)
"""

from __future__ import annotations
import json, numpy as np
from pathlib import Path
from typing import List

class DriftVolAggregator:
    def __init__(self, baseline_path: str | Path, tickers: List[str]):
        with open(baseline_path, "r") as f:
            base = json.load(f)

        self.base_mu = np.array([base["mu"][t] for t in tickers]) / 100
        self.base_sig = np.array([base["sigma"][t] for t in tickers]) / 100
        rho = base.get("rho", 0.75)
        n = len(tickers)
        self.base_cov = rho * np.ones((n, n))
        np.fill_diagonal(self.base_cov, 1.0)
        self.base_cov *= np.outer(self.base_sig, self.base_sig)

    # ------------------------------------------------------------------
    def combine(self,
                event_drift: np.ndarray,   # shape (years, n_assets)
                vol_mult:   np.ndarray     # shape (years,)
               ) -> dict:
        """
        Adds drift deltas and scales covariance per year.
        """
        years, n = event_drift.shape
        mu = self.base_mu + event_drift                    # (years, n)
        cov = np.empty((years, n, n))
        for y in range(years):
            scale = (1 + vol_mult[y])
            cov[y] = self.base_cov * scale**2
        return {"mu": mu, "cov": cov}
