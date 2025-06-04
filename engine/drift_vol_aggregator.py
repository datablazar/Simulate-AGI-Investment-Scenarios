# AI_Scenario_Sim/engine/drift_vol_aggregator.py
# ----------------------------------------------
"""
DriftVolAggregator
==================

Combines baseline μ/σ and event-driven deltas into 
year-by-year drift and covariance matrices.

Usage:
    agg = DriftVolAggregator("data/asset_baseline.json", tickers)
    combo = agg.combine(event_drift, event_volmul)
    # combo["mu"] is (years × n_assets)
    # combo["cov"] is (years × n_assets × n_assets)
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List

class DriftVolAggregator:
    def __init__(self, baseline_path: str | Path, tickers: List[str]):
        with open(baseline_path, "r") as f:
            base = json.load(f)

        # Load baseline drifts (in %), convert to decimals
        self.base_mu  = np.array([base["mu"][t]    for t in tickers]) / 100
        self.base_sig = np.array([base["sigma"][t] for t in tickers]) / 100
        rho = base.get("rho", 0.75)

        # Build baseline covariance matrix with equal correlation (simple)
        n = len(tickers)
        corr_mat = rho * np.ones((n, n))
        np.fill_diagonal(corr_mat, 1.0)
        self.base_cov = corr_mat * np.outer(self.base_sig, self.base_sig)

    def combine(self,
                event_drift: np.ndarray,    # (years × n_assets) additive
                vol_mult:   np.ndarray      # (years,) multiplicative factor
               ) -> dict:
        """
        Given:
          • event_drift: array of shape (years, n_assets),
            holding annual drift deltas (in decimals) from events.
          • vol_mult: array of shape (years,), each entry ≥ 0
            representing the total volatility multiplier for that year.

        Returns:
          {
            "mu":  np.ndarray (years × n_assets),   # base_mu + event_drift
            "cov": np.ndarray (years × n_assets × n_assets)
          }
        """
        years, n = event_drift.shape

        # 1) Add baseline mu + event-driven deltas
        mu = self.base_mu + event_drift        # shape (years, n)

        # 2) Scale covariance each year
        cov = np.empty((years, n, n))
        for y in range(years):
            scale = vol_mult[y]                 # Use vol_mult directly (no +1)
            cov[y] = self.base_cov * (scale**2)

        return {"mu": mu, "cov": cov}
