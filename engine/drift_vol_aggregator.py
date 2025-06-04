# AI_Scenario_Sim/engine/drift_vol_aggregator.py
# ----------------------------------------------
"""
DriftVolAggregator
==================

Combines baseline μ/σ and event-driven deltas into 
year-by-year drift and covariance matrices, using
an array-based stochastic noise model.

Public API
----------
agg = DriftVolAggregator("data/asset_baseline.json", tickers)
combo = agg.combine(event_drift_array, vol_mult_array)
#   - event_drift_array: np.ndarray of shape (years × n_assets),
#       containing additive annual drift deltas (in decimals) from fired events.
#   - vol_mult_array:     np.ndarray of shape (years,),
#       each entry ≥ 0 representing the total volatility multiplier for that year.
#
# Returns:
#   {
#     "mu":  np.ndarray shape (years × n_assets),
#     "cov": np.ndarray shape (years × n_assets × n_assets)
#   }
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

        # Load baseline drifts (in %) and volatilities (in %), convert to decimals
        self.base_mu  = np.array([base["mu"][t]    for t in tickers]) / 100.0
        self.base_sig = np.array([base["sigma"][t] for t in tickers]) / 100.0
        rho = base.get("rho", 0.75)

        # Build baseline covariance matrix assuming equal correlation = rho
        n = len(tickers)
        corr_mat = rho * np.ones((n, n))
        np.fill_diagonal(corr_mat, 1.0)
        self.base_cov = corr_mat * np.outer(self.base_sig, self.base_sig)

        # Master RNG placeholder (BatchRunner will assign .rng)
        self.rng = None  # type: ignore

    def combine(self,
                event_drift: np.ndarray,   # shape: (years × n_assets)
                vol_mult:    np.ndarray    # shape: (years,)
               ) -> dict:
        """
        Combine baseline and event deltas into a drift & covariance series.

        Steps:
          1) Initialize mu_series = baseline μ repeated for each year.
          2) For each cell where event_drift != 0, draw noise ~ Normal(0, 0.4*|event_drift|).
             Add (drift + noise) to base μ for that year and asset.
          3) Build covariance for each year by scaling base_cov by (vol_mult[y])^2.

        Returns dict with keys "mu" and "cov".
        """
        years, n_assets = event_drift.shape

        # 1) Initialize mu_series with baseline mu repeated for each year
        mu_series = np.tile(self.base_mu, (years, 1))  # shape (years, n_assets)

        # 2) Introduce stochastic noise proportional to |event_drift|
        noise = np.zeros_like(event_drift)
        # Mask where event_drift is nonzero
        nonzero_mask = np.abs(event_drift) > 0.0
        # Standard deviation per cell = 0.4 * |event_drift|
        sigma_matrix = 0.4 * np.abs(event_drift)
        # Draw noise only where event_drift != 0
        noise[nonzero_mask] = self.rng.normal(
            loc=0.0,
            scale=sigma_matrix[nonzero_mask]
        )
        # Final drift = baseline + event_drift + noise
        mu_series += event_drift + noise

        # 3) Build covariance series year-by-year by scaling base_cov
        cov_series = np.empty((years, n_assets, n_assets))
        for y in range(years):
            scale = vol_mult[y]  # volumetric multiplier for year y
            cov_series[y] = self.base_cov * (scale ** 2)

        return {"mu": mu_series, "cov": cov_series}
