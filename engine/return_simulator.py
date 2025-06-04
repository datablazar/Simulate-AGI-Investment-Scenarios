# AI_Scenario_Sim/engine/return_simulator.py
# ------------------------------------------
"""
ReturnSimulator
===============

• Accepts one path's yearly drift (mu) and covariance (cov) matrices
  plus an ETF-weight vector and monthly contribution amount.
• Generates annual portfolio returns via multivariate normal draws.
• Compounds wealth with equal end-of-month deposits.
• Outputs wealth series, annual-return series, CAGR, and max draw-down.

Assumes arithmetic annual returns; geometric comp handled implicitly.
"""

from __future__ import annotations
from typing import Dict
import numpy as np

class ReturnSimulator:
    def __init__(self,
                 weights: np.ndarray,        # n-asset weights, sum to 1
                 monthly_contrib: float = 100.0,
                 seed: int = 0):
        self.w = weights
        self.contrib = monthly_contrib
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def simulate_path(self,
                      mu: np.ndarray,        # (years, n_assets)
                      cov: np.ndarray        # (years, n_assets, n_assets)
                     ) -> Dict:
        years, n = mu.shape
        assert cov.shape == (years, n, n)

        ann_ret = np.empty(years)

        # --- draw correlated returns year by year --------------------
        for y in range(years):
            chol = np.linalg.cholesky(cov[y])
            z = self.rng.standard_normal(n)
            asset_r = mu[y] + chol @ z           # arithmetic annual returns
            ann_ret[y] = asset_r @ self.w

        # --- wealth compounding with monthly deposits ----------------
        wealth_t = np.zeros(years + 1)
        wealth = 0.0

        for y in range(years):
            r_annual = max(ann_ret[y], -0.999)          # hard floor at –99.9 %
            if r_annual <= -1.0 + 1e-9:                 # shouldn’t happen after clamp
                r_month = -1.0
            else:
                r_month = (1 + r_annual) ** (1/12) - 1

            # future value of 12 equal deposits
            cf = (self.contrib * 12 if r_month == 0
                  else self.contrib * ((1 + r_month)**12 - 1) / r_month)
            wealth = wealth * (1 + r_annual) + cf
            wealth_t[y + 1] = wealth

        # --- metrics -------------------------------------------------
        cagr = (wealth / (self.contrib * 12 * years)) ** (1/years) - 1
        peak = np.maximum.accumulate(wealth_t[1:])      # skip t = 0 (wealth=0)
        drawdown = (wealth_t[1:] - peak) / peak
        max_dd = drawdown.min() if len(drawdown) else 0.0



        return {
            "wealth_series": wealth_t,
            "ann_returns": ann_ret,
            "terminal_wealth": wealth,
            "cagr": cagr,
            "max_drawdown": max_dd
        }
