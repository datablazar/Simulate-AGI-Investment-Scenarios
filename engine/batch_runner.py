# AI_Scenario_Sim/engine/batch_runner.py
# --------------------------------------
"""
BatchRunner
===========

Monte-Carlo driver:
  • draws a capability timeline
  • triggers the event tree
  • aggregates drift/vol (with stochastic noise)
  • simulates wealth path
  • stores summary metrics

Saves a JSON file like:
  results/scenario_run_20250604_1432.json
"""

from __future__ import annotations
from pathlib import Path
import json, time, numpy as np
from typing import Dict, List

from .timeline_sampler     import TimelineSampler
from .event_tree_engine    import EventTreeEngine
from .drift_vol_aggregator import DriftVolAggregator
from .return_simulator     import ReturnSimulator
from .macro               import MACRO_STATES, MACRO_PROBS, MACRO_MAP

class BatchRunner:
    def __init__(self,
                 n_paths: int,
                 tickers: List[str],
                 monthly_contrib: float = 100.0,
                 seed: int = 0):

        self.n_paths = n_paths
        self.tickers = tickers

        # --- Use InvestEngine allocations ---
        TARGET_ALLOC = {
            "PACW": 0.25, "SEMI": 0.06, "VAGS": 0.05, "EMIM": 0.05, "XAIX": 0.05,
            "ITPS": 0.05, "IWFQ": 0.04, "RBTX": 0.04, "WLDS": 0.04, "MINV": 0.04,
            "IWFV": 0.04, "MEUD": 0.03, "EQQQ": 0.03, "PRIJ": 0.03, "XCX5": 0.03,
            "WDEP": 0.03, "LCUK": 0.03, "WCOM": 0.03, "ISPY": 0.03, "NUCG": 0.03,
            "SGLN": 0.02
        }
        self.weights = np.array([TARGET_ALLOC[t] for t in tickers])

        # Create a single master RNG
        self.master_rng = np.random.default_rng(seed)

        # Initialize modules without giving them their own seeds…
        self.ts = TimelineSampler("data/timeline_buckets.json")
        self.et = EventTreeEngine("data/events_catalogue.json",
                                  tickers=tickers,
                                  horizon_years=40)
        self.agg = DriftVolAggregator("data/asset_baseline.json", tickers)
        self.rs  = ReturnSimulator(self.weights, monthly_contrib)

        # Override each module's .rng to use the shared master RNG
        self.ts.rng  = self.master_rng
        self.et.rng  = self.master_rng
        self.agg.rng = self.master_rng    # ← Ensure aggregator.rng is set
        self.rs.rng  = self.master_rng


    # -----------------------------------------------------------------
    def run(self, store_paths: bool = False) -> Dict:
        wealths, cagrs, maxdds = [], [], []
        if store_paths:
            wealth_matrix = []
        bucket_counts = {b["name"]:0 for b in
                         json.load(open("data/timeline_buckets.json"))["buckets"]}
        fired_counts  = {}

        for i in range(self.n_paths):
            # 1) Sample AI capability timeline
            tl = self.ts.sample_timeline()
            bucket_counts[tl["bucket"]] += 1

            # 2) Simulate event tree based on that timeline
            ev_out = self.et.simulate(tl)
            for ev in ev_out["fired"]:
                fired_counts[ev.id] = fired_counts.get(ev.id, 0) + 1

            # 3) Draw one macro-state for this path
            macro_choice = self.master_rng.choice(MACRO_STATES, p=MACRO_PROBS)
            mu_shift, sigma_mult = MACRO_MAP[macro_choice]

            # 4) Combine baseline + event-induced drifts/vols
            combo = self.agg.combine(ev_out["drift"], ev_out["volmul"])
            mu_arr  = combo["mu"]   # shape: (years, n_assets)
            cov_arr = combo["cov"]  # shape: (years, n_assets, n_assets)

            # 5a) Apply macro-state mu_shift to every year & asset
            mu_arr = mu_arr + mu_shift

            # 5b) Scale covariance by sigma_mult^2
            cov_arr = cov_arr * (sigma_mult ** 2)

            # 6) Simulate returns & contributions
            res = self.rs.simulate_path(mu_arr, cov_arr)
            if store_paths:
                wealth_matrix.append(res["wealth_series"])

            wealths.append(res["terminal_wealth"])
            cagrs.append(res["cagr"])
            maxdds.append(res["max_drawdown"])

        if store_paths:
            wealth_matrix = np.vstack(wealth_matrix)

        wealths = np.array(wealths)
        cagrs   = np.array(cagrs)
        maxdds  = np.array(maxdds)

        summary = {
            "n_paths": self.n_paths,
            "terminal_wealth_percentiles": np.percentile(
                wealths, [5,25,50,75,95]).round(0).tolist(),
            "cagr_percentiles": (np.percentile(cagrs, [5,25,50,75,95])*100
                                 ).round(2).tolist(),
            "max_dd_percentiles": (np.percentile(maxdds, [5,50,95])*100
                                   ).round(1).tolist(),
            "bucket_frequency": {k: round(v/self.n_paths*100,2)
                                 for k,v in bucket_counts.items()},
            "event_frequency": {k: round(v/self.n_paths*100,2)
                                for k,v in fired_counts.items()}
        }

        if store_paths:
            summary["wealth_matrix"] = wealth_matrix.tolist()
            summary["cagrs_raw"]     = cagrs.tolist()

        return summary


    # -----------------------------------------------------------------
    def save(self, summary: Dict, out_dir: str | Path = "results") -> Path:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M")
        path = Path(out_dir)/f"scenario_run_{ts}.json"
        json.dump(summary, open(path,"w"), indent=2)
        return path
