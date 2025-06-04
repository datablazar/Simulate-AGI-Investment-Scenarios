# AI_Scenario_Sim/engine/batch_runner.py
# --------------------------------------
"""
BatchRunner
===========

Monte-Carlo driver:
  • draws a capability timeline
  • triggers the event tree
  • aggregates drift/vol
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

class BatchRunner:
    def __init__(self,
                 n_paths: int,
                 tickers: List[str],
                 monthly_contrib: float = 100.0,
                 seed: int = 0):

        self.n_paths = n_paths
        self.tickers = tickers
        self.weights = np.array([1/len(tickers)]*len(tickers))
        self.ts = TimelineSampler("data/timeline_buckets.json", seed=seed)
        self.et = EventTreeEngine("data/events_catalogue.json",
                                  tickers=tickers,
                                  horizon_years=40,
                                  seed=seed)
        self.agg = DriftVolAggregator("data/asset_baseline.json", tickers)
        self.rs  = ReturnSimulator(self.weights, monthly_contrib, seed)
        self.rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------
    def run(self, store_paths: bool = False) -> Dict:
        wealths, cagrs, maxdds = [], [], []
        if store_paths:
            wealth_matrix = []
        bucket_counts = {b["name"]:0 for b in
                         json.load(open("data/timeline_buckets.json"))["buckets"]}
        fired_counts  = {}

        for _ in range(self.n_paths):
            tl = self.ts.sample_timeline()
            bucket_counts[tl["bucket"]] += 1

            ev_out = self.et.simulate(tl)
            for ev in ev_out["fired"]:
                fired_counts[ev.id] = fired_counts.get(ev.id, 0) + 1

            combo = self.agg.combine(ev_out["drift"], ev_out["volmul"])
            res   = self.rs.simulate_path(combo["mu"], combo["cov"])
        if store_paths:
            wealth_matrix.append(res["wealth_series"])

            wealths.append(res["terminal_wealth"])
            cagrs.append(res["cagr"])
            maxdds.append(res["max_drawdown"])

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
            summary["wealth_matrix"] = np.vstack(wealth_matrix).tolist()
            summary["cagrs_raw"]     = cagrs.tolist()

        return summary
    

    # -----------------------------------------------------------------
    def save(self, summary: Dict, out_dir: str | Path = "results") -> Path:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M")
        path = Path(out_dir)/f"scenario_run_{ts}.json"
        json.dump(summary, open(path,"w"), indent=2)
        return path
