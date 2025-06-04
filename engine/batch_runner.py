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
  results/scenario_run_YYYYMMDD_HHMM.json
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
        # Inside BatchRunner.__init__, replace your weights initialization with:

        TARGET_ALLOC = {
            "PACW": 0.25,  # Amundi Prime All Country World
            "SEMI": 0.06,  # iShares MSCI Global Semiconductors
            "VAGS": 0.05,  # Vanguard Global Aggregate Bonds
            "EMIM": 0.05,  # iShares MSCI Emerging Markets IMI
            "XAIX": 0.05,  # Xtrackers Artificial Intelligence & Big Data
            "ITPS": 0.05,  # iShares $ ITPS
            "IWFQ": 0.04,  # iShares Edge MSCI World Quality Factor
            "RBTX": 0.04,  # iShares Automation and Robotics
            "WLDS": 0.04,  # iShares MSCI World Small Cap
            "MINV": 0.04,  # iShares MSCI World Minimum Volatility
            "IWFV": 0.04,  # iShares Edge MSCI World Value Factor
            "EQQQ": 0.03,  # Invesco Nasdaq 100
            "MEUD": 0.03,  # Amundi Stoxx Europe 600
            "PRIJ": 0.03,  # Amundi Prime Japan
            "XCX5": 0.03,  # Xtrackers MSCI India Swap
            "WDEP": 0.03,  # WisdomTree Europe Defence
            "LCUK": 0.03,  # Amundi Core UK Equity All Cap
            "WCOM": 0.03,  # WisdomTree Enhanced Commodity
            "ISPY": 0.03,  # L&G Cyber Security
            "NUCG": 0.03,  # VanEck Uranium and Nuclear Technologies
            "SGLN": 0.02   # iShares Physical Gold
        }

        self.weights = np.array([TARGET_ALLOC[t] for t in tickers])

        # Create a single master RNG
        self.master_rng = np.random.default_rng(seed)

        # Initialize modules WITHOUT passing seeds
        self.ts = TimelineSampler("data/timeline_buckets.json")
        self.et = EventTreeEngine("data/events_catalogue.json",
                                  tickers=tickers,
                                  horizon_years=40)
        self.agg = DriftVolAggregator("data/asset_baseline.json", tickers)
        self.rs  = ReturnSimulator(self.weights, monthly_contrib)

        # Override each module's .rng to use the shared master RNG
        self.ts.rng = self.master_rng
        self.et.rng = self.master_rng
        self.rs.rng = self.master_rng

    # -----------------------------------------------------------------
    def run(self, store_paths: bool = False) -> Dict:
        wealths, cagrs, maxdds = [], [], []
        if store_paths:
            wealth_matrix = []

        # Prepare counters for timeline buckets and fired events
        buckets = json.load(open("data/timeline_buckets.json"))["buckets"]
        bucket_counts = {b["name"]: 0 for b in buckets}
        fired_counts  = {}

        for _ in range(self.n_paths):
            # 1) Sample timeline and count bucket
            tl = self.ts.sample_timeline()
            bucket_counts[tl["bucket"]] += 1

            # 2) Simulate event tree
            ev_out = self.et.simulate(tl)
            for ev in ev_out["fired"]:
                fired_counts[ev.id] = fired_counts.get(ev.id, 0) + 1

            # 3) Build drift & cov from aggregator
            combo = self.agg.combine(ev_out["drift"], ev_out["volmul"])

            # 4) Run return simulator
            res = self.rs.simulate_path(combo["mu"], combo["cov"])

            # 5) Store results
            if store_paths:
                wealth_matrix.append(res["wealth_series"])

            wealths.append(res["terminal_wealth"])
            cagrs.append(res["cagr"])
            maxdds.append(res["max_drawdown"])

        wealths = np.array(wealths)
        cagrs   = np.array(cagrs)
        maxdds  = np.array(maxdds)

        # Build summary dict
        summary = {
            "n_paths": self.n_paths,
            "terminal_wealth_percentiles": (
                np.percentile(wealths, [5,25,50,75,95])
                  .round(0)
                  .tolist()
            ),
            "cagr_percentiles": (
                np.percentile(cagrs, [5,25,50,75,95]) * 100
            ).round(2).tolist(),
            "max_dd_percentiles": (
                np.percentile(maxdds, [5,50,95]) * 100
            ).round(1).tolist(),
            "bucket_frequency": {
                k: round(v / self.n_paths * 100, 2)
                for k, v in bucket_counts.items()
            },
            "event_frequency": {
                k: round(v / self.n_paths * 100, 2)
                for k, v in fired_counts.items()
            }
        }

        if store_paths:
            summary["wealth_matrix"] = np.vstack(wealth_matrix).tolist()
            summary["cagrs_raw"]     = cagrs.tolist()

        return summary

    # -----------------------------------------------------------------
    def save(self, summary: Dict, out_dir: str | Path = "results") -> Path:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M")
        path = Path(out_dir) / f"scenario_run_{ts}.json"
        json.dump(summary, open(path, "w"), indent=2)
        return path
