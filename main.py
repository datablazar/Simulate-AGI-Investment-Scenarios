from engine.timeline_sampler import TimelineSampler

ts = TimelineSampler("data/timeline_buckets.json")
tl = ts.sample_timeline()
print("Timeline draw:", tl)
print("Stage in year 12:", ts.stage_for_year(tl, 12))

from engine.timeline_sampler import TimelineSampler
from engine.event_tree_engine import EventTreeEngine
from engine.event_schema import Event
import json, numpy as np

tickers = list(json.load(open("data/asset_baseline.json"))["mu"].keys())

ts = TimelineSampler("data/timeline_buckets.json", seed=1)
tl = ts.sample_timeline()
print("Timeline →", tl)

et = EventTreeEngine("data/events_catalogue.json", tickers, seed=1)
r = et.simulate(tl)
print("Fired IDs :", [e.id for e in r["fired"]])
print("Drift delta first 5 yrs:\n", r["drift"][:5])
print("Vol multiplier first 5 yrs:", r["volmul"][:5])

from engine.drift_vol_aggregator import DriftVolAggregator

agg = DriftVolAggregator("data/asset_baseline.json", tickers)
combo = agg.combine(r["drift"], r["volmul"])

print("Year-0 mu  :", combo["mu"][0][:5])       # first 5 assets
print("Year-0 cov :", combo["cov"][0][:3, :3])  # 3×3 slice

from engine.timeline_sampler import TimelineSampler
from engine.event_tree_engine import EventTreeEngine
from engine.drift_vol_aggregator import DriftVolAggregator
from engine.return_simulator import ReturnSimulator
import json

# ---------- load master asset list & weights ------------------------
base = json.load(open("data/asset_baseline.json"))
tickers = list(base["mu"].keys())
weights = np.array([1/len(tickers)]*len(tickers))   # equal-weight demo

# ---------- 1) sample capability timeline --------------------------
ts = TimelineSampler("data/timeline_buckets.json", seed=4)
timeline = ts.sample_timeline()

# ---------- 2) walk event tree -------------------------------------
et = EventTreeEngine("data/events_catalogue.json",
                     tickers=tickers,
                     horizon_years=40,
                     seed=4)
event_out = et.simulate(timeline)

# ---------- 3) build yearly mu, cov --------------------------------
agg = DriftVolAggregator("data/asset_baseline.json", tickers)
combo = agg.combine(event_out["drift"], event_out["volmul"])

# ---------- 4) run return simulator --------------------------------
rs = ReturnSimulator(weights, monthly_contrib=100, seed=4)
result = rs.simulate_path(combo["mu"], combo["cov"])

print("Timeline :", timeline)
print("Events   :", [e.id for e in event_out["fired"]])
print("Terminal wealth £:", round(result["terminal_wealth"], 0))
print("CAGR % :", round(result["cagr"]*100, 2))
print("Max draw-down %:", round(result["max_drawdown"]*100, 1))

import json, numpy as np
from pathlib import Path
from engine.batch_runner import BatchRunner

from engine.batch_runner import BatchRunner
from engine.reporting    import save_summary_csv, fan_chart, cagr_histogram
import json, numpy as np, time, os

# ---- ticker list ----------------------------
base = json.load(open("data/asset_baseline.json"))
tickers = list(base["mu"].keys())

# ---- run Monte-Carlo with paths -------------
runner = BatchRunner(n_paths=20_000,
                     tickers=tickers,
                     monthly_contrib=100,
                     seed=0)
summary = runner.run(store_paths=True)
run_file = runner.save(summary)

# ---- save CSV master log --------------------
csv_path = save_summary_csv(summary, "results")

# ---- make plots -----------------------------
stamp = time.strftime("%Y%m%d_%H%M")
fan_chart( np.array(summary["wealth_matrix"]),
           f"results/fan_chart_{stamp}.png")
cagr_histogram( np.array(summary["cagrs_raw"]),
                f"results/cagr_hist_{stamp}.png")

print("\nMonte-Carlo complete ✅")
print("Summary JSON :", run_file)
print("Master CSV   :", csv_path)
print("Fan chart    :", f'results/fan_chart_{stamp}.png')
print("CAGR hist    :", f'results/cagr_hist_{stamp}.png')
