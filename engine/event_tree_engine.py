"""
EventTreeEngine
===============

• Loads Event objects.
• Given a timeline (ai, agi, asi) and RNG, decides which events fire.
• Returns:
      - fired_events: List[str]  (event IDs in firing order)
      - yearly_drift  [years] × [assets]
      - yearly_volmul [years]   scalar
"""

from __future__ import annotations
from typing import Dict, List
import json
import numpy as np
from pathlib import Path
from .event_schema import Event

class EventTreeEngine:
    def __init__(self,
                 events_path: str | Path,
                 tickers: List[str],
                 horizon_years: int = 40,
                 seed: int = 0):
        self.events = self._load_events(events_path)
        self.tickers = tickers
        self.horizon = horizon_years
        self.rng = np.random.default_rng(seed)

        # Build lookup by id for quick access
        self._by_id: Dict[str, Event] = {e.id: e for e in self.events}

        # Group root events by stage
        self._roots: Dict[str, List[Event]] = {}
        for ev in self.events:
            if ev.condition == ["ALWAYS"]:
                self._roots.setdefault(ev.stage, []).append(ev)

    # ------------------------------------------------------------------
    def _load_events(self, path: str | Path) -> List[Event]:
        with open(path, "r") as f:
            raw = json.load(f)
        return [Event(**e) for e in raw]

    # ------------------------------------------------------------------
    def simulate(self, timeline: Dict[str, int]) -> Dict:
        """
        Returns dict with keys:
            'fired'  : list of Event
            'drift'  : np.ndarray shape (horizon, n_assets) drift deltas
            'volmul' : np.ndarray shape (horizon,) multiplicative vol factors
        """
        n = len(self.tickers)
        drift = np.zeros((self.horizon, n))
        vol   = np.zeros(self.horizon)

        fired: List[Event] = []

        # helper to add deltas
        def apply(event: Event, start_year: int):
            nonlocal drift, vol
            year = start_year + event.year_offset
            if year >= self.horizon:
                return
            # apply delta from this year onward
            idx = [self.tickers.index(t) for t in event.delta_drift]
            drift[year:, idx] += np.array(list(event.delta_drift.values())) / 100
            vol[year:] += event.delta_vol

        # Stage start years
        stage_start = {
            "Pre-AGI": timeline["ai"],
            "AGI-Rollout": timeline["agi"],
            "Self-Improving": timeline["asi"],
        }

        # 1️⃣ handle root events for each stage
        stack: List[Event] = []
        for stage, roots in self._roots.items():
            for ev in roots:
                if self.rng.random() <= ev.base_prob:
                    stack.append(ev)

        # 2️⃣ depth-first walk through children
        while stack:
            ev = stack.pop()
            if ev in fired:
                continue
            fired.append(ev)
            apply(ev, stage_start.get(ev.stage, 0))

            # enqueue children
            for child_id, p in ev.next_events:
                if self.rng.random() <= p:
                    child = self._by_id[child_id]
                    stack.append(child)

        return {"fired": fired,
                "drift": drift,
                "volmul": 1 + vol}   # convert additive to multiplicative

# ---------------------------------------------------------------------------
# Stand-alone test (assumes data files exist)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from timeline_sampler import TimelineSampler
    tickers = list(Event.__annotations__)  # quick hack; replace with real list
    ts = TimelineSampler(Path(__file__).parents[1]/"data"/"timeline_buckets.json")
    et = EventTreeEngine(Path(__file__).parents[1]/"data"/"events_catalogue.json",
                         tickers=tickers,
                         seed=42)

    tl = ts.sample_timeline()
    out = et.simulate(tl)
    print("Fired events:", [e.id for e in out["fired"]])
    print("Drift shape :", out["drift"].shape)
