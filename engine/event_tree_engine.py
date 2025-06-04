# AI_Scenario_Sim/engine/event_tree_engine.py
# -------------------------------------------
"""
EventTreeEngine
===============
• Loads Event objects.
• Given a timeline (ai, agi, asi) and RNG, decides which events fire.
• Respects each event’s `condition` list (prerequisites).
• Returns:
    - fired: List[Event]       (events that actually fired)
    - drift: np.ndarray        (shape: horizon × n_assets) additive drifts
    - volmul: np.ndarray       (shape: horizon) multiplicative vol factors
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
        # Load all events from JSON
        self.events = self._load_events(events_path)
        self.tickers = tickers
        self.horizon = horizon_years
        self.rng = np.random.default_rng(seed)

        # Map id → Event for quick lookup
        self._by_id: Dict[str, Event] = {e.id: e for e in self.events}

        # Identify root events (condition == ["ALWAYS"])
        self._roots: Dict[str, List[Event]] = {}
        for ev in self.events:
            if ev.condition == ["ALWAYS"]:
                self._roots.setdefault(ev.stage, []).append(ev)

        # Validate that every referenced child exists
        missing = []
        for ev in self.events:
            for (child_id, _) in ev.next_events:
                if child_id not in self._by_id:
                    missing.append(f"{ev.id} → {child_id}")
        if missing:
            raise ValueError(
                "EventTreeEngine: missing event definitions for:\n  "
                + "\n  ".join(missing)
            )

    def _load_events(self, path: str | Path) -> List[Event]:
        with open(path, "r") as f:
            raw = json.load(f)
        return [Event(**e) for e in raw]

    def simulate(self, timeline: Dict[str, int]) -> Dict:
        """
        Simulate which events fire given a timeline.
        Returns a dict:
            'fired'  : list of Event objects (in firing order)
            'drift'  : np.ndarray shape (horizon, n_assets) (additive)
            'volmul' : np.ndarray shape (horizon,) (multiplicative factor)
        """
        n = len(self.tickers)
        drift = np.zeros((self.horizon, n))
        vol   = np.zeros(self.horizon)
        fired: List[Event] = []

        # Helper to apply an event's drifts and vol from its start onward
        def apply_event(event: Event, start_year: int):
            year = start_year + event.year_offset
            if year >= self.horizon:
                return
            # Add drift deltas for this and subsequent years
            indices = [self.tickers.index(t) for t in event.delta_drift]
            drift[year:, indices] += np.array(list(event.delta_drift.values())) / 100
            # Add volatility delta for this and subsequent years
            vol[year:] += event.delta_vol

        # Determine the calendar year when each stage begins
        stage_start = {
            "Pre-AGI":     timeline["ai"],
            "AGI-Rollout": timeline["agi"],
            "Self-Improving": timeline["asi"],
        }

        # 1️⃣ Enqueue all root events whose base_prob succeeds
        stack: List[Event] = []
        for stage, roots in self._roots.items():
            for ev in roots:
                if self.rng.random() <= ev.base_prob:
                    stack.append(ev)

        # 2️⃣ Depth-first traversal, respecting conditions
        while stack:
            ev = stack.pop()
            if ev in fired:
                continue  # already processed

            # Check that all prerequisites (ev.condition) are satisfied
            # (skip "ALWAYS" since roots were pre-filtered)
            if ev.condition != ["ALWAYS"]:
                prerequisites_met = all(
                    cond_id in {e.id for e in fired} for cond_id in ev.condition
                )
                if not prerequisites_met:
                    continue  # cannot fire until all prerequisites have fired

            # Fire this event
            fired.append(ev)
            # Apply its drift/vol starting at its stage start year
            start_yr = stage_start.get(ev.stage, 0)
            apply_event(ev, start_yr)

            # Enqueue children based on their conditional probabilities
            for child_id, prob in ev.next_events:
                if self.rng.random() <= prob:
                    child = self._by_id[child_id]
                    stack.append(child)

        return {
            "fired": fired,
            "drift": drift,
            "volmul": 1 + vol   # convert additive vol to multiplicative factor
        }


# ---------------------------------- #
# if run as a script for testing:
# ---------------------------------- #
if __name__ == "__main__":
    from engine.timeline_sampler import TimelineSampler

    # Example tickers (replace with actual list as needed)
    tickers = ["PACW","SEMI","VAGS","EMIM","XAIX","ITPS",
               "IWFQ","RBTX","WLDS","MINV","IWFV","EQQQ",
               "MEUD","PRIJ","XCX5","WDEP","LCUK","WCOM",
               "ISPY","NUCG","SGLN"]

    ts = TimelineSampler(
        Path(__file__).parents[1] / "data" / "timeline_buckets.json",
        seed=42
    )
    tl = ts.sample_timeline()
    print("Sampled timeline:", tl)

    et = EventTreeEngine(
        Path(__file__).parents[1] / "data" / "events_catalogue.json",
        tickers=tickers,
        seed=42
    )
    out = et.simulate(tl)
    print("Fired events:", [e.id for e in out["fired"]])
    print("Drift matrix shape:", out["drift"].shape)
    print("Vol multipliers:", out["volmul"][:5])
