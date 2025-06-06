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
    - triggered_ep3_event_id: str | None (NEW: ID of the last triggered Post-ASI event, if any)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import json
import numpy as np
from pathlib import Path
from .event_schema import Event

class EventTreeEngine:
    def __init__(self,
                 events_path: str | Path,
                 tickers: List[str],
                 horizon_years: int = 40,
                 seed: int | None = None):
        # Load all events from JSON
        self.events = self._load_events(events_path)
        self.tickers = tickers
        self.horizon = horizon_years
        # seed=None → nondeterministic RNG
        self.rng = np.random.default_rng(seed)

        # Map id → Event for quick lookup
        self._by_id: Dict[str, Event] = {e.id: e for e in self.events}

        # Identify root events (condition == ["ALWAYS"])
        self._root_events = [e for e in self.events if e.condition == ["ALWAYS"]]

        # NEW: Store the last triggered E-P3 event ID
        self.triggered_ep3_event_id: str | None = None

    def _load_events(self, events_path: str | Path) -> List[Event]:
        """Loads events from a JSON file."""
        with open(events_path, "r") as f:
            raw_events = json.load(f)
        return [Event(**data) for data in raw_events]

    def simulate_events(self,
                        timeline: Dict[str, int | None],
                        event_stats_map: Dict[str, Tuple[float, float]], # Pass event_stats_map here
                        current_year: int
                        ) -> Dict[str, Any]:
        """
        Simulates events for a given year based on timeline and current state.

        Args:
            timeline: Dictionary with 'ai_year', 'agi_year', 'asi_year'.
            event_stats_map: Map of event ID to (mu, sigma) from event_stats.json.
            current_year: The current year of the simulation.

        Returns:
            A dictionary containing:
                - "fired": List of Event objects that fired in the current year.
                - "drift": numpy array of additive drifts for assets.
                - "volmul": Multiplicative volatility factor.
                - "triggered_ep3_event_id": The ID of the last triggered E-P3 event, if any.
        """
        fired_events_this_year: List[Event] = []
        # Initialize drifts and volatility for this year
        year_drift = np.zeros(len(self.tickers))
        year_vol_change = 0.0 # additive change

        # --- NEW: Check if an E-P3 event has already been persistently triggered ---
        # If an E-P3 event was triggered in a previous year and is meant to persist,
        # we do not re-run the event tree for new E-P3 possibilities.
        # This simplifies logic, assuming once a 'final' AGI outcome is reached, it sticks.
        if self.triggered_ep3_event_id is not None:
             # Find the event object for the persistently triggered E-P3 event
            triggered_ep3_event = self._by_id.get(self.triggered_ep3_event_id)
            if triggered_ep3_event:
                # Apply its effects if it's the current year for its year_offset
                # or if its effects are ongoing. For simplicity, we assume its effects
                # are continuously applied once triggered in this model's context.
                # The actual application of its drift/volmul will be handled by
                # drift_vol_aggregator based on the returned triggered_ep3_event_id
                # and the macro state mapping.
                pass # The impact is handled via the macro state selected by macro.py
            return {
                "fired": [], # No new events fire if E-P3 is persistent
                "drift": year_drift,
                "volmul": 1.0, # Will be overridden by macro state effects
                "triggered_ep3_event_id": self.triggered_ep3_event_id
            }

        # Events that have already fired in previous years and their effects might persist
        # (This engine focuses on yearly triggers, persistence handled by drift_vol_aggregator)
        # For simplicity, we assume events fire in the year they occur, not over multiple years here.

        # Decide which events to consider for firing this year
        # This requires tracking which events have fired in previous years
        # to correctly evaluate conditions and prevent re-firing of one-off events.
        # A more robust system would involve self.fired_events_history: List[Event] = []
        # for simplicity, let's assume events are processed once they are applicable.

        # For this example, we will focus on what events *can* fire this year
        # based on timeline and conditions.
        
        # Events are usually designed to fire once. The effects might persist.
        # This method is called per year. We need to maintain a history of fired events
        # across simulation years to correctly check conditions.
        
        # In a complete simulation, you'd have a way to query `all_fired_events` from the simulation state
        # that persist across years within a single simulation run.
        # For now, let's assume `EventTreeEngine` is reset yearly or its state is managed externally.
        # For the purpose of this modification, we'll assume the caller (e.g., main.py)
        # provides context on what events have *already* fired.
        
        # Since this method is called per year, we need a way to know what was fired previously.
        # Let's assume a 'fired_events_history' parameter is passed to simulate_events
        # or stored as an instance variable across years if EventTreeEngine is not re-instantiated.
        # For this modification, let's assume the context of `self.fired_events_history`
        # which would be updated by the caller. This class currently doesn't manage that.

        # To simplify and show the logic, we will assume self._fired_events_history exists
        # and is updated externally or passed in a more complex setup.
        # For this specific modification, we are adding the EP3 tracking.
        
        # Events are processed in stages and depend on timeline.
        # The original code's _fire_events seems to run the full tree once for a given timeline.
        # We need to adapt it to run YEARLY and track what has fired.

        # Let's assume the timeline already gives us years for AI/AGI/ASI
        ai_year = timeline.get("ai_year")
        agi_year = timeline.get("agi_year")
        asi_year = timeline.get("asi_year")

        # Events that might fire this year based on their year_offset
        # We need a set of active events, typically updated yearly.
        # This would usually be managed by the main simulation loop that calls this engine.
        # For now, let's assume 'fired_events_history' contains IDs of all events fired up to prev year.
        # This is a conceptual placeholder for a more complex state management in the full simulation.
        # For this example, we are simulating a single year's event triggering.
        
        # This is an adaptation from the typical use of EventTreeEngine which might
        # run through the whole tree once. Here, it's run per year.
        
        # This revised simulate_events needs to:
        # 1. Take a list of already_fired_event_ids to manage dependencies.
        # 2. Return new events fired this year + updated list of all fired events.

        # Let's assume `EventTreeEngine` is initialized once per simulation run,
        # and `self._fired_events_history` tracks all events fired in *this* run.
        if not hasattr(self, '_fired_events_history'):
            self._fired_events_history = set() # Store IDs of all events fired in this run

        # Identify candidate events for this year
        candidate_events = []
        for event in self.events:
            # Check if event has already fired in this simulation run
            if event.id in self._fired_events_history:
                continue

            # Check year offset relative to timeline anchors (AI, AGI, ASI)
            event_base_year = 0
            if event.stage == "Pre-AGI":
                # Pre-AGI events often tied to start of simulation or early AI milestones
                event_base_year = 0 # Relative to sim start
            elif event.stage == "AGI-Rollout" and agi_year is not None:
                event_base_year = agi_year # Relative to AGI breakthrough
            elif event.stage == "Self-Improving" and agi_year is not None:
                event_base_year = agi_year # Relative to AGI breakthrough
            elif event.stage == "Post-ASI" and asi_year is not None:
                event_base_year = asi_year # Relative to ASI emergence

            # Check if event is scheduled for this year
            if current_year == event_base_year + event.year_offset:
                # Check conditions
                conditions_met = True
                if "ALWAYS" not in event.condition:
                    for cond_id in event.condition:
                        if cond_id not in self._fired_events_history:
                            conditions_met = False
                            break
                if conditions_met:
                    candidate_events.append(event)
        
        # Shuffle candidates to randomize order if multiple can fire
        self.rng.shuffle(candidate_events)

        # Decide which events actually fire this year
        newly_fired_events_this_year = []
        for event in candidate_events:
            # If the event has a base_prob and a condition, evaluate if it fires
            if event.base_prob > 0 and self.rng.random() <= event.base_prob:
                newly_fired_events_this_year.append(event)
                self._fired_events_history.add(event.id) # Mark as fired for this run

                # Aggregate drift and vol from this event
                for ticker, drift_val in event.delta_drift.items():
                    try:
                        idx = self.tickers.index(ticker)
                        year_drift[idx] += drift_val / 100.0 # Convert to decimal
                    except ValueError:
                        print(f"Warning: Ticker {ticker} in event {event.id} not found in asset list.")
                year_vol_change += event.delta_vol # Additive vol change

                # --- NEW: Track triggered E-P3 event ---
                if event.stage == "Post-ASI":
                    self.triggered_ep3_event_id = event.id
                    # Once an E-P3 event fires, it typically becomes the dominant long-term outcome.
                    # We might want to stop processing other events that could lead to conflicting E-P3s
                    # but for this specific method, we simply set the ID.
                    # The main simulation loop will use this to manage future macro states.

                # Enqueue children based on their conditional probabilities for THIS year
                # This logic is simplified for yearly step-by-step processing.
                # In a full simulation, child events might fire in subsequent years.
                # Here, we only enqueue those that could fire *this* year based on their offset.
                # For more robust event propagation, this would usually be handled by the caller
                # which adds these children to future years' candidate pool.
                pass # Event Tree Engine itself doesn't need to enqueue for future years here,
                     # it just returns what fired this year.

        # The volmul is 1 + total_additive_vol_change
        final_vol_multiplier = 1.0 + year_vol_change

        return {
            "fired": newly_fired_events_this_year,
            "drift": year_drift,
            "volmul": final_vol_multiplier,
            "triggered_ep3_event_id": self.triggered_ep3_event_id # Return the tracked E-P3 ID
        }

# ---------------------------------- #
# if run as a script for testing (remains largely the same, but note the changes needed in caller)
# ---------------------------------- #
if __name__ == "__main__":
    from engine.timeline_sampler import TimelineSampler
    from engine.event_stats import EVENT_STATS_MAP # Assuming you have a way to load this

    # Example tickers (replace with actual list as needed)
    tickers = ["PACW","SEMI","VAGS","EMIM","XAIX","ITPS",
               "IWFQ","RBTX","WLDS","MINV","IWFV","EQQQ",
               "MEUD","PRIJ","XCX5","WDEP","LCUK","WCOM",
               "ISPY","NUCG","SGLN"]

    ts = TimelineSampler(
        Path(__file__).parents[1] / "data" / "timeline_buckets.json"
    )
    tl = ts.sample_timeline()
    print("Sampled timeline:", tl)

    # Instantiate EventTreeEngine
    # In a full simulation, this would be part of a loop.
    # For testing, we'll simulate a few years.
    horizon_years = 10
    event_engine = EventTreeEngine(
        Path(__file__).parents[1] / "data" / "events_catalogue.json",
        tickers,
        horizon_years=horizon_years
    )

    # Simulate year by year
    agi_breakthrough_year = tl.get("agi_year")
    current_ep3_event = None # This will be passed to macro.py

    print("\n--- Simulating Events Year by Year ---")
    for year in range(horizon_years):
        print(f"\nYear {year}:")
        
        # Pass the current_ep3_event to the simulate_events for persistence logic
        # and update it based on the return
        event_results = event_engine.simulate_events(
            timeline=tl,
            event_stats_map=EVENT_STATS_MAP, # Assuming EVENT_STATS_MAP is available
            current_year=year
        )
        
        newly_fired = event_results["fired"]
        year_drift = event_results["drift"]
        year_volmul = event_results["volmul"]
        
        # Update the current_ep3_event for next year's simulation based on what fired this year
        current_ep3_event = event_results["triggered_ep3_event_id"]

        if newly_fired:
            print(f"  Fired Events: {[e.id for e in newly_fired]}")
        else:
            print("  No new events fired this year.")
        
        print(f"  Aggregate Drift this year: {year_drift}")
        print(f"  Vol Multiplier this year: {year_volmul}")
        print(f"  Current E-P3 Event ID (for macro state): {current_ep3_event}")
        
        # Example of how macro state would be determined (conceptual, requires macro.py)
        # from . import macro # Assuming macro.py is in the same engine directory
        # current_macro_state = macro.get_current_macro_state(
        #     current_year=year,
        #     agi_breakthrough_year=agi_breakthrough_year,
        #     triggered_ep3_event_id=current_ep3_event
        # )
        # print(f"  Determined Macro State: {current_macro_state}")
