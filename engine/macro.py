"""Utilities for loading macro-state parameters used in simulations."""
from __future__ import annotations
import json
from pathlib import Path
import random

# Path to the data directory two levels up from this file
BASE = Path(__file__).resolve().parent.parent / "data"
_macro_path = BASE / "macro_states.json"

with open(_macro_path, "r") as f:
    _raw_macros = json.load(f)

# Public constants loaded from macro_states.json
MACRO_STATES = [entry["state"] for entry in _raw_macros]
MACRO_PROBS = [entry["prob"] for entry in _raw_macros] # These are now BASE probabilities, potentially overridden
# Map state -> (mu_shift_decimal, sigma_mult)
MACRO_MAP = {
    entry["state"]: (entry["mu_shift"] / 100.0, entry["sigma_mult"])
    for entry in _raw_macros
}

# --- NEW FUNCTION FOR DYNAMIC MACRO STATE SELECTION ---

def get_current_macro_state(
    current_year: int,
    agi_breakthrough_year: int | None,
    triggered_ep3_event_id: str | None
) -> str:
    """
    Determines the current macro state dynamically based on AGI events.

    Args:
        current_year: The current year of the simulation.
        agi_breakthrough_year: The year AGI breakthrough occurred (if any),
                               or None if no breakthrough has occurred yet.
        triggered_ep3_event_id: The ID of the Post-ASI (E-P3) event that has been triggered and is ongoing,
                                  or None if no such event has triggered yet.

    Returns:
        The selected macro state for the current year.
    """
    # Define mapping from E-P3 event IDs to corresponding long-term macro states.
    # This mapping must align with the E-P3 events defined in events_catalogue.json
    # and the states defined in macro_states.json.
    ep3_to_macro_state_map = {
        "E-P3-Abundance": "Abundance",
        "E-P3-Oligopoly": "Oligopoly",
        "E-P3-Managed": "Managed",
        "E-P3-Conflict": "Conflict",
        "E-P3-EquitableDistribution": "Managed" # Assuming EquitableDistribution leads to a 'Managed' economic state
    }

    # If a specific E-P3 event has been triggered and is ongoing, its corresponding macro state dominates.
    # This assumes that once an E-P3 event's macro state is triggered, it persists.
    # You could add more complex logic here (e.g., decay, transition to other E-P3s over time)
    # but for initial dynamic probabilities, a direct override is effective.
    if triggered_ep3_event_id and triggered_ep3_event_id in ep3_to_macro_state_map:
        return ep3_to_macro_state_map[triggered_ep3_event_id]
    else:
        # Before AGI breakthrough or if no specific E-P3 event has taken hold,
        # sample from the base probabilities from macro_states.json.
        # We need to adjust probabilities if 'NoBreakthrough' state should not occur after AGI.
        
        available_states = []
        available_probs = []

        # If AGI has not yet broken through, or it's the year of breakthrough,
        # we consider all base macro states.
        if agi_breakthrough_year is None or current_year < agi_breakthrough_year:
            available_states = MACRO_STATES
            available_probs = MACRO_PROBS
        else:
            # If AGI breakthrough has already happened (current_year >= agi_breakthrough_year),
            # the 'NoBreakthrough' state should typically no longer be possible.
            for i, state in enumerate(MACRO_STATES):
                if state == "NoBreakthrough":
                    continue # Exclude NoBreakthrough after AGI has occurred
                available_states.append(state)
                available_probs.append(MACRO_PROBS[i])

            # Re-normalize probabilities for the remaining states
            total_prob = sum(available_probs)
            if total_prob > 0: # Avoid division by zero if all probabilities are zero
                normalized_probs = [p / total_prob for p in available_probs]
            else:
                # Fallback: if no valid states remain (should not happen with typical setups)
                # or if probabilities sum to zero, pick a default or raise an error.
                # For robustness, we could default to 'Managed' or raise.
                # For this example, let's just return 'NoBreakthrough' as a safe, though likely incorrect, default.
                print("Warning: No valid macro states with non-zero probabilities after AGI breakthrough. Defaulting to NoBreakthrough.")
                return "NoBreakthrough"

        # If no states are available after filtering (e.g., if MACRO_STATES was empty), handle gracefully
        if not available_states:
            print("Error: No available macro states to choose from. Defaulting to NoBreakthrough.")
            return "NoBreakthrough"

        return random.choices(available_states, weights=normalized_probs, k=1)[0]
