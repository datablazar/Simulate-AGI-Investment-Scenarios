"""Utilities for loading macro-state parameters used in simulations."""
from __future__ import annotations
import json
from pathlib import Path

# Path to the data directory two levels up from this file
BASE = Path(__file__).resolve().parent.parent / "data"
_macro_path = BASE / "macro_states.json"

with open(_macro_path, "r") as f:
    _raw_macros = json.load(f)

# Public constants
MACRO_STATES = [entry["state"] for entry in _raw_macros]
MACRO_PROBS = [entry["prob"] for entry in _raw_macros]
# Map state -> (mu_shift_decimal, sigma_mult)
MACRO_MAP = {
    entry["state"]: (entry["mu_shift"] / 100.0, entry["sigma_mult"])
    for entry in _raw_macros
}
