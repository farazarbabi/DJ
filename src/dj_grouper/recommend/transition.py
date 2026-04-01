"""Transition compatibility scoring between track structures."""

from __future__ import annotations

# Flow type compatibility matrix: (from, to) -> bonus/penalty
_FLOW_COMPAT: dict[tuple[str, str], float] = {
    ("G", "G"): 0.15, ("G", "H"): 0.10, ("G", "D"): 0.0,  ("G", "B"): 0.05, ("G", "L"): 0.10,
    ("H", "G"): 0.10, ("H", "H"): 0.15, ("H", "D"): -0.10, ("H", "B"): 0.0,  ("H", "L"): 0.10,
    ("D", "G"): 0.0,  ("D", "H"): -0.10, ("D", "D"): 0.05, ("D", "B"): 0.0,  ("D", "L"): -0.10,
    ("B", "G"): 0.05, ("B", "H"): 0.0,  ("B", "D"): 0.0,  ("B", "B"): 0.10, ("B", "L"): 0.0,
    ("L", "G"): 0.10, ("L", "H"): 0.10, ("L", "D"): -0.10, ("L", "B"): 0.0,  ("L", "L"): 0.15,
}


def flow_compatibility(flow_a: str | None, flow_b: str | None) -> float:
    """Score the compatibility between two flow types."""
    a = flow_a or "H"
    b = flow_b or "H"
    return _FLOW_COMPAT.get((a, b), 0.0)


def intro_compatibility(bars_a: int | None, bars_b: int | None) -> float:
    """Score the compatibility between intro bar lengths."""
    a = bars_a or 32
    b = bars_b or 32
    diff = abs(a - b)
    if diff == 0:
        return 0.05
    if diff <= 16:
        return 0.0
    return -0.05  # large mismatch (e.g., 16 vs 64)


def structure_compatibility(
    flow_a: str | None,
    bars_a: int | None,
    flow_b: str | None,
    bars_b: int | None,
) -> float:
    """Combined structure compatibility score."""
    return flow_compatibility(flow_a, flow_b) + intro_compatibility(bars_a, bars_b)
