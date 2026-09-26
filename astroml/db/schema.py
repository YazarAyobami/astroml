"""Backward-compatible re-export of ORM models (issue #571).

This module provides backward compatibility by re-exporting all ORM models
from astroml.db.models. New code should import directly from db.models.

Dependencies:
- astroml.db.models: ORM model definitions
"""

from astroml.db.models import *  # noqa: F401, F403

import math
from collections.abc import Mapping


def validate_router_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Validate router weights and return them normalised to sum to 1.0 (issue #980).

    Raises:
        ValueError: if ``weights`` is empty, a key is blank, a weight is
            not a finite non-negative number, or all weights are zero.
    """
    if not weights:
        raise ValueError("router weights must not be empty")
    for name, weight in weights.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"invalid route name: {name!r}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"weight for {name!r} must be a number, got {weight!r}")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"weight for {name!r} must be finite and >= 0, got {weight!r}")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("router weights must not all be zero")
    return {name: float(weight) / total for name, weight in weights.items()}
