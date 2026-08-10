"""Quantitative evaluation metrics for structural persistence."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any


def evaluate_persistence(
    mse_sequence: Sequence[float],
    stability_threshold: float = 0.05,
) -> dict[str, Any]:
    """Calculate stability metrics for an extended persistence rollout.

    Args:
        mse_sequence: Sequence of Mean Squared Error values recorded at
            intervals during an extended NCA rollout.
        stability_threshold: The maximum MSE allowed before the structure
            is considered to have collapsed or degraded into noise.

    Returns:
        A dictionary containing persistence metrics: final_mse, drift,
        variance, maximum error, and a boolean stability flag.

    Raises:
        ValueError: If the sequence is empty or contains negative values.
    """
    if not mse_sequence:
        raise ValueError("MSE sequence cannot be empty.")

    for val in mse_sequence:
        if val < 0.0:
            raise ValueError("MSE values cannot be negative.")

    final_mse = float(mse_sequence[-1])
    start_mse = float(mse_sequence[0])

    # Drift: Change in error from the start of the persistence phase to the end.
    # Positive drift indicates degradation; negative drift indicates stabilization.
    drift = final_mse - start_mse

    # Variance: Measures structural oscillation or instability over time.
    if len(mse_sequence) > 1:
        variance = statistics.variance(mse_sequence)
    else:
        variance = 0.0

    max_mse = max(mse_sequence)

    # Structure is stable if it never breached the acceptable error threshold
    # during the entire extended simulation.
    is_stable = bool(max_mse <= stability_threshold)

    return {
        "final_mse": final_mse,
        "drift": drift,
        "variance": variance,
        "max_mse": max_mse,
        "is_stable": is_stable,
    }