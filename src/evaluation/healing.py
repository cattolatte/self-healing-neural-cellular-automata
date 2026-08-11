"""Quantitative evaluation metrics for self-healing capability."""

from __future__ import annotations

from typing import Any



def evaluate_healing_performance(
    pre_damage_mse: float,
    post_damage_mse: float,
    final_recovery_mse: float,
) -> dict[str, Any]:
    """Calculate aggregate performance metrics for a healing rollout.

    Args:
        pre_damage_mse: Baseline error before damage was applied.
        post_damage_mse: Error immediately after the wound mask was applied.
        final_recovery_mse: Error after the recovery rollout completed.

    Returns:
        A dictionary containing the damage severity, recovery efficiency,
        and a boolean flag indicating if full structural recovery was achieved.

    Raises:
        ValueError: If any MSE value is negative.
    """
    _validate_non_negative_mse(pre_damage_mse, "pre_damage_mse")
    _validate_non_negative_mse(post_damage_mse, "post_damage_mse")
    _validate_non_negative_mse(final_recovery_mse, "final_recovery_mse")

    # How much did the wound increase the error relative to the baseline?
    damage_severity = post_damage_mse - pre_damage_mse

    # How much error was eliminated during the recovery rollout?
    recovered_error = post_damage_mse - final_recovery_mse

    # Did the wound actually damage the tissue? Avoid division by zero.
    if damage_severity <= 0:
        recovery_efficiency = 0.0
    else:
        # 1.0 means perfect recovery to pre-damage baseline. 
        # < 0.0 means the model continued to degrade after damage.
        recovery_efficiency = recovered_error / damage_severity

    # Is the final structure practically indistinguishable from the target?
    # (Threshold is typically tuned based on specific experiment configurations)
    full_recovery_achieved = bool(final_recovery_mse < 0.05)

    return {
        "damage_severity": damage_severity,
        "recovery_efficiency": recovery_efficiency,
        "full_recovery_achieved": full_recovery_achieved,
    }


def _validate_non_negative_mse(value: float, name: str) -> None:
    if not isinstance(value, float | int) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    if value < 0.0:
        raise ValueError(f"{name} cannot be negative.")