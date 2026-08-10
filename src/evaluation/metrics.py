"""Quantitative metrics for Neural Cellular Automata evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional

from src.data.preprocess import validate_tensor
from src.model.state import DEFAULT_VISIBLE_CHANNELS, extract_visible_channels


def compute_mse(
    state: torch.Tensor,
    target: torch.Tensor,
    visible_channels: Sequence[int] = DEFAULT_VISIBLE_CHANNELS,
) -> float:
    """Compute the Mean Squared Error between the visible state and a target.

    Args:
        state: Evolved automata state tensor in ``(B, C, H, W)`` layout.
        target: Normalized RGB target tensor in ``(1, 3, H, W)`` or 
            ``(B, 3, H, W)`` layout.
        visible_channels: State-channel indices to compare against the target.

    Returns:
        The scalar Mean Squared Error (MSE) value as a Python float.

    Raises:
        ValueError: If the target spatial dimensions do not match the state,
            or if the number of visible channels does not match the target.
    """
    validate_tensor(state, name="evaluation state")
    validate_tensor(target, name="evaluation target", channels=len(visible_channels))

    if state.shape[2:] != target.shape[2:]:
        raise ValueError(
            f"Spatial dimensions of state {tuple(state.shape[2:])} and "
            f"target {tuple(target.shape[2:])} must match."
        )

    visible_state = extract_visible_channels(state, visible_channels)

    if target.shape[0] != visible_state.shape[0]:
        if target.shape[0] != 1:
            raise ValueError(
                "Target batch size must be 1 or match the state batch size."
            )
        target = target.expand_as(visible_state)

    mse_tensor = functional.mse_loss(visible_state, target)
    return float(mse_tensor.item())