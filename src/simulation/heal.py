"""Healing simulation rollouts for trained Neural Cellular Automata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.evaluation.metrics import compute_mse
from src.simulation.seed import create_seed_state
from src.simulation.wound import apply_wound


def run_healing_experiment(
    model: nn.Module,
    target: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Execute a grow-wound-recover rollout and evaluate self-healing quality.

    Args:
        model: The trained NeuralCellularAutomaton model.
        target: Normalized RGB target tensor in ``(1, 3, H, W)`` layout.
        config: Resolved experiment configuration mapping.
        device: Execution device for tensors.

    Returns:
        A dictionary containing evaluated metrics and key state snapshots
        (grown, damaged, and recovered tensors) for visualization.
    """
    model.eval()

    grid_size = config["model"]["grid_size"]
    channels = config["model"]["channels"]
    visible_channels = config["model"]["visible_channels"]
    rollout_steps = config["evaluation"]["rollout_steps"]

    wound_shape = config["damage"]["shape"]
    wound_fraction = config["damage"]["fraction"]

    current_state = create_seed_state(
        grid_size=grid_size,
        channels=channels,
        batch_size=1,
        device=device,
    )

    with torch.no_grad():
        # Phase 1: Pre-damage Growth
        grown_state = model(current_state, steps=rollout_steps)
        pre_damage_mse = compute_mse(
            state=grown_state,
            target=target,
            visible_channels=visible_channels,
        )

        # Phase 2: Explicit Damage Event
        damaged_state = apply_wound(
            grown_state,
            shape=wound_shape,
            fraction=wound_fraction,
        )
        post_damage_mse = compute_mse(
            state=damaged_state,
            target=target,
            visible_channels=visible_channels,
        )

        # Phase 3: Recovery Rollout
        recovered_state = model(damaged_state, steps=rollout_steps)
        final_recovery_mse = compute_mse(
            state=recovered_state,
            target=target,
            visible_channels=visible_channels,
        )

    return {
        "pre_damage_mse": pre_damage_mse,
        "post_damage_mse": post_damage_mse,
        "final_recovery_mse": final_recovery_mse,
        "grown_state": grown_state.cpu(),
        "damaged_state": damaged_state.cpu(),
        "recovered_state": recovered_state.cpu(),
    }