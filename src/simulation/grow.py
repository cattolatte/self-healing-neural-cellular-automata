"""Growth simulation rollouts for trained Neural Cellular Automata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.evaluation.metrics import compute_mse
from src.simulation.seed import create_seed_state


def run_growth_experiment(
    model: nn.Module,
    target: torch.Tensor,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Execute a seed-to-target growth rollout and evaluate morphogenesis quality.

    Args:
        model: The trained NeuralCellularAutomaton model.
        target: Normalized RGB target tensor in ``(1, 3, H, W)`` layout.
        config: Resolved experiment configuration mapping.
        device: Execution device for tensors.

    Returns:
        A dictionary containing the evaluated metrics and the final evolved state.
    """
    model.eval()

    # 1. Initialize Seed
    seed_state = create_seed_state(
        grid_size=config["model"]["grid_size"],
        channels=config["model"]["channels"],
        batch_size=1,
        device=device,
    )

    rollout_steps = config["evaluation"]["rollout_steps"]

    # 2. Rollout NCA (No Gradients for Evaluation)
    with torch.no_grad():
        # The NCA forward pass computes evolution through local interactions
        evolved_state = model(seed_state, steps=rollout_steps)

    # 3. Evaluate Morphogenesis Quality
    visible_channels = config["model"]["visible_channels"]
    mse_value = compute_mse(
        state=evolved_state,
        target=target,
        visible_channels=visible_channels,
    )

    return {
        "mse": mse_value,
        "final_state": evolved_state.cpu(),
    }