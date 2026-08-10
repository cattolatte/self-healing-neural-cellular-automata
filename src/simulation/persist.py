"""Persistence simulation rollouts for trained Neural Cellular Automata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.evaluation.metrics import compute_mse


def run_persistence_experiment(
    model: nn.Module,
    initial_state: torch.Tensor,
    target: torch.Tensor,
    config: Mapping[str, Any],
    *,
    persistence_steps: int = 1000,
    evaluation_interval: int = 50,
) -> dict[str, Any]:
    """Execute a long rollout to evaluate structural persistence.

    Args:
        model: The trained NeuralCellularAutomaton model.
        initial_state: The grown state tensor to test for stability.
        target: Normalized RGB target tensor in ``(1, 3, H, W)`` layout.
        config: Resolved experiment configuration mapping.
        persistence_steps: Total number of steps to simulate for the persistence test.
        evaluation_interval: Number of steps between metric recordings.

    Returns:
        A dictionary containing the sequence of recorded MSE values and the
        final state tensor after the extended rollout.
    """
    model.eval()
    visible_channels = config["model"]["visible_channels"]

    # Detach and clone to ensure we do not retain massive BPTT graphs in memory
    current_state = initial_state.detach().clone()
    mse_sequence = []

    with torch.no_grad():
        # Record the initial baseline MSE at step 0
        initial_mse = compute_mse(
            state=current_state,
            target=target,
            visible_channels=visible_channels,
        )
        mse_sequence.append(initial_mse)

        # Evolve the state in chunks to minimize Python iteration overhead
        for step in range(0, persistence_steps, evaluation_interval):
            chunk_steps = min(evaluation_interval, persistence_steps - step)
            current_state = model(current_state, steps=chunk_steps)

            mse = compute_mse(
                state=current_state,
                target=target,
                visible_channels=visible_channels,
            )
            mse_sequence.append(mse)

    return {
        "mse_sequence": mse_sequence,
        "final_state": current_state.cpu(),
    }