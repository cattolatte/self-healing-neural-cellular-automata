"""Shared neural update rule for local neural cellular automata dynamics."""

from __future__ import annotations

import torch
from torch import nn

from src.data.preprocess import validate_tensor


class UpdateRule(nn.Module):
    """Predict a local state delta from perception features using 1x1 convolutions."""

    def __init__(
        self,
        perception_channels: int,
        state_channels: int,
        hidden_channels: int | None = None,
    ) -> None:
        """Initialize the shared pointwise neural update network.
        Args:
            perception_channels: Number of input perception features.
            state_channels: Number of predicted state-delta channels.
            hidden_channels: Width of the intermediate pointwise layer. When
                omitted, the perception feature count is used.
        Raises:
            ValueError: If any channel count is not positive.
        """
        super().__init__()
        _validate_positive_integer(perception_channels, "perception_channels")
        _validate_positive_integer(state_channels, "state_channels")
        
        if hidden_channels is None:
            hidden_channels = perception_channels
            
        _validate_positive_integer(hidden_channels, "hidden_channels")
        
        self.perception_channels = perception_channels
        self.state_channels = state_channels
        
        self.network = nn.Sequential(
            nn.Conv2d(perception_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, state_channels, kernel_size=1),
        )

        # FIX: Zero-initialize the final layer so the model starts as an identity function
        with torch.no_grad():
            self.network[2].weight.data.fill_(0.0)
            self.network[2].bias.data.fill_(0.0)

    def forward(self, perception: torch.Tensor) -> torch.Tensor:
        """Predict residual state deltas from local perception features.

        Args:
            perception: Float32 tensor in ``(B, perception_channels, H, W)``
                layout.

        Returns:
            A float32 state delta in ``(B, state_channels, H, W)`` layout.
        """
        validate_tensor(
            perception,
            name="perception tensor",
            channels=self.perception_channels,
        )
        return self.network(perception)


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
