"""Reproducible damage and wound simulation operations."""

from __future__ import annotations

from typing import Literal

import torch

from src.data.preprocess import validate_tensor

WoundShape = Literal["circle", "square", "half_right"]


def apply_wound(
    state: torch.Tensor,
    *,
    shape: WoundShape = "circle",
    fraction: float = 0.25,
    center_row: int | None = None,
    center_col: int | None = None,
) -> torch.Tensor:
    """Apply a localized zero-mask to simulate physical tissue damage.

    Args:
        state: Automata state tensor in ``(B, C, H, W)`` layout.
        shape: Geometric shape of the wound ('circle', 'square', 'half_right').
        fraction: Fraction of the grid dimension used to scale the wound severity.
        center_row: Optional explicit row coordinate for the wound center.
        center_col: Optional explicit column coordinate for the wound center.

    Returns:
        A new state tensor with the damaged cells set to zero across all channels.

    Raises:
        ValueError: If the fraction is invalid or shape is unsupported.
    """
    validate_tensor(state, name="automata state")

    if not 0.0 < fraction <= 1.0:
        raise ValueError("Wound fraction must be strictly between 0 and 1.")

    batch_size, channels, height, width = state.shape
    device = state.device

    # Default to the absolute center of the grid if no coordinates are provided
    row_origin = height // 2 if center_row is None else center_row
    col_origin = width // 2 if center_col is None else center_col

    survivor_mask = _generate_survivor_mask(
        height=height,
        width=width,
        shape=shape,
        fraction=fraction,
        row_origin=row_origin,
        col_origin=col_origin,
        device=device,
    )

    # Multiply state by the mask to physically destroy the target cells
    # without breaking the PyTorch autograd graph for the surviving cells.
    return state * survivor_mask.to(dtype=state.dtype)


def _generate_survivor_mask(
    height: int,
    width: int,
    shape: str,
    fraction: float,
    row_origin: int,
    col_origin: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a 2D boolean mask indicating which cells survive the damage."""
    y = torch.arange(height, device=device).view(-1, 1).expand(height, width)
    x = torch.arange(width, device=device).view(1, -1).expand(height, width)

    if shape == "circle":
        # Radius is based on a fraction of the grid height
        radius = (height * fraction) / 2.0
        distance_sq = (y - row_origin) ** 2 + (x - col_origin) ** 2
        survivor_mask = distance_sq > (radius ** 2)

    elif shape == "square":
        half_side = (height * fraction) / 2.0
        in_row = (y >= row_origin - half_side) & (y <= row_origin + half_side)
        in_col = (x >= col_origin - half_side) & (x <= col_origin + half_side)
        wound_mask = in_row & in_col
        survivor_mask = ~wound_mask

    elif shape == "half_right":
        # Systematically destroys the entire right side of the grid
        wound_mask = x >= (width // 2)
        survivor_mask = ~wound_mask

    else:
        raise ValueError(f"Unsupported wound shape: '{shape}'")

    # Reshape to canonical (1, 1, H, W) to cleanly broadcast across batches and channels
    return survivor_mask.unsqueeze(0).unsqueeze(0)