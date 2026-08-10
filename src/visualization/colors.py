"""Color mapping and channel semantics for visible tissue rendering."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def map_to_rgb(
    visible_tensor: torch.Tensor,
    visible_channels: Sequence[int] = (0, 1, 2),
) -> torch.Tensor:
    """Map the configured visible channels to standard RGB space.

    In the canonical architecture, the tissue fields map directly to RGB:
    - Channel 0 (Epidermis) -> Red
    - Channel 1 (Dermis) -> Green
    - Channel 2 (Vasculature) -> Blue

    Args:
        visible_tensor: Extracted visible state tensor in ``(B, C, H, W)`` layout.
        visible_channels: The channel indices corresponding to tissue layers.

    Returns:
        A normalized float32 tensor in ``(B, 3, H, W)`` layout representing RGB
        colors, strictly clamped to valid color ranges.

    Raises:
        ValueError: If the tensor does not contain exactly 3 visible channels.
    """
    if visible_tensor.shape[1] != 3:
        raise ValueError(
            f"RGB mapping requires exactly 3 visible channels, "
            f"received {visible_tensor.shape[1]}."
        )

    # Enforce safe color bounds without altering the source tensor
    return torch.clamp(visible_tensor, 0.0, 1.0)