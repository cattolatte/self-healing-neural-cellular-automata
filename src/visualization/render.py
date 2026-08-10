"""Rendering utilities to convert NCA states into visible images."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image

from src.data.preprocess import tensor_to_image
from src.model.state import DEFAULT_VISIBLE_CHANNELS, extract_visible_channels
from src.visualization.colors import map_to_rgb


def render_state(
    state: torch.Tensor,
    visible_channels: Sequence[int] = DEFAULT_VISIBLE_CHANNELS,
) -> Image.Image:
    """Render an automata state tensor into an RGB image.

    This function safely extracts the visible channels, maps them using the 
    canonical color mapping, and returns a human-viewable image. It does not 
    mutate the input state tensor.

    Args:
        state: State tensor in ``(B, C, H, W)`` layout. If a batched tensor is 
            provided, only the first item in the batch is rendered.
        visible_channels: State-channel indices corresponding to visible tissue.

    Returns:
        An RGB Pillow image representing the visible tissue morphology.
    """
    # Isolate the first item if a batch is provided to ensure safe 2D rendering
    if state.ndim == 4 and state.shape[0] > 1:
        state = state[0:1]

    # Extract only the specified visible channels (stripping hidden latent state)
    visible_tensor = extract_visible_channels(state, visible_channels)

    # Apply the documented experiment color mapping
    rgb_tensor = map_to_rgb(visible_tensor, visible_channels)

    # Delegate to the established preprocessing utility for consistent conversion
    return tensor_to_image(rgb_tensor)