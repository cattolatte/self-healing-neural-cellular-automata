"""Unit tests for visualization rendering and RGB extraction."""

import pytest
import torch
from PIL import Image

from src.visualization.render import render_state


def test_render_state_rgb_extraction():
    """Verify state tensors are correctly mapped to RGB Pillow images."""
    batch_size, channels, grid_size = 1, 16, 32
    
    # Create a synthetic state tensor with distinct values
    state = torch.zeros((batch_size, channels, grid_size, grid_size))
    
    # Fill visible channels (0, 1, 2) with valid color data
    state[:, 0, :, :] = 1.0  # Red
    state[:, 1, :, :] = 0.5  # Green
    state[:, 2, :, :] = 0.0  # Blue
    
    # Fill hidden channels with extreme values that would break rendering
    # if they were not correctly stripped out
    state[:, 3:15, :, :] = 999.0 
    
    # Render the state
    image = render_state(state, visible_channels=(0, 1, 2))
    
    # Verify the output is a valid PIL Image
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (grid_size, grid_size)
    
    # Sample a pixel to ensure correct mapping and clamping
    pixel = image.getpixel((16, 16))
    
    # Pillow converts [0.0, 1.0] floats to [0, 255] integers
    expected_red = int(1.0 * 255)
    expected_green = int(0.5 * 255)
    expected_blue = int(0.0 * 255)
    
    assert pixel == (expected_red, expected_green, expected_blue)


def test_render_state_batch_handling():
    """Verify the renderer safely isolates the first item of a batched tensor."""
    batch_size, channels, grid_size = 5, 16, 8
    state = torch.rand((batch_size, channels, grid_size, grid_size))
    
    # Should not raise an error despite batch_size > 1
    image = render_state(state)
    
    assert isinstance(image, Image.Image)
    assert image.size == (grid_size, grid_size)


def test_render_state_invalid_channels():
    """Verify that an invalid number of visible channels raises an error."""
    state = torch.zeros((1, 16, 10, 10))
    
    with pytest.raises(ValueError, match="exactly 3 visible channels"):
        # Passing 4 channels instead of 3
        render_state(state, visible_channels=(0, 1, 2, 3))