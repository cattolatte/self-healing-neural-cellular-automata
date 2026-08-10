"""Unit tests for fixed local perception filters."""

import pytest
import torch

from src.model.perception import Perception


def test_perception_output_shape_and_channel_ordering():
    """Verify that the perception layer expands channels by a factor of 3."""
    batch_size, channels, height, width = 2, 16, 32, 32
    model = Perception(channels)
    
    # Assert public property is correct
    assert model.output_channels == channels * 3
    
    state = torch.zeros((batch_size, channels, height, width))
    output = model(state)
    
    # Assert tensor shape matches expectation (Identity, Sobel X, Sobel Y per channel)
    assert output.shape == (batch_size, channels * 3, height, width)


def test_perception_invalid_initialization():
    """Verify that invalid channel counts raise actionable errors."""
    with pytest.raises(ValueError, match="must be a positive integer"):
        Perception(0)
        
    with pytest.raises(ValueError, match="must be a positive integer"):
        Perception(-5)


def test_perception_device_transfer():
    """Verify that fixed kernels correctly follow the module to specified devices."""
    channels = 16
    model = Perception(channels)
    
    # We test on CPU to maintain hermetic, hardware-independent tests.
    # The PyTorch .to() API behavior is identical across CPU/CUDA/MPS.
    device = torch.device("cpu")
    model.to(device)
    
    assert model.kernels.device == device
    
    state = torch.zeros((1, channels, 8, 8), device=device)
    output = model(state)
    
    assert output.device == device


def test_perception_known_local_responses():
    """Verify that Sobel filters produce the mathematically correct local gradients."""
    channels = 1
    model = Perception(channels)
    
    # Create a 3x3 synthetic grid with a horizontal line of active cells in the middle
    # [0.0, 0.0, 0.0]
    # [1.0, 1.0, 1.0]
    # [0.0, 0.0, 0.0]
    state = torch.zeros((1, channels, 3, 3))
    state[0, 0, 1, :] = 1.0
    
    output = model(state)
    
    # The 3 output channels for our single input channel
    identity_out = output[0, 0]
    sobel_x_out = output[0, 1]
    sobel_y_out = output[0, 2]
    
    # 1. Identity filter should perfectly mirror the input state
    assert torch.allclose(identity_out, state[0, 0])
    
    # 2. Sobel X measures horizontal change. 
    # Because our line is perfectly horizontal, there is no horizontal gradient 
    # at the center pixel. The convolution [-1, 0, 1] over [1, 1, 1] == 0.
    assert sobel_x_out[1, 1].item() == 0.0
    
    # 3. Sobel Y measures vertical change.
    # The center pixel itself sits on the flat line, so vertical change across 
    # its exact symmetrical center is 0.
    assert sobel_y_out[1, 1].item() == 0.0
    
    # Above the line (row 0, col 1): The Sobel Y bottom edge [1, 2, 1] hits the [1, 1, 1] line.
    # 1(1) + 2(1) + 1(1) = 4.0
    assert sobel_y_out[0, 1].item() == 4.0
    
    # Below the line (row 2, col 1): The Sobel Y top edge [-1, -2, -1] hits the [1, 1, 1] line.
    # -1(1) - 2(1) - 1(1) = -4.0
    assert sobel_y_out[2, 1].item() == -4.0