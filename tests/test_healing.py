"""Unit tests for reproducible wound simulation operations."""

import pytest
import torch

from src.simulation.wound import apply_wound


def test_apply_wound_circle():
    """Verify the circle wound mask correctly zeros out the center."""
    batch_size, channels, height, width = 1, 16, 32, 32
    state = torch.ones((batch_size, channels, height, width))
    
    # Apply a circle wound with 0.5 fraction (radius = 8)
    damaged_state = apply_wound(state, shape="circle", fraction=0.5)
    
    # The exact center should be dead (0.0) across all 16 channels
    assert damaged_state[0, :, 16, 16].sum().item() == 0.0
    
    # The far corners should be completely alive (1.0 * 16 channels = 16.0)
    assert damaged_state[0, :, 0, 0].sum().item() == 16.0
    assert damaged_state[0, :, 31, 31].sum().item() == 16.0


def test_apply_wound_square():
    """Verify the square wound mask correctly zeros out a central block."""
    batch_size, channels, height, width = 1, 16, 32, 32
    state = torch.ones((batch_size, channels, height, width))
    
    # Apply a square wound with 0.5 fraction (16x16 block in center)
    # Center is at 16,16. Half-side is 8.
    # Wound should cover rows 8 to 24, cols 8 to 24
    damaged_state = apply_wound(state, shape="square", fraction=0.5)
    
    # Center is dead
    assert damaged_state[0, :, 16, 16].sum().item() == 0.0
    
    # Just outside the wound block is alive
    assert damaged_state[0, :, 7, 16].sum().item() == 16.0
    assert damaged_state[0, :, 25, 16].sum().item() == 16.0


def test_apply_wound_half_right():
    """Verify the half_right mask destroys the right side of the grid."""
    batch_size, channels, height, width = 1, 16, 32, 32
    state = torch.ones((batch_size, channels, height, width))
    
    damaged_state = apply_wound(state, shape="half_right")
    
    # Left side should be alive
    assert damaged_state[0, :, 16, 15].sum().item() == 16.0
    
    # Right side should be completely dead
    assert damaged_state[0, :, 16, 16].sum().item() == 0.0
    assert damaged_state[0, :, 16, 31].sum().item() == 0.0


def test_apply_wound_validation():
    """Verify wound application validates fractions and shapes properly."""
    state = torch.ones((1, 16, 32, 32))
    
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        apply_wound(state, fraction=1.5)
        
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        apply_wound(state, fraction=-0.1)
        
    with pytest.raises(ValueError, match="Unsupported wound shape"):
        apply_wound(state, shape="triangle")  # type: ignore