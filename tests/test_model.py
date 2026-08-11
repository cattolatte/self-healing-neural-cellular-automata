"""Unit tests for NCA dynamics, masking, and gradient flow."""

import torch

from src.model.alive_mask import AliveMask
from src.model.nca import NeuralCellularAutomaton
from src.model.stochastic import StochasticUpdate


def test_residual_update_behavior():
    """Verify that state updates are purely residual."""
    channels = 16
    grid_size = 16
    
    # Initialize the model with 100% update rate to bypass stochastic dropping
    # and a 0.0 threshold so no cells are killed by the alive mask.
    model = NeuralCellularAutomaton(
        channels=channels,
        grid_size=grid_size,
        update_rate=1.0,
        alive_threshold=0.0,
    )
    
    # Force the neural network to output exactly zero by zeroing all parameters
    for param in model.parameters():
        param.data.fill_(0.0)
        
    initial_state = torch.rand((1, channels, grid_size, grid_size))
    
    # Run a forward pass
    evolved_state = model(initial_state, steps=1)
    
    # Because the network outputs 0, the residual update (state + delta) 
    # should be exactly equal to the initial state.
    assert torch.allclose(evolved_state, initial_state), "Update is not purely residual."


def test_stochastic_update_masking():
    """Verify Bernoulli masking drops the exact percentage of cells expected."""
    update_rate = 0.5
    model = StochasticUpdate(update_rate=update_rate)
    
    # Create a large delta tensor to reduce statistical noise
    batch_size, channels, height, width = 1, 16, 100, 100
    state_delta = torch.ones((batch_size, channels, height, width))
    
    # Use a fixed generator for absolute deterministic reproducibility
    generator = torch.Generator().manual_seed(42)
    masked_delta = model(state_delta, generator=generator)
    
    # Check that approximately 50% of the ones were preserved
    retention_ratio = masked_delta.mean().item()
    assert 0.45 < retention_ratio < 0.55, f"Expected ~50% retention, got {retention_ratio}"
    
    # Crucially, ensure the exact same spatial mask was applied to all 16 channels 
    # of a single cell (e.g., channel 0 and channel 15 match perfectly).
    assert torch.allclose(masked_delta[:, 0, :, :], masked_delta[:, 15, :, :])

    # Test edge cases
    model_zero = StochasticUpdate(update_rate=0.0)
    assert model_zero(state_delta).sum().item() == 0.0

    model_one = StochasticUpdate(update_rate=1.0)
    assert torch.allclose(model_one(state_delta), state_delta)


def test_alive_mask_logic():
    """Verify the max-pooling alive mask logic against a hand-checkable grid."""
    channels = 4
    # Define channels 0 and 1 as visible. Threshold > 0.1 means alive.
    model = AliveMask(
        channels=channels,
        threshold=0.1,
        visible_channels=(0, 1),
        neighborhood_size=3,
    )
    
    # Create a 5x5 grid filled entirely with 1.0s in all channels
    state = torch.ones((1, channels, 5, 5))
    
    # Zero out the visible channels to "kill" the entire grid
    state[:, 0:2, :, :] = 0.0
    
    # Ignite a single "living" pixel in the exact center of visible channel 0
    state[0, 0, 2, 2] = 1.0
    
    output = model(state)
    
    # Due to a 3x3 neighborhood, the center pixel and its immediate neighbors 
    # [rows 1:4, cols 1:4] should be preserved. Everything else should be killed (0.0).
    
    # Check hidden channel 3 to verify surviving state data
    # 1. The center pixel is alive
    assert output[0, 3, 2, 2].item() == 1.0
    # 2. An immediately adjacent neighbor is alive
    assert output[0, 3, 1, 1].item() == 1.0
    # 3. A distant corner pixel is dead
    assert output[0, 3, 0, 0].item() == 0.0
    assert output[0, 3, 4, 4].item() == 0.0


def test_bptt_gradient_flow():
    """Ensure computational graphs remain intact across unrolled forward passes."""
    channels = 16
    grid_size = 16
    model = NeuralCellularAutomaton(
        channels=channels,
        grid_size=grid_size,
        update_rate=0.5,
        alive_threshold=0.1,
    )
    
    # Create an initial state requiring gradients
    state = torch.rand((2, channels, grid_size, grid_size), requires_grad=True)
    
    # Execute a multi-step rollout to unroll the graph through time
    evolved_state = model(state, steps=4)
    
    # Backpropagate a dummy scalar loss
    dummy_loss = evolved_state.mean()
    dummy_loss.backward()
    
    # 1. Ensure the input state received gradients
    assert state.grad is not None
    assert not torch.allclose(state.grad, torch.zeros_like(state.grad))
    
    # 2. Ensure the neural network parameters received gradients
    has_param_grads = any(
        p.grad is not None and not torch.allclose(p.grad, torch.zeros_like(p.grad))
        for p in model.parameters()
    )
    assert has_param_grads, "Gradients failed to flow back into model parameters."