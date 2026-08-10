"""Unit tests for training coordination and sample pool management."""

import pytest
import torch

from src.training.pool import SamplePool


def test_sample_pool_initialization():
    """Verify the pool initializes correctly from a single seed state."""
    batch_size, channels, grid_size = 1, 16, 32
    initial_state = torch.zeros((batch_size, channels, grid_size, grid_size))
    
    # Mark the seed uniquely
    initial_state[0, 0, 16, 16] = 99.0
    
    pool_size = 100
    pool = SamplePool(initial_state, size=pool_size)
    
    assert pool.size == pool_size
    assert pool.state_shape == (channels, grid_size, grid_size)
    
    # The pool should be fully populated with identical copies of the initial state
    snapshot = pool.snapshot()
    assert snapshot.shape == (pool_size, channels, grid_size, grid_size)
    assert torch.all(snapshot[:, 0, 16, 16] == 99.0)


def test_sample_pool_sampling_and_replacement():
    """Verify that batches can be sampled and successfully replaced."""
    initial_state = torch.zeros((1, 16, 32, 32))
    pool = SamplePool(initial_state, size=10)
    
    # Sample a batch
    batch = pool.sample(batch_size=4)
    assert batch.states.shape[0] == 4
    assert batch.indices.shape[0] == 4
    
    # Evolve the sampled states (simulate a forward pass)
    evolved_states = batch.states + 1.0
    
    # Replace the states back into the pool
    pool.replace(batch, evolved_states)
    
    # Verify the pool now contains the evolved values at the exact indices
    snapshot = pool.snapshot()
    for idx in batch.indices:
        assert torch.all(snapshot[idx] == 1.0)


def test_sample_pool_reseeding_policy():
    """Verify that a specific fraction of samples are forcefully reseeded."""
    initial_state = torch.zeros((1, 16, 32, 32))
    # Give the seed a unique value
    initial_state[0, 0, 0, 0] = -1.0
    
    pool = SamplePool(initial_state, size=100, reseed_fraction=0.25)
    
    # Mutate the entire pool so we can tell the difference between old states and fresh seeds
    mock_batch = pool.sample(100)
    mutated_states = mock_batch.states + 5.0
    pool.replace(mock_batch, mutated_states)
    
    # Now sample a regular training batch
    generator = torch.Generator().manual_seed(42)
    batch = pool.sample(batch_size=8, generator=generator)
    
    # 25% of 8 is exactly 2. We should have 2 fresh seeds and 6 old states.
    num_seeds = (batch.states[:, 0, 0, 0] == -1.0).sum().item()
    num_old = (batch.states[:, 0, 0, 0] == 4.0).sum().item() # 0 - 1 + 5 = 4
    
    assert num_seeds == 2, f"Expected 2 reseeded samples, got {num_seeds}."
    assert num_old == 6, f"Expected 6 persisted samples, got {num_old}."


def test_sample_pool_capacity_limits():
    """Verify that the pool rejects invalid sample requests."""
    initial_state = torch.zeros((1, 16, 32, 32))
    pool = SamplePool(initial_state, size=50)
    
    with pytest.raises(ValueError, match="cannot exceed pool size"):
        pool.sample(batch_size=51)
        
    with pytest.raises(ValueError, match="must be a positive integer"):
        pool.sample(batch_size=0)