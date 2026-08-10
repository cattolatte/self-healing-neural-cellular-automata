"""State lifecycle management for pooled NCA training samples."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.preprocess import validate_tensor
from src.simulation.wound import WoundShape, apply_wound


@dataclass(frozen=True)
class PoolBatch:
    """A sampled pool batch and the indices it originated from."""

    indices: torch.Tensor
    states: torch.Tensor


class SamplePool:
    """Manage reusable, partially evolved states without model dependencies."""

    def __init__(
        self,
        initial_state: torch.Tensor,
        *,
        size: int,
        reseed_fraction: float = 0.0,
        damage_enabled: bool = False,
        wound_shape: WoundShape = "circle",
        wound_fraction: float = 0.25,
    ) -> None:
        """Initialize a fixed-size pool from a single seed state.

        Args:
            initial_state: Float32 seed state with shape (1, C, H, W).
            size: Number of states retained in the pool.
            reseed_fraction: Fraction of each sampled batch replaced with a seed.
            damage_enabled: Whether to apply damage to a subset of sampled states.
            wound_shape: The geometric shape of the training wound.
            wound_fraction: Severity/size of the training wound.

        Raises:
            ValueError: If pool configuration or initial-state shape is invalid.
        """
        validate_tensor(initial_state, name="initial pool state", batch_size=1)
        _validate_positive_integer(size, "size")
        _validate_fraction(reseed_fraction, "reseed_fraction")
        _validate_fraction(wound_fraction, "wound_fraction")

        self._seed_state = initial_state.detach().clone()
        self._states = self._seed_state.repeat(size, 1, 1, 1)
        self._reseed_fraction = float(reseed_fraction)
        
        self._damage_enabled = bool(damage_enabled)
        self._wound_shape = wound_shape
        self._wound_fraction = float(wound_fraction)

    @property
    def size(self) -> int:
        """Return the number of states retained by the pool."""
        return self._states.shape[0]

    @property
    def device(self) -> torch.device:
        """Return the device that stores pooled states."""
        return self._states.device

    @property
    def state_shape(self) -> tuple[int, int, int]:
        """Return the pooled per-sample shape as (channels, height, width)."""
        channels, height, width = self._states.shape[1:]
        return channels, height, width

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> PoolBatch:
        """Sample a batch of states, applying configured reseeding and damage.

        Args:
            batch_size: Number of states to sample without replacement.
            generator: Optional generator compatible with the pool device.

        Returns:
            Sampled state tensors and their pool indices.

        Raises:
            ValueError: If batch_size exceeds the pool size.
        """
        _validate_positive_integer(batch_size, "batch_size")
        if batch_size > self.size:
            raise ValueError(
                f"batch_size ({batch_size}) cannot exceed pool size ({self.size})."
            )

        indices = torch.randperm(
            self.size,
            device=self.device,
            generator=generator,
        )[:batch_size]

        states = self._states.index_select(0, indices).clone()
        states = self._reseed_states(states, generator)
        
        if self._damage_enabled:
            states = self._damage_states(states, generator)

        return PoolBatch(indices=indices, states=states)

    def replace(self, batch: PoolBatch, evolved_states: torch.Tensor) -> None:
        """Store evolved states for a previously sampled pool batch.

        Detachment is intentional here: persisted pool entries are state
        snapshots between optimization steps and must not retain old BPTT graphs.

        Args:
            batch: Original sampled pool batch.
            evolved_states: Evolved replacement states with matching shape.

        Raises:
            ValueError: If replacement indices or state shape are invalid.
        """
        validate_tensor(evolved_states, name="evolved pool states")
        if batch.indices.ndim != 1 or batch.indices.shape[0] != evolved_states.shape[0]:
            raise ValueError("Pool batch indices must match evolved batch size.")
        if evolved_states.shape[1:] != self._states.shape[1:]:
            raise ValueError(
                "Evolved state shape must match pooled state channels and spatial size."
            )
        if evolved_states.device != self.device:
            raise ValueError(
                f"Evolved states must be on {self.device}, "
                f"received {evolved_states.device}."
            )

        with torch.no_grad():
            self._states.index_copy_(0, batch.indices, evolved_states.detach())

    def snapshot(self) -> torch.Tensor:
        """Return a detached copy of all pooled states for inspection or testing."""
        return self._states.detach().clone()

    def _reseed_states(
        self,
        states: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        reseed_count = int(states.shape[0] * self._reseed_fraction)
        if reseed_count == 0:
            return states

        reseed_indices = torch.randperm(
            states.shape[0],
            device=states.device,
            generator=generator,
        )[:reseed_count]

        seed_states = self._seed_state.expand_as(states.index_select(0, reseed_indices))
        return states.index_copy(0, reseed_indices, seed_states)

    def _damage_states(
        self,
        states: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Apply a wound mask to a random subset of the batch."""
        # Damage up to 50% of the sampled batch to force regeneration learning
        damage_count = max(1, states.shape[0] // 2) if states.shape[0] > 1 else 1

        damage_indices = torch.randperm(
            states.shape[0],
            device=states.device,
            generator=generator,
        )[:damage_count]

        states_to_damage = states.index_select(0, damage_indices)
        
        damaged_subset = apply_wound(
            states_to_damage,
            shape=self._wound_shape,
            fraction=self._wound_fraction,
        )
        
        return states.index_copy(0, damage_indices, damaged_subset)


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_fraction(value: float, name: str) -> None:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be between 0 and 1.")