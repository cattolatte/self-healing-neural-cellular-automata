"""Orchestration for rollout-based neural cellular automata training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from src.data.preprocess import validate_target_tensor
from src.model.state import extract_visible_channels
from src.simulation.wound import WoundShape
from src.training.losses import create_loss
from src.training.optimizer import create_optimizer
from src.training.pool import SamplePool
from src.training.scheduler import Scheduler, create_scheduler
from src.utils.random_seed import set_seed


@dataclass(frozen=True)
class TrainingStepResult:
    """Loss and batch information produced by one optimizer step."""

    loss: float
    batch_size: int


class Trainer:
    """Coordinate pool sampling, BPTT rollout, loss, and optimization."""

    def __init__(
        self,
        *,
        model: nn.Module,
        target: torch.Tensor,
        pool: SamplePool,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout_steps: int,
        batch_size: int,
        visible_channels: Sequence[int],
        scheduler: Scheduler | None = None,
    ) -> None:
        """Initialize a training coordinator with explicit dependencies.

        Args:
            model: NCA model that accepts state and steps arguments.
            target: Float32 RGB target tensor in (1, 3, H, W) layout.
            pool: State lifecycle manager for training batches.
            loss_function: Reconstruction loss module.
            optimizer: Optimizer for model parameters.
            rollout_steps: Number of differentiable NCA steps per update.
            batch_size: Default number of pooled states per training update.
            visible_channels: State channels compared against the RGB target.
            scheduler: Optional scheduler stepped after each training update.
        """
        _validate_positive_integer(rollout_steps, "rollout_steps")
        _validate_positive_integer(batch_size, "batch_size")
        _validate_visible_channels(visible_channels)
        validate_target_tensor(target)

        self.model = model
        self.target = target
        self.pool = pool
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.visible_channels = tuple(visible_channels)
        self.scheduler = scheduler

        self._validate_device_compatibility()

    @classmethod
    def from_config(
        cls,
        *,
        model: nn.Module,
        target: torch.Tensor,
        initial_state: torch.Tensor,
        config: Mapping[str, Any],
    ) -> "Trainer":
        """Construct a trainer and dependencies from resolved configuration.

        Args:
            model: NCA model to optimize.
            target: Normalized RGB target tensor.
            initial_state: Single seeded state used to populate the sample pool.
            config: Resolved YAML configuration mapping.

        Returns:
            A configured trainer with reproducibility settings applied.

        Raises:
            KeyError: If required training configuration sections are absent.
        """
        _require_sections(
            config,
            ("seed", "model", "training", "optimizer", "pool", "loss", "damage"),
        )

        set_seed(
            seed=int(config["seed"]["value"]),
            deterministic=bool(config["seed"]["deterministic"]),
        )

        optimizer = create_optimizer(model.parameters(), config["optimizer"])
        scheduler = create_scheduler(optimizer, config.get("scheduler"))

        pool = SamplePool(
            initial_state,
            size=int(config["pool"]["size"]),
            reseed_fraction=float(config["pool"]["reseed_fraction"]),
            damage_enabled=bool(config["damage"]["enabled"]),
            wound_shape=cast(WoundShape, str(config["damage"]["shape"])),
            wound_fraction=float(config["damage"]["fraction"]),
        )

        return cls(
            model=model,
            target=target,
            pool=pool,
            loss_function=create_loss(config["loss"]),
            optimizer=optimizer,
            rollout_steps=int(config["training"]["rollout_steps"]),
            batch_size=int(config["training"]["batch_size"]),
            visible_channels=config["model"]["visible_channels"],
            scheduler=scheduler,
        )

    def train_step(
        self,
        batch_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> TrainingStepResult:
        """Run one differentiable rollout, optimizer update, and pool replacement.

        Args:
            batch_size: Optional number of pooled states to optimize. When omitted,
                the configured default batch size is used.
            generator: Optional random generator for pool and NCA stochasticity.

        Returns:
            Scalar loss and effective batch size for the completed update.
        """
        effective_batch_size = self.batch_size if batch_size is None else batch_size
        _validate_positive_integer(effective_batch_size, "batch_size")

        pool_batch = self.pool.sample(effective_batch_size, generator)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        evolved_states = self.model(
            pool_batch.states,
            steps=self.rollout_steps,
            generator=generator,
        )

        visible_states = extract_visible_channels(
            evolved_states,
            self.visible_channels,
        )

        target_batch = self.target.expand_as(visible_states)
        loss = self.loss_function(visible_states, target_batch)

        if loss.ndim != 0:
            raise ValueError("Training loss must be reduced to a scalar tensor.")

        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.pool.replace(pool_batch, evolved_states)

        return TrainingStepResult(
            loss=float(loss.detach().item()),
            batch_size=effective_batch_size,
        )

    def _validate_device_compatibility(self) -> None:
        model_device = _get_model_device(self.model)

        if self.target.device != model_device:
            raise ValueError(
                f"Target tensor must be on {model_device}, "
                f"received {self.target.device}."
            )

        if self.pool.device != model_device:
            raise ValueError(
                f"Sample pool must be on {model_device}, "
                f"received {self.pool.device}."
            )

        if self.target.shape[2:] != self.pool.state_shape[1:]:
            raise ValueError(
                "Target spatial dimensions must match the sampled automata state."
            )


def _get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError(
            "Model must expose at least one trainable parameter."
        ) from error


def _require_sections(config: Mapping[str, Any], sections: tuple[str, ...]) -> None:
    missing_sections = [section for section in sections if section not in config]
    if missing_sections:
        raise KeyError(
            "Training configuration is missing section(s): "
            f"{', '.join(missing_sections)}."
        )


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_visible_channels(visible_channels: Sequence[int]) -> None:
    if len(visible_channels) != 3:
        raise ValueError("Exactly three visible channels are required for RGB targets.")
    if any(
        not isinstance(channel, int) or isinstance(channel, bool)
        for channel in visible_channels
    ):
        raise TypeError("visible_channels must contain integer indices.")
    if len(set(visible_channels)) != len(visible_channels):
        raise ValueError("visible_channels must be unique.")