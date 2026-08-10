"""Unit tests for experiment configuration validation."""

import copy
from typing import Any

import pytest

from src.utils.config import validate_config


@pytest.fixture
def valid_config() -> dict[str, Any]:
    """Provide a structurally complete and valid configuration mapping."""
    return {
        "run": {"name": "test_run", "mode": "train"},
        "paths": {"outputs": "out", "checkpoints": "chk", "logs": "log"},
        "device": {"preference": "cpu"},
        "seed": {"value": 42, "deterministic": True},
        "model": {
            "grid_size": 32,
            "channels": 16,
            "visible_channels": [0, 1, 2],
            "hidden_channels": list(range(3, 16)),
            "perception": {"filters": ["identity", "sobel_x", "sobel_y"]},
            "update_rate": 0.5,
            "alive_threshold": 0.1,
        },
        "training": {
            "epochs": 100,
            "batch_size": 8,
            "rollout_steps": 64,
            "checkpoint_interval": 10,
        },
        "optimizer": {"name": "adam", "learning_rate": 0.001, "weight_decay": 0.0},
        "evaluation": {
            "rollout_steps": 128,
            "num_trials": 5,
            "stochastic_updates": True,
        },
        "visualization": {"enabled": True, "save_animations": True, "dpi": 150},
        "logging": {"level": "INFO", "file_enabled": True},
        "pool": {"enabled": True, "size": 128, "reseed_fraction": 0.25},
        "damage": {"enabled": True, "shape": "circle", "fraction": 0.25},
    }


def test_valid_config_passes(valid_config: dict[str, Any]) -> None:
    """Verify that a properly structured configuration raises no errors."""
    # Should execute without throwing an exception
    validate_config(valid_config)


def test_missing_required_section(valid_config: dict[str, Any]) -> None:
    """Verify that missing a top-level section raises a ValueError."""
    invalid_config = copy.deepcopy(valid_config)
    del invalid_config["model"]
    
    with pytest.raises(ValueError, match="missing required section"):
        validate_config(invalid_config)


def test_missing_required_field(valid_config: dict[str, Any]) -> None:
    """Verify that missing a specific field within a section raises a ValueError."""
    invalid_config = copy.deepcopy(valid_config)
    del invalid_config["training"]["epochs"]
    
    with pytest.raises(ValueError, match="missing required field"):
        validate_config(invalid_config)


def test_invalid_range_negative_grid(valid_config: dict[str, Any]) -> None:
    """Verify that grid size bounds are strictly enforced."""
    invalid_config = copy.deepcopy(valid_config)
    invalid_config["model"]["grid_size"] = -5
    
    with pytest.raises(ValueError, match="must be greater than zero"):
        validate_config(invalid_config)


def test_invalid_range_probability(valid_config: dict[str, Any]) -> None:
    """Verify that probability parameters are bounded between 0.0 and 1.0."""
    invalid_config = copy.deepcopy(valid_config)
    invalid_config["model"]["update_rate"] = 1.5
    
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        validate_config(invalid_config)


def test_model_channels_mismatch(valid_config: dict[str, Any]) -> None:
    """Verify that visible and hidden channels correctly sum to the total channel count."""
    invalid_config = copy.deepcopy(valid_config)
    
    # We have 16 total channels, but we are removing channel 15 from the hidden list
    invalid_config["model"]["hidden_channels"] = list(range(3, 15))
    
    with pytest.raises(ValueError, match="exactly 'model.channels' entries"):
        validate_config(invalid_config)