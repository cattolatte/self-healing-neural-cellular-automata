"""Command-line entry point for NCA growth experiments."""

import argparse
import sys
from pathlib import Path

from src.data.load_target import load_target
from src.model.nca import NeuralCellularAutomaton
from src.simulation.grow import run_growth_experiment
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logger import get_logger
from src.utils.random_seed import set_seed


def main() -> None:
    """Parse arguments and execute the growth experiment."""
    parser = argparse.ArgumentParser(description="Run an NCA growth experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML experiment configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Path to the PNG target image.",
    )

    args = parser.parse_args()

    # 1. Configuration & Logging
    try:
        config = load_config(args.config)
    except Exception as error:
        sys.exit(f"Failed to load configuration: {error}")

    logger = get_logger(__name__, level=config["logging"]["level"])
    logger.info(f"Starting growth experiment using config: {args.config}")

    # 2. Hardware & Reproducibility
    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(
        seed=config["seed"]["value"],
        deterministic=config["seed"]["deterministic"],
    )

    # 3. Target Loading
    logger.info(f"Loading target image from: {args.target}")
    target_tensor = load_target(
        args.target,
        grid_size=config["model"]["grid_size"],
        device=device,
    )

    # 4. Model Initialization & Checkpoint Loading
    logger.info("Initializing Neural Cellular Automaton...")
    model = NeuralCellularAutomaton(
        channels=config["model"]["channels"],
        grid_size=config["model"]["grid_size"],
        update_rate=config["model"]["update_rate"],
        alive_threshold=config["model"]["alive_threshold"],
        visible_channels=config["model"]["visible_channels"],
    ).to(device)

    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint_data = load_checkpoint(
        args.checkpoint,
        model=model,
        map_location=device,
    )
    logger.info(f"Restored model from epoch {checkpoint_data.epoch}.")

    # 5. Run Growth Experiment
    rollout_steps = config["evaluation"]["rollout_steps"]
    logger.info(f"Running growth rollout for {rollout_steps} steps...")
    
    results = run_growth_experiment(
        model=model,
        target=target_tensor,
        config=config,
        device=device,
    )

    # 6. Report Results
    logger.info("Growth experiment complete.")
    logger.info(f"Final Mean Squared Error (MSE): {results['mse']:.6f}")


if __name__ == "__main__":
    main()