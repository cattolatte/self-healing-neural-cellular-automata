"""Command-line entry point for generating NCA rollout MP4 videos."""

import argparse
import sys
from pathlib import Path

import torch

from src.model.nca import NeuralCellularAutomaton
from src.simulation.seed import create_seed_state
from src.simulation.wound import apply_wound
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logger import get_logger
from src.visualization.mp4 import save_mp4
from src.visualization.render import render_state


def main() -> None:
    """Parse arguments and generate an MP4 video of the rollout."""
    parser = argparse.ArgumentParser(description="Render NCA rollout as an MP4.")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML configuration."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["growth", "healing"],
        default="growth",
        help="Simulation scenario to render (growth or healing).",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as error:
        sys.exit(f"Failed to load configuration: {error}")

    logger = get_logger(__name__, level=config["logging"]["level"])
    device = get_device()
    logger.info(f"Using device: {device}")

    model = NeuralCellularAutomaton(
        channels=config["model"]["channels"],
        grid_size=config["model"]["grid_size"],
        update_rate=config["model"]["update_rate"],
        alive_threshold=config["model"]["alive_threshold"],
        visible_channels=config["model"]["visible_channels"],
    ).to(device)

    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    state = create_seed_state(
        grid_size=config["model"]["grid_size"],
        channels=config["model"]["channels"],
        batch_size=1,
        device=device,
    )

    frames = []
    visible_channels = config["model"]["visible_channels"]
    rollout_steps = config["evaluation"]["rollout_steps"]

    logger.info(f"Generating frames for {args.scenario} scenario...")

    with torch.no_grad():
        # Phase 1: Growth
        for _ in range(rollout_steps):
            frames.append(render_state(state, visible_channels))
            state = model(state, steps=1)

        # Phase 2: Healing
        if args.scenario == "healing":
            logger.info("Applying damage and capturing recovery...")
            state = apply_wound(
                state,
                shape=config["damage"]["shape"],
                fraction=config["damage"]["fraction"],
            )
            for _ in range(rollout_steps):
                frames.append(render_state(state, visible_channels))
                state = model(state, steps=1)

        frames.append(render_state(state, visible_channels))

    output_dir = Path(config["paths"]["outputs"])
    save_path = output_dir / f"{args.scenario}_animation.mp4"

    logger.info("Encoding MP4 video...")
    saved_path = save_mp4(frames, save_path, fps=15)
    logger.info(f"Success! Video saved to: {saved_path}")


if __name__ == "__main__":
    main()