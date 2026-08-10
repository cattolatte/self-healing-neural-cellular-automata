"""Command-line entry point for exporting publication-quality evaluation figures."""

import argparse
import sys
from pathlib import Path

from src.data.load_target import load_target
from src.model.nca import NeuralCellularAutomaton
from src.simulation.heal import run_healing_experiment
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logger import get_logger
from src.visualization.plots import plot_healing_evaluation


def main() -> None:
    """Parse arguments and generate static evaluation figures."""
    parser = argparse.ArgumentParser(description="Export evaluation figures.")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML configuration."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--target", type=Path, required=True, help="Path to the PNG target image."
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as error:
        sys.exit(f"Failed to load configuration: {error}")

    logger = get_logger(__name__, level=config["logging"]["level"])
    device = get_device()
    logger.info(f"Using device: {device}")

    target_tensor = load_target(
        args.target, grid_size=config["model"]["grid_size"], device=device
    )

    model = NeuralCellularAutomaton(
        channels=config["model"]["channels"],
        grid_size=config["model"]["grid_size"],
        update_rate=config["model"]["update_rate"],
        alive_threshold=config["model"]["alive_threshold"],
        visible_channels=config["model"]["visible_channels"],
    ).to(device)

    load_checkpoint(args.checkpoint, model=model, map_location=device)

    logger.info("Executing evaluation sequence...")
    # Obtain purely quantitative metrics via the decoupled simulation layer
    results = run_healing_experiment(model, target_tensor, config, device)

    output_dir = Path(config["paths"]["outputs"])
    save_path = output_dir / "healing_evaluation_bar_chart.png"

    logger.info("Exporting diagnostic figures...")
    saved_path = plot_healing_evaluation(
        pre_damage_mse=results["pre_damage_mse"],
        post_damage_mse=results["post_damage_mse"],
        recovery_mse=results["final_recovery_mse"],
        save_path=save_path,
        dpi=config["visualization"]["dpi"],
    )
    
    logger.info(f"Success! Figure saved to: {saved_path}")


if __name__ == "__main__":
    main()