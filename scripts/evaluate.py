"""Command-line entry point for running the full NCA evaluation suite."""

import argparse
import json
import sys
from pathlib import Path

from src.data.load_target import load_target
from src.evaluation import (
    evaluate_healing_performance,
    evaluate_persistence,
)
from src.model.nca import NeuralCellularAutomaton
from src.simulation.grow import run_growth_experiment
from src.simulation.heal import run_healing_experiment
from src.simulation.persist import run_persistence_experiment
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.io import ensure_directory
from src.utils.logger import get_logger
from src.utils.random_seed import set_seed


def main() -> None:
    """Parse arguments and execute the standardized evaluation suite."""
    parser = argparse.ArgumentParser(description="Run the full NCA evaluation suite.")
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

    # 1. Configuration & Setup
    try:
        config = load_config(args.config)
    except Exception as error:
        sys.exit(f"Failed to load configuration: {error}")

    logger = get_logger(__name__, level=config["logging"]["level"])
    device = get_device()
    logger.info(f"Starting standard evaluation suite. Using device: {device}")

    set_seed(
        seed=config["seed"]["value"],
        deterministic=config["seed"]["deterministic"],
    )

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

    # ==========================================
    # BENCHMARK 1: GROWTH
    # ==========================================
    logger.info("Executing Benchmark 1: Morphogenesis Growth...")
    growth_results = run_growth_experiment(model, target_tensor, config, device)
    growth_mse = float(growth_results["mse"])
    logger.info(f"Growth final MSE: {growth_mse:.6f}")

    # ==========================================
    # BENCHMARK 2: PERSISTENCE
    # ==========================================
    logger.info("Executing Benchmark 2: Structural Persistence...")
    # The growth module returns a CPU tensor; push it back to the active device
    grown_state = growth_results["final_state"].to(device)
    
    persistence_results = run_persistence_experiment(
        model=model,
        initial_state=grown_state,
        target=target_tensor,
        config=config,
    )
    
    stability_metrics = evaluate_persistence(persistence_results["mse_sequence"])
    logger.info(f"Persistence drift: {stability_metrics['drift']:.6f} | Stable: {stability_metrics['is_stable']}")

    # ==========================================
    # BENCHMARK 3: HEALING
    # ==========================================
    logger.info("Executing Benchmark 3: Damage and Recovery...")
    healing_results = run_healing_experiment(model, target_tensor, config, device)
    
    healing_metrics = evaluate_healing_performance(
        pre_damage_mse=healing_results["pre_damage_mse"],
        post_damage_mse=healing_results["post_damage_mse"],
        final_recovery_mse=healing_results["final_recovery_mse"],
    )
    logger.info(f"Recovery efficiency: {healing_metrics['recovery_efficiency']:.2%}")

    # ==========================================
    # REPORT AGGREGATION & EXPORT
    # ==========================================
    logger.info("Aggregating metrics into final report...")
    
    report = {
        "growth": {
            "final_mse": growth_mse,
        },
        "persistence": {
            "final_mse": float(stability_metrics["final_mse"]),
            "drift": float(stability_metrics["drift"]),
            "variance": float(stability_metrics["variance"]),
            "max_mse": float(stability_metrics["max_mse"]),
            "is_stable": bool(stability_metrics["is_stable"]),
        },
        "healing": {
            "damage_severity": float(healing_metrics["damage_severity"]),
            "recovery_efficiency": float(healing_metrics["recovery_efficiency"]),
            "full_recovery_achieved": bool(healing_metrics["full_recovery_achieved"]),
        }
    }

    output_dir = Path(config["paths"]["outputs"])
    ensure_directory(output_dir)
    report_path = output_dir / "evaluation_metrics.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Evaluation complete. Metrics saved to {report_path}")


if __name__ == "__main__":
    main()