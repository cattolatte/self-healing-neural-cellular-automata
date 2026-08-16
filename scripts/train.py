"""Command-line entry point for training the Neural Cellular Automata."""
import argparse
import sys
from pathlib import Path

from src.data.load_target import load_target
from src.model.nca import NeuralCellularAutomaton
from src.simulation.seed import create_seed_state
from src.training.checkpoint import save_checkpoint
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logger import get_logger


def main() -> None:
    """Parse arguments and execute the NCA training loop."""
    parser = argparse.ArgumentParser(description="Train the NCA model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--target", type=Path, required=True, help="Path to target PNG.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (ValueError, KeyError, OSError, TypeError) as error:
        sys.exit(f"Failed to load configuration: {error}")

    logger = get_logger(__name__, level=config["logging"]["level"])
    device = get_device()
    logger.info(f"Starting training on device: {device}")

    # 1. Load Target
    target_tensor = load_target(args.target, grid_size=config["model"]["grid_size"], device=device)

    # 2. Initialize Model
    model = NeuralCellularAutomaton(
        channels=config["model"]["channels"],
        grid_size=config["model"]["grid_size"],
        update_rate=config["model"]["update_rate"],
        alive_threshold=config["model"]["alive_threshold"],
        visible_channels=config["model"]["visible_channels"],
    ).to(device)

    # 3. Create Seed State for Pool
    initial_state = create_seed_state(
        grid_size=config["model"]["grid_size"],
        channels=config["model"]["channels"],
        batch_size=1,
        device=device,
    )

    # 4. Initialize Trainer
    trainer = Trainer.from_config(
        model=model,
        target=target_tensor,
        initial_state=initial_state,
        config=config,
    )

    # 5. Training Loop
    epochs = int(config["training"]["epochs"])
    checkpoint_interval = int(config["training"]["checkpoint_interval"])
    output_dir = Path(config["paths"]["checkpoints"])

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        result = trainer.train_step()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{epochs} | Loss: {result.loss:.6f}")

        if epoch % checkpoint_interval == 0 or epoch == epochs:
            ckpt_path = output_dir / f"model_epoch_{epoch}.pt"
            save_checkpoint(
                path=ckpt_path,
                model=model,
                epoch=epoch,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                config=config,
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Training complete!")

if __name__ == "__main__":
    main()