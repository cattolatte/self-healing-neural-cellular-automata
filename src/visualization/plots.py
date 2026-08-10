"""Publication-quality plotting for training and evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import ensure_directory


def plot_loss_curve(
    losses: Sequence[float],
    save_path: str | Path,
    *,
    title: str = "Training Reconstruction Loss",
    dpi: int = 150,
) -> Path:
    """Generate and save a loss curve plot.

    Args:
        losses: Sequence of sequential loss values recorded during training.
        save_path: Destination file path for the plot (e.g., .png or .pdf).
        title: Title of the chart.
        dpi: Resolution of the saved figure.

    Returns:
        The normalized path to the saved figure.
    """
    path = Path(save_path).resolve()
    ensure_directory(path.parent)

    plt.figure(figsize=(8, 5))
    plt.plot(losses, color="#2c3e50", linewidth=2.0, label="MSE Loss")
    
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Training Step / Epoch", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return path


def plot_healing_evaluation(
    pre_damage_mse: float,
    post_damage_mse: float,
    recovery_mse: float,
    save_path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """Generate a bar chart comparing phases of the healing experiment.

    Args:
        pre_damage_mse: Baseline error before damage.
        post_damage_mse: Error immediately after the wound mask.
        recovery_mse: Error after the final recovery rollout.
        save_path: Destination file path for the plot.
        dpi: Resolution of the saved figure.

    Returns:
        The normalized path to the saved figure.
    """
    path = Path(save_path).resolve()
    ensure_directory(path.parent)

    phases = ["Pre-Damage", "Wounded", "Recovered"]
    values = [pre_damage_mse, post_damage_mse, recovery_mse]
    colors = ["#27ae60", "#c0392b", "#2980b9"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(phases, values, color=colors, width=0.6)

    plt.title("Tissue Regeneration Evaluation", fontsize=14, fontweight="bold")
    plt.ylabel("Mean Squared Error (vs. Target)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Attach numerical labels to each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return path