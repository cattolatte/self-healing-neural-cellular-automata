"""Utilities for exporting individual rollout frames to disk."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from src.utils.io import ensure_directory


def export_frames(
    frames: Sequence[Image.Image],
    output_directory: str | Path,
    *,
    prefix: str = "frame_",
) -> Path:
    """Save a sequence of PIL Images as sequentially numbered PNG files.

    Args:
        frames: Sequence of PIL Images representing animation frames.
        output_directory: Directory where the PNG frames will be saved.
        prefix: String prefix for each generated filename.

    Returns:
        The normalized path to the output directory.

    Raises:
        ValueError: If the frame sequence is empty.
    """
    if not frames:
        raise ValueError("Cannot export frames from an empty sequence.")

    directory = Path(output_directory).resolve()
    ensure_directory(directory)

    # Dynamically calculate padding based on total frames (e.g., 01, 10, 99)
    pad_length = len(str(len(frames) - 1))

    for index, frame in enumerate(frames):
        filename = f"{prefix}{str(index).zfill(pad_length)}.png"
        file_path = directory / filename
        frame.save(file_path, format="PNG")

    return directory