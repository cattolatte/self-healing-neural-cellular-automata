"""Animation export utilities for NCA rollouts as MP4 videos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

from src.utils.io import ensure_directory


def save_mp4(
    frames: Sequence[Image.Image],
    save_path: str | Path,
    *,
    fps: int = 10,
) -> Path:
    """Compile a sequence of PIL Images into an MP4 video.

    Args:
        frames: Sequence of PIL Images representing animation frames.
        save_path: Destination file path for the .mp4 file.
        fps: Frames per second.

    Returns:
        The normalized path to the saved MP4.

    Raises:
        ValueError: If the frame sequence is empty or path has an invalid extension.
    """
    if not frames:
        raise ValueError("Cannot generate an MP4 from an empty frame sequence.")

    path = Path(save_path).resolve()
    if path.suffix.lower() != ".mp4":
        raise ValueError(f"Destination path must have an .mp4 extension, got {path.suffix}")

    ensure_directory(path.parent)

    # Use imageio to write the video sequence frame-by-frame
    with imageio.get_writer(path, fps=fps, macro_block_size=None) as writer:
        for frame in frames:
            # Convert PIL Image to RGB numpy array for imageio encoding
            frame_array = np.array(frame.convert("RGB"))
            writer.append_data(frame_array)

    return path