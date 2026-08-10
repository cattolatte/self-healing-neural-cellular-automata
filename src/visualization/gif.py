"""Animation export utilities for NCA rollouts as GIFs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from src.utils.io import ensure_directory


def save_gif(
    frames: Sequence[Image.Image],
    save_path: str | Path,
    *,
    fps: int = 10,
    loop: int = 0,
) -> Path:
    """Compile a sequence of PIL Images into an animated GIF.

    Args:
        frames: Sequence of PIL Images representing animation frames.
        save_path: Destination file path for the .gif file.
        fps: Frames per second.
        loop: Number of times to loop (0 means infinite loop).

    Returns:
        The normalized path to the saved GIF.

    Raises:
        ValueError: If the frame sequence is empty or path has an invalid extension.
    """
    if not frames:
        raise ValueError("Cannot generate a GIF from an empty frame sequence.")

    path = Path(save_path).resolve()
    if path.suffix.lower() != ".gif":
        raise ValueError(f"Destination path must have a .gif extension, got {path.suffix}")
        
    ensure_directory(path.parent)

    duration = int(1000 / fps)

    # Save the first frame, appending the rest as an animated sequence
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=False,
    )

    return path