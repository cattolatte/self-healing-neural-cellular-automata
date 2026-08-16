"""Image preprocessing and tensor validation utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image


def resize_image(image: Image.Image, size: int | Sequence[int]) -> Image.Image:
    """Resize an image to the requested height and width.

    Args:
        image: Pillow image to resize.
        size: Square edge length or a two-item ``(height, width)`` sequence.

    Returns:
        A resized Pillow image.

    Raises:
        TypeError: If ``image`` is not a Pillow image.
        ValueError: If ``size`` does not describe positive dimensions.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance.")

    height, width = _parse_image_size(size)
    return image.resize((width, height), resample=Image.Resampling.LANCZOS)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB Pillow image to a normalized channel-first tensor.

    Args:
        image: RGB Pillow image.

    Returns:
        A float32 tensor with shape ``(3, H, W)`` and values in ``[0, 1]``.

    Raises:
        TypeError: If ``image`` is not a Pillow image.
        ValueError: If ``image`` is not RGB.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance.")
    if image.mode != "RGB":
        raise ValueError(f"Expected an RGB image, received mode '{image.mode}'.")

    image_array = np.array(image, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    return normalize_image_tensor(tensor)


def normalize_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Convert an unsigned-byte image tensor to normalized float32 values.

    Args:
        tensor: Image tensor with unsigned-byte values in ``[0, 255]`` or
            floating-point values already in ``[0, 1]``.

    Returns:
        A float32 tensor with values in ``[0, 1]``.

    Raises:
        TypeError: If ``tensor`` is not a tensor or uses an unsupported dtype.
        ValueError: If floating-point values fall outside ``[0, 1]``.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor.")
    if tensor.dtype == torch.uint8:
        return tensor.to(dtype=torch.float32).div(255.0)
    if not tensor.is_floating_point():
        raise TypeError("Image tensor must use torch.uint8 or a floating dtype.")
    if tensor.numel() and (tensor.amin().item() < 0.0 or tensor.amax().item() > 1.0):
        raise ValueError("Floating image tensor values must be within [0, 1].")
    return tensor.to(dtype=torch.float32)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized RGB tensor to a Pillow image.

    Args:
        tensor: Tensor with shape ``(3, H, W)`` or ``(1, 3, H, W)`` and values
            in ``[0, 1]``.

    Returns:
        An RGB Pillow image.

    Raises:
        ValueError: If the tensor shape or values are invalid for RGB output.
    """
    image_tensor = _remove_optional_batch_dimension(tensor)
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(
            "RGB tensor must have shape (3, H, W) or (1, 3, H, W)."
        )

    normalized_tensor = normalize_image_tensor(image_tensor)
    image_array = (
        normalized_tensor.detach()
        .to(device="cpu")
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .numpy()
    )
    return Image.fromarray(image_array, mode="RGB")


def move_to_device(tensor: torch.Tensor, device: torch.device | str | None) -> torch.Tensor:
    """Move a tensor to a requested device when one is supplied.

    Args:
        tensor: Tensor to move.
        device: Destination device, or ``None`` to preserve the current device.

    Returns:
        The tensor on the requested device.

    Raises:
        TypeError: If ``tensor`` is not a tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor.")
    if device is None:
        return tensor
    return tensor.to(device=torch.device(device))


def validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str = "tensor",
    batch_size: int | None = None,
    channels: int | None = None,
    height: int | None = None,
    width: int | None = None,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> None:
    """Validate a batched tensor using the canonical ``(B, C, H, W)`` layout.

    Args:
        tensor: Tensor to validate.
        name: Human-readable name used in error messages.
        batch_size: Optional expected batch dimension.
        channels: Optional expected channel dimension.
        height: Optional expected height.
        width: Optional expected width.
        dtype: Optional expected dtype. Use ``None`` to skip dtype validation.
        device: Optional expected device. Use ``None`` to skip device validation.

    Raises:
        TypeError: If the object is not a tensor or has an unexpected dtype.
        ValueError: If rank, shape, or device requirements are not satisfied.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.ndim != 4:
        raise ValueError(
            f"{name} must use (B, C, H, W) layout with rank 4; "
            f"received shape {tuple(tensor.shape)}."
        )
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(f"{name} must use dtype {dtype}, received {tensor.dtype}.")
    if device is not None and tensor.device.type != torch.device(device).type:
        raise ValueError(
            f"{name} must be on device type {torch.device(device).type}, "
            f"received {tensor.device.type}."
        )

    _validate_dimension(tensor.shape[0], batch_size, "batch size", name)
    _validate_dimension(tensor.shape[1], channels, "channel count", name)
    _validate_dimension(tensor.shape[2], height, "height", name)
    _validate_dimension(tensor.shape[3], width, "width", name)


def validate_target_tensor(
    tensor: torch.Tensor,
    *,
    grid_size: int | None = None,
    device: torch.device | str | None = None,
) -> None:
    """Validate a normalized, batched RGB target tensor.

    Args:
        tensor: Target tensor in ``(B, 3, H, W)`` layout.
        grid_size: Optional expected square image size.
        device: Optional expected device.

    Raises:
        TypeError: If the tensor is not float32.
        ValueError: If shape, device, or normalized value requirements fail.
    """
    validate_tensor(
        tensor,
        name="target tensor",
        batch_size=1,
        channels=3,
        height=grid_size,
        width=grid_size,
        device=device,
    )
    if tensor.numel() and (tensor.amin().item() < 0.0 or tensor.amax().item() > 1.0):
        raise ValueError("target tensor values must be within [0, 1].")


def _parse_image_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        dimensions = (size, size)
    else:
        dimensions = tuple(size)
        if len(dimensions) != 2:
            raise ValueError("Image size must be an integer or a (height, width) pair.")

    height, width = dimensions
    if not isinstance(height, int) or not isinstance(width, int):
        raise TypeError("Image dimensions must be integers.")
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be greater than zero.")
    return height, width


def _remove_optional_batch_dimension(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor.")
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Batched RGB tensor must have a batch size of 1.")
        return tensor[0]
    return tensor


def _validate_dimension(
    actual_value: int,
    expected_value: int | None,
    dimension_name: str,
    tensor_name: str,
) -> None:
    if expected_value is not None and actual_value != expected_value:
        raise ValueError(
            f"{tensor_name} {dimension_name} must be {expected_value}, "
            f"received {actual_value}."
        )
