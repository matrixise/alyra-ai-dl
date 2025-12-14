"""Device detection utilities for model inference."""

import enum

import torch


class DeviceEnum(enum.StrEnum):
    """Enumeration of supported device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def detect_device(device: str | None = None) -> DeviceEnum:
    """
    Detect the best available device for model inference.

    Args:
        device: Specific device to use. If None, auto-detects the best available device.

    Returns:
        DeviceEnum: The detected or specified device.

    Example:
        >>> device = detect_device()
        >>> print(f"Using device: {device.value}")
    """
    if device:
        return DeviceEnum(device)
    if torch.cuda.is_available():
        return DeviceEnum.CUDA
    if torch.backends.mps.is_available():
        return DeviceEnum.MPS
    return DeviceEnum.CPU
