"""Core functionality for the alyra_ai_dl package."""

from alyra_ai_dl.core.device import DeviceEnum, detect_device
from alyra_ai_dl.core.model import DEFAULT_MODEL_PATH, create_classifier

__all__ = ["DeviceEnum", "detect_device", "create_classifier", "DEFAULT_MODEL_PATH"]
