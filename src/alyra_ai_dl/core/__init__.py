"""Core functionality for the alyra_ai_dl package."""

from alyra_ai_dl.core.device import DeviceEnum, detect_device
from alyra_ai_dl.core.model import DEFAULT_MODEL_PATH, create_classifier

__all__ = ["DEFAULT_MODEL_PATH", "DeviceEnum", "create_classifier", "detect_device"]
