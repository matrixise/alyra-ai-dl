"""Package alyra_ai_dl pour classification de maladies à partir de symptômes."""

from alyra_ai_dl.core.device import DeviceEnum, detect_device
from alyra_ai_dl.core.model import DEFAULT_MODEL_PATH, create_classifier
from alyra_ai_dl.inference.pipeline import predict_with_threshold

__version__ = "0.1.0"
__all__ = [
    "DeviceEnum",
    "detect_device",
    "create_classifier",
    "predict_with_threshold",
    "DEFAULT_MODEL_PATH",
]
