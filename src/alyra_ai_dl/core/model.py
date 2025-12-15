"""Model loading and configuration utilities."""

import pathlib

from transformers import pipeline

from alyra_ai_dl.core.device import DeviceEnum, detect_device

DEFAULT_MODEL_PATH = pathlib.Path("./models/symptom_classifier-old-trainer/final/")


def create_classifier(
    model_path: pathlib.Path | str | None = None,
    device: str | DeviceEnum | None = None,
    top_k: int | None = None,
):
    """
    Create a text classification pipeline for symptom classification.

    Args:
        model_path: Path to the trained model (default: DEFAULT_MODEL_PATH)
        device: Device to use for inference (default: auto-detection)
        top_k: Number of predictions to return (default: None = all scores)

    Returns:
        Configured text classification pipeline

    Example:
        >>> classifier = create_classifier(top_k=2)
        >>> results = classifier("hip pain, back pain")
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if device is None:
        device = detect_device()

    # Convert DeviceEnum to string if necessary
    device_str = device.value if hasattr(device, "value") else device

    kwargs = {
        "model": str(model_path),
        "tokenizer": str(model_path),
        "device": device_str,
    }

    # Add top_k (None = all scores, otherwise top K results)
    kwargs["top_k"] = top_k

    return pipeline("text-classification", **kwargs)
