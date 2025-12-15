"""Visualization functions for model evaluation metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list,
    output_path: Path,
) -> None:
    """
    Save confusion matrix as PNG image.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: List of class names for axis labels
        output_path: Path where to save the confusion matrix image

    Example:
        >>> save_confusion_matrix(
        ...     y_true=np.array([0, 1, 2, 0, 1]),
        ...     y_pred=np.array([0, 1, 1, 0, 1]),
        ...     target_names=["anxiety", "pneumonia", "cystitis"],
        ...     output_path=Path("confusion_matrix.png")
        ... )
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
