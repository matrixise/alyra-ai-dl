"""Prediction pipeline with confidence thresholding."""


def predict_with_threshold(
    classifier_pipe,
    symptoms: str,
    threshold: float = 0.55,
) -> dict:
    """
    Predict disease from symptoms with confidence thresholding.

    Args:
        classifier_pipe: Text classification pipeline
        symptoms: Comma-separated symptoms string
        threshold: Minimum confidence threshold (default: 0.55)

    Returns:
        Dict containing prediction, confidence, and all probabilities

    Example:
        >>> from alyra_ai_dl import create_classifier, predict_with_threshold
        >>> classifier = create_classifier(top_k=None)
        >>> result = predict_with_threshold(classifier, "hip pain, back pain", 0.55)
        >>> print(f"Disease: {result['disease']}, Confidence: {result['confidence']:.2%}")
    """
    # The pipeline returns a list of results (one per text)
    # Each result is a list of dicts {label: str, score: float}
    results = classifier_pipe(symptoms)[0]

    # Find the best prediction
    best_pred = max(results, key=lambda x: x["score"])
    confidence = best_pred["score"]
    predicted_disease = best_pred["label"]

    # Threshold logic for "unknown"
    if threshold is not None and confidence < threshold:
        return {
            "disease": "unknown",
            "confidence": confidence,
            "suggestion": "Symptoms don't match known diseases with sufficient confidence",
            "all_probs": {r["label"]: r["score"] for r in results},
            "symptoms": symptoms,
            "threshold": threshold,
        }
    else:
        return {
            "disease": predicted_disease,
            "confidence": confidence,
            "all_probs": {r["label"]: r["score"] for r in results},
            "symptoms": symptoms,
            "threshold": threshold,
        }
