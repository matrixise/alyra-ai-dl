import pathlib
import typing

import typer

from alyra_ai_dl import create_classifier, detect_device, predict_with_threshold

ModelPath = typing.Annotated[
    pathlib.Path,
    typer.Argument(exists=True, dir_okay=True),
]

TEST_CASES = [
    (
        "spondylolisthesis",
        "low back pain, problems with movement, paresthesia, leg cramps or spasms, leg weakness",
    ),
    (
        "spondylolisthesis",
        "hip pain, back pain, neck pain, low back pain, problems with movement, loss of sensation, leg cramps or spasms",
    ),
    (
        "herniated disk",
        "arm pain, back pain, neck pain, paresthesia, shoulder pain, arm weakness",
    ),
    ("herniated disk", "loss of sensation, paresthesia, shoulder pain"),
    (
        "panic disorder",
        "depressive or psychotic symptoms, irregular heartbeat, breathing fast",
    ),
    ("panic disorder", "insomnia, palpitations, irregular heartbeat"),
    (
        "panic disorder",
        "anxiety and nervousness, shortness of breath, depressive or psychotic symptoms, chest tightness, palpitations, irregular heartbeat, breathing fast",
    ),
]


def main(
    model_path: ModelPath,
    threshold: float = 0.5,
    top_k: int | None = None,
):
    print(f"Loading model from: {model_path}")

    # Utilise la fonction factory pour créer le classifier
    classifier = create_classifier(
        model_path=model_path,
        device=detect_device(),
        top_k=top_k,
    )

    # Afficher les labels disponibles
    print(f"Available labels: {classifier.model.config.id2label}\n")

    for expected_disease, test_symptoms in TEST_CASES:
        result = predict_with_threshold(classifier, test_symptoms, threshold)
        match = "✓" if result["disease"] == expected_disease else "✗"
        print(
            f"{match} Expected: {expected_disease:20s} | "
            f"Predicted: {result['disease']:20s} | "
            f"Confidence: {result['confidence']:.2%}"
        )


if __name__ == "__main__":
    typer.run(main)
