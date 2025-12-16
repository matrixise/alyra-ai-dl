import json
import pathlib
import typing

import typer

from alyra_ai_dl import create_classifier, detect_device, predict_with_threshold

ModelPath = typing.Annotated[
    pathlib.Path,
    typer.Argument(exists=True, dir_okay=True),
]


def main(
    model_path: ModelPath,
    threshold: float = 0.5,
    top_k: int | None = None,
    test_cases_path: typing.Annotated[pathlib.Path, typer.Option(exists=True, file_okay=True)] = pathlib.Path(
        "data/test_cases.json"
    ),
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

    test_cases = [
        (
            record["disease"],
            record["symptoms"],
        )
        for record in json.loads(test_cases_path.read_text())
    ]

    for expected_disease, test_symptoms in test_cases:
        result = predict_with_threshold(classifier, test_symptoms, threshold)
        match = "✓" if result["disease"] == expected_disease else "✗"
        print(
            f"{match} Expected: {expected_disease:20s} | "
            f"Predicted: {result['disease']:20s} | "
            f"Confidence: {result['confidence']:.2%}"
        )


if __name__ == "__main__":
    typer.run(main)
