from __future__ import annotations
import json
import pathlib
import typing

# import pytest
import torch
# from rich.pretty import pprint
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
)

from alyra_ai_dl import detect_device

if typing.TYPE_CHECKING:
    from alyra_ai_dl import DeviceEnum


class ModelAndTokenizerTuple(typing.NamedTuple):
    model: BertForSequenceClassification
    tokenizer: BertTokenizerFast


def load_model_and_tokenizer(
    model_path: pathlib.Path,
    device: DeviceEnum,
) -> ModelAndTokenizerTuple:
    tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(model_path)
    model: BertForSequenceClassification = (
        AutoModelForSequenceClassification.from_pretrained(model_path)
    )
    model.to(device.value)
    model.eval()
    return ModelAndTokenizerTuple(model, tokenizer)


class DiseaseClassifier:
    def __init__(
        self,
        model: BertForSequenceClassification,
        tokenizer: BertTokenizerFast,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = self.model.device
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        # print(f"Loaded model from {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Labels: {list(self.label2id.keys())}")

    def predict(self, text: str, threshold: float | None = None) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        # reassign to the right device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # pprint(inputs)
        # pprint(probs)
        # pprint(outputs)

        # print(f'{probs=}')
        pred_idx = probs.argmax().item()
        confidence = probs[0][pred_idx].item()

        is_unknown = False
        if threshold is not None and confidence < threshold:
            is_unknown = True

        if is_unknown:
            result = {
                "disease": "unknown",
                "confidence": confidence,
                "suggestion": "Symptoms don't match known diseases with sufficient confidence",
            }
        else:
            result = {
                "disease": self.id2label[pred_idx],
                "confidence": confidence,
            }
        result["all_probs"] = {
            self.id2label[i]: probs[0][i].item() for i in range(len(self.id2label))
        }

        result["symptoms"] = text
        result["threshold"] = threshold

        return result


model, tokenizer = load_model_and_tokenizer(
    model_path=pathlib.Path("./models/symptom_classifier-old-trainer/final/"),
    device=detect_device(),
)

# print(f'{model=}')
# pprint(tokenizer)
# print(f'{tokenizer=}')
# print(f'{model.config.id2label=}')
# print(f'{model.config.label2id=}')
classifier = DiseaseClassifier(model, tokenizer)
# symptoms = "leg pain"
symptoms = "hip pain, back pain, neck pain, low back pain, problems with movement, loss of sensation, leg cramps or spasms"
threshold = 0.55
print(f"{symptoms=}")
print(f"{threshold=}")

predictions = classifier.predict(symptoms, threshold)
# pprint(predictions)

# Load test cases from JSON file
test_cases_path = pathlib.Path("./data/test_cases.json")
with open(test_cases_path, encoding="utf-8") as f:
    data = json.load(f)

for test_case in data:
    expected_disease = test_case["disease"]
    symptoms = test_case["symptoms"]
    result = classifier.predict(symptoms, threshold=threshold)
    has_matched = expected_disease == result["disease"]
    confidence = result["confidence"]
    print(f"{expected_disease} -> {has_matched} {confidence:.2%} ")


# @pytest.fixture
# def classifier(
#     model: BertForSequenceClassification,
#     tokenizer: BertTokenizerFast,
# ) -> DiseaseClassifier:
#     return DiseaseClassifier(model, tokenizer)
#
#
# @pytest.mark.parametrize(
#     "test_case",
#     data,
#     ids=[f"{tc['disease'][:15]}..." for tc in data],
# )
# def test_detect_disease(
#     test_case: dict,
#     classifier: DiseaseClassifier,
# ) -> None:
#     expected_disease = test_case["disease"]
#     symptoms = test_case["symptoms"]
#     result = classifier.predict(symptoms, threshold=threshold)
#     assert result["disease"] == expected_disease
#     assert result["confidence"] > 0.55
