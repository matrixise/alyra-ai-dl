"""
Chainlit app for symptom-to-disease classification.

Architecture:
1. User text -> Bio_ClinicalBERT (prediction)
2. Prediction -> LLM (generate explanation)

Run with: chainlit run apps/chainlit_app.py

Configuration:
- Set LLM_BACKEND env var to choose backend (ollama, openai, lightning)
- Default: ollama
"""

import os

import chainlit as cl
from llm_processor import generate_response

from alyra_ai_dl import (
    DEFAULT_MODEL_PATH,
    create_classifier,
    detect_device,
    predict_with_threshold,
)

# Load LLM backend configuration from environment
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# Load Bio_ClinicalBERT model at startup
classifier = None


@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    global classifier

    # Load model if not already loaded
    if classifier is None:
        await cl.Message(content="Loading DiagnosIA model...").send()
        device = detect_device()
        classifier = create_classifier(
            model_path=DEFAULT_MODEL_PATH,
            device=device,
            top_k=None,  # Return all probabilities
        )

    await cl.Message(
        content="Hello! I'm a clinical decision support assistant for general practitioners. "
        "Describe your patient's clinical presentation and I'll provide a differential diagnosis analysis.\n\n"
        "**Important**: This is a decision support tool for healthcare professionals only. "
        "All diagnoses must be validated through proper clinical examination."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""
    user_text = message.content

    # Step 1: Bio_ClinicalBERT predicts disease from user text directly
    async with cl.Step(name="Analyzing with DiagnosIA...") as step:
        prediction = predict_with_threshold(classifier, user_text, threshold=0.55)
        step.output = f"Predicted: {prediction['disease']} ({prediction['confidence']:.1%})"

    # Step 2: LLM generates patient-friendly explanation
    async with cl.Step(name=f"Generating response with {LLM_BACKEND}...") as step:
        try:
            response = generate_response(user_text, prediction, backend=LLM_BACKEND)
            step.output = f"Response generated using {LLM_BACKEND}"
        except ValueError as e:
            # API key missing
            response = f"⚠️ LLM unavailable: {e}\n\n**BERT Prediction Only:**\n"
            response += f"- Disease: {prediction['disease']}\n"
            response += f"- Confidence: {prediction['confidence']:.1%}"
            step.output = f"LLM unavailable: {e}"
        except ConnectionError as e:
            # LLM server unreachable
            response = f"⚠️ LLM connection failed: {e}\n\n**BERT Prediction Only:**\n"
            response += f"- Disease: {prediction['disease']}\n"
            response += f"- Confidence: {prediction['confidence']:.1%}"
            step.output = f"LLM connection failed: {e}"

    # Show technical details
    details = f"""
---
**Technical Details:**
- Predicted: {prediction["disease"]}
- Confidence: {prediction["confidence"]:.1%}
"""

    if "suggestion" in prediction:
        details += f"\n- Note: {prediction['suggestion']}"

    await cl.Message(content=response + details).send()
