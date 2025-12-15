"""
Chainlit app for symptom-to-disease classification.

Architecture:
1. User text -> Bio_ClinicalBERT (prediction)
2. Prediction -> Ollama (generate explanation)

Run with: chainlit run apps/chainlit_app.py
"""

from pathlib import Path

import chainlit as cl

from alyra_ai_dl import (
    DEFAULT_MODEL_PATH,
    create_classifier,
    detect_device,
    predict_with_threshold,
)
from apps.llm_processor import generate_response

# Load Bio_ClinicalBERT model at startup
classifier = None


@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    global classifier

    # Load model if not already loaded
    if classifier is None:
        await cl.Message(content="Loading Bio_ClinicalBERT model...").send()
        device = detect_device()
        classifier = create_classifier(
            model_path=DEFAULT_MODEL_PATH,
            device=device,
            top_k=None  # Return all probabilities
        )

    await cl.Message(
        content="Hello! I'm a medical symptom analyzer. "
        "Describe your symptoms and I'll try to help identify potential conditions.\n\n"
        "**Disclaimer**: This is not medical advice. Always consult a doctor."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""
    user_text = message.content

    # Step 1: Bio_ClinicalBERT predicts disease from user text directly
    async with cl.Step(name="Analyzing with Bio_ClinicalBERT...") as step:
        prediction = predict_with_threshold(classifier, user_text, threshold=0.55)
        step.output = f"Predicted: {prediction['disease']} ({prediction['confidence']:.1%})"

    # Step 2: Ollama generates patient-friendly explanation
    async with cl.Step(name="Generating response...") as step:
        response = generate_response(user_text, prediction)
        step.output = "Response generated"

    # Show technical details
    details = f"""
---
**Technical Details:**
- Predicted: {prediction['disease']}
- Confidence: {prediction['confidence']:.1%}
"""

    if 'suggestion' in prediction:
        details += f"\n- Note: {prediction['suggestion']}"

    await cl.Message(content=response + details).send()
