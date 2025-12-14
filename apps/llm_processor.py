"""
LLM processor for generating patient-friendly responses.

IMPORTANT: The LLM does NOT extract symptoms or make predictions.
Bio_ClinicalBERT handles all symptom understanding and disease prediction.
Ollama ONLY generates explanatory text for the patient.
"""

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"


def get_llm(
    base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL
) -> OllamaLLM:
    """Create an OllamaLLM instance with the specified URL and model."""
    return OllamaLLM(model=model, base_url=base_url)


RESPONSE_PROMPT = PromptTemplate.from_template("""
You are a helpful medical assistant. Based on the AI analysis results below,
provide a clear, empathetic response in English to the patient.

Patient's description: {user_text}

AI Analysis - All possible conditions:
{all_probs_text}

Top prediction: {disease} ({confidence:.0%} confidence)

Guidelines:
- Present ALL the possible conditions with their probabilities
- Explain what the probabilities mean in simple terms
- If the top confidence is low, emphasize the uncertainty
- If the disease is "unknown", explain that the symptoms don't match known patterns
- Always remind them this is not a medical diagnosis
- Suggest consulting a healthcare professional
- Be empathetic and reassuring
- Keep the response concise and easy to understand

Response:""")


def generate_response(
    user_text: str,
    prediction: dict,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate a patient-friendly response based on BERT prediction.

    Args:
        user_text: Original text from user
        prediction: Dict from DiseaseClassifier.predict() with keys:
                   - disease: predicted disease or "unknown"
                   - confidence: float 0-1
                   - all_probs: dict of {disease: probability}
        base_url: Ollama server URL (default: http://localhost:11434)
        model: Ollama model name (default: llama3)

    Returns:
        Human-friendly response string
    """
    # Format all probabilities for the prompt (filter out < 1%)
    all_probs = prediction.get("all_probs", {})
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    # Only include probabilities >= 1%
    filtered_probs = [(disease, prob) for disease, prob in sorted_probs if prob >= 0.01]
    all_probs_text = "\n".join(
        f"- {disease}: {prob:.1%}" for disease, prob in filtered_probs
    )

    llm = get_llm(base_url=base_url, model=model)
    chain = RESPONSE_PROMPT | llm
    return chain.invoke(
        {
            "user_text": user_text,
            "disease": prediction["disease"],
            "confidence": prediction["confidence"],
            "all_probs_text": all_probs_text,
        }
    )
