"""
LLM processor for generating clinical summaries for general practitioners.

IMPORTANT: The LLM does NOT extract symptoms or make predictions.
Bio_ClinicalBERT handles all symptom understanding and disease prediction.
Ollama ONLY generates professional clinical summaries for healthcare professionals.
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
You are a clinical decision support assistant for general practitioners.
Based on the AI analysis results below, provide a professional clinical summary in English.

Clinical presentation: {user_text}

AI Analysis - Differential diagnosis probabilities:
{all_probs_text}

Primary diagnosis: {disease} ({confidence:.0%} confidence)

Guidelines:
- Present the differential diagnosis with all significant probabilities
- Provide a clinical interpretation of the confidence levels
- If confidence is low (<60%), emphasize the diagnostic uncertainty and suggest alternative approaches
- If the disease is "unknown", indicate that the symptom pattern doesn't match the training dataset
- Suggest relevant clinical examinations or diagnostic tests to confirm/rule out the top conditions
- Highlight any red flags or urgent findings that warrant immediate attention
- Maintain a professional, evidence-based tone
- Keep the response concise and actionable for clinical decision-making

Clinical Summary:""")


def generate_response(
    user_text: str,
    prediction: dict,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate a clinical summary for general practitioners based on BERT prediction.

    Args:
        user_text: Original clinical presentation text
        prediction: Dict from DiseaseClassifier.predict() with keys:
                   - disease: predicted disease or "unknown"
                   - confidence: float 0-1
                   - all_probs: dict of {disease: probability}
        base_url: Ollama server URL (default: http://localhost:11434)
        model: Ollama model name (default: llama3)

    Returns:
        Professional clinical summary string for practitioners
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
