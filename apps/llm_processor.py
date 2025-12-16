"""
LLM processor for generating clinical summaries for general practitioners.

IMPORTANT: The LLM does NOT extract symptoms or make predictions.
Bio_ClinicalBERT handles all symptom understanding and disease prediction.
The LLM ONLY generates professional clinical summaries for healthcare professionals.

Supports multiple LLM backends: Ollama (local), OpenAI, and Lightning AI.
"""

import os
from enum import StrEnum

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

# Ollama defaults (local)
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"

# OpenAI defaults
DEFAULT_OPENAI_MODEL = "gpt-4"

# Lightning AI defaults (cloud)
DEFAULT_LIGHTNING_URL = "https://lightning.ai/api/v1/"
DEFAULT_LIGHTNING_MODEL = "lightning-ai/llama-3.3-70b"


def get_ollama_config(
    base_url: str | None = None,
    model: str | None = None,
) -> dict:
    """
    Get Ollama configuration from environment or parameters.

    Args:
        base_url: Override for Ollama URL
        model: Override for model name

    Returns:
        Dict with 'base_url' and 'model' keys

    Environment Variables:
        - OLLAMA_BASE_URL: Ollama server URL
        - OLLAMA_MODEL: Ollama model name
    """
    return {
        "base_url": base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        "model": model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
    }


def get_lightning_config(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Get Lightning AI configuration from environment or parameters.

    Args:
        base_url: Override for API URL
        model: Override for model name
        api_key: Override for API key

    Returns:
        Dict with 'base_url', 'model', and 'api_key' keys

    Raises:
        ValueError: If no API key is provided or found in environment

    Environment Variables:
        - LIGHTNING_BASE_URL: Lightning AI API URL
        - LIGHTNING_MODEL: Lightning AI model name
        - LIGHTNING_API_KEY: Lightning AI API key (required)
    """
    key = api_key or os.getenv("LIGHTNING_API_KEY")

    if not key:
        raise ValueError(
            "Lightning AI backend requires an API key. "
            "Set LIGHTNING_API_KEY environment variable or pass api_key parameter."
        )

    return {
        "base_url": os.getenv("LIGHTNING_BASE_URL", DEFAULT_LIGHTNING_URL),
        "model": os.getenv("LIGHTNING_MODEL", DEFAULT_LIGHTNING_MODEL),
        "api_key": key,
    }


class LLMBackend(StrEnum):
    """Supported LLM backends."""

    OLLAMA = "ollama"
    LIGHTNING = "lightning"


def get_llm(
    backend: LLMBackend | str = LLMBackend.OLLAMA,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> BaseLanguageModel:
    """
    Create an LLM instance with the specified backend and configuration.

    Configuration is read from environment variables if not provided.

    Args:
        backend: LLM backend to use ("ollama" or "lightning")
        base_url: Base URL for the API (uses env or defaults if None)
        model: Model name (uses env or defaults if None)
        api_key: API key for Lightning (uses env if None)

    Returns:
        LLM instance compatible with LangChain

    Raises:
        ValueError: If Lightning backend is used without API key

    Environment Variables:
        - OLLAMA_BASE_URL: Ollama server URL
        - OLLAMA_MODEL: Ollama model name
        - LIGHTNING_BASE_URL: Lightning AI API URL
        - LIGHTNING_MODEL: Lightning AI model name
        - LIGHTNING_API_KEY: Lightning AI API key (required)

    Examples:
        >>> # Use Ollama with defaults from environment
        >>> llm = get_llm()

        >>> # Use Lightning AI
        >>> llm = get_llm(backend="lightning")

        >>> # Override configuration
        >>> llm = get_llm(backend="ollama", base_url="http://custom:11434")
    """
    backend = LLMBackend(backend) if isinstance(backend, str) else backend

    if backend == LLMBackend.OLLAMA:
        config = get_ollama_config(base_url, model)
        return OllamaLLM(model=config["model"], base_url=config["base_url"])

    elif backend == LLMBackend.LIGHTNING:
        config = get_lightning_config(base_url, model, api_key)
        return ChatOpenAI(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
        )

    else:
        raise ValueError(f"Unsupported backend: {backend}")


RESPONSE_PROMPT = PromptTemplate.from_template("""
You are a clinical decision support assistant for general practitioners.
Based on the AI analysis results below, provide a professional clinical summary in English.

Clinical presentation: {user_text}

AI Analysis - Differential diagnosis probabilities:
{all_probs_text}

Primary diagnosis: {disease} ({confidence:.0%} confidence)

Guidelines:
- **IMPORTANT**: Always emphasize that this is a decision support tool only and
  that a proper medical consultation and clinical examination by a qualified healthcare
  professional is REQUIRED for diagnosis and treatment
- Present the differential diagnosis with all significant probabilities
- Provide a clinical interpretation of the confidence levels
- If confidence is low (<60%), emphasize the diagnostic uncertainty and suggest alternative approaches
- If the disease is "unknown", indicate that the symptom pattern doesn't match the training dataset
- Suggest relevant clinical examinations or diagnostic tests to confirm/rule out the top conditions
- Highlight any red flags or urgent findings that warrant immediate attention
- Maintain a professional, evidence-based tone
- Keep the response concise and actionable for clinical decision-making
- End with a reminder that professional medical advice is essential

Clinical Summary:""")


def generate_response(
    user_text: str,
    prediction: dict,
    backend: LLMBackend | str = LLMBackend.OLLAMA,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    """
    Generate a clinical summary for general practitioners based on BERT prediction.

    Args:
        user_text: Original clinical presentation text
        prediction: Dict from DiseaseClassifier.predict() with keys:
                   - disease: predicted disease or "unknown"
                   - confidence: float 0-1
                   - all_probs: dict of {disease: probability}
        backend: LLM backend to use ("ollama", "openai", "lightning")
        base_url: Base URL for the API (uses backend defaults if None)
        model: Model name (uses backend defaults if None)

    Returns:
        Professional clinical summary string for practitioners

    Raises:
        ValueError: If backend requires API key but none is available
        ConnectionError: If backend server is unreachable

    Examples:
        >>> # Use Ollama (default)
        >>> response = generate_response(text, prediction)
        >>> # Use Lightning AI
        >>> response = generate_response(text, prediction, backend="lightning")
        >>> # Custom Ollama server
        >>> response = generate_response(
        ...     text, prediction,
        ...     backend="ollama",
        ...     base_url="http://custom:11434",
        ...     model="mistral"
        ... )
    """
    # Format all probabilities for the prompt (filter out < 1%)
    all_probs = prediction.get("all_probs", {})
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    # Only include probabilities >= 1%
    filtered_probs = [(disease, prob) for disease, prob in sorted_probs if prob >= 0.01]
    all_probs_text = "\n".join(f"- {disease}: {prob:.1%}" for disease, prob in filtered_probs)

    llm = get_llm(backend=backend, base_url=base_url, model=model)
    chain = RESPONSE_PROMPT | llm
    result = chain.invoke(
        {
            "user_text": user_text,
            "disease": prediction["disease"],
            "confidence": prediction["confidence"],
            "all_probs_text": all_probs_text,
        }
    )

    # ChatOpenAI returns AIMessage, OllamaLLM returns str
    # Extract content if it's an AIMessage object
    if hasattr(result, "content"):
        return result.content
    return result
