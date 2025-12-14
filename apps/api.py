"""
API FastAPI pour la classification de maladies à partir de symptômes.
"""

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import uvicorn

from alyra_ai_dl.core import create_classifier, detect_device
from alyra_ai_dl.inference import predict_with_threshold


# ============================================================================
# Configuration
# ============================================================================


class Settings(BaseSettings):
    """Configuration de l'application depuis variables d'environnement."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Model settings
    model_path: str = "./models/symptom_classifier-mini/final"

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_title: str = "Disease Classifier API"
    api_version: str = "1.0.0"

    # Prediction settings
    default_threshold: float = 0.55
    max_batch_size: int = 100

    # Logging
    log_level: str = "INFO"


settings = Settings()

# Configuration du logger
logger.remove()  # Retirer le handler par défaut
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level,
)


# ============================================================================
# Pydantic Models
# ============================================================================


class PredictRequest(BaseModel):
    """Requête de prédiction pour un ensemble de symptômes."""

    symptoms: str = Field(
        ..., description="Symptômes séparés par des virgules", min_length=1
    )
    threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Seuil de confiance minimum (défaut depuis config)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symptoms": "hip pain, back pain, neck pain, low back pain",
                    "threshold": 0.55,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """Réponse de prédiction."""

    disease: str
    confidence: float
    threshold: float
    all_probs: dict[str, float]
    symptoms: str
    suggestion: str | None = None


class BatchPredictRequest(BaseModel):
    """Requête de prédiction batch."""

    items: list[PredictRequest] = Field(
        ..., min_length=1, max_length=settings.max_batch_size
    )


class BatchPredictResponse(BaseModel):
    """Réponse de prédiction batch."""

    results: list[PredictResponse]
    total: int
    success_count: int


class HealthResponse(BaseModel):
    """Réponse du health check."""

    status: str
    model_loaded: bool
    device: str | None = None


class ModelInfoResponse(BaseModel):
    """Informations sur le modèle."""

    model_path: str
    device: str
    supported_diseases: list[str]
    num_labels: int


# ============================================================================
# Lifespan Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application (startup/shutdown)."""
    # Startup: Charger le modèle
    logger.info("🚀 Loading model from {}", settings.model_path)
    device = detect_device()
    app.state.classifier = create_classifier(
        model_path=settings.model_path,
        top_k=None,
    )
    app.state.model_info = {
        "model_path": settings.model_path,
        "device": str(device.value),
        "labels": list(app.state.classifier.model.config.id2label.values()),
    }
    logger.success("✅ Model loaded on {} device", device.value)
    logger.info("📋 Supported diseases: {}", ", ".join(app.state.model_info["labels"]))

    yield

    # Shutdown: Nettoyer
    logger.info("🛑 Shutting down API...")
    if hasattr(app.state, "classifier"):
        del app.state.classifier
    logger.success("✅ Shutdown complete")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title=settings.api_title,
    description="API de classification de maladies à partir de symptômes utilisant un modèle transformers",
    version=settings.api_version,
    lifespan=lifespan,
)


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/", tags=["Root"])
async def root():
    """Page d'accueil de l'API."""
    return {
        "message": "Disease Classifier API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "batch": "/predict/batch",
            "info": "/info",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health(request: Request):
    """Vérifie que l'API et le modèle sont opérationnels."""
    model_loaded = hasattr(request.app.state, "classifier")
    device = None

    if model_loaded:
        device = request.app.state.model_info.get("device")

    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": device,
    }


@app.get("/info", response_model=ModelInfoResponse, tags=["Info"])
async def model_info(request: Request):
    """Retourne les informations sur le modèle chargé."""
    if not hasattr(request.app.state, "classifier"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = request.app.state.model_info
    return {
        "model_path": info["model_path"],
        "device": info["device"],
        "supported_diseases": info["labels"],
        "num_labels": len(info["labels"]),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: Request, payload: PredictRequest):
    """Prédit une maladie à partir de symptômes."""
    if not hasattr(request.app.state, "classifier"):
        logger.error("Classifier not loaded in app.state")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        classifier = request.app.state.classifier
        threshold = (
            payload.threshold
            if payload.threshold is not None
            else settings.default_threshold
        )

        logger.debug("Predicting for symptoms: {}", payload.symptoms[:50])
        result = predict_with_threshold(classifier, payload.symptoms, threshold)
        logger.info(
            "Prediction result: {} (confidence: {:.2%})",
            result["disease"],
            result["confidence"],
        )
        return result
    except Exception as e:
        logger.exception("Prediction error: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: Request, payload: BatchPredictRequest):
    """Prédit des maladies pour plusieurs ensembles de symptômes."""
    if not hasattr(request.app.state, "classifier"):
        logger.error("Classifier not loaded in app.state")
        raise HTTPException(status_code=503, detail="Model not loaded")

    classifier = request.app.state.classifier
    results = []
    success_count = 0
    batch_size = len(payload.items)

    logger.info("Processing batch of {} predictions", batch_size)

    for idx, item in enumerate(payload.items, 1):
        try:
            threshold = (
                item.threshold
                if item.threshold is not None
                else settings.default_threshold
            )
            result = predict_with_threshold(classifier, item.symptoms, threshold)
            results.append(result)
            if result["disease"] != "unknown":
                success_count += 1
            logger.debug("[{}/{}] Predicted: {}", idx, batch_size, result["disease"])
        except Exception as e:
            # En cas d'erreur, retourner un résultat d'erreur
            logger.warning("[{}/{}] Prediction failed: {}", idx, batch_size, str(e))
            results.append(
                {
                    "disease": "error",
                    "confidence": 0.0,
                    "threshold": item.threshold or settings.default_threshold,
                    "all_probs": {},
                    "symptoms": item.symptoms,
                    "suggestion": f"Error: {str(e)}",
                }
            )

    logger.info(
        "Batch complete: {}/{} successful predictions", success_count, batch_size
    )

    return {
        "results": results,
        "total": len(results),
        "success_count": success_count,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting {} v{}", settings.api_title, settings.api_version)
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
