# Applications

This directory contains user-facing applications using the alyra_ai_dl package.

## Available Apps

### 1. CLI - Interactive Terminal Interface

**File:** `cli.py`
**Description:** Interactive command-line interface for symptom analysis with optional Ollama LLM explanations.

**Usage:**

```bash
# Basic usage (default model and settings)
python apps/cli.py

# Custom model
python apps/cli.py --model ./models/my-model

# Adjust confidence threshold
python apps/cli.py --threshold 0.7

# Skip LLM (BERT only mode)
python apps/cli.py --no-llm

# Custom Ollama configuration
python apps/cli.py --ollama-url http://192.168.1.100:11434 --ollama-model llama2
```

**Features:**

- Interactive REPL for continuous symptom input
- Rich terminal output with tables and panels
- Optional Ollama LLM explanations in English
- Device auto-detection (CPU/CUDA/MPS)
- Confidence thresholding for unknown detection
- Display all disease probabilities

**Requirements:**

- Ollama running locally (`ollama serve`) if LLM features are used
- Default model at `./models/symptom_classifier-mini/final` or specify with `--model`

**Command-Line Options:**

- `--model`, `-m` - Path to the trained model (default: `models/symptom_classifier-mini/final`)
- `--threshold`, `-t` - Confidence threshold 0.0-1.0 (default: 0.55)
- `--no-llm` - Skip Ollama response generation (BERT only mode)
- `--ollama-url`, `-u` - Ollama server URL (default: `http://localhost:11434`)
- `--ollama-model` - Ollama model name (default: `llama3`)

---

### 2. API - FastAPI REST Service

**File:** `api.py`
**Description:** REST API for disease classification with batch prediction support.

**Usage:**

```bash
# Start the API server
python apps/api.py

# Or with uvicorn directly
uvicorn apps.api:app --reload
```

**Endpoints:**

- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /health` - Health check
- `GET /info` - Model information

**API Documentation:**

- Interactive docs: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json

**Environment Variables:**

- `MODEL_PATH` - Path to the model (default: `models/symptom_classifier-mini/final`)
- `THRESHOLD` - Confidence threshold (default: 0.55)
- `API_TITLE` - API title (default: "Disease Classification API")
- `API_VERSION` - API version (default: "1.0.0")

---

### 3. Streamlit - Web UI

**File:** `streamlit_app.py`
**Description:** Interactive web interface for symptom analysis with visual charts.

**Usage:**

```bash
# Start the Streamlit app
streamlit run apps/streamlit_app.py

# Or with task
task app:streamlit:dev
```

**Features:**

- Web-based UI with model configuration
- Interactive symptom input with examples
- Probability charts and tables
- Device information display
- Configurable threshold and top-k parameters

**Access:**

- Default URL: http://localhost:8501

---

## Shared Components

### LLM Processor

**File:** `llm_processor.py`
**Description:** Ollama integration for generating patient-friendly medical explanations in English.

**Functions:**

- `generate_response(user_text, prediction, base_url, model)` - Generate explanation from prediction

**Requirements:**

- Ollama server running (`ollama serve`)
- LLM model installed (`ollama pull llama3`)

**Note:** The LLM does NOT make medical predictions. It only generates explanatory text based on Bio_ClinicalBERT predictions.

---

## Dependencies

All applications use the `alyra_ai_dl` package for core functionality:

- **Core:** `create_classifier()`, `detect_device()`, `DEFAULT_MODEL_PATH`
- **Inference:** `predict_with_threshold()`

Additional dependencies:

- **CLI:** typer, rich, langchain-core, langchain-ollama
- **API:** fastapi, uvicorn, pydantic-settings
- **Streamlit:** streamlit

---

## Quick Start

### 1. Install Dependencies

```bash
# Initialize virtual environment and install all dependencies
task venv:init
source .venv/bin/activate
```

### 2. Prepare Model

Make sure you have a trained model available:

```bash
# Default location
./models/symptom_classifier-mini/final/

# Or specify custom path with --model flag
```

### 3. Run an Application

```bash
# CLI
python apps/cli.py

# API
python apps/api.py

# Streamlit
streamlit run apps/streamlit_app.py
```

### 4. (Optional) Setup Ollama for LLM Features

```bash
# Install Ollama (see https://ollama.ai)
# Start Ollama server
ollama serve

# Pull llama3 model
ollama pull llama3

# Now CLI and other apps can use LLM features
python apps/cli.py  # Will include LLM explanations
```

---

## Development

### Code Quality

```bash
# Format code
task code:format

# Check linting
task code:lint

# Run all checks
task code:check

# Auto-fix all issues
task code:fix
```

### Testing

```bash
# Test CLI
python apps/cli.py --no-llm
# Input: hip pain, back pain, neck pain
# Expected: Prediction table with spondylolisthesis

# Test API
python apps/api.py
# Visit http://localhost:8000/docs
# Try /predict endpoint

# Test Streamlit
streamlit run apps/streamlit_app.py
# Visit http://localhost:8501
```

---

## Troubleshooting

### CLI Issues

**Model not found:**

```bash
python apps/cli.py --model ./models/symptom_classifier-mini/final
```

**Ollama not running:**

```bash
# Start Ollama
ollama serve

# Or skip LLM
python apps/cli.py --no-llm
```

### API Issues

**Port already in use:**

```bash
# Use different port
uvicorn apps.api:app --port 8001
```

**Model loading error:**

```bash
# Check MODEL_PATH environment variable
export MODEL_PATH=./models/your-model
python apps/api.py
```

### Streamlit Issues

**Port conflict:**

```bash
# Use different port
streamlit run apps/streamlit_app.py --server.port 8502
```

---

## Architecture

All applications follow the same pattern:

1. **Model Loading:** Use `create_classifier()` with device auto-detection
2. **Inference:** Use `predict_with_threshold()` for predictions
3. **LLM (optional):** Use `generate_response()` for explanations

This ensures consistency across all interfaces (CLI, API, Web UI).
