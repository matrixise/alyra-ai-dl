# Applications

This directory contains user-facing applications using the alyra_ai_dl package.

## Available Apps

### Task Commands (Quick Start)

```bash
task app:cli              # CLI interactive terminal
task app:api:dev          # FastAPI REST server
task app:streamlit:dev    # Streamlit web UI
task app:chainlit:dev     # Chainlit chat interface
```

---

## Environment Configuration

### LLM Backend Configuration (Required for CLI and Chainlit)

Applications that use LLM features (CLI, Chainlit) **require** LLM backend configuration via environment variables.

**Configure in `.env` file:**

```bash
# LLM Backend Selection (REQUIRED)
LLM_BACKEND=lightning  # Options: ollama, lightning

# Lightning AI Configuration (if using lightning backend)
LIGHTNING_API_KEY=your-api-key-here

# Ollama Configuration (if using ollama backend)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

**Available Backends:**

1. **Ollama (local)** - Free, runs locally
   - Requires: Ollama server running (`ollama serve`)
   - Set: `LLM_BACKEND=ollama`

2. **Lightning AI (cloud)** - Hosted service
   - Requires: `LIGHTNING_API_KEY` environment variable
   - Set: `LLM_BACKEND=lightning`

**⚠️ Important:** Without proper LLM configuration, applications will fail or run in BERT-only mode (no clinical summaries).

**CLI Override:**

```bash
# Use specific backend for CLI
task app:cli -- --llm-backend lightning
task app:cli -- --llm-backend ollama
```

---

### 1. CLI - Interactive Terminal Interface

**File:** `cli.py`
**Description:** Interactive command-line interface for clinical decision support with optional LLM explanations.

**Usage:**

```bash
# Basic usage (requires LLM_BACKEND environment variable)
python apps/cli.py

# Specify LLM backend explicitly
python apps/cli.py --llm-backend lightning
python apps/cli.py --llm-backend ollama

# Custom model
python apps/cli.py --model ./models/my-model

# Adjust confidence threshold
python apps/cli.py --threshold 0.7

# Skip LLM (BERT only mode)
python apps/cli.py --no-llm

# Custom Ollama configuration
python apps/cli.py --llm-backend ollama --ollama-url http://192.168.1.100:11434 --ollama-model llama2
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
- `--llm-backend` - LLM backend to use: `ollama`, `lightning` (default: from `LLM_BACKEND` env var)
- `--no-llm` - Skip LLM response generation (BERT only mode)
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

### 4. Chainlit - Chat Interface

**File:** `chainlit_app.py`
**Description:** Interactive chat-based interface for clinical decision support with step-by-step visualization.

**Usage:**

```bash
# Start Chainlit app
chainlit run apps/chainlit_app.py

# Or with task
task app:chainlit:dev
```

**Features:**

- Chat-based conversational interface for general practitioners
- Step-by-step analysis visualization
- Bio_ClinicalBERT predictions with confidence scores
- LLM-generated professional clinical summaries
- Differential diagnosis with probabilities
- Technical details display (disease, confidence, suggestions)

**Requirements:**

- **LLM Backend Configuration (REQUIRED)** - See [Environment Configuration](#environment-configuration)
  - Set `LLM_BACKEND` environment variable (ollama or lightning)
  - Configure API keys or Ollama server accordingly
- Bio_ClinicalBERT model at `DEFAULT_MODEL_PATH` or configured model
- Chainlit installed (`pip install chainlit` or via `pyproject.toml`)

**Access:**

- Default URL: http://localhost:8000 (Chainlit default port)
- Configuration: `.chainlit/config.toml`

**Example Interaction:**

```
User: I have rapid heartbeat, sweating, trembling, shortness of breath

[BERT Analysis]
Predicted: panic disorder (86.1%)

[Ollama Response]
Based on your symptoms, the analysis suggests...
```

---

## Shared Components

### LLM Processor

**File:** `llm_processor.py`
**Description:** Multi-backend LLM integration for generating professional clinical summaries in English.

**Supported Backends:**
- **Ollama** (local) - Free, requires local Ollama server
- **Lightning AI** (cloud) - Hosted service, requires API key

**Functions:**

- `generate_response(user_text, prediction, backend, base_url, model)` - Generate clinical summary from BERT prediction

**Configuration:**

- **REQUIRED:** Set `LLM_BACKEND` environment variable (see [Environment Configuration](#environment-configuration))
- **Lightning AI:** Requires `LIGHTNING_API_KEY` environment variable
- **Ollama:** Requires Ollama server running (`ollama serve`) and model installed (`ollama pull llama3`)

**Note:** The LLM does NOT make medical predictions. It only generates professional clinical summaries based on Bio_ClinicalBERT predictions for general practitioners.

---

## Dependencies

All applications use the `alyra_ai_dl` package for core functionality:

- **Core:** `create_classifier()`, `detect_device()`, `DEFAULT_MODEL_PATH`
- **Inference:** `predict_with_threshold()`

Additional dependencies:

- **CLI:** typer, rich, langchain-core, langchain-ollama
- **API:** fastapi, uvicorn, pydantic-settings
- **Streamlit:** streamlit
- **Chainlit:** chainlit, langchain-core, langchain-ollama

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
task app:cli
# or: python -m apps.cli

# API
task app:api:dev
# or: python apps/api.py

# Streamlit
task app:streamlit:dev
# or: streamlit run apps/streamlit_app.py

# Chainlit
task app:chainlit:dev
# or: chainlit run apps/chainlit_app.py
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
