# alyra-ai-dl

Medical symptom analyzer using Bio_ClinicalBERT for disease classification. This project provides multiple interfaces (CLI, API, Web UI) to analyze symptoms and predict potential diseases using transformer-based models.

## Prerequisites

This project uses [asdf](https://asdf-vm.com/) for version management. Make sure you have asdf installed with the following plugins:

```bash
asdf plugin add python
asdf plugin add task
asdf plugin add uv
```

## Installation

### 1. Install tool versions

The project specifies exact versions in `.tool-versions`:
- Python: 3.12.12
- Task: 3.45.5
- uv: 0.9.17

Install them with:

```bash
asdf install
```

### 2. Initialize the virtual environment

```bash
task venv:init
```

This will:
- Create a Python virtual environment in `.venv/`
- Install `uv` package manager
- Sync dependencies from `uv.lock`

### 3. Install project dependencies

```bash
task venv:install
```

This installs the project in editable mode with all dev dependencies.

## Tools Used

- **[uv](https://github.com/astral-sh/uv)**: Fast Python package manager (10-100x faster than pip)
- **[ruff](https://github.com/astral-sh/ruff)**: Modern Python linter and formatter (replaces isort, black, flake8)
- **[Task](https://taskfile.dev/)**: Modern task runner (replaces Makefile)
- **[asdf](https://asdf-vm.com/)**: Version manager for Python and CLI tools

## Configuration

### Environment Variables

The project supports configuration via environment variables using a `.env` file.

**Setup:**

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to customize settings:
   ```bash
   # Model path configuration
   MODEL_PATH=./models/symptom_classifier-mini/final/
   ```

**Available Variables:**

- `MODEL_PATH` - Path to the trained model directory (default: `./models/symptom_classifier-old-trainer/final/`)

## Models

The project uses **Bio_ClinicalBERT** fine-tuned for disease classification from symptom descriptions.

**Model Path Configuration:**
- The model path can be configured via the `MODEL_PATH` environment variable in `.env`
- Default: `models/symptom_classifier-old-trainer/final/`
- Alternative models: `models/symptom_classifier-mini/final/`

### Model Features
- Multi-class disease classification
- Confidence scoring with adjustable thresholds
- Support for unknown symptoms (fallback handling)
- Device auto-detection (CUDA, MPS, or CPU)

## Applications

The project includes four application interfaces:

### 1. CLI (Command-Line Interface)

Interactive terminal interface using Bio_ClinicalBERT + Ollama for symptom analysis:

```bash
# Run the CLI
python -m apps.cli

# With custom model
python -m apps.cli --model models/my-model

# Skip LLM response (BERT only)
python -m apps.cli --no-llm

# Custom confidence threshold
python -m apps.cli --threshold 0.7
```

**Features:**
- Interactive symptom input
- BERT-based disease prediction
- Ollama-powered patient-friendly explanations
- Confidence scoring and threshold filtering

### 2. FastAPI (REST API)

Production-ready API with health checks and batch predictions:

```bash
# Run the API
python -m apps.api

# Or with custom settings
API_PORT=8080 MODEL_PATH=./models/my-model python -m apps.api
```

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /info` - Model information
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

**API Documentation:** Visit `http://localhost:8000/docs` after starting the server.

### 3. Streamlit (Web UI)

Interactive web interface with visualizations:

```bash
# Run the Streamlit app
streamlit run apps/streamlit_app.py
```

**Features:**
- User-friendly web interface
- Probability visualizations (charts, tables)
- Adjustable confidence threshold
- Real-time predictions

### 4. Chainlit (Chat Interface)

Interactive chat interface with step-by-step analysis:

```bash
# Run Chainlit
task app:chainlit:dev

# Or directly
chainlit run apps/chainlit_app.py
```

**Features:**
- Chat-based conversational interface
- Step-by-step analysis visualization
- Bio_ClinicalBERT predictions
- Ollama LLM explanations
- Medical disclaimer and guidance

## Examples

### CLI Example

```bash
$ python -m apps.cli

You: I have rapid heartbeat, sweating, trembling, shortness of breath, chest pain

[BERT] Analyzing symptoms...
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric     ┃ Value          ┃
┣━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━┣
┃ Prediction │ panic disorder │
┃ Confidence │ 86.1%          ┃
┗━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━┛

[Ollama] Generating response...
Based on your symptoms, the analysis suggests a possible panic disorder...
```

### API Example

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": "hip pain, back pain, neck pain, low back pain",
    "threshold": 0.55
  }'

# Response
{
  "disease": "spondylolisthesis",
  "confidence": 0.89,
  "threshold": 0.55,
  "all_probs": {
    "spondylolisthesis": 0.89,
    "herniated_disk": 0.08,
    "...": "..."
  }
}
```

## Quick Start

### Code Quality

```bash
# Run linting checks
task code:lint

# Auto-fix linting issues
task code:lint:fix

# Format code
task code:format

# Run all checks (lint + format check) - for CI
task code:check

# Auto-fix everything
task code:fix

# Show ruff configuration
task code:show:config

# Clean cache files
task code:clean
```

### Dependency Management

```bash
# Add a production dependency
task venv:add <package-name>

# Add a dev dependency
task venv:add:dev <package-name>

# Sync dependencies from uv.lock
task venv:sync

# Check for outdated packages
task venv:outdated

# Upgrade all dependencies to latest versions
task venv:upgrade

# Upgrade pip and uv tools
task venv:upgrade:tools
```

## Available Commands

Run `task --list` to see all available commands. Commands are organized by category:

- **`venv:*`** - Virtual environment and dependency management
- **`code:*`** - Code quality (linting, formatting, testing)

## Project Structure

```
alyra-ai-dl/
├── apps/                      # Application interfaces
│   ├── cli.py                 # Command-line interface (BERT + Ollama)
│   ├── api.py                 # FastAPI REST API
│   ├── streamlit_app.py       # Streamlit web interface
│   └── llm_processor.py       # Ollama LLM integration
├── src/
│   └── alyra_ai_dl/           # Core package
│       ├── core/              # Model and device management
│       ├── inference/         # Prediction pipeline
│       └── training/          # Training utilities
├── .tool-versions             # asdf version specifications
├── Taskfile.yml               # Task automation configuration
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Locked dependency versions
├── CLAUDE.md                  # Claude Code integration documentation
├── MEMORY.md                  # Project decisions and session history
└── .venv/                     # Virtual environment (not in git)
```

## Development Workflow

1. Make changes to your code
2. Run `task code:fix` to auto-fix linting and formatting
3. Run `task code:check` to verify everything passes
4. Commit your changes

## Configuration

### Ruff Configuration

Ruff is configured in `pyproject.toml`:
- Target: Python 3.12+
- Line length: 88
- Rules: E, W, F, I, N, UP, B, C4, SIM, RUF

### Task Configuration

All tasks are defined in `Taskfile.yml` with organized prefixes:
- Tasks use variables for tool paths (`.venv/bin/`)
- Tasks automatically ensure virtualenv exists via `deps: [venv:init]`

## Technologies utilisées

### Langage et runtime
- **Python 3.12.12** (géré via asdf)

### Bibliothèques principales (ML/AI)
- **transformers** - Modèles transformers (Bio_ClinicalBERT) avec support PyTorch
- **datasets** - Gestion et manipulation de datasets
- **scikit-learn** - Machine learning et analyses statistiques

### Applications Web et API
- **fastapi** - Framework API REST moderne et performant
- **streamlit** - Interface web interactive pour ML
- **uvicorn** - Serveur ASGI haute performance

### LLM Integration
- **langchain-core** - Framework pour applications LLM
- **langchain-ollama** - Intégration Ollama pour génération de texte

### Interface utilisateur
- **typer** - Framework CLI moderne basé sur type hints
- **rich** - Formatage terminal (tables, panels, couleurs)

### Utilitaires
- **loguru** - Logging simplifié et puissant
- **pydantic-settings** - Gestion de configuration
- **python-dotenv** - Variables d'environnement

### Outils de développement
- **uv** - Gestionnaire de paquets Python rapide (10-100x faster than pip)
- **ruff** - Linter et formateur Python moderne
- **Task** - Task runner moderne
- **asdf** - Gestionnaire de versions d'outils
- **jupyter** - Notebooks interactifs
- **tensorboard** - Visualisation d'entraînement de modèles
- **plotly** - Visualisations interactives
- **seaborn** - Visualisations statistiques avancées
- **umap-learn** - Réduction de dimensionnalité
- **lightning-sdk** - Intégration Lightning Studio
- **watchdog** - Monitoring de fichiers

## Documentation

- **CLAUDE.md** - Claude Code integration, conventions, and guidelines
- **MEMORY.md** - Project decisions, technical choices, and session history

## Python Version

This project requires **Python 3.12.12** or higher (specified in `pyproject.toml` as `>=3.12`).

## License

Projet académique - Formation Alyra

## Contributing

Stéphane Wirtel <stephane@wirtel.be>
