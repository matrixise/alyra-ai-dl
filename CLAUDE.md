# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📖 Context and Memory

**IMPORTANT:** Always read `MEMORY.md` first to understand the project history, decisions, and conventions. This file contains:
- Technical decisions and their rationales
- Project conventions (task naming, tool choices)
- Session history and what has been accomplished
- Future TODOs and ideas

### Memory Management Commands

Use these slash commands to update MEMORY.md:

```bash
# Add a TODO item (uses haiku - fast)
/add-todo Add pytest configuration

# Add a technical decision (uses haiku)
/add-decision Use PostgreSQL instead of SQLite

# Add a session entry (uses haiku)
/add-session Implemented user authentication system

# Add a question to address (uses haiku)
/add-question Should we use Redis for caching?

# Comprehensive session update (uses sonnet - thorough)
/update-memory Completed authentication feature with OAuth2
```

**When to use each command:**
- `/add-todo` - Quick TODO additions during development
- `/add-decision` - Document important technical choices with rationale
- `/add-session` - At the end of a work session to log what was accomplished
- `/add-question` - Note questions that need discussion or research
- `/update-memory` - Comprehensive update of multiple sections at once

---

## Project Overview

AI/Deep Learning project repository (alyra-ai-dl) using Python 3.12.12, uv for dependency management, and task for task automation.

## Tool Versions

Managed via asdf (`.tool-versions`):
- Python: 3.12.12
- uv: 0.9.17
- task: 3.45.5

Python tools (in `.venv/`):
- ruff: 0.14.8 (linting, formatting, import sorting)

## Environment Setup

### Quick Start (Recommended)

```bash
# Initialize virtual environment and install dependencies in one command
task venv:init

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

The `task venv:init` command will:
1. Create a virtual environment in `.venv/` with upgraded pip/setuptools
2. Install `uv` package manager
3. Synchronize all dependencies from `pyproject.toml`

### Manual Setup (Alternative)

```bash
# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install dependencies (once defined)
uv pip install -r requirements.txt
# or
uv sync  # if using pyproject.toml with uv
```

## Dependency Management with uv

### Quick Commands (Recommended)

```bash
# Add a production dependency
task venv:add <package-name>

# Add a dev dependency
task venv:add:dev <package-name>

# Sync dependencies from uv.lock
task venv:sync

# Install project in editable mode with dev dependencies
task venv:install

# Check for outdated packages
task venv:outdated

# Upgrade all dependencies to latest versions
task venv:upgrade

# Upgrade pip and uv tools
task venv:upgrade:tools
```

### Manual Commands (Alternative)

```bash
# Add a package
uv pip install <package-name>

# Add a package and update requirements
uv pip install <package-name> && uv pip freeze > requirements.txt

# Install from requirements.txt
uv pip install -r requirements.txt

# Upgrade a package
uv pip install --upgrade <package-name>
```

## Task Automation

This project uses [Task](https://taskfile.dev/) for automation. Check `Taskfile.yml` for available tasks.

```bash
# List available tasks
task --list

# Run a specific task
task <task-name>
```

### Task Naming Convention

**IMPORTANT**: Tasks MUST be organized by category using prefixes:

**Code operations** (linting, formatting, type checking, etc.) → `code:` prefix:
- ✅ `code:lint` - correct
- ✅ `code:format` - correct
- ✅ `code:test` - correct (for future test tasks)
- ❌ `lint` - incorrect
- ❌ `format` - incorrect

**Virtual environment operations** → `venv:` prefix:
- ✅ `venv:init` - initialize virtual environment
- ✅ `venv:add` - add production dependency
- ✅ `venv:add:dev` - add dev dependency
- ✅ `venv:sync` - sync dependencies from lockfile
- ✅ `venv:install` - install project in editable mode
- ✅ `venv:outdated` - check for outdated packages
- ✅ `venv:upgrade` - upgrade all dependencies
- ✅ `venv:upgrade:tools` - upgrade pip and uv tools
- ❌ `init` - incorrect
- ❌ `add` - incorrect
- ❌ `outdated` - incorrect

**Lightning AI operations** → `lightning:` prefix:
- ✅ `lightning:login` - authenticate with Lightning AI
- ✅ `lightning:studio:list` - list your studios
- ✅ `lightning:studio:start` - start default studio
- ✅ `lightning:studio:stop` - stop default studio
- ✅ `lightning:studio:ssh` - SSH into studio
- ❌ `login` - incorrect
- ❌ `studio-start` - incorrect

**Machine learning operations** → `ml:` prefix:
- ✅ `ml:train:standard` - train on all diseases
- ✅ `ml:train:quick` - quick training test
- ✅ `ml:prepare:dataset` - prepare dataset
- ✅ `ml:tensorboard` - launch TensorBoard
- ❌ `train` - incorrect
- ❌ `tensorboard` - incorrect

**Demo operations** → `demo:` prefix:
- ✅ `demo:cli:record` - record CLI demo
- ✅ `demo:cli:gif` - convert to GIF
- ✅ `demo:cli:all` - record and convert
- ❌ `record` - incorrect

When creating new tasks, always use the appropriate category prefix.

### Task Conventions

**IMPORTANT**: When creating or modifying tasks in `Taskfile.yml`:

- **Task descriptions**: Keep descriptions concise and focused on what the task does
- **Do NOT include usage examples** in the `desc` field (e.g., avoid "usage: task name -- args")
- Usage information should be documented in this file or discovered via `task --list`

Examples:
- ✅ `desc: "Add one or more packages to dependencies"` - concise, clear
- ❌ `desc: "Add packages (usage: task venv:add -- package1 package2)"` - includes usage, too verbose

## Code Quality Tools

### Tool Paths

All Python tools are installed in `.venv/bin/`:
- Ruff: `.venv/bin/ruff`
- Task: Available via system (asdf: `task`)

Claude Code can run these tools directly:
- `task` commands (e.g., `task lint`, `task check`)
- `.venv/bin/ruff` commands directly if needed

### Linting and Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting, formatting, and import sorting.

**Quick commands:**

```bash
# Before committing - check everything
task code:check

# Auto-fix all issues
task code:fix

# Individual operations
task code:lint            # Check for linting errors
task code:lint:fix        # Fix linting errors automatically
task code:format          # Format code
task code:format:check    # Check if code is formatted
task code:clean           # Clean cache files
task code:show:config     # Show ruff configuration
task code:validate:pyproject  # Validate pyproject.toml against PEP standards
```

### Ruff Configuration

Configuration in `pyproject.toml` under `[tool.ruff]`:
- **Rules**: pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, bugbear, comprehensions, simplify
- **Target**: Python 3.12+
- **Line length**: 88 characters
- **Import sorting**: Built-in (replaces isort)

### Development Workflow

```bash
# During development
task code:format        # Format code
task code:lint          # Check for issues

# Before committing
task code:check         # Verify everything is clean
# or
task code:fix          # Auto-fix all issues
```

### IDE Integration

**VS Code**: Install the Ruff extension and add to settings.json:
```json
{
  "ruff.enable": true,
  "ruff.organizeImports": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true,
    "source.fixAll": true
  }
}
```

## Model Configuration

The project uses **Bio_ClinicalBERT** fine-tuned for disease classification from symptom descriptions.

### Model Path

Configure the model location via environment variables:

```bash
# Create .env file from example
cp .env.example .env

# Edit .env to set model path
MODEL_PATH=./models/symptom_classifier-mini/final/
```

**Available Models:**
- `models/symptom_classifier-old-trainer/final/` - Default full model
- `models/symptom_classifier-mini/final/` - Smaller alternative model
- `models/diagnosia/` - Alternative variant

**Environment Variables:**
- `MODEL_PATH` - Path to trained model directory (default: `./models/symptom_classifier-old-trainer/final/`)

### Model Features

- Multi-class disease classification (6 diseases: anxiety, cystitis, herniated disk, panic disorder, pneumonia, spondylolisthesis)
- Confidence scoring with adjustable thresholds
- Support for unknown symptoms (fallback handling)
- Device auto-detection (CUDA, MPS, or CPU)

## Lightning AI Integration

This project supports training on Lightning AI studios with GPU acceleration.

### Quick Commands

```bash
# Authentication
task lightning:login
task lightning:logout

# Studio management
task lightning:studio:list          # List your studios
task lightning:studio:list:all      # List all studios in teamspace
task lightning:studio:start         # Start default studio (alyra-dl)
task lightning:studio:start:t4      # Start with T4 GPU
task lightning:studio:stop          # Stop default studio
task lightning:studio:delete        # Delete default studio

# SSH access
task lightning:studio:ssh           # SSH into default studio

# File transfer
task lightning:studio:cp -- <source> <dest>
task lightning:studio:download:model         # Download trained model
task lightning:studio:download:results       # Download all results

# Custom studio operations
task lightning:studio:start:custom -- --name my-studio
task lightning:studio:stop:custom -- --name my-studio
```

### Configuration

Default studio configuration in `Taskfile.yml`:
- **Studio name**: `alyra-dl`
- **Teamspace**: `stephanewirtel/tool-exploration-project`
- **Default machine**: CPU (use `:t4` variants for GPU)

### Workflow

See `LIGHTNING_WORKFLOW.md` for detailed training workflow on Lightning AI.

**Typical workflow:**
1. `task lightning:studio:start:t4` - Start studio with GPU
2. `task lightning:studio:ssh` - SSH into studio
3. Train model on studio
4. `task lightning:studio:download:results` - Download results
5. `task lightning:studio:stop` - Stop studio to save costs

## Machine Learning Tasks

### Training

```bash
# Train standard model on all 6 diseases
task ml:train:standard

# Quick training test (2 diseases, 20 epochs) for development
task ml:train:quick

# Custom training with additional arguments
task ml:train:standard -- --epochs 50 --batch-size 32
```

**Training details:**
- **Standard training**: 6 diseases (anxiety, cystitis, herniated disk, panic disorder, pneumonia, spondylolisthesis)
- **Quick training**: 2 diseases (anxiety, pneumonia) for faster testing
- **Output**: `models/symptom_classifier/` (standard) or `models/symptom_classifier_quick/` (quick)
- **Features**: Data augmentation, early stopping, confusion matrix visualization

### Data Preparation

```bash
# Prepare dataset from augmented source
task ml:prepare:dataset

# With custom diseases
task ml:prepare:dataset -- --diseases "anxiety,pneumonia"
```

**Preparation details:**
- **Input**: `data/Final_Augmented_dataset_Diseases_and_Symptoms.csv`
- **Output**: `data/prepared-dataset.csv`
- **Process**: Filtering by diseases, deduplication, validation

### Visualization and Analysis

```bash
# Launch TensorBoard for training visualization
task ml:tensorboard
# Access at http://localhost:6006

# Custom TensorBoard directory
task ml:tensorboard:custom -- --logdir models/my-model

# Analyze training logs and generate plots
task ml:analyze

# Compare multiple training runs
task ml:analyze:compare -- run1 run2
```

### Scripts

Training and data scripts are located in `scripts/`:

**Data Preparation:**
- `scripts/data/prepare_dataset.py` - Dataset filtering, deduplication, and preparation

**Training:**
- `scripts/training/train_standard.py` - Main training script with augmentation and evaluation
- `scripts/training/train_model.py` - Legacy training script

**Training Experiments:**

See `EXPERIMENTS_README.md` for 4 training variants to handle imbalanced classes:
1. Standard training (baseline)
2. Class weights (balanced loss)
3. Focal loss (hard examples)
4. SMOTE oversampling (synthetic data)

### Notebooks

Jupyter notebooks for exploratory analysis in `notebooks/`:

```bash
# Start Jupyter
jupyter notebook notebooks/

# Available notebooks:
# - eda_diseases_symptoms.ipynb - Disease/symptom relationships
# - advanced_eda_statistical_analysis.ipynb - Statistical analysis
# - eda_dataset_original_kaggle.ipynb - Original dataset
# - eda_cleaned_dataset.ipynb - Cleaned dataset exploration
```

## Applications

This project includes four user-facing applications:

### Quick Commands

```bash
# CLI - Interactive terminal interface
task app:cli

# API - FastAPI REST server
task app:api:dev

# Streamlit - Web UI with visualizations
task app:streamlit:dev

# Chainlit - Chat interface with LLM
task app:chainlit:dev
```

### Application Details

**CLI** (`apps/cli.py`):
- Interactive REPL for symptom analysis
- BERT predictions + optional Ollama explanations
- Rich terminal output
- Usage: `task app:cli` or `python -m apps.cli`

**API** (`apps/api.py`):
- FastAPI REST service
- Endpoints: `/predict`, `/predict/batch`, `/health`, `/info`
- Interactive docs at http://localhost:8000/docs
- Usage: `task app:api:dev`

**Streamlit** (`apps/streamlit_app.py`):
- Web-based UI with charts and tables
- Model configuration interface
- Access at http://localhost:8501
- Usage: `task app:streamlit:dev`

**Chainlit** (`apps/chainlit_app.py`):
- Chat-based interface
- BERT predictions + Ollama LLM explanations
- Step-by-step analysis display
- Usage: `task app:chainlit:dev`

### Task Naming for Applications

**Application tasks** → `app:` prefix:
- ✅ `app:cli` - CLI interface
- ✅ `app:api:dev` - API development server
- ✅ `app:streamlit:dev` - Streamlit development server
- ✅ `app:chainlit:dev` - Chainlit chat interface
- ❌ `run-cli` - incorrect
- ❌ `start-api` - incorrect

## LLM Backend Configuration

This project supports multiple LLM backends for generating clinical summaries from Bio_ClinicalBERT predictions.

### Available Backends

- **Ollama** (default): Local LLM server for privacy and offline use
- **OpenAI**: Cloud-based GPT models (gpt-4, gpt-3.5-turbo)
- **Lightning AI**: Lightning.ai hosted models (llama-3.3-70b)

### Quick Start

**CLI Usage:**
```bash
# Default (Ollama)
task app:cli

# Use OpenAI
export OPENAI_API_KEY="sk-..."
task app:cli -- --llm-backend openai

# Use Lightning AI
export LIGHTNING_API_KEY="your-key"
task app:cli -- --llm-backend lightning
```

**Chainlit Usage:**
```bash
# Uses default backend from .env (ollama)
task app:chainlit:dev
```

### Configuration

**Environment Variables** (`.env`):
```bash
# Backend selection
LLM_BACKEND=ollama  # or openai, lightning

# Ollama settings (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# OpenAI settings (cloud)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# Lightning AI settings (cloud)
LIGHTNING_API_KEY=your-key
LIGHTNING_MODEL=lightning-ai/llama-3.3-70b
```

**CLI Options:**
- `--llm-backend`: Choose backend (ollama, openai, lightning)
- `--no-llm`: Skip LLM entirely, BERT predictions only
- `--ollama-url`: Custom Ollama server URL
- `--ollama-model`: Custom Ollama model name

### API Keys Security

**⚠️ IMPORTANT:**
- Never commit API keys to git
- Store keys in `.env` file (already in `.gitignore`)
- Use environment variables only
- Lightning AI keys are project-specific

**Setting API Keys:**
```bash
# Option 1: Export in shell session
export OPENAI_API_KEY="sk-..."
export LIGHTNING_API_KEY="..."

# Option 2: Add to .env file (recommended)
echo 'OPENAI_API_KEY=sk-...' >> .env
echo 'LIGHTNING_API_KEY=...' >> .env
```

### Backend Comparison

| Feature | Ollama | OpenAI | Lightning AI |
|---------|--------|--------|--------------|
| **Privacy** | ✅ Local | ❌ Cloud | ❌ Cloud |
| **Cost** | ✅ Free | 💰 Pay-per-use | 💰 Pay-per-use |
| **Setup** | Requires local install | API key only | API key only |
| **Speed** | Depends on hardware | Fast | Fast |
| **Models** | llama3, mistral, etc | gpt-4, gpt-3.5 | llama-3.3-70b |
| **Offline** | ✅ Yes | ❌ No | ❌ No |

### Troubleshooting

**Ollama Connection Error:**
```bash
# Check Ollama is running
ollama serve

# List available models
ollama list

# Pull a model if needed
ollama pull llama3
```

**OpenAI/Lightning API Key Error:**
```bash
# Verify key is set
echo $OPENAI_API_KEY
echo $LIGHTNING_API_KEY

# Check .env file
cat .env | grep API_KEY
```

**Wrong Model Error:**
```bash
# For Ollama, check installed models
ollama list

# For OpenAI/Lightning, verify model name in .env
```

### Programming with Multiple Backends

**Python Example:**
```python
from apps.llm_processor import generate_response, LLMBackend

# Use Ollama (default)
response = generate_response(user_text, prediction)

# Use OpenAI
response = generate_response(
    user_text,
    prediction,
    backend=LLMBackend.OPENAI
)

# Use Lightning AI
response = generate_response(
    user_text,
    prediction,
    backend=LLMBackend.LIGHTNING
)

# Custom configuration
response = generate_response(
    user_text,
    prediction,
    backend="ollama",
    base_url="http://custom-server:11434",
    model="mistral"
)
```

## Demo Recording

This project includes tools for recording CLI demos and converting them to various formats.

### Prerequisites

Install recording tools:

```bash
# asciinema - Terminal session recorder
brew install asciinema  # macOS
# or
apt-get install asciinema  # Linux

# agg - Convert asciinema to GIF
brew install agg  # macOS
# or download from: https://github.com/asciinema/agg

# ffmpeg - Convert GIF to MP4
brew install ffmpeg  # macOS
# or
apt-get install ffmpeg  # Linux
```

### Recording Commands

```bash
# Record a CLI demo
task demo:cli:record
# Starts recording with: task app:cli -- --llm --llm-backend=lightning
# Press Ctrl+D to stop recording

# Record with custom filename
task demo:cli:record -- my-demo

# Convert recording to GIF
task demo:cli:gif
# or with custom filename:
task demo:cli:gif -- my-demo

# Convert GIF to MP4
task demo:cli:mp4
# or with custom filename:
task demo:cli:mp4 -- my-demo

# Do everything in one command
task demo:cli:all
# or with custom filename:
task demo:cli:all -- my-demo
```

### Output Files

- `<filename>` - asciinema recording (JSON format)
- `<filename>.gif` - Animated GIF for documentation
- `<filename>.mp4` - Video file for presentations

**Default filename:** `alyra-ai-dl-cli`

### Recording Tips

- Keep demos short (1-2 minutes)
- Use clear, representative examples
- Pause briefly between commands for readability
- Test the recording before final version
- GIF files can be large - optimize with shorter recordings

## Additional Notes

- To add a new dependency for development: `task venv:add:dev -- DEPENDENCY`
- All issues I create must be written in French, as well as comments