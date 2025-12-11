# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- ✅ `venv:clean` - clean virtual environment (for future)
- ❌ `init` - incorrect

When creating new tasks, always use the appropriate category prefix.

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
- to memorize