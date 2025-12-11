# alyra-ai-dl

A Python AI/Deep Learning project with modern tooling and automation.

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
├── .tool-versions       # asdf version specifications
├── Taskfile.yml         # Task automation configuration
├── pyproject.toml       # Project metadata and tool configuration
├── uv.lock              # Locked dependency versions
├── CLAUDE.md            # Claude Code integration documentation
├── MEMORY.md            # Project decisions and session history
└── .venv/               # Virtual environment (not in git)
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

## Documentation

- **CLAUDE.md** - Claude Code integration, conventions, and guidelines
- **MEMORY.md** - Project decisions, technical choices, and session history

## Python Version

This project requires **Python 3.12.12** or higher (specified in `pyproject.toml` as `>=3.12`).

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
