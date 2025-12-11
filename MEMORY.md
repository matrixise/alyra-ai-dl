# MEMORY.md

This file contains the memory of conversations and decisions made during development. Claude Code instances should read this file at startup to understand the project context.

**Last Updated:** 2025-12-11

---

## 📋 Project Conventions

### Task Naming Convention
All tasks MUST use category prefixes:
- `code:*` → Code quality operations (linting, formatting, testing)
- `venv:*` → Virtual environment operations (init, add, sync, install)

Examples:
- ✅ `code:lint`, `code:format`, `venv:add`
- ❌ `lint`, `format`, `add`

### Tool Configuration
- **Ruff**: Single tool for linting, formatting, and import sorting (replaced isort)
- **uv**: Package manager for dependency management
- **Task**: Automation tool with organized tasks by prefix

### Git Commit Messages
- **DO NOT** add Claude Code footer or co-author attribution in commit messages
- The `/commit` command generates commits without this footer
- Keep commits clean and consistent with the project's commit style

---

## 🎯 Technical Decisions

### 2025-12-11: Ruff as single linting/formatting tool
**Decision:** Use ruff exclusively instead of multiple tools (isort, black, flake8, etc.)

**Rationale:**
- Modern standard (2025 best practice)
- 10-100x faster than traditional tools
- Single tool reduces complexity
- Built-in import sorting compatible with isort
- Comprehensive rule coverage (700+ rules)

**Configuration:** `pyproject.toml` → `[tool.ruff]`
- Target: Python 3.12+
- Line length: 88
- Rules: E, W, F, I, N, UP, B, C4, SIM, RUF

### 2025-12-11: Task organization with category prefixes
**Decision:** All Task commands must use category prefixes (code:, venv:)

**Rationale:**
- Better organization as project grows
- Clear separation of concerns
- Easy to find related commands
- Extensible for future categories (docs:, db:, etc.)

---

## 📅 Session History

### Session 2025-12-11: Initial project setup
**What was done:**
1. Created project structure with CLAUDE.md, Taskfile.yml, pyproject.toml
2. Configured ruff for linting/formatting (removed isort dependency)
3. Added 13 tasks organized by category:
   - 5 venv tasks: init, install, sync, add, add:dev
   - 8 code tasks: lint, lint:fix, format, format:check, check, fix, clean, show:config
4. Documented all conventions in CLAUDE.md
5. Created initial commits:
   - `1458e9f` - Initial project setup with ruff and Task
   - `bbbb222` - Added dependency management tasks

**Key decisions:**
- Ruff only (no isort, black, flake8)
- Task automation with category prefixes
- uv for package management
- Python 3.12.12 via asdf

**Conventions established:**
- Task prefixes: `code:*`, `venv:*`
- All tools in `.venv/bin/`
- VENV_DIR variable for flexibility

---

## 💡 TODOs and Future Ideas

### Potential improvements
- [ ] Add pre-commit hooks for automatic code quality checks
- [ ] Add pytest configuration and test tasks (code:test, code:test:coverage)
- [ ] Add mypy for type checking (code:typecheck)
- [ ] Consider adding jupyter notebook support if needed
- [ ] Add docs:* category for documentation tasks
- [ ] Add CI/CD workflow with GitHub Actions

### Questions to address
- Will this project need Lightning Studio integration?
- Do we need API development tasks (FastAPI)?
- Should we add database tasks (db:*)?

---

## 📝 Notes

- This file should be updated after significant sessions or decisions
- Keep the history concise but informative
- Update "Last Updated" date at the top when modifying
- Claude Code will read this automatically via CLAUDE.md reference
