# MEMORY.md

This file contains the memory of conversations and decisions made during development. Claude Code instances should read this file at startup to understand the project context.

**Last Updated:** 2025-12-16

---

## 📋 Project Conventions

### Task Naming Convention
All tasks MUST use category prefixes:
- `code:*` → Code quality operations (linting, formatting, testing, validation)
- `venv:*` → Virtual environment operations (init, add, sync, install, upgrade)
- `lightning:*` → Lightning AI studio operations (login, start, stop, ssh, download)
- `ml:*` → Machine learning operations (train, prepare dataset, tensorboard, analyze)
- `app:*` → Application operations (cli, api, streamlit, chainlit)
- `demo:*` → Demo recording operations (record, convert to gif/mp4)

Examples:
- ✅ `code:lint`, `venv:add`, `lightning:studio:start`, `ml:train:standard`, `app:cli`, `demo:cli:record`
- ❌ `lint`, `add`, `train`, `cli`, `record`

### Tool Configuration
- **Ruff**: Single tool for linting, formatting, and import sorting (replaced isort)
- **uv**: Package manager for dependency management
- **Task**: Automation tool with organized tasks by prefix

### Git Commit Messages
- **DO NOT** add Claude Code footer or co-author attribution in commit messages
- The `/commit` command generates commits without this footer
- Keep commits clean and consistent with the project's commit style

### File Creation and Encoding
- **ALWAYS** use `cat > file << 'EOF'` with heredoc for creating text files (especially markdown, config files)
- **NEVER** use the Write tool for text files as it can introduce binary characters and encoding issues
- The Write tool has caused encoding problems before (null bytes, control characters like `\x1c`)
- Git will treat files with binary characters as "data" instead of "text"
- Verify file encoding after creation: `file <filename>` should show "UTF-8 text" not "data"

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

### Session 2025-12-16: Comprehensive CLAUDE.md documentation update
**What was done:**
1. Rescanned entire codebase using Explore agent (very thorough mode)
2. Identified gaps in CLAUDE.md documentation
3. Added 4 major new sections to CLAUDE.md:
   - Model Configuration (model paths, environment variables, features)
   - Lightning AI Integration (15 tasks, workflow, studio management)
   - Machine Learning Tasks (training, data prep, visualization, notebooks)
   - Demo Recording (asciinema, GIF/MP4 conversion)
4. Updated Task Naming Convention section with all prefixes:
   - Added `lightning:*`, `ml:*`, `app:*`, `demo:*` prefixes
   - Now documents 40+ tasks across 6 categories
5. Added `code:validate:pyproject` to code quality tools
6. Documented all 4 applications (CLI, API, Streamlit, Chainlit)
7. Updated MEMORY.md with expanded task conventions

**Documentation improvements:**
- ~200 lines of new documentation added
- 26 new tasks documented
- 4 major sections added
- Complete Lightning AI workflow documented
- Training and ML pipeline fully documented

**Files modified:**
- `CLAUDE.md` - Major expansion with new sections
- `MEMORY.md` - Updated conventions and session history

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
- [x] ~~Consider adding jupyter notebook support if needed~~ - Already has Jupyter notebooks in notebooks/
- [ ] Add docs:* category for documentation tasks
- [ ] Add CI/CD workflow with GitHub Actions
- [ ] Consider adding training metrics visualization to Streamlit app
- [ ] Add more training experiments (see EXPERIMENTS_README.md)

### Questions to address
- [x] ~~Will this project need Lightning Studio integration?~~ - Yes, fully integrated with 15 lightning:* tasks
- [x] ~~Do we need API development tasks (FastAPI)?~~ - Yes, API app exists at apps/api.py
- [ ] Should we add database tasks (db:*)?
- [ ] Should we add more model variants or fine-tuning options?
- [ ] Do we need automated model evaluation tasks?

---

## 📝 Notes

- This file should be updated after significant sessions or decisions
- Keep the history concise but informative
- Update "Last Updated" date at the top when modifying
- Claude Code will read this automatically via CLAUDE.md reference
