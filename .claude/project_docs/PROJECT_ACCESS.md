# PROJECT ACCESS
> Connection details and environment configuration

## Remote Access
- SSH connection: `ssh -i ~/.ssh/id_ed25519_wsl2 graham@192.168.86.49`
- Authentication: Uses ED25519 key from WSL2 environment

## Environment Setup
- Python version: 3.11.12
- Package manager: uv (not pip)
- Environment files:
  - `.env` - Contains environment variables
  - `pyproject.toml` - Project configuration and dependencies
  - `requirements.txt` - Additional dependency specifications

## Project Structure
- Project root: `/home/graham/workspace/experiments/pdf_extractor/`
- Main configuration: `/home/graham/workspace/experiments/pdf_extractor/pyproject.toml`
- Main task list: `/home/graham/workspace/experiments/pdf_extractor/task.md`

### Key Directories
- `src/` - Primary source code directory
  - `input/` - PDF input files
  - `output/` - Generated output files
- `docs/` - Documentation files
- `build/` - Build artifacts (don't modify directly)

## Connection Workflow
1. Establish SSH connection: `ssh -i ~/.ssh/id_ed25519_wsl2 graham@192.168.86.49`
2. Navigate to project root: `cd /home/graham/workspace/experiments/pdf_extractor/`
3. Activate virtual environment (if not auto-activated): `.venv/bin/activate`
4. Verify environment setup: `uv run python -m pytest -xvs tests/`

## First-Time Setup
If connecting for the first time:
1. Verify SSH key permissions: `chmod 600 ~/.ssh/id_ed25519_wsl2`
2. Create and activate virtual environment:
   ```
   uv venv
   source .venv/bin/activate
   ```
3. Install project dependencies: `llm install -e .`
4. Verify environment variables in `.env` file

## Project Purpose
The pdf_extractor project converts PDF documents to ordered JSON objects, with specialized handling for tables, markdown extraction, and other structural elements.

## Finding Your Way Around
Instead of relying on a static directory listing that may become outdated, use these commands to explore the current project structure:
- `find . -type f -name "*.py" | sort` - List all Python files
- `grep -r "function_name" --include="*.py" .` - Find where a function is defined/used
- `find ./src -type d -not -path "*/\.*" | sort` - List all non-hidden subdirectories
