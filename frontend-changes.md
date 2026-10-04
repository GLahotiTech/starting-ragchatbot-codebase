# Frontend changes: code quality tooling

No files under `frontend/` (HTML/CSS/JS) were modified. Black is a Python
formatter, so this feature only touches the backend and dev workflow.

## What changed
- Added `black` as a dev dependency (`uv add --dev black`) and a `[tool.black]`
  section in `pyproject.toml` (line length 88, py313, excludes `chroma_db`).
- Added `scripts/format.sh` (auto-format `backend/` and `main.py`) and
  `scripts/check.sh` (`black --check` plus the pytest suite).
- Reformatted all Python files with black (16 files; formatting only).
- All 61 tests pass after reformatting.

## Usage
- `./scripts/format.sh`: format the code
- `./scripts/check.sh`: verify formatting and run the tests

Frontend formatting (e.g. Prettier for JS/CSS) was not added, since it wasn't requested.
