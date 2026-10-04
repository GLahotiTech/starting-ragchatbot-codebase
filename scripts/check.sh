#!/bin/bash
# Run quality checks: formatting (black --check) and the test suite
set -e
cd "$(dirname "$0")/.."
echo "==> Checking formatting with black"
uv run black --check backend main.py
echo "==> Running tests"
cd backend && uv run pytest tests
