"""
CLI Entry Point (`backend/main.py`) for the `XAU/USD` Python Backend API Server.

Run from the repository root:
    .venv/bin/python -m backend.main
or:
    .venv/bin/python backend/main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure both `backend/` and repository root are on sys.path
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
for p in (str(BACKEND_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.api_server import run_backend_server  # noqa: E402


def main() -> None:
    host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("BACKEND_PORT", "8050"))
    run_backend_server(host=host, port=port)


if __name__ == "__main__":
    main()
