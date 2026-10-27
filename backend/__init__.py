"""
Python Backend Package (`backend/`) for the `XAU/USD` Quantitative AI & macOS MetaTrader5 Terminal.

Modules:
- `backend/main.py`: CLI entry point launching the backend HTTP API server (`http://127.0.0.1:8050`).
- `backend/api_server.py`: REST API server (`/api/tick`, `/api/signal`, `/api/order_check`, `/api/pillars`, `/api/benchmarks`).
- `backend/signal_engine.py`: Real-time 4-Pillar `XAU/USD` Quantitative Trading Signal Engine.
- `backend/research_service.py`: Empirical pipeline runner & cache for Pillars `01`–`04`.
"""

from pathlib import Path
import sys

_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

