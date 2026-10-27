# `backend/` — XAU/USD Quantitative AI & macOS MT5 Backend API

This directory contains the modular Python backend serving live London/Swissquote OTC Spot `XAU/USD` BBO ticks (Exness price parity), the 4-Pillar Live Signal Engine, the macOS `MetaTrader5` order simulator, and the empirical research pipelines (`01`–`04`).

## Folder Structure

```text
backend/
├── __init__.py           # Package initializer
├── main.py               # Entry point (runs on http://127.0.0.1:8050)
├── api_server.py         # REST API handler (/api/tick, /api/signal, /api/order_check, /api/pillars, /api/benchmarks)
├── signal_engine.py      # Real-time 4-Pillar XAU/USD Trading Signal Engine (Entry, SL, TP1, TP2, TP3)
├── research_service.py   # Empirical pipeline runner & cache for Pillars 01–04 and 8 IEEE/Scopus Q1 PDFs
└── README.md             # Backend documentation
```

## Quick Start

From the project root:

```bash
# 1. Start the Python Backend (Port 8050)
.venv/bin/python -m backend.main

# 2. Start the Next.js Frontend (Port 3000)
cd frontend && npm run dev
```

## REST API Endpoints (`http://127.0.0.1:8050`)

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Health check and active endpoint list |
| `/api/tick` | `GET` | Live OTC Spot `XAU/USD` BBO tick (Exness parity), 1m Spot bars, and live 4-Pillar Signal |
| `/api/signal` | `GET` | Dedicated 4-Pillar `XAU/USD` trade signal (`BUY`/`SELL`, `Entry`, `SL`, `TP1`, `TP2`, `TP3`) |
| `/api/order_check` | `GET` | Guarded `mt5.order_check()` simulator (`?side=buy&lots=5.0&max_spread_bps=3.5`) |
| `/api/pillars` | `GET` | Full empirical results across Pillars `01`–`04` and 8 IEEE/Scopus Q1 benchmark papers |
| `/api/benchmarks` | `GET` | Summary benchmark metrics dictionary |
