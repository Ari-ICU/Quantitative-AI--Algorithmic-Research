"""
REST API Server (`backend/api_server.py`) for the `XAU/USD` Quantitative AI & macOS MT5 Terminal.

Endpoints:
- `GET /api/tick`: Live London/Swissquote OTC Spot `XAU/USD` BBO tick (Exness parity), 1m bars, and 4-Pillar Signal.
- `GET /api/signal`: Dedicated 4-Pillar `XAU/USD` Live Trading Signal (`BUY`/`SELL`, Entry, SL, TP1, TP2, TP3, Meta-Labeling).
- `GET /api/order_check?side=buy&lots=5.0&max_spread_bps=2.5`: Guarded `mt5.order_check()` simulation.
- `GET /api/pillars`: Non-blocking JSON empirical metrics & time-series across Pillars `01`–`04` (cached via `ProcessPoolExecutor`).
- `GET /api/benchmarks`: Summary benchmark metrics dictionary.
- `GET /figures/*`: Static comparison figures.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from urllib.parse import parse_qs, urlparse

from backend.research_service import (
    CACHE_JSON_PATH,
    PROJECT_ROOT,
    get_cached_benchmarks,
    get_cached_pillars_payload,
    trigger_async_pillars_recompute,
)
from backend.signal_engine import compute_live_xauusd_4pillar_signal
from common.macos_mt5_bridge import TIMEFRAME_M1, MacOSMetaTraderBridge

BRIDGE = MacOSMetaTraderBridge(mode="live_yahoo_ecn")
BRIDGE.initialize()


class XAUUSDBackendAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the `XAU/USD` JSON API endpoints and figures."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, payload: object, status: int = 200) -> None:
        raw = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/api/health"):
            self._send_json(
                {
                    "status": "ok",
                    "service": "XAU/USD Quantitative AI & macOS MT5 Backend API",
                    "frontend_url": "http://localhost:3000",
                    "cache_ready": CACHE_JSON_PATH.is_file(),
                    "endpoints": [
                        "/api/tick",
                        "/api/signal",
                        "/api/order_check",
                        "/api/pillars",
                        "/api/benchmarks",
                    ],
                }
            )
            return

        if path == "/api/tick":
            tick = BRIDGE.symbol_info_tick("XAU/USD")
            bars_df = BRIDGE.copy_rates_from_pos("XAU/USD", TIMEFRAME_M1, 0, 90, prefer_live=True)
            sig = compute_live_xauusd_4pillar_signal(
                tick=tick,
                bars_df=bars_df,
                live_tick_buffer=BRIDGE._live_tick_buffer,
            )
            bars_list = [
                {
                    "timestamp": str(row["timestamp"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
                for _, row in bars_df.iterrows()
            ]
            self._send_json({"tick": tick.to_dict(), "bars": bars_list, "signal": sig})
            return

        if path == "/api/signal":
            tick = BRIDGE.symbol_info_tick("XAU/USD")
            bars_df = BRIDGE.copy_rates_from_pos("XAU/USD", TIMEFRAME_M1, 0, 90, prefer_live=True)
            sig = compute_live_xauusd_4pillar_signal(
                tick=tick,
                bars_df=bars_df,
                live_tick_buffer=BRIDGE._live_tick_buffer,
            )
            self._send_json(sig)
            return

        if path == "/api/order_check":
            side = "sell" if qs.get("side", ["buy"])[0].lower() == "sell" else "buy"
            lots = float(qs.get("lots", ["5.0"])[0])
            max_spread = float(qs.get("max_spread_bps", ["2.5"])[0])
            chk = BRIDGE.order_check(
                side=side,  # type: ignore[arg-type]
                volume_lots=lots,
                max_allowed_spread_bps=max_spread,
            )
            self._send_json(chk)
            return

        if path == "/api/benchmarks":
            self._send_json(get_cached_benchmarks())
            return

        if path == "/api/pillars":
            if qs.get("refresh", ["false"])[0].lower() == "true":
                worker_state = trigger_async_pillars_recompute()
                payload = dict(get_cached_pillars_payload())
                payload["worker_status"] = worker_state
                self._send_json(payload)
                return
            self._send_json(get_cached_pillars_payload())
            return

        if path.startswith("/figures/"):
            rel = path.lstrip("/")
            file_path = (PROJECT_ROOT / rel).resolve()
            if file_path.is_file() and str(file_path).startswith(str(PROJECT_ROOT)):
                mime, _ = mimetypes.guess_type(str(file_path))
                raw = file_path.read_bytes()
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", mime or "application/octet-stream")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

        self._send_json({"error": f"Not found: {path}"}, status=404)


def run_backend_server(host: str = "127.0.0.1", port: int = 8050) -> None:
    """Start the multi-threaded `XAU/USD` Python backend API server & pre-warm ProcessPoolExecutor cache."""
    if not CACHE_JSON_PATH.is_file():
        trigger_async_pillars_recompute()
    server = ThreadingHTTPServer((host, port), XAUUSDBackendAPIHandler)
    print(f"=== XAU/USD Python Backend API Server Running at http://{host}:{port} ===")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down XAU/USD Backend API Server...")
        server.server_close()
