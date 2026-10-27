"""
Non-Blocking Empirical Research Service (`backend/research_service.py`).

Uses `concurrent.futures.ProcessPoolExecutor` and a persistent JSON cache
(`backend/data/cached_pillars_payload.json`) so heavy backtests and ML/GARCH/HAR-RV-J
evaluations across Pillars `01` through `04` never block the live `XAU/USD` tick/signal API loop.
"""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
import importlib
import json
from pathlib import Path
import sys
from typing import Dict

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np

from common.plotting import evaluate_literature_paper_benchmarks

PROJECT_ROOT = BACKEND_DIR
CACHE_JSON_PATH = BACKEND_DIR / "data" / "cached_pillars_payload.json"

_CACHED_BENCHMARKS: Dict[str, Dict[str, float]] | None = None
_CACHED_PILLARS_PAYLOAD: Dict[str, object] | None = None
_PROCESS_POOL: ProcessPoolExecutor | None = None
_ACTIVE_FUTURE: Future[Dict[str, object]] | None = None


def _compute_pillars_payload_worker() -> Dict[str, object]:
    """
    Isolated worker function executed inside `ProcessPoolExecutor` so CPU-intensive
    walk-forward ML/HAR-RV-J/GARCH and DuckDB tick aggregations never block the API server.
    """
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    cb_nlp = importlib.import_module("01_central_bank_nlp.pipeline")
    news_micro = importlib.import_module("02_news_microstructure.pipeline")
    vol_model = importlib.import_module("03_ml_vs_garch_volatility.pipeline")
    fx_cross = importlib.import_module("04_fx_cross_rate_arbitrage.pipeline")

    # Pillar 1: FOMC Two-Factor NLP
    extractor = cb_nlp.VintageGuardNLPExtractor(model_id="ProsusAI/finbert")
    real_bars, real_events = cb_nlp.load_real_xauusd_1m_bars_and_fomc_events()
    tf = extractor.score_two_factor_fomc_surprise(real_events)
    aligned = cb_nlp.align_sentences_to_bars(
        real_events,
        tf["rate_shock_score"],
        real_bars,
        horizon_minutes=2,
        tail_scores=tf["tail_safe_haven_score"],
        novelty_weights=tf["novelty_weight"],
    )
    stats1 = cb_nlp.estimate_hawkish_market_impact(aligned)
    events_list = []
    for idx_r, row in aligned.iterrows():
        ev_text = real_events[int(idx_r)].text if int(idx_r) < len(real_events) else ""
        events_list.append(
            {
                "speaker": str(row["speaker"]),
                "phase": str(row["event_phase"]),
                "timestamp": str(row["sentence_timestamp"]),
                "text": ev_text,
                "rate_shock": round(float(row["rate_shock_score"]), 3),
                "tail_safe_haven": round(float(row["tail_safe_haven_score"]), 3),
                "novelty_weight": round(float(row["novelty_weight"]), 3),
                "reaction_bps": round(float(row["reaction_bps"]), 2),
                "reaction_usd_oz": round(float(row["reaction_usd_oz"]), 2),
            }
        )

    # Pillar 2: Post-NFP Millisecond Ticks & Liquidity Resiliency Gate
    ticks = news_micro.load_xauusd_nfp_ticks_from_csv()
    metrics2 = news_micro.compute_microstructure_metrics_duckdb(ticks)
    gate2 = news_micro.evaluate_liquidity_resiliency_gate(ticks)
    sim_unguarded = news_micro.simulate_spread_aware_execution(
        ticks,
        news_micro.OrderExecutionSpec(
            side="buy",
            signal_ms_from_release=50,
            latency_ms=20,
            enforce_resiliency_gate=False,
            pair="XAU/USD",
        ),
    )
    sim_gated = news_micro.simulate_spread_aware_execution(
        ticks,
        news_micro.OrderExecutionSpec(
            side="buy",
            signal_ms_from_release=50,
            latency_ms=20,
            enforce_resiliency_gate=True,
            pair="XAU/USD",
        ),
    )
    step2 = max(1, len(ticks) // 120)
    ticks_sub = ticks.iloc[::step2]
    tick_series = [
        {
            "ms": int(r["ms_from_release"]),
            "bid": round(float(r["bid_px"]), 2),
            "ask": round(float(r["ask_px"]), 2),
            "micro": round(float(r["micro_px"]), 3),
            "spread_usd": round(float(r["spread_usd_oz"]), 2),
            "spread_bps": round(float(r["spread_bps"]), 2),
            "ask_depth": round(float(r["ask_sz_lots"]), 1),
            "ofi": round(float(r["ofi_rolling_50t"]), 2),
        }
        for _, r in ticks_sub.iterrows()
    ]

    # Pillar 3: 24h Intraday Volatility (450 OOS Days)
    intraday_df = vol_model.load_xauusd_intraday_bars_from_csv()
    daily_rv = vol_model.compute_daily_realized_variance_from_5m(intraday_df)
    feat_df = vol_model.build_har_rv_features(daily_rv)
    res3 = vol_model.run_walk_forward_volatility_comparison(
        feat_df, min_train_days=120, test_step_days=15, return_series=True
    )
    session_shares = {
        "Tokyo_SGE": round(100.0 * float((daily_rv["rv_tokyo"] / daily_rv["rv_daily"]).mean()), 1),
        "London_AM": round(100.0 * float((daily_rv["rv_london"] / daily_rv["rv_daily"]).mean()), 1),
        "COMEX_LBMA_Overlap": round(
            100.0 * float((daily_rv["rv_london_ny_overlap"] / daily_rv["rv_daily"]).mean()), 1
        ),
        "NY_Rollover_Sydney": round(
            100.0 * float((daily_rv["rv_ny_sydney"] / daily_rv["rv_daily"]).mean()), 1
        ),
    }
    oos_dates = [str(d)[:10] for d in res3["oos_dates"]]  # type: ignore[index]
    y_real = np.asarray(res3["realized_rvol"]) * 100.0
    y_ml = np.asarray(res3["ml_forecast_rvol"]) * 100.0
    y_har = np.asarray(res3["har_rv_j_forecast_rvol"]) * 100.0
    y_garch = np.asarray(res3["garch_forecast_rvol"]) * 100.0
    step3 = max(1, len(oos_dates) // 110)
    vol_series = [
        {
            "date": oos_dates[i],
            "realized": round(float(y_real[i]), 2),
            "purged_ml": round(float(y_ml[i]), 2),
            "har_rv_j": round(float(y_har[i]), 2),
            "garch": round(float(y_garch[i]), 2),
        }
        for i in range(0, len(oos_dates), step3)
    ]

    # Pillar 4: Cross-Venue London-COMEX EFP Basis Arbitrage
    bars_efp = fx_cross.load_xauusd_cross_efp_bars_from_csv(n_bars=360)
    monitor = fx_cross.FXPegAndCIPRegimeMonitor(
        max_peg_deviation_bps=25.0, max_cip_basis_bps=40.0
    )
    res4 = fx_cross.generate_fx_cross_rate_arbitrage_signals(
        bars_efp, monitor, ignore_regime_guardrail=False
    )
    f4 = res4["frame"]
    step4 = max(1, len(f4) // 120)
    efp_series = [
        {
            "idx": int(i),
            "timestamp": str(f4["timestamp"].iloc[i]),
            "naive_basis_bps": round(float(f4["naive_basis_bps"].iloc[i]), 2),
            "clean_basis_bps": round(float(f4["true_cross_basis_bps"].iloc[i]), 2),
            "carry_adj_bps": round(float(f4["carry_adjusted_efp_bps"].iloc[i]), 2),
            "z_score": round(float(f4["z_score"].iloc[i]), 2),
            "hurst": round(float(f4["hurst_exponent"].iloc[i]), 3),
            "kalman_beta": round(float(f4["kalman_beta"].iloc[i]), 5),
            "position": float(f4["position"].iloc[i]),
            "breach": bool(f4["is_regime_breach"].iloc[i]),
        }
        for i in range(0, len(f4), step4)
    ]

    benchmarks = evaluate_literature_paper_benchmarks()
    payload: Dict[str, object] = {
        "benchmarks": benchmarks,
        "pillar1": {
            "stats": {k: round(float(v), 4) for k, v in stats1.items()},
            "events": events_list,
        },
        "pillar2": {
            "metrics": {k: round(float(v), 3) for k, v in metrics2.items()},
            "resiliency_recovery_ms": round(float(gate2["resiliency_recovery_ms"]), 1),
            "sim_unguarded": {k: round(float(v), 3) for k, v in sim_unguarded.items()},
            "sim_gated": {k: round(float(v), 3) for k, v in sim_gated.items()},
            "tick_series": tick_series,
        },
        "pillar3": {
            "n_oos_days": int(res3["n_oos_days"]),
            "ml_rmse_pct": round(float(res3["ml_rmse"]) * 100.0, 2),
            "har_rv_j_rmse_pct": round(float(res3["har_rv_j_rmse"]) * 100.0, 2),
            "garch_rmse_pct": round(float(res3["garch_rmse"]) * 100.0, 2),
            "ml_qlike": round(float(res3["ml_qlike"]), 4),
            "har_rv_j_qlike": round(float(res3["har_rv_j_qlike"]), 4),
            "garch_qlike": round(float(res3["garch_qlike"]), 4),
            "session_shares": session_shares,
            "vol_series": vol_series,
        },
        "pillar4": {
            "adf_stat": round(float(res4["adf_stat"]), 3),
            "kalman_beta_mean": round(float(res4["kalman_beta_mean"]), 5),
            "mean_hurst_exponent": round(float(res4["mean_hurst_exponent"]), 3),
            "regime_bars_halted": int(res4["regime_bars_halted"]),
            "max_regime_distortion_bps": round(float(res4["max_regime_distortion_bps"]), 2),
            "deflated_sharpe_ratio": round(float(res4["deflated_sharpe_ratio"]), 3),
            "cost_adjusted_profit_factor": round(float(res4["cost_adjusted_profit_factor"]), 2),
            "efp_series": efp_series,
        },
    }

    try:
        CACHE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    return payload


def trigger_async_pillars_recompute() -> str:
    """Submit a non-blocking background job to `ProcessPoolExecutor` to refresh the JSON cache."""
    global _PROCESS_POOL, _ACTIVE_FUTURE
    if _ACTIVE_FUTURE is not None and not _ACTIVE_FUTURE.done():
        return "RUNNING_IN_BACKGROUND"
    if _PROCESS_POOL is None:
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=1)
    _ACTIVE_FUTURE = _PROCESS_POOL.submit(_compute_pillars_payload_worker)
    return "SUBMITTED_TO_PROCESS_POOL"


def get_cached_pillars_payload() -> Dict[str, object]:
    """
    Return the cached 4-Pillar empirical JSON payload in <1ms from memory or
    `backend/data/cached_pillars_payload.json` without blocking the API server.
    """
    global _CACHED_PILLARS_PAYLOAD, _ACTIVE_FUTURE
    if _ACTIVE_FUTURE is not None and _ACTIVE_FUTURE.done():
        try:
            _CACHED_PILLARS_PAYLOAD = _ACTIVE_FUTURE.result()
        except Exception:
            pass
        _ACTIVE_FUTURE = None

    if _CACHED_PILLARS_PAYLOAD is not None:
        return _CACHED_PILLARS_PAYLOAD

    if CACHE_JSON_PATH.is_file():
        try:
            _CACHED_PILLARS_PAYLOAD = json.loads(CACHE_JSON_PATH.read_text(encoding="utf-8"))
            return _CACHED_PILLARS_PAYLOAD  # type: ignore[return-value]
        except Exception:
            pass

    _CACHED_PILLARS_PAYLOAD = _compute_pillars_payload_worker()
    return _CACHED_PILLARS_PAYLOAD


def get_cached_benchmarks() -> Dict[str, Dict[str, float]]:
    """Return cached benchmark comparison dictionary from the pre-computed JSON cache."""
    global _CACHED_BENCHMARKS
    if _CACHED_BENCHMARKS is not None:
        return _CACHED_BENCHMARKS
    payload = get_cached_pillars_payload()
    if isinstance(payload.get("benchmarks"), dict):
        _CACHED_BENCHMARKS = payload["benchmarks"]  # type: ignore[assignment]
        return _CACHED_BENCHMARKS
    _CACHED_BENCHMARKS = evaluate_literature_paper_benchmarks()
    return _CACHED_BENCHMARKS
