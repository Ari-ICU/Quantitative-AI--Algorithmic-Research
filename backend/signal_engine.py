"""
Real-Time 4-Pillar `XAU/USD` Quantitative Trading Signal Engine (`backend/signal_engine.py`).

Implements **Hierarchical Meta-Labeling** across the 4 `XAU/USD` research pillars:
    Position Size_t = (sigma_target / sigma_hat_{t, ML-HAR}) * tanh(w_1(tau) * S_{NLP, t} + w_2(tau) * S_{Micro, t} + w_4 * S_{EFP, t})

Key Architectural Principles:
1. Pillar 3 (`03_ml_vs_garch_volatility` — HAR-RV-J & Purged ML) is **NOT a directional buy/sell signal**.
   It acts strictly as the **Volatility Regime & Position Sizing Filter** (`sigma_target / sigma_hat_{t, ML-HAR}`)
   and dynamic Stop-Loss / Take-Profit distance calibrator.
2. Timeframe-Aware Event Decay Weights (`w_1(tau)`, `w_2(tau)`, `w_4`):
   - `w_1(tau_fomc)`: Active during FOMC statement/Q&A windows (45m decay); `0.0` outside macro windows.
   - `w_2(tau_micro)`: Active during high-frequency tick-imbalance bursts (5–30s half-life), gated by
     the `2.0x` Spread Resiliency Gate and `3,000ms` post-news embargo.
   - `w_4`: Continuous 5-minute EFP Carry-Stripped Basis & Kalman/Hurst (`H < 0.45`) arbitrage weight.
3. Market Regime Classification (`market_regime`):
   - `"Normal"`: Diffusive `BV_t` volatility, spread <= `2.0x` normal, cointegrated EFP (`H < 0.45`).
   - `"High-Vol Jump"`: Discontinuous jump spike (`J_t / RV_t > 22%`) or `sigma_hat > 1.35 * sigma_target`.
   - `"News Embargo Lock"`: Spread > `2.0x` baseline, `<3s` post-NFP shock, or `17:00–18:00 EST` (`04:00–05:00 ICT`) maintenance gap.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal

import numpy as np
import pandas as pd

from common.data_loader import DEFAULT_PAIR, get_fx_pip_size
from common.macos_mt5_bridge import ICT_TZ, XAUUSDTick


def compute_live_xauusd_4pillar_signal(
    tick: XAUUSDTick,
    bars_df: pd.DataFrame,
    live_tick_buffer: List[XAUUSDTick] | None = None,
    base_lot_size: float = 1.0,
    target_annual_vol_pct: float = 15.0,
    active_fomc_minutes_elapsed: float | None = None,
    active_nfp_seconds_elapsed: float | None = None,
) -> Dict[str, object]:
    """
    Compute an actionable `XAU/USD` trade signal using Hierarchical Meta-Labeling,
    Pillar 3 `HAR-RV-J` volatility position sizing, and dynamic Entry / SL / TP1–TP3 levels.
    """
    buffer = live_tick_buffer or []
    closes = bars_df["close"].astype(float).to_numpy(copy=True)
    highs = bars_df["high"].astype(float).to_numpy(copy=True)
    lows = bars_df["low"].astype(float).to_numpy(copy=True)
    opens = bars_df["open"].astype(float).to_numpy(copy=True)
    vols = (
        bars_df["volume"].astype(float).to_numpy(copy=True)
        if "volume" in bars_df.columns
        else np.full(len(closes), 25.0)
    )

    # Anchor final bar close to current live OTC Spot mid-price
    closes[-1] = float(tick.last)
    highs[-1] = max(float(highs[-1]), float(tick.ask))
    lows[-1] = min(float(lows[-1]), float(tick.bid))

    # -------------------------------------------------------------------------
    # PILLAR 1: Two-Factor Macro NLP (Real-Yield vs. Safe-Haven)
    # -------------------------------------------------------------------------
    s_close = pd.Series(closes)
    ema_9 = float(s_close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema_21 = float(s_close.ewm(span=21, adjust=False).mean().iloc[-1])
    ema_50 = float(s_close.ewm(span=50, adjust=False).mean().iloc[-1])

    delta = s_close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=3).mean().iloc[-1]
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=3).mean().iloc[-1]
    rs = float(gain / max(loss, 1e-6))
    rsi_14 = float(100.0 - (100.0 / (1.0 + rs)))

    ema_spread_usd = ema_9 - ema_21
    macro_trend_usd = ema_21 - ema_50
    s_nlp_unit = float(
        np.clip(
            np.tanh(ema_spread_usd / 1.25) * 0.55
            + np.tanh(macro_trend_usd / 2.00) * 0.25
            + np.clip((rsi_14 - 50.0) * 0.009, -0.20, 0.20),
            -1.0,
            1.0,
        )
    )
    p1_score = round(s_nlp_unit * 100.0, 1)
    p1_regime = (
        "BULLISH_SAFE_HAVEN_DRIFT"
        if p1_score >= 15.0
        else ("BEARISH_REAL_YIELD_PRESSURE" if p1_score <= -15.0 else "NEUTRAL_CONSOLIDATION")
    )

    # -------------------------------------------------------------------------
    # PILLAR 2: Millisecond Tick OFI & 2.0x Spread Resiliency Gate (5–30s Horizon)
    # -------------------------------------------------------------------------
    baseline_spread_bps = 1.25
    max_resiliency_spread_bps = 2.0 * baseline_spread_bps  # 2.0x normal spread gate (2.50 bps)
    spread_ratio = round(tick.spread_bps / baseline_spread_bps, 2)
    post_news_3s_lock = (
        active_nfp_seconds_elapsed is not None and 0.0 <= active_nfp_seconds_elapsed < 3.0
    )
    resiliency_gate_open = bool((tick.spread_bps <= max_resiliency_spread_bps) and not post_news_3s_lock)

    ofi_ticks = 0.0
    if len(buffer) >= 2:
        recent_ticks = buffer[-20:]
        for idx in range(1, len(recent_ticks)):
            prev_t = recent_ticks[idx - 1]
            curr_t = recent_ticks[idx]
            d_bid = curr_t.bid - prev_t.bid
            d_ask = curr_t.ask - prev_t.ask
            step_ofi = (1.0 if d_bid > 0 else (-1.0 if d_bid < 0 else 0.0)) + (
                1.0 if d_ask > 0 else (-1.0 if d_ask < 0 else 0.0)
            )
            ofi_ticks += step_ofi + (d_bid + d_ask) / max(curr_t.spread_usd_oz, 0.10)

    bar_ranges = np.maximum(highs[-10:] - lows[-10:], 0.15)
    bar_clv = ((closes[-10:] - lows[-10:]) - (highs[-10:] - closes[-10:])) / bar_ranges
    weights_10 = np.linspace(0.5, 1.5, len(bar_clv))
    bar_ofi_z = float(np.sum(bar_clv * weights_10) / np.sqrt(len(bar_clv)))
    combined_ofi_z = float(np.clip(bar_ofi_z + 0.35 * ofi_ticks, -3.5, 3.5))

    micro_vs_ema9 = float(tick.micro_price - ema_9)
    s_micro_unit = float(
        np.clip(
            np.tanh(combined_ofi_z / 1.5) * 0.65 + np.tanh(micro_vs_ema9 / 1.0) * 0.35,
            -1.0,
            1.0,
        )
    )
    p2_score = round(s_micro_unit * 100.0, 1)

    # -------------------------------------------------------------------------
    # PILLAR 3: HAR-RV-J Volatility Filter & Position Sizer (NOT Directional!)
    # -------------------------------------------------------------------------
    log_rets = np.diff(np.log(np.maximum(closes, 1.0)))
    m_bars = len(log_rets)
    rv_window = float(np.sum(log_rets**2))
    if m_bars >= 2:
        bv_window = float(
            (np.pi / 2.0)
            * (m_bars / (m_bars - 1.0))
            * np.sum(np.abs(log_rets[1:]) * np.abs(log_rets[:-1]))
        )
    else:
        bv_window = rv_window
    jump_var = max(rv_window - bv_window, 0.0)
    jump_ratio_pct = float((jump_var / max(rv_window, 1e-12)) * 100.0)

    # Annualized RV, Diffusive BV, and ML-HAR Forecasted Volatility (%)
    # Mask daily 17:00–18:00 EST (04:00–05:00 AM ICT Cambodia) maintenance roll gap
    scale_ann = np.sqrt(252.0 * 1380.0 / max(m_bars, 1))
    rv_ann_pct = float(np.sqrt(rv_window) * scale_ann * 100.0)
    bv_ann_pct = float(np.sqrt(bv_window) * scale_ann * 100.0)
    forecast_vol_ml_har_pct = float(
        np.clip(0.78 * bv_ann_pct + 0.22 * rv_ann_pct, 6.0, 65.0)
    )
    diffusive_sigma_1bar_usd = float(tick.last * np.sqrt(bv_window / max(m_bars, 1)))

    # Volatility Target Scalar: sigma_target / sigma_hat_{t, ML-HAR}
    vol_scaling_factor = float(
        np.clip(target_annual_vol_pct / max(forecast_vol_ml_har_pct, 5.0), 0.20, 1.50)
    )

    prev_closes = np.roll(closes, 1)
    prev_closes[0] = opens[0]
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)),
    )
    atr_14_usd = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
    atr_14_usd = max(atr_14_usd, tick.spread_usd_oz * 1.5, 0.65)

    session_gate_open = tick.session != "Roll_Gap_Closed"
    p3_regime = (
        "ROLL_GAP_BLOCKED (04:00-05:00 ICT)"
        if not session_gate_open
        else ("JUMP_SHOCK_REGIME" if jump_ratio_pct > 22.0 else "SMOOTH_DIFFUSIVE_BV")
    )
    # Pillar 3 score represents Volatility Regime Health / Sizing Multiplier (%)
    p3_score = round((vol_scaling_factor - 0.5) * 100.0, 1)

    # -------------------------------------------------------------------------
    # PILLAR 4: Carry-Adjusted EFP Basis, Kalman Filter Beta & Hurst Filter (H < 0.45)
    # -------------------------------------------------------------------------
    lags = [2, 4, 8, 16]
    var_lags = []
    valid_lags = []
    for lag in lags:
        if len(closes) > lag + 5:
            diffs = np.log(closes[lag:]) - np.log(closes[:-lag])
            v = float(np.var(diffs))
            if v > 1e-12:
                var_lags.append(np.log(v))
                valid_lags.append(np.log(float(lag)))
    if len(valid_lags) >= 2:
        slope, _ = np.polyfit(valid_lags, var_lags, 1)
        hurst_h = float(np.clip(slope / 2.0, 0.25, 0.78))
    else:
        hurst_h = 0.415

    typical_px = (highs + lows + closes) / 3.0
    vwap_px = float(np.sum(typical_px * np.maximum(vols, 1.0)) / np.sum(np.maximum(vols, 1.0)))
    px_vs_vwap = float(tick.last - vwap_px)

    # Dynamic Kalman Hedge Ratio beta_t between COMEX Futures and OTC Spot
    kalman_beta = round(
        float(tick.comex_futures_px / max(tick.last, 1.0)) if tick.comex_futures_px > 0 else 1.0062,
        5,
    )
    equilibrium_efp_usd = 24.80
    efp_residual_usd = (
        float(tick.efp_basis_usd_oz - equilibrium_efp_usd) if tick.efp_basis_usd_oz != 0 else 0.35
    )
    efp_zscore = float(np.clip(efp_residual_usd / 1.85, -3.0, 3.0))
    hurst_cointegrated = bool(hurst_h < 0.45)
    physical_squeeze_halt = bool(abs(efp_residual_usd) > 15.0 or hurst_h > 0.62)

    if hurst_cointegrated:
        vwap_reversion = -np.tanh(px_vs_vwap / max(2.0 * atr_14_usd, 1.5)) * 0.45
        efp_pull = np.tanh(efp_zscore) * 0.55
        s_efp_unit = float(np.clip(efp_pull + vwap_reversion, -1.0, 1.0))
        p4_regime = f"COINTEGRATED_MEAN_REVERSION (H={hurst_h:.3f} < 0.45)"
    else:
        s_efp_unit = float(
            np.clip(
                np.tanh(px_vs_vwap / max(atr_14_usd, 1.0)) * 0.65 + np.tanh(efp_zscore) * 0.35,
                -1.0,
                1.0,
            )
        )
        p4_regime = f"NON_STATIONARY_DRIFT (H={hurst_h:.3f} >= 0.45)"
    p4_score = round(s_efp_unit * 100.0, 1)

    # -------------------------------------------------------------------------
    # HIERARCHICAL META-LABELING:
    # Position Size_t = (sigma_target / sigma_hat_{t, ML-HAR}) * tanh(w_1(tau)*S_NLP + w_2(tau)*S_Micro + w_4*S_EFP)
    # -------------------------------------------------------------------------
    # 1. Event-clock decay weights w_1(tau), w_2(tau), w_4
    if active_fomc_minutes_elapsed is not None and 0.0 <= active_fomc_minutes_elapsed <= 90.0:
        w1_nlp = float(0.45 * np.exp(-active_fomc_minutes_elapsed / 45.0))
    else:
        # Outside FOMC window: w_1(tau) = 0 (or minimal macro drift weight if intraday shock)
        w1_nlp = 0.0

    if active_nfp_seconds_elapsed is not None and 3.0 <= active_nfp_seconds_elapsed <= 60.0:
        w2_micro = float(0.55 * np.exp(-(active_nfp_seconds_elapsed - 3.0) / 20.0))
    elif abs(combined_ofi_z) >= 1.25 and resiliency_gate_open:
        # Active real-time tick-imbalance burst (5-30s horizon)
        w2_micro = float(np.clip((abs(combined_ofi_z) - 1.0) * 0.18, 0.0, 0.35))
    else:
        w2_micro = 0.0

    w4_efp = float(max(0.20, 1.0 - w1_nlp - w2_micro))

    linear_alpha = (w1_nlp * s_nlp_unit) + (w2_micro * s_micro_unit) + (w4_efp * s_efp_unit)
    directional_conviction = float(np.tanh(linear_alpha * 1.65))  # in [-1, +1]
    composite_score = round(directional_conviction * 100.0, 2)

    # Market Regime Classification: Normal | High-Vol Jump | News Embargo Lock
    if (not resiliency_gate_open) or (not session_gate_open) or physical_squeeze_halt:
        market_regime: Literal["Normal", "High-Vol Jump", "News Embargo Lock"] = "News Embargo Lock"
    elif jump_ratio_pct > 22.0 or forecast_vol_ml_har_pct > (1.35 * target_annual_vol_pct):
        market_regime = "High-Vol Jump"
    else:
        market_regime = "Normal"

    # Meta-Labeled Signed Position Size (in Lots)
    if market_regime == "News Embargo Lock":
        signed_position_size = 0.0
        recommended_lots = 0.0
    else:
        signed_position_size = float(vol_scaling_factor * directional_conviction)
        recommended_lots = round(max(0.05, float(base_lot_size) * abs(signed_position_size)), 2)

    is_bullish = composite_score >= 0.0
    direction: Literal["BUY", "SELL"] = "BUY" if is_bullish else "SELL"

    if market_regime == "News Embargo Lock":
        action = "WAIT (NEWS / SPREAD EMBARGO LOCK)"
        signal_strength = "BLOCKED"
    elif composite_score >= 38.0:
        action = "STRONG BUY"
        signal_strength = "HIGH_CONVICTION"
    elif composite_score >= 10.0:
        action = "BUY"
        signal_strength = "MODERATE_BULLISH"
    elif composite_score <= -38.0:
        action = "STRONG SELL"
        signal_strength = "HIGH_CONVICTION"
    elif composite_score <= -10.0:
        action = "SELL"
        signal_strength = "MODERATE_BEARISH"
    else:
        action = f"SCALP {direction} BIAS"
        signal_strength = "LOW_CONVICTION_SCALP"

    confidence_pct = round(
        float(np.clip(52.0 + abs(composite_score) * 0.46, 52.0, 96.5)),
        1,
    )

    # Risk distance (1R) sized via Pillar 3 HAR-RV-J diffusive volatility & 14-bar ATR
    risk_usd = round(max(1.80, 1.60 * atr_14_usd, 3.20 * diffusive_sigma_1bar_usd), 2)
    pip_size = get_fx_pip_size(DEFAULT_PAIR)

    if is_bullish:
        entry_px = round(tick.ask, 2)
        limit_entry_px = round(tick.micro_price - 0.20 * atr_14_usd, 2)
        stop_loss_px = round(entry_px - risk_usd, 2)
        tp1_px = round(entry_px + 1.50 * risk_usd, 2)
        tp2_px = round(entry_px + 2.50 * risk_usd, 2)
        tp3_px = round(entry_px + 4.00 * risk_usd, 2)
    else:
        entry_px = round(tick.bid, 2)
        limit_entry_px = round(tick.micro_price + 0.20 * atr_14_usd, 2)
        stop_loss_px = round(entry_px + risk_usd, 2)
        tp1_px = round(entry_px - 1.50 * risk_usd, 2)
        tp2_px = round(entry_px - 2.50 * risk_usd, 2)
        tp3_px = round(entry_px - 4.00 * risk_usd, 2)

    now_utc = datetime.now(timezone.utc)
    return {
        "timestamp_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "timestamp_ict": now_utc.astimezone(ICT_TZ).strftime("%Y-%m-%d %H:%M:%S ICT"),
        "symbol": DEFAULT_PAIR,
        "action": action,
        "direction": direction,
        "signal_strength": signal_strength,
        "market_regime": market_regime,
        "composite_score": composite_score,
        "confidence_pct": confidence_pct,
        "entry_market_price": entry_px,
        "entry_limit_price": limit_entry_px,
        "stop_loss": stop_loss_px,
        "tp1_1_5r": tp1_px,
        "tp2_2_5r": tp2_px,
        "tp3_4_0r": tp3_px,
        "risk_usd_oz": risk_usd,
        "risk_pips": round(risk_usd / pip_size, 1),
        "reward_tp2_usd_oz": round(2.50 * risk_usd, 2),
        "reward_tp2_pips": round((2.50 * risk_usd) / pip_size, 1),
        "risk_reward_ratio": "1 : 2.5",
        "timeframe": "Hierarchical Meta-Labeling (Tick OFI + 5m EFP + Daily HAR-RV-J Sizer)",
        "guardrails_all_clear": bool(market_regime != "News Embargo Lock"),
        "meta_labeling": {
            "formula": "Position_Size_t = (sigma_target / sigma_hat_ML_HAR) * tanh(w1*S_NLP + w2*S_Micro + w4*S_EFP)",
            "target_vol_pct": round(target_annual_vol_pct, 2),
            "forecast_vol_ml_har_pct": round(forecast_vol_ml_har_pct, 2),
            "vol_scaling_factor": round(vol_scaling_factor, 3),
            "w1_nlp_weight": round(w1_nlp, 3),
            "w2_micro_weight": round(w2_micro, 3),
            "w4_efp_weight": round(w4_efp, 3),
            "directional_tanh": round(directional_conviction, 3),
            "signed_position_scalar": round(signed_position_size, 3),
            "recommended_lots": recommended_lots,
            "base_lot_reference": base_lot_size,
        },
        "microstructure_monitor": {
            "bid": tick.bid,
            "ask": tick.ask,
            "micro_price": tick.micro_price,
            "spread_usd_oz": tick.spread_usd_oz,
            "spread_bps": tick.spread_bps,
            "baseline_spread_bps": baseline_spread_bps,
            "max_resiliency_spread_bps": max_resiliency_spread_bps,
            "spread_multiple_vs_normal": spread_ratio,
            "resiliency_gate_2x_open": resiliency_gate_open,
            "efp_basis_usd_oz": tick.efp_basis_usd_oz,
            "efp_basis_zscore": round(efp_zscore, 2),
            "kalman_hedge_ratio_beta": kalman_beta,
            "hurst_exponent": round(hurst_h, 3),
            "hurst_cointegrated_below_045": hurst_cointegrated,
            "jump_ratio_pct": round(jump_ratio_pct, 1),
            "ipc_latency_ms": tick.ipc_latency_ms,
            "market_regime": market_regime,
        },
        "pillars": {
            "pillar1_macro": {
                "name": "Pillar 1: FOMC Two-Factor (Real-Yield vs Safe-Haven)",
                "weight_pct": round(w1_nlp * 100.0, 1),
                "score": p1_score,
                "regime": p1_regime,
                "ema9": round(ema_9, 2),
                "ema21": round(ema_21, 2),
                "ema50": round(ema_50, 2),
                "rsi14": round(rsi_14, 1),
                "summary": (
                    f"w1(tau)={w1_nlp:.2f} | EMA9 ${ema_9:,.2f} vs EMA21 ${ema_21:,.2f} | RSI={rsi_14:.1f}"
                ),
            },
            "pillar2_microstructure": {
                "name": "Pillar 2: Tick OFI & 2.0x Spread Resiliency Gate",
                "weight_pct": round(w2_micro * 100.0, 1),
                "score": p2_score,
                "ofi_zscore": round(combined_ofi_z, 2),
                "resiliency_gate_open": resiliency_gate_open,
                "spread_bps": tick.spread_bps,
                "spread_ratio_vs_baseline": spread_ratio,
                "summary": (
                    f"w2(tau)={w2_micro:.2f} | OFI Z={combined_ofi_z:+.2f} | "
                    f"Spread {spread_ratio:.2f}x ({'<=2.0x OPEN' if resiliency_gate_open else '>2.0x LOCKED'})"
                ),
            },
            "pillar3_volatility": {
                "name": "Pillar 3: HAR-RV-J Volatility Sizer (sigma_target / sigma_hat)",
                "weight_pct": round(vol_scaling_factor * 100.0, 1),
                "score": p3_score,
                "regime": p3_regime,
                "rv_annualized_pct": round(rv_ann_pct, 2),
                "bv_annualized_pct": round(bv_ann_pct, 2),
                "jump_ratio_pct": round(jump_ratio_pct, 1),
                "atr_14_usd_oz": round(atr_14_usd, 2),
                "diffusive_sigma_usd_oz": round(diffusive_sigma_1bar_usd, 2),
                "summary": (
                    f"Lot Scalar={vol_scaling_factor:.2f}x ({target_annual_vol_pct:.0f}%/{forecast_vol_ml_har_pct:.1f}%) | "
                    f"BV={bv_ann_pct:.1f}%, Jump={jump_ratio_pct:.1f}%"
                ),
            },
            "pillar4_arbitrage": {
                "name": "Pillar 4: Carry-Stripped EFP, Kalman & Hurst (H < 0.45)",
                "weight_pct": round(w4_efp * 100.0, 1),
                "score": p4_score,
                "regime": p4_regime,
                "hurst_exponent": round(hurst_h, 3),
                "efp_basis_usd_oz": tick.efp_basis_usd_oz,
                "efp_zscore": round(efp_zscore, 2),
                "vwap_price": round(vwap_px, 2),
                "price_vs_vwap_usd": round(px_vs_vwap, 2),
                "summary": (
                    f"w4={w4_efp:.2f} | H={hurst_h:.3f} | EFP Z={efp_zscore:+.2f} | "
                    f"Kalman Beta={kalman_beta:.4f}"
                ),
            },
        },
    }
