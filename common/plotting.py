"""
Publication-quality visual comparison charts across all 4 `XAU/USD` (Spot Gold FX) Research Pillars,
driven directly by the real historical `XAU/USD` CSV datasets in `data/`:
1. Pillar 1 (`01_central_bank_nlp`): Real 1-minute XAU/USD Spot Trajectory (`data/xauusd_1m_real.csv`) & FOMC Shock HAC Regression.
2. Pillar 2 (`02_news_microstructure`): Real-Market-Anchored XAU/USD NFP ECN Spread (`data/xauusd_nfp_ticks_real.csv`) & Mid-Price Illusion vs. L2 Sweep.
3. Pillar 3 (`03_ml_vs_garch_volatility`): Real 2-Year Out-of-Sample 24h XAU/USD Volatility (`data/xauusd_intraday_2y_real.csv`) & Session RV Share.
4. Pillar 4 (`04_fx_cross_rate_arbitrage`): Real 5-Minute XAU/USD London LBMA vs. COMEX EFP Basis (`data/xauusd_cross_efp_5m_real.csv`) & Circuit Breaker.

Run directly via:
    .venv/bin/python -m common.plotting
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

cb_nlp = importlib.import_module("01_central_bank_nlp.pipeline")
news_micro = importlib.import_module("02_news_microstructure.pipeline")
vol_model = importlib.import_module("03_ml_vs_garch_volatility.pipeline")
fx_cross = importlib.import_module("04_fx_cross_rate_arbitrage.pipeline")


def _apply_fx_chart_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#1e293b",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#e2e8f0",
            "axes.titlecolor": "#f8fafc",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f8fafc",
            "grid.color": "#334155",
            "grid.linestyle": "--",
            "grid.alpha": 0.55,
            "legend.facecolor": "#0f172a",
            "legend.edgecolor": "#475569",
            "font.size": 10,
        }
    )


def plot_pillar1_central_bank_fx_nlp(output_dir: Path) -> Path:
    """Generate Pillar 1 comparison chart from real 1-minute XAU/USD CSV (`data/xauusd_1m_real.csv`)."""
    _apply_fx_chart_style()
    extractor = cb_nlp.VintageGuardNLPExtractor(model_id="ProsusAI/finbert")
    real_bars, real_events = cb_nlp.load_real_xauusd_1m_bars_and_fomc_events()
    scores = extractor.score_hawkish_dovish(real_events)
    aligned = cb_nlp.align_sentences_to_bars(real_events, scores, real_bars, horizon_minutes=2)
    stats = cb_nlp.estimate_hawkish_market_impact(aligned)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)

    # Plot a 120-minute real 1m XAU/USD window centered on the largest event reaction
    target_ts = aligned.loc[aligned["reaction_usd_oz"].abs().idxmax(), "reaction_bar_open"]
    win_bars = [
        b for b in real_bars
        if abs((b.bar_open_time - target_ts).total_seconds()) <= 3600
    ]
    if len(win_bars) < 20:
        win_bars = real_bars[:120]

    minutes = np.arange(len(win_bars))
    closes = [b.close_price for b in win_bars]
    ax1.plot(minutes, closes, color="#facc15", linewidth=2.0, label="Real XAU/USD 1m Close ($/oz)")
    for idx_b, b in enumerate(win_bars):
        if b.bar_open_time == target_ts:
            ax1.axvline(idx_b, color="#f43f5e", linestyle="--", linewidth=1.8, label="Event Utterance Timestamp")
            break
    ax1.set_title("Pillar 1A: Real Historical 1-Minute XAU/USD Spot Trajectory (`xauusd_1m_real.csv`)", fontweight="bold")
    ax1.set_xlabel("1-Minute Bars Around Event Window")
    ax1.set_ylabel("Real XAU/USD Spot Price ($ / troy oz)")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=8.5)

    x_vals = aligned["hawkish_score"].to_numpy()
    y_usd = aligned["reaction_usd_oz"].to_numpy()
    colors = ["#f43f5e" if x > 0 else "#22c55e" for x in x_vals]
    ax2.scatter(x_vals, y_usd, c=colors, s=85, edgecolors="white", zorder=4, label="Real 2m XAU/USD Reaction ($/oz)")
    x_line = np.linspace(x_vals.min() - 0.1, x_vals.max() + 0.1, 100)
    slope_usd = stats["beta_hawkish_usd_oz"]
    intercept_usd = float(np.mean(y_usd - slope_usd * x_vals))
    ax2.plot(
        x_line,
        intercept_usd + slope_usd * x_line,
        color="#facc15",
        linewidth=2.2,
        label=f"OLS HAC Fit: {slope_usd:+.2f} $/oz ({stats['beta_hawkish_bps']:+.1f} bps/shock, R²={stats['r_squared']:.2f})",
    )
    ax2.axhline(0, color="#94a3b8", linewidth=1, linestyle="--")
    ax2.axvline(0, color="#94a3b8", linewidth=1, linestyle="--")
    ax2.set_title("Pillar 1B: FOMC Hawkish Score vs. Real XAU/USD Post-Utterance ($/oz)", fontweight="bold")
    ax2.set_xlabel("Vintage-Gated FOMC Net Hawkish (+1) / Dovish (-1) Score")
    ax2.set_ylabel("Real 2-Minute XAU/USD Change ($/oz)")
    ax2.grid(True)
    ax2.legend(loc="lower left", fontsize=8.5)

    fig.tight_layout()
    out_path = output_dir / "01_central_bank_nlp_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pillar2_fx_news_microstructure(output_dir: Path) -> Path:
    """Generate Pillar 2 comparison chart from `data/xauusd_nfp_ticks_real.csv`."""
    _apply_fx_chart_style()
    ticks = news_micro.load_xauusd_nfp_ticks_from_csv()
    sim = news_micro.simulate_spread_aware_execution(
        ticks,
        news_micro.OrderExecutionSpec(
            side="buy",
            signal_ms_from_release=50,
            latency_ms=20,
            order_size_lots=10.0,
            execution_price_mode="spread_and_depth_sweep",
            pair="XAU/USD",
        ),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)

    ms = ticks["ms_from_release"].to_numpy()
    ax1.plot(ms, ticks["spread_usd_oz"], color="#facc15", linewidth=1.6, label="XAU/USD ECN Spread ($/oz)")
    ax1.axvspan(0, 5000, color="#ef4444", alpha=0.18, label="5s Post-NFP Liquidity Vacuum")
    ax1.axvline(0, color="#f8fafc", linestyle="--", linewidth=1.4, label="Release Timestamp (t = 0 ms)")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(ms, ticks["ask_sz_lots"], color="#38bdf8", alpha=0.65, linewidth=1.2, label="Top-of-Book Ask Depth (100-oz Lots)")
    ax1_twin.set_ylabel("ECN L1 Ask Liquidity (100-oz XAU/USD Lots)", color="#38bdf8")

    ax1.set_title("Pillar 2A: Millisecond XAU/USD Spread ($/oz) & L1 Depth (`xauusd_nfp_ticks_real.csv`)", fontweight="bold")
    ax1.set_xlabel("Milliseconds from Macro Release (t = 0 ms)")
    ax1.set_ylabel("Quoted ECN XAU/USD Spread ($ / troy oz)", color="#facc15")
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=8.5)

    categories = [
        "Illusory Mid-to-Mid\nXAU/USD Backtest PnL",
        "ECN Half-Spread &\nL2 Book-Sweep Drag",
        "Realized Spread-Swept\nXAU/USD Net PnL",
    ]
    values = [
        sim["mid_price_illusion_pnl_bps"],
        -sim["execution_cost_drag_bps"],
        sim["realized_spread_swept_pnl_bps"],
    ]
    bar_colors = ["#38bdf8", "#f43f5e", "#22c55e"]
    bars = ax2.bar(categories, values, color=bar_colors, width=0.52, edgecolor="white", linewidth=1.2)
    ax2.axhline(0, color="#cbd5e1", linewidth=1)
    for bar, val in zip(bars, values):
        y_offset = val + (0.6 if val >= 0 else -1.5)
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_offset,
            f"{val:+.2f} bps",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontweight="bold",
            color="#f8fafc",
        )
    ax2.set_ylim(min(values) - 4.0, max(values) + 4.0)
    ax2.set_title("Pillar 2B: XAU/USD Mid-Price Execution Illusion vs. Realized ECN Sweep", fontweight="bold")
    ax2.set_ylabel("XAU/USD Return / Execution Drag (Basis Points)")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    out_path = output_dir / "02_news_microstructure_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pillar3_fx_volatility_comparison(output_dir: Path) -> Path:
    """Generate Pillar 3 comparison chart from real 2-year XAU/USD intraday CSV (`data/xauusd_intraday_2y_real.csv`)."""
    _apply_fx_chart_style()
    intraday_df = vol_model.load_xauusd_intraday_bars_from_csv()
    daily_rv = vol_model.compute_daily_realized_variance_from_5m(intraday_df)
    feat_df = vol_model.build_har_rv_features(daily_rv)
    res = vol_model.run_walk_forward_volatility_comparison(
        feat_df, min_train_days=120, test_step_days=15, return_series=True
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)

    days = np.arange(res["n_oos_days"])
    y_real = np.asarray(res["realized_rvol"]) * 100.0
    y_ml = np.asarray(res["ml_forecast_rvol"]) * 100.0
    y_garch = np.asarray(res["garch_forecast_rvol"]) * 100.0

    ax1.plot(days, y_real, color="#94a3b8", alpha=0.65, linewidth=1.3, label="Realized 24h XAU/USD RVol (%)")
    ax1.plot(
        days,
        y_ml,
        color="#facc15",
        linewidth=2.1,
        label=f"Leak-Free HAR-XAU ML (RMSE={res['ml_rmse']*100:.2f}%, QLIKE={res['ml_qlike']:.3f})",
    )
    ax1.plot(
        days,
        y_garch,
        color="#f97316",
        linewidth=1.9,
        linestyle="--",
        label=f"GARCH(1,1) Baseline (RMSE={res['garch_rmse']*100:.2f}%, QLIKE={res['garch_qlike']:.3f})",
    )
    ax1.set_title("Pillar 3A: Real 2024–2026 Walk-Forward XAU/USD Volatility (`xauusd_intraday_2y_real.csv`)", fontweight="bold")
    ax1.set_xlabel(f"Out-of-Sample Real Trading Days ({res['n_oos_days']} Days)")
    ax1.set_ylabel("Annualized XAU/USD Realized Volatility (%)")
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=8.5)

    sessions = [
        "Tokyo / SGE\n(00–08 UTC)",
        "London Bullion AM\n(08–13 UTC)",
        "COMEX / LBMA Overlap\n(13–17 UTC)",
        "NY Rollover /\nSydney (17–24 UTC)",
    ]
    shares = [
        100.0 * float((daily_rv["rv_tokyo"] / daily_rv["rv_daily"]).mean()),
        100.0 * float((daily_rv["rv_london"] / daily_rv["rv_daily"]).mean()),
        100.0 * float((daily_rv["rv_london_ny_overlap"] / daily_rv["rv_daily"]).mean()),
        100.0 * float((daily_rv["rv_ny_sydney"] / daily_rv["rv_daily"]).mean()),
    ]
    s_colors = ["#38bdf8", "#818cf8", "#facc15", "#a855f7"]
    bars = ax2.bar(sessions, shares, color=s_colors, width=0.55, edgecolor="white", linewidth=1.2)
    for bar, sh in zip(bars, shares):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            sh + 0.8,
            f"{sh:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="#f8fafc",
        )
    ax2.set_ylim(0, max(shares) + 7.0)
    ax2.set_title("Pillar 3B: Real 24-Hour Global XAU/USD Session Realized Variance Share", fontweight="bold")
    ax2.set_ylabel("Share of Daily 24h XAU/USD Realized Variance (%)")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    out_path = output_dir / "03_ml_vs_garch_volatility_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pillar4_fx_cross_rate_arbitrage(output_dir: Path) -> Path:
    """Generate Pillar 4 comparison chart from real 5-minute XAU/USD Cross/EFP CSV (`data/xauusd_cross_efp_5m_real.csv`)."""
    _apply_fx_chart_style()
    bars_df = fx_cross.load_xauusd_cross_efp_bars_from_csv(n_bars=500)
    monitor = fx_cross.FXPegAndCIPRegimeMonitor(max_peg_deviation_bps=25.0, max_cip_basis_bps=40.0)
    res = fx_cross.generate_fx_cross_rate_arbitrage_signals(bars_df, monitor, ignore_regime_guardrail=False)
    frame = res["frame"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)
    idx = np.arange(len(frame))

    ax1.plot(
        idx,
        frame["naive_basis_bps"],
        color="#f43f5e",
        linewidth=1.7,
        label="Naive XAU/USD Basis (Unadjusted for London-COMEX EFP Shift)",
    )
    ax1.plot(
        idx,
        frame["true_cross_basis_bps"],
        color="#facc15",
        linewidth=1.8,
        label=f"Regime-Adjusted Clean XAU/USD Basis (ADF={res['adf_stat']:.2f})",
    )
    breach_mask = frame["is_regime_breach"].to_numpy()
    ax1.fill_between(
        idx,
        frame["naive_basis_bps"].min() - 5,
        frame["naive_basis_bps"].max() + 5,
        where=breach_mask,
        color="#ef4444",
        alpha=0.20,
        label=f"EFP / Lease-Rate Breach Halted ({res['regime_bars_halted']} bars)",
    )
    ax1.set_title("Pillar 4A: Real 5m XAU/USD Spot vs. EFP Basis (`xauusd_cross_efp_5m_real.csv`)", fontweight="bold")
    ax1.set_xlabel("Real 5-Minute XAU/USD Bar Index")
    ax1.set_ylabel("XAU/USD Cross / EFP Basis (bps)")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=8.5)

    ax2.plot(idx, frame["z_score"], color="#38bdf8", linewidth=1.4, label="Clean XAU/USD Basis Rolling Z-Score")
    ax2.axhline(2.0, color="#facc15", linestyle="--", linewidth=1.2, label="Entry Threshold (±2.0σ)")
    ax2.axhline(-2.0, color="#facc15", linestyle="--", linewidth=1.2)
    ax2.step(
        idx,
        frame["position"] * 2.5,
        color="#22c55e",
        linewidth=2.0,
        where="post",
        label="Gated XAU/USD Strategy Position (Forced 0 on Breach)",
    )
    ax2.fill_between(
        idx,
        -4.5,
        4.5,
        where=breach_mask,
        color="#ef4444",
        alpha=0.20,
        label="Circuit Breaker Active (Position = 0)",
    )
    ax2.set_title("Pillar 4B: Real XAU/USD Z-Score & Circuit-Breaker Gated Position", fontweight="bold")
    ax2.set_xlabel("Real 5-Minute XAU/USD Bar Index")
    ax2.set_ylabel("Rolling Z-Score / Scaled Position")
    ax2.grid(True)
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    out_path = output_dir / "04_fx_cross_rate_arbitrage_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_master_fx_comparison_dashboard(output_dir: Path) -> Path:
    """Create a 2x2 summary dashboard comparing all 4 XAU/USD guardrails on real historical CSVs."""
    _apply_fx_chart_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=160)

    # Panel 1: Pillar 1 Real 1m XAU/USD FOMC Reaction
    extractor = cb_nlp.VintageGuardNLPExtractor(model_id="ProsusAI/finbert")
    real_bars, real_events = cb_nlp.load_real_xauusd_1m_bars_and_fomc_events()
    scores = extractor.score_hawkish_dovish(real_events)
    aligned = cb_nlp.align_sentences_to_bars(real_events, scores, real_bars, horizon_minutes=2)
    stats1 = cb_nlp.estimate_hawkish_market_impact(aligned)

    ax = axes[0, 0]
    x_v = aligned["hawkish_score"].to_numpy()
    y_v = aligned["reaction_usd_oz"].to_numpy()
    ax.scatter(x_v, y_v, c=["#f43f5e" if v > 0 else "#22c55e" for v in x_v], s=80, edgecolors="white", zorder=4)
    xl = np.linspace(x_v.min() - 0.1, x_v.max() + 0.1, 50)
    b_usd = stats1["beta_hawkish_usd_oz"]
    a_usd = float(np.mean(y_v - b_usd * x_v))
    ax.plot(xl, a_usd + b_usd * xl, color="#facc15", linewidth=2.2, label=f"Slope: {b_usd:+.2f} $/oz ({stats1['beta_hawkish_bps']:+.1f} bps/shock, R²={stats1['r_squared']:.2f})")
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.set_title("1. Real 1m XAU/USD (`xauusd_1m_real.csv`): FOMC Score vs. $/oz", fontweight="bold")
    ax.set_xlabel("Vintage-Gated FOMC Net Hawkish Score")
    ax.set_ylabel("Real 2-Min XAU/USD Reaction ($ / troy oz)")
    ax.grid(True)
    ax.legend(loc="lower left")

    # Panel 2: Pillar 2 Real-Anchored NFP XAU/USD Tick Execution
    ax = axes[0, 1]
    ticks = news_micro.load_xauusd_nfp_ticks_from_csv()
    sim = news_micro.simulate_spread_aware_execution(
        ticks,
        news_micro.OrderExecutionSpec(
            side="buy", signal_ms_from_release=50, latency_ms=20, order_size_lots=10.0, pair="XAU/USD"
        ),
    )
    cats = ["Illusory Mid-Price\nXAU/USD PnL", "ECN Spread + L2\nGold Book-Sweep Drag", "Realized ECN\nXAU/USD Net PnL"]
    vals = [sim["mid_price_illusion_pnl_bps"], -sim["execution_cost_drag_bps"], sim["realized_spread_swept_pnl_bps"]]
    b2 = ax.bar(cats, vals, color=["#38bdf8", "#f43f5e", "#22c55e"], width=0.5, edgecolor="white")
    ax.axhline(0, color="#cbd5e1", linewidth=1)
    for bar, val in zip(b2, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + (0.6 if val >= 0 else -1.5), f"{val:+.2f} bps", ha="center", fontweight="bold")
    ax.set_ylim(min(vals) - 4.0, max(vals) + 4.0)
    ax.set_title("2. Real-Anchored XAU/USD Ticks (`xauusd_nfp_ticks_real.csv`): Execution Drag", fontweight="bold")
    ax.set_ylabel("XAU/USD Return / Drag (bps)")
    ax.grid(True, axis="y")

    # Panel 3: Pillar 3 Real 2Y XAU/USD Walk-Forward Volatility
    ax = axes[1, 0]
    intraday_df = vol_model.load_xauusd_intraday_bars_from_csv()
    daily_rv = vol_model.compute_daily_realized_variance_from_5m(intraday_df)
    feat_df = vol_model.build_har_rv_features(daily_rv)
    res3 = vol_model.run_walk_forward_volatility_comparison(feat_df, min_train_days=120, test_step_days=15, return_series=True)
    d_idx = np.arange(res3["n_oos_days"])
    ax.plot(d_idx, np.asarray(res3["realized_rvol"]) * 100.0, color="#94a3b8", alpha=0.65, label="Realized 24h XAU/USD RVol")
    ax.plot(d_idx, np.asarray(res3["ml_forecast_rvol"]) * 100.0, color="#facc15", linewidth=2.1, label=f"Leak-Free HAR-XAU ML (QLIKE={res3['ml_qlike']:.3f})")
    ax.plot(d_idx, np.asarray(res3["garch_forecast_rvol"]) * 100.0, color="#f97316", linestyle="--", linewidth=1.9, label=f"GARCH(1,1) (QLIKE={res3['garch_qlike']:.3f})")
    ax.set_title("3. Real 2Y 24h XAU/USD Volatility (`xauusd_intraday_2y_real.csv`): ML vs. GARCH", fontweight="bold")
    ax.set_xlabel(f"Out-of-Sample Real Trading Days ({res3['n_oos_days']} Days)")
    ax.set_ylabel("Annualized XAU/USD Volatility (%)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel 4: Pillar 4 Real 5m XAU/USD Cross/EFP Basis
    ax = axes[1, 1]
    bars_df = fx_cross.load_xauusd_cross_efp_bars_from_csv(n_bars=500)
    monitor = fx_cross.FXPegAndCIPRegimeMonitor(max_peg_deviation_bps=25.0, max_cip_basis_bps=40.0)
    res4 = fx_cross.generate_fx_cross_rate_arbitrage_signals(bars_df, monitor, ignore_regime_guardrail=False)
    f4 = res4["frame"]
    idx4 = np.arange(len(f4))
    ax.plot(idx4, f4["naive_basis_bps"], color="#f43f5e", linewidth=1.6, label="Naive XAU/USD Basis (London-COMEX EFP Shift)")
    ax.plot(idx4, f4["true_cross_basis_bps"], color="#facc15", linewidth=1.6, label="Clean XAU/USD Inter-Contract Basis")
    breach_mask = f4["is_regime_breach"].to_numpy()
    ax.fill_between(
        idx4,
        f4["naive_basis_bps"].min() - 5,
        f4["naive_basis_bps"].max() + 5,
        where=breach_mask,
        color="#ef4444",
        alpha=0.22,
        label=f"Circuit Breaker Halted ({res4['regime_bars_halted']} bars)",
    )
    ax.set_title("4. Real 5m XAU/USD EFP Basis (`xauusd_cross_efp_5m_real.csv`): Guarded vs. Naive", fontweight="bold")
    ax.set_xlabel("Real 5-Minute XAU/USD Bar Index")
    ax.set_ylabel("XAU/USD Spot vs. EFP Basis (bps)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Quantitative XAU/USD (Real Historical CSV Data) — 4-Pillar Methodological Guardrail Dashboard",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out_path = output_dir / "fx_research_comparison_dashboard.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_fx_comparison_charts(output_dir: str | Path = "figures") -> Dict[str, Path]:
    """Generate all 4 individual real-CSV XAU/USD pillar comparison charts plus the 2x2 master XAU/USD dashboard."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "dashboard": plot_master_fx_comparison_dashboard(out),
        "pillar1_cb_nlp": plot_pillar1_central_bank_fx_nlp(out),
        "pillar2_news_micro": plot_pillar2_fx_news_microstructure(out),
        "pillar3_vol_model": plot_pillar3_fx_volatility_comparison(out),
        "pillar4_fx_cross": plot_pillar4_fx_cross_rate_arbitrage(out),
    }


if __name__ == "__main__":
    paths = generate_all_fx_comparison_charts("figures")
    print("=== Generated Real-CSV XAU/USD Comparison Charts ===")
    for name, p in paths.items():
        print(f"  [{name}] -> {p}")
