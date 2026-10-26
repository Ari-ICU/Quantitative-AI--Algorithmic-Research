"""
Topic 2: XAU/USD News Microstructure — Millisecond ECN Tick Processing & Post-NFP Spread-Widening Execution.

Primary Libraries: polars, duckdb, scipy
Granularity: Real Historical XAU/USD ECN tick CSV (`data/xauusd_nfp_ticks_real.csv`, Bid/Ask $/oz and 100-oz lots)
Pitfall Prevented: Execution illusion (assuming mid-price XAU/USD fills during the 5 seconds after US NFP/CPI
when quoted ECN bullion spreads widen >12x and top-of-book L1 gold liquidity evaporates).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal
import numpy as np
import pandas as pd

from common.data_loader import XAUUSD_NFP_TICKS_CSV, load_real_xauusd_ticks_csv
from common.validation import (
    DEFAULT_PAIR,
    ExecutionIllusionError,
    assert_xauusd_pair,
    get_fx_default_spot_price,
    get_fx_pip_size,
)


@dataclass(frozen=True)
class OrderExecutionSpec:
    """Parameters governing realistic ECN XAU/USD market-order execution during macro releases (NFP/CPI)."""

    side: Literal["buy", "sell"]
    signal_ms_from_release: int
    latency_ms: int = 15
    order_size_lots: float = 10.0  # 1 standard XAU/USD lot = 100 troy oz (10 lots = 1,000 oz clip)
    l2_book_slope_bps_per_lot: float = 0.45
    execution_price_mode: Literal["spread_and_depth_sweep", "mid"] = "spread_and_depth_sweep"
    pair: str = DEFAULT_PAIR


def load_xauusd_nfp_ticks_from_csv(
    csv_path: str | Path = XAUUSD_NFP_TICKS_CSV,
    n_ticks: int | None = None,
    pair: str = DEFAULT_PAIR,
) -> pd.DataFrame:
    """
    Load real-market-anchored `XAU/USD` millisecond Bid/Ask ticks from `data/xauusd_nfp_ticks_real.csv`
    (or any external MT5 / Dukascopy / LMAX `XAU/USD` tick CSV).
    """
    assert_xauusd_pair(pair)
    df = load_real_xauusd_ticks_csv(csv_path)
    if n_ticks is not None and n_ticks < len(df):
        idx = np.linspace(0, len(df) - 1, n_ticks).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
    return df


def generate_synthetic_nfp_ticks(
    n_ticks: int = 2000,
    seed: int = 42,
    pair: str = DEFAULT_PAIR,
    csv_path: str | Path = XAUUSD_NFP_TICKS_CSV,
) -> pd.DataFrame:
    """
    Load the real-market-anchored `XAU/USD` tick dataset from `data/xauusd_nfp_ticks_real.csv`
    when available, or fall back to calibrated `XAU/USD` tick simulation if the CSV is absent.
    """
    pair = assert_xauusd_pair(pair)
    p = Path(csv_path)
    if p.exists():
        return load_xauusd_nfp_ticks_from_csv(p, n_ticks=n_ticks, pair=pair)

    rng = np.random.default_rng(seed)
    ms_offsets = np.linspace(-2000, 10000, n_ticks).astype(int)
    pip_size = get_fx_pip_size(pair)

    base_mid = get_fx_default_spot_price(pair)
    fair_mid = np.full(n_ticks, base_mid, dtype=float)
    base_spread_bps = 1.15
    spread_bps = np.ones(n_ticks, dtype=float) * base_spread_bps
    ask_size_lots = np.full(n_ticks, 18.0, dtype=float)
    bid_size_lots = np.full(n_ticks, 18.0, dtype=float)

    for i, t_ms in enumerate(ms_offsets):
        if t_ms >= 0:
            jump = base_mid * 0.0052 * (1.0 - np.exp(-max(t_ms, 1) / 80.0))
            fair_mid[i] = base_mid + jump + rng.normal(0, base_mid * 0.00022)
            if t_ms <= 5000:
                widening_factor = 13.5 * np.exp(-max(0, t_ms - 500) / 4500.0)
                spread_bps[i] = max(1.8, widening_factor + rng.uniform(-0.3, 0.5))
                ask_size_lots[i] = max(0.8, 1.8 + rng.exponential(0.9))
                bid_size_lots[i] = max(0.8, 1.8 + rng.exponential(0.9))
            else:
                decay = np.exp(-(t_ms - 5000) / 2500.0)
                spread_bps[i] = base_spread_bps + 3.5 * decay + rng.uniform(0, 0.2)
                ask_size_lots[i] = 12.0 + 6.0 * (1.0 - decay)
                bid_size_lots[i] = 12.0 + 6.0 * (1.0 - decay)
        else:
            fair_mid[i] = base_mid + rng.normal(0, base_mid * 0.000028)
            spread_bps[i] = base_spread_bps + rng.uniform(-0.05, 0.10)

    half_spread_price = fair_mid * (spread_bps * 1e-4) / 2.0
    bid_px = fair_mid - half_spread_price
    ask_px = fair_mid + half_spread_price
    mid_px = (bid_px + ask_px) / 2.0
    spread_usd_oz = ask_px - bid_px
    spread_pips = spread_usd_oz / pip_size
    total_sz = bid_size_lots + ask_size_lots
    micro_px = (bid_size_lots * ask_px + ask_size_lots * bid_px) / total_sz
    obi = (bid_size_lots - ask_size_lots) / total_sz

    return pd.DataFrame(
        {
            "pair": pair,
            "ms_from_release": ms_offsets,
            "bid_px": bid_px,
            "ask_px": ask_px,
            "mid_px": mid_px,
            "micro_px": micro_px,
            "spread_bps": spread_bps,
            "spread_usd_oz": spread_usd_oz,
            "spread_pips": spread_pips,
            "bid_sz_lots": bid_size_lots,
            "ask_sz_lots": ask_size_lots,
            "obi": obi,
        }
    )


def compute_microstructure_metrics_duckdb(ticks_df: pd.DataFrame) -> Dict[str, float]:
    """
    Process millisecond ECN `XAU/USD` tick data using DuckDB/Polars (when installed) and `scipy.stats`
    to quantify the exact spread widening ratio (in bps, $/oz, and pips) and L1 liquidity drop during [0ms, +5000ms].
    """
    from scipy import stats

    has_pips = "spread_pips" in ticks_df.columns
    if not has_pips:
        ticks_df = ticks_df.copy()
        ticks_df["spread_pips"] = (ticks_df["ask_px"] - ticks_df["bid_px"]) / 0.01

    try:
        import duckdb

        rel = duckdb.sql(
            """
            SELECT
                AVG(CASE WHEN ms_from_release < 0 THEN spread_bps END) AS pre_spread_bps,
                AVG(CASE WHEN ms_from_release BETWEEN 0 AND 5000 THEN spread_bps END) AS shock_5s_spread_bps,
                MAX(CASE WHEN ms_from_release BETWEEN 0 AND 5000 THEN spread_bps END) AS max_shock_spread_bps,
                AVG(CASE WHEN ms_from_release < 0 THEN spread_pips END) AS pre_spread_pips,
                MAX(CASE WHEN ms_from_release BETWEEN 0 AND 5000 THEN spread_pips END) AS max_shock_spread_pips,
                AVG(CASE WHEN ms_from_release < 0 THEN ask_sz_lots END) AS pre_ask_depth,
                AVG(CASE WHEN ms_from_release BETWEEN 0 AND 5000 THEN ask_sz_lots END) AS shock_5s_ask_depth
            FROM ticks_df
            """
        ).fetchall()[0]
        (
            pre_spread,
            shock_spread,
            max_spread,
            pre_pips,
            max_pips,
            pre_depth,
            shock_depth,
        ) = [float(x) for x in rel]
    except ImportError:
        pre_mask = ticks_df["ms_from_release"] < 0
        shock_mask = (ticks_df["ms_from_release"] >= 0) & (ticks_df["ms_from_release"] <= 5000)
        pre_spread = float(ticks_df.loc[pre_mask, "spread_bps"].mean())
        shock_spread = float(ticks_df.loc[shock_mask, "spread_bps"].mean())
        max_spread = float(ticks_df.loc[shock_mask, "spread_bps"].max())
        pre_pips = float(ticks_df.loc[pre_mask, "spread_pips"].mean())
        max_pips = float(ticks_df.loc[shock_mask, "spread_pips"].max())
        pre_depth = float(ticks_df.loc[pre_mask, "ask_sz_lots"].mean())
        shock_depth = float(ticks_df.loc[shock_mask, "ask_sz_lots"].mean())

    pre_samples = ticks_df.loc[ticks_df["ms_from_release"] < 0, "spread_bps"].to_numpy()
    shock_samples = ticks_df.loc[
        (ticks_df["ms_from_release"] >= 0) & (ticks_df["ms_from_release"] <= 5000),
        "spread_bps",
    ].to_numpy()
    ks_stat, ks_pvalue = stats.ks_2samp(shock_samples, pre_samples)

    return {
        "pre_spread_bps": pre_spread,
        "shock_5s_spread_bps": shock_spread,
        "max_shock_spread_bps": max_spread,
        "pre_spread_pips": pre_pips,
        "max_shock_spread_pips": max_pips,
        "pre_spread_usd_oz": pre_pips * 0.01,
        "max_shock_spread_usd_oz": max_pips * 0.01,
        "spread_widening_multiple": max_spread / max(pre_spread, 1e-9),
        "depth_drop_pct": 100.0 * (1.0 - shock_depth / max(pre_depth, 1e-9)),
        "ks_stat_spread_shift": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
    }


def simulate_spread_aware_execution(
    ticks_df: pd.DataFrame,
    spec: OrderExecutionSpec,
    horizon_ms: int = 5000,
) -> Dict[str, float]:
    """
    Execute an aggressive ECN `XAU/USD` market order after a news release with strict spread & L2 sweep modeling.

    Raises `ExecutionIllusionError` if `spec.execution_price_mode == "mid"`.
    """
    assert_xauusd_pair(spec.pair)
    if spec.execution_price_mode == "mid":
        raise ExecutionIllusionError(
            "Execution Illusion Guardrail: Mid-price XAU/USD execution is forbidden during news releases. "
            "During the 5 seconds following NFP/CPI, ECN XAU/USD quoted spreads widen >12x and L1 gold liquidity "
            "evaporates. Use `execution_price_mode='spread_and_depth_sweep'`."
        )

    pip_size = get_fx_pip_size(spec.pair)
    arrival_ms = spec.signal_ms_from_release + spec.latency_ms
    eligible_entry = ticks_df.loc[ticks_df["ms_from_release"] >= arrival_ms]
    if eligible_entry.empty:
        raise ValueError(f"No tick found at or after arrival timestamp {arrival_ms}ms.")

    entry_tick = eligible_entry.iloc[0]
    exit_target_ms = int(entry_tick["ms_from_release"]) + horizon_ms
    eligible_exit = ticks_df.loc[ticks_df["ms_from_release"] >= exit_target_ms]
    exit_tick = eligible_exit.iloc[0] if not eligible_exit.empty else ticks_df.iloc[-1]

    if spec.side == "buy":
        avail_l1 = float(entry_tick["ask_sz_lots"])
        excess_lots = max(0.0, spec.order_size_lots - avail_l1)
        sweep_impact_px = float(entry_tick["mid_px"]) * (
            excess_lots * spec.l2_book_slope_bps_per_lot * 1e-4
        )
        realized_entry_px = float(entry_tick["ask_px"]) + sweep_impact_px
        realized_exit_px = float(exit_tick["bid_px"])
        illusion_pnl_bps = (
            (float(exit_tick["mid_px"]) - float(entry_tick["mid_px"]))
            / float(entry_tick["mid_px"])
            * 1e4
        )
        realized_pnl_bps = (realized_exit_px - realized_entry_px) / realized_entry_px * 1e4
        realized_pnl_usd_oz = realized_exit_px - realized_entry_px
        realized_pnl_pips = realized_pnl_usd_oz / pip_size
    else:
        avail_l1 = float(entry_tick["bid_sz_lots"])
        excess_lots = max(0.0, spec.order_size_lots - avail_l1)
        sweep_impact_px = float(entry_tick["mid_px"]) * (
            excess_lots * spec.l2_book_slope_bps_per_lot * 1e-4
        )
        realized_entry_px = float(entry_tick["bid_px"]) - sweep_impact_px
        realized_exit_px = float(exit_tick["ask_px"])
        illusion_pnl_bps = (
            (float(entry_tick["mid_px"]) - float(exit_tick["mid_px"]))
            / float(entry_tick["mid_px"])
            * 1e4
        )
        realized_pnl_bps = (realized_entry_px - realized_exit_px) / realized_entry_px * 1e4
        realized_pnl_usd_oz = realized_entry_px - realized_exit_px
        realized_pnl_pips = realized_pnl_usd_oz / pip_size

    entry_spread_usd_oz = float(entry_tick["ask_px"]) - float(entry_tick["bid_px"])
    entry_spread_pips = entry_spread_usd_oz / pip_size

    return {
        "arrival_ms": float(entry_tick["ms_from_release"]),
        "entry_spread_bps": float(entry_tick["spread_bps"]),
        "entry_spread_usd_oz": float(entry_spread_usd_oz),
        "entry_spread_pips": float(entry_spread_pips),
        "l1_available_lots": avail_l1,
        "l2_excess_swept_lots": excess_lots,
        "mid_price_illusion_pnl_bps": float(illusion_pnl_bps),
        "realized_spread_swept_pnl_bps": float(realized_pnl_bps),
        "realized_spread_swept_pnl_usd_oz": float(realized_pnl_usd_oz),
        "realized_spread_swept_pnl_pips": float(realized_pnl_pips),
        "execution_cost_drag_bps": float(illusion_pnl_bps - realized_pnl_bps),
    }


def _run_demo() -> None:
    """Self-contained demonstration of Topic 2 loading real XAU/USD tick CSV (`data/xauusd_nfp_ticks_real.csv`)."""
    print("=== Topic 2: XAU/USD News Microstructure (Real XAU/USD Tick CSV `data/xauusd_nfp_ticks_real.csv`) ===")
    ticks = load_xauusd_nfp_ticks_from_csv()
    metrics = compute_microstructure_metrics_duckdb(ticks)
    print(
        f"Loaded {len(ticks):,d} XAU/USD ticks from `{XAUUSD_NFP_TICKS_CSV.name}` (Spot ~${ticks['mid_px'].iloc[0]:,.2f}/oz):\n"
        f"  Pre-NFP XAU/USD Spread: {metrics['pre_spread_bps']:.2f} bps (${metrics['pre_spread_usd_oz']:.2f}/oz) | "
        f"Max 5s Post-NFP Spread: {metrics['max_shock_spread_bps']:.2f} bps (${metrics['max_shock_spread_usd_oz']:.2f}/oz, "
        f"{metrics['spread_widening_multiple']:.1f}x widening) | "
        f"ECN L1 Gold Depth Drop: {metrics['depth_drop_pct']:.1f}%"
    )

    # 1. Verify mid-price execution guardrail raises ExecutionIllusionError
    try:
        simulate_spread_aware_execution(
            ticks,
            OrderExecutionSpec(side="buy", signal_ms_from_release=50, execution_price_mode="mid", pair="XAU/USD"),
        )
    except ExecutionIllusionError as exc:
        print(f"[GUARDRAIL VERIFIED] {exc}")

    # 2. Compare illusory mid-price PnL vs. realistic ECN spread+sweep PnL at t=+50ms post-NFP
    sim = simulate_spread_aware_execution(
        ticks,
        OrderExecutionSpec(
            side="buy",
            signal_ms_from_release=50,
            latency_ms=20,
            order_size_lots=10.0,
            execution_price_mode="spread_and_depth_sweep",
            pair="XAU/USD",
        ),
    )
    print(
        f"XAU/USD Order Arrival at +{sim['arrival_ms']:.0f}ms (Spread={sim['entry_spread_bps']:.2f} bps / "
        f"${sim['entry_spread_usd_oz']:.2f}/oz, L1 Avail={sim['l1_available_lots']:.1f} lots):"
    )
    print(
        f"  Illusory Mid-to-Mid XAU/USD PnL: {sim['mid_price_illusion_pnl_bps']:+.2f} bps\n"
        f"  Realized ECN Ask-Sweep to Bid XAU/USD PnL: {sim['realized_spread_swept_pnl_bps']:+.2f} bps "
        f"({sim['realized_spread_swept_pnl_usd_oz']:+.2f} $/oz | Execution Cost Drag: {sim['execution_cost_drag_bps']:.2f} bps)"
    )


if __name__ == "__main__":
    _run_demo()
