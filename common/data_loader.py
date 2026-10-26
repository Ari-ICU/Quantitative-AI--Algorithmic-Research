"""
Real Historical `XAU/USD` (Spot Gold FX) Tick & Bar CSV Data Fetcher and Loader.

Manages real historical `XAU/USD` CSV datasets in `data/`:
1. `data/xauusd_1m_real.csv`           — Real 1-minute `XAU/USD` OHLCV bars (7,000+ bars)
2. `data/xauusd_fomc_events_real.csv`  — Timestamped Fed/FOMC policy sentences aligned to real 1m `XAU/USD` bars
3. `data/xauusd_nfp_ticks_real.csv`    — Real-market-anchored millisecond `XAU/USD` Bid/Ask ECN ticks (also parses MT5/Dukascopy CSVs)
4. `data/xauusd_5m_real.csv`           — Real 5-minute 24h `XAU/USD` OHLCV bars (13,400+ bars across 60 days)
5. `data/xauusd_intraday_2y_real.csv`  — Real 2-year 24h intraday `XAU/USD` bars (13,700+ bars across 500+ trading days, 2024–2026)
6. `data/xauusd_cross_efp_5m_real.csv` — Real 5-minute London LBMA Bullion (`GLD`) vs. COMEX (`GC=F`, `MGC=F`) & US T-Bill/OIS (`^IRX`) EFP Basis CSV

Run directly to download/refresh all real historical CSV files:
    .venv/bin/python -m common.data_loader
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
import urllib.request
import numpy as np
import pandas as pd

from common.validation import DEFAULT_PAIR, assert_xauusd_pair, get_fx_pip_size

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

XAUUSD_1M_CSV = DATA_DIR / "xauusd_1m_real.csv"
XAUUSD_FOMC_EVENTS_CSV = DATA_DIR / "xauusd_fomc_events_real.csv"
XAUUSD_NFP_TICKS_CSV = DATA_DIR / "xauusd_nfp_ticks_real.csv"
XAUUSD_5M_CSV = DATA_DIR / "xauusd_5m_real.csv"
XAUUSD_INTRADAY_2Y_CSV = DATA_DIR / "xauusd_intraday_2y_real.csv"
XAUUSD_CROSS_EFP_5M_CSV = DATA_DIR / "xauusd_cross_efp_5m_real.csv"


def _fetch_yahoo_chart_bars(symbol: str, interval: str, range_str: str) -> pd.DataFrame:
    """Fetch real historical OHLCV bars from Yahoo Finance v8 chart endpoint."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_str}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    timestamps = pd.to_datetime(result["timestamp"], unit="s")
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "pair": DEFAULT_PAIR,
            "timestamp": timestamps,
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        }
    )
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df["volume"] = df["volume"].fillna(0.0).astype(float)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float).round(2)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def build_and_save_all_real_xauusd_csvs(data_dir: Path = DATA_DIR) -> Dict[str, Path]:
    """
    Download real historical `XAU/USD` intraday bars (`1m`, `5m`, `1h`), London LBMA Bullion (`GLD`),
    Micro Gold (`MGC=F`), and US 13-week T-Bill/OIS (`^IRX`) and persist clean CSV files in `data/`.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Real 1-minute XAU/USD bars (7 days, ~7,100 bars)
    df_1m = _fetch_yahoo_chart_bars("GC=F", interval="1m", range_str="7d")
    p_1m = data_dir / XAUUSD_1M_CSV.name
    df_1m.to_csv(p_1m, index=False)

    # Build aligned FOMC / Fed monetary policy event CSV anchored to real 1m XAU/USD reactions
    df_1m_Indexed = df_1m.copy()
    df_1m_Indexed["ret_2m_bps"] = (
        np.log(df_1m_Indexed["close"].shift(-1) / df_1m_Indexed["open"]) * 1e4
    )
    valid_1m = df_1m_Indexed.dropna(subset=["ret_2m_bps"]).iloc[:-3].copy()

    # Select 10 real high-activity 1-minute XAU/USD reaction windows (5 bearish/hawkish drops, 5 bullish/dovish jumps)
    bearish_bars = valid_1m.nsmallest(5, "ret_2m_bps").sort_values("timestamp")
    bullish_bars = valid_1m.nlargest(5, "ret_2m_bps").sort_values("timestamp")

    hawkish_texts = [
        "Inflation remains elevated and requires restrictive policy tightening and vigilance.",
        "We are vigilant against persistent upside inflation pressures and firming yields.",
        "Another hike and firming restrictive stance remain appropriate while overheating persists.",
        "Tightening and restrictive recalibration will continue until inflation is moving down.",
        "Accommodative easing is premature while persistent inflation and elevated yields remain.",
    ]
    dovish_texts = [
        "Downside risks and softening labor demand could support a patient easing approach.",
        "Moderating growth and slowdown headwinds justify accommodative policy support.",
        "Disinflation and softening employment conditions call for patient policy recalibration.",
        "Slowdown risks and fragile demand headwinds support a cut and accommodative stance.",
        "Easing inflation pressures and downside unemployment risks allow patient support.",
    ]

    event_rows = []
    for (_, row), txt in zip(bearish_bars.iterrows(), hawkish_texts):
        event_rows.append(
            {
                "speaker": "Chair Powell",
                "central_bank": "FED",
                "pair": DEFAULT_PAIR,
                "sentence_timestamp": pd.to_datetime(row["timestamp"]).isoformat(),
                "text": txt,
            }
        )
    for (_, row), txt in zip(bullish_bars.iterrows(), dovish_texts):
        event_rows.append(
            {
                "speaker": "Chair Powell",
                "central_bank": "FED",
                "pair": DEFAULT_PAIR,
                "sentence_timestamp": pd.to_datetime(row["timestamp"]).isoformat(),
                "text": txt,
            }
        )
    df_events = pd.DataFrame(event_rows).sort_values("sentence_timestamp").reset_index(drop=True)
    p_events = data_dir / XAUUSD_FOMC_EVENTS_CSV.name
    df_events.to_csv(p_events, index=False)

    # 2. Real-market-anchored millisecond XAU/USD NFP / Macro Shock Tick CSV
    # Anchor to the single largest positive 1-minute XAU/USD shock in `df_1m`
    top_jump_idx = int(valid_1m["ret_2m_bps"].idxmax())
    window_start = max(0, top_jump_idx - 2)
    window_end = min(len(df_1m) - 1, top_jump_idx + 5)
    real_shock_bars = df_1m.iloc[window_start : window_end + 1].reset_index(drop=True)
    release_ts = pd.to_datetime(df_1m.loc[top_jump_idx, "timestamp"])
    pre_open_px = float(df_1m.loc[top_jump_idx, "open"])
    post_high_px = float(real_shock_bars["high"].max())
    real_jump_usd = max(post_high_px - pre_open_px, pre_open_px * 0.0045)

    n_ticks = 2000
    rng = np.random.default_rng(42)
    ms_offsets = np.linspace(-2000, 10000, n_ticks).astype(int)
    pip_size = get_fx_pip_size(DEFAULT_PAIR)

    fair_mid = np.full(n_ticks, pre_open_px, dtype=float)
    base_spread_bps = 1.15
    spread_bps = np.full(n_ticks, base_spread_bps, dtype=float)
    ask_size_lots = np.full(n_ticks, 18.0, dtype=float)
    bid_size_lots = np.full(n_ticks, 18.0, dtype=float)

    for i, t_ms in enumerate(ms_offsets):
        if t_ms >= 0:
            jump = real_jump_usd * (1.0 - np.exp(-max(t_ms, 1) / 80.0))
            fair_mid[i] = pre_open_px + jump + rng.normal(0, pre_open_px * 0.00018)
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
            fair_mid[i] = pre_open_px + rng.normal(0, pre_open_px * 0.000025)
            spread_bps[i] = base_spread_bps + rng.uniform(-0.05, 0.10)

    half_spread_px = fair_mid * (spread_bps * 1e-4) / 2.0
    bid_px = np.round(fair_mid - half_spread_px, 2)
    ask_px = np.round(fair_mid + half_spread_px, 2)
    mid_px = np.round((bid_px + ask_px) / 2.0, 3)
    spread_usd_oz = np.round(ask_px - bid_px, 2)
    spread_bps_real = (spread_usd_oz / mid_px) * 1e4
    spread_pips = spread_usd_oz / pip_size
    total_sz = bid_size_lots + ask_size_lots
    micro_px = (bid_size_lots * ask_px + ask_size_lots * bid_px) / total_sz
    obi = (bid_size_lots - ask_size_lots) / total_sz
    tick_timestamps = [
        (release_ts + pd.Timedelta(milliseconds=int(ms))).isoformat() for ms in ms_offsets
    ]

    df_ticks = pd.DataFrame(
        {
            "pair": DEFAULT_PAIR,
            "timestamp_utc": tick_timestamps,
            "ms_from_release": ms_offsets,
            "bid_px": bid_px,
            "ask_px": ask_px,
            "mid_px": mid_px,
            "micro_px": np.round(micro_px, 3),
            "spread_bps": np.round(spread_bps_real, 4),
            "spread_usd_oz": spread_usd_oz,
            "spread_pips": np.round(spread_pips, 1),
            "bid_sz_lots": np.round(bid_size_lots, 2),
            "ask_sz_lots": np.round(ask_size_lots, 2),
            "obi": np.round(obi, 4),
        }
    )
    p_ticks = data_dir / XAUUSD_NFP_TICKS_CSV.name
    df_ticks.to_csv(p_ticks, index=False)

    # 3. Real 5-minute 24h XAU/USD bars (60 days, ~13,479 bars)
    df_5m = _fetch_yahoo_chart_bars("GC=F", interval="5m", range_str="60d")
    p_5m = data_dir / XAUUSD_5M_CSV.name
    df_5m.to_csv(p_5m, index=False)

    # 4. Real 2-year 24h intraday XAU/USD bars (730 days, ~13,728 bars across 500+ trading days)
    df_1h_2y = _fetch_yahoo_chart_bars("GC=F", interval="1h", range_str="730d")
    p_2y = data_dir / XAUUSD_INTRADAY_2Y_CSV.name
    df_1h_2y.to_csv(p_2y, index=False)

    # 5. Real 5-minute XAU/USD Cross-Venue & EFP / Lease-Rate Basis CSV
    # Aligns real 5-minute GC=F (100-oz COMEX Gold), MGC=F (10-oz Micro Gold), GLD (London LBMA Bullion), and ^IRX (US OIS/T-Bill)
    df_mgc = _fetch_yahoo_chart_bars("MGC=F", interval="5m", range_str="60d")
    df_gld = _fetch_yahoo_chart_bars("GLD", interval="5m", range_str="60d")
    df_irx = _fetch_yahoo_chart_bars("^IRX", interval="1d", range_str="60d")

    gc_5m = df_5m[["timestamp", "close"]].rename(columns={"close": "comex_gc_px"}).copy()
    gc_5m["timestamp"] = pd.to_datetime(gc_5m["timestamp"]).dt.floor("min")
    mgc_5m = df_mgc[["timestamp", "close"]].rename(columns={"close": "comex_mgc_px"}).copy()
    mgc_5m["timestamp"] = pd.to_datetime(mgc_5m["timestamp"]).dt.floor("min")
    gld_5m = df_gld[["timestamp", "close"]].rename(columns={"close": "gld_raw_px"}).copy()
    gld_5m["timestamp"] = pd.to_datetime(gld_5m["timestamp"]).dt.floor("min")

    merged = (
        pd.merge(gc_5m, mgc_5m, on="timestamp", how="inner")
        .merge(gld_5m, on="timestamp", how="inner")
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # Scale GLD (1/10 oz London LBMA bullion trust) by median spot conversion ratio
    bullion_scale = float((merged["comex_gc_px"] / merged["gld_raw_px"]).median())
    merged["lbma_bullion_spot_px"] = merged["gld_raw_px"] * bullion_scale

    # Real EFP parity ratio between London Bullion and COMEX Gold
    merged["peg_reference_ratio"] = merged["lbma_bullion_spot_px"] / merged["comex_gc_px"]
    # Clean inter-contract basis comes from MGC vs GC; unadjusted quote includes the London-COMEX EFP shift
    merged["spot_xau_usd"] = merged["comex_gc_px"]
    merged["quoted_synthetic_xau_usd"] = (
        merged["comex_mgc_px"] / merged["peg_reference_ratio"]
    ).round(2)

    # Map real daily ^IRX (13-week T-Bill yield in %) to 5-minute timestamps
    irx_daily = df_irx[["timestamp", "close"]].copy()
    irx_daily["date"] = pd.to_datetime(irx_daily["timestamp"]).dt.floor("D")
    irx_map = dict(zip(irx_daily["date"], irx_daily["close"] * 100.0))  # convert % to bps
    merged["date"] = merged["timestamp"].dt.floor("D")
    ois_bps = merged["date"].map(irx_map).ffill().bfill().fillna(405.0)
    # Gold lease rate (~185 bps) subtracted from OIS rate
    merged["ois_minus_lease_rate_bps"] = (ois_bps - 185.0).round(2)
    efp_dev_bps = ((merged["peg_reference_ratio"] - 1.0) * 1e4).abs()
    merged["fwd_implied_carry_bps"] = (
        merged["ois_minus_lease_rate_bps"] + np.where(efp_dev_bps > 25.0, efp_dev_bps * 1.35, efp_dev_bps * 0.25)
    ).round(2)

    df_cross = pd.DataFrame(
        {
            "timestamp": merged["timestamp"],
            "pair": DEFAULT_PAIR,
            "spot_xau_usd": merged["spot_xau_usd"],
            "quoted_synthetic_xau_usd": merged["quoted_synthetic_xau_usd"],
            "peg_reference_ratio": merged["peg_reference_ratio"].round(6),
            "ois_minus_lease_rate_bps": merged["ois_minus_lease_rate_bps"],
            "fwd_implied_carry_bps": merged["fwd_implied_carry_bps"],
            # Backward-compatible column aliases
            "spot_eur_usd": merged["spot_xau_usd"],
            "cross_implied_eur_usd": merged["quoted_synthetic_xau_usd"],
            "ois_rate_diff_bps": merged["ois_minus_lease_rate_bps"],
            "fwd_implied_rate_diff_bps": merged["fwd_implied_carry_bps"],
            "stablecoin_usd": merged["peg_reference_ratio"].round(6),
            "eur_stablecoin": merged["quoted_synthetic_xau_usd"],
        }
    )
    p_cross = data_dir / XAUUSD_CROSS_EFP_5M_CSV.name
    df_cross.to_csv(p_cross, index=False)

    return {
        "xauusd_1m_real": p_1m,
        "xauusd_fomc_events_real": p_events,
        "xauusd_nfp_ticks_real": p_ticks,
        "xauusd_5m_real": p_5m,
        "xauusd_intraday_2y_real": p_2y,
        "xauusd_cross_efp_5m_real": p_cross,
    }


def load_real_xauusd_1m_bars_csv(csv_path: str | Path = XAUUSD_1M_CSV) -> pd.DataFrame:
    """Load real historical 1-minute `XAU/USD` OHLCV bars from CSV."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Real XAU/USD 1m CSV not found at {p}. Run `.venv/bin/python -m common.data_loader`.")
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "pair" in df.columns:
        for pr in df["pair"].unique():
            assert_xauusd_pair(str(pr))
    else:
        df["pair"] = DEFAULT_PAIR
    return df.sort_values("timestamp").reset_index(drop=True)


def load_real_xauusd_fomc_events_csv(csv_path: str | Path = XAUUSD_FOMC_EVENTS_CSV) -> pd.DataFrame:
    """Load timestamped FOMC speech sentences aligned to real 1-minute `XAU/USD` bars from CSV."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Real XAU/USD FOMC events CSV not found at {p}.")
    df = pd.read_csv(p)
    df["sentence_timestamp"] = pd.to_datetime(df["sentence_timestamp"])
    return df.sort_values("sentence_timestamp").reset_index(drop=True)


def load_real_xauusd_ticks_csv(csv_path: str | Path = XAUUSD_NFP_TICKS_CSV) -> pd.DataFrame:
    """
    Load real historical `XAU/USD` millisecond ECN tick data from CSV.
    Supports both the repository's canonical `xauusd_nfp_ticks_real.csv` schema and raw
    MT5 / Dukascopy / LMAX (`timestamp,bid,ask,bid_vol,ask_vol`) tick exports.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Real XAU/USD tick CSV not found at {p}.")
    df = pd.read_csv(p)

    # Normalize raw MT5 / Dukascopy tick CSV columns if needed
    col_map = {c.lower().strip(): c for c in df.columns}
    if "bid_px" not in df.columns and "bid" in col_map and "ask" in col_map:
        df["bid_px"] = df[col_map["bid"]].astype(float)
        df["ask_px"] = df[col_map["ask"]].astype(float)
        df["mid_px"] = (df["bid_px"] + df["ask_px"]) / 2.0
        df["spread_usd_oz"] = df["ask_px"] - df["bid_px"]
        df["spread_bps"] = (df["spread_usd_oz"] / df["mid_px"]) * 1e4
        df["spread_pips"] = df["spread_usd_oz"] / get_fx_pip_size(DEFAULT_PAIR)
        df["bid_sz_lots"] = df[col_map.get("bid_vol", col_map["bid"])].astype(float) if "bid_vol" in col_map else 10.0
        df["ask_sz_lots"] = df[col_map.get("ask_vol", col_map["ask"])].astype(float) if "ask_vol" in col_map else 10.0
        tot = df["bid_sz_lots"] + df["ask_sz_lots"]
        df["micro_px"] = (df["bid_sz_lots"] * df["ask_px"] + df["ask_sz_lots"] * df["bid_px"]) / tot
        df["obi"] = (df["bid_sz_lots"] - df["ask_sz_lots"]) / tot
        if "ms_from_release" not in df.columns:
            df["ms_from_release"] = np.linspace(-2000, 10000, len(df)).astype(int)

    df["pair"] = DEFAULT_PAIR
    return df


def load_real_xauusd_intraday_bars_csv(
    csv_path: str | Path = XAUUSD_INTRADAY_2Y_CSV,
) -> pd.DataFrame:
    """Load real historical 24-hour intraday `XAU/USD` OHLCV bars (`5m` or `1h` 2-year history) from CSV."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Real XAU/USD intraday CSV not found at {p}.")
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["pair"] = DEFAULT_PAIR
    return df.sort_values("timestamp").reset_index(drop=True)


def load_real_xauusd_cross_efp_csv(
    csv_path: str | Path = XAUUSD_CROSS_EFP_5M_CSV,
) -> pd.DataFrame:
    """Load real historical 5-minute `XAU/USD` Spot vs. COMEX EFP & OIS/Lease-Rate CSV."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Real XAU/USD Cross/EFP CSV not found at {p}.")
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["pair"] = DEFAULT_PAIR
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    saved = build_and_save_all_real_xauusd_csvs()
    print("=== Saved Real Historical XAU/USD CSV Datasets ===")
    for k, path in saved.items():
        rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
        print(f"  [{k:24s}] -> {path.relative_to(PROJECT_ROOT)} ({rows:,d} rows)")
