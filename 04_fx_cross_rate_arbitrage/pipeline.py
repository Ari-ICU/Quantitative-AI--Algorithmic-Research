"""
Topic 4: XAU/USD Cross-Venue & EFP / Gold Lease-Rate (CIP) Basis Arbitrage —
Real Historical 5-Minute London LBMA Bullion (`GLD`) vs. COMEX (`GC=F`, `MGC=F`) & US T-Bill/OIS (`^IRX`)
with Physical Bullion EFP & Lease-Rate Squeeze Circuit Breaker.

Primary Libraries: vectorbt (backtesting), statsmodels, scipy
Granularity: Real Historical 5-minute XAU/USD Spot & EFP Basis CSV (`data/xauusd_cross_efp_5m_real.csv`)
Pitfall Prevented: Physical EFP & Gold Lease-Rate Squeeze Risk — Mistaking a London-to-COMEX
Exchange-for-Physical (EFP) delivery/roll squeeze or Gold Lease Rate spike for a stationary
OTC XAU/USD microstructure dislocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

from common.data_loader import XAUUSD_CROSS_EFP_5M_CSV, load_real_xauusd_cross_efp_csv
from common.validation import (
    DEFAULT_PAIR,
    DepegRiskError,
    FXRegimeOrPegBreakError,
    get_fx_default_spot_price,
)


@dataclass(frozen=True)
class FXPegAndCIPRegimeMonitor:
    """
    Monitors two structural `XAU/USD` tail-risk regimes that invalidate statistical
    spot-vs-synthetic / EFP relative-value cointegration strategies:
    1. Physical Bullion EFP / Contract-Roll Regime Shock (`peg_reference_ratio` deviation exceeding
       `max_peg_deviation_bps`).
    2. Gold Lease-Rate / USD Covered Interest Parity (CIP) Swap Squeeze (`abs(cip_funding_basis_bps)`
       exceeding `max_cip_basis_bps`).
    """

    max_peg_deviation_bps: float = 25.0
    max_cip_basis_bps: float = 40.0

    def evaluate_fx_regime_health(
        self,
        peg_reference_ratio: pd.Series,
        cip_funding_basis_bps: pd.Series | None = None,
    ) -> pd.DataFrame:
        peg_dev_bps = ((peg_reference_ratio.astype(float) - 1.0) * 1e4).abs()
        if cip_funding_basis_bps is not None:
            cip_abs_bps = cip_funding_basis_bps.astype(float).abs()
        else:
            cip_abs_bps = pd.Series(0.0, index=peg_reference_ratio.index, dtype=float)

        is_breach = (peg_dev_bps > self.max_peg_deviation_bps) | (
            cip_abs_bps > self.max_cip_basis_bps
        )
        return pd.DataFrame(
            {
                "peg_reference_ratio": peg_reference_ratio,
                "peg_deviation_bps": peg_dev_bps,
                "cip_basis_abs_bps": cip_abs_bps,
                "is_regime_breach": is_breach,
                "is_depegged": is_breach,
            },
            index=peg_reference_ratio.index,
        )

    def evaluate_peg_health(self, peg_reference_ratio: pd.Series) -> pd.DataFrame:
        return self.evaluate_fx_regime_health(peg_reference_ratio)


StablecoinPegMonitor = FXPegAndCIPRegimeMonitor


def decompose_fx_cross_and_cip_basis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose the real historical `XAU/USD` Spot vs. COMEX EFP-Implied `XAU/USD` surface into:
    1. Direct Spot / Benchmark Gold (`spot_xau_usd`, $/troy oz).
    2. Regime-Adjusted Implied `XAU/USD` (`implied_xau_usd`):
       Strips out physical EFP delivery / contract-roll shocks (`peg_reference_ratio`).
    3. Pure Inter-Contract `XAU/USD` Basis (`true_cross_basis_bps`):
       Stationary microstructure dislocation (`MGC=F` vs. `GC=F`) suitable for mean-reversion.
    4. Naive Unadjusted `XAU/USD` Basis (`naive_basis_bps`) & EFP/Lease-Rate Distortion (`regime_distortion_bps`).
    """
    out = df.copy()

    spot_col = "spot_xau_usd" if "spot_xau_usd" in out.columns else "spot_eur_usd"
    synth_col = "quoted_synthetic_xau_usd" if "quoted_synthetic_xau_usd" in out.columns else (
        "cross_implied_eur_usd" if "cross_implied_eur_usd" in out.columns else "eur_stablecoin"
    )
    peg_col = "peg_reference_ratio" if "peg_reference_ratio" in out.columns else "stablecoin_usd"

    out["spot_xau_usd"] = out[spot_col]
    out["quoted_synthetic_xau_usd"] = out[synth_col]
    out["peg_reference_ratio"] = out[peg_col]

    out["implied_xau_usd"] = out["quoted_synthetic_xau_usd"] * out["peg_reference_ratio"]
    out["naive_basis_bps"] = (out["quoted_synthetic_xau_usd"] / out["spot_xau_usd"] - 1.0) * 1e4
    out["true_cross_basis_bps"] = (out["implied_xau_usd"] / out["spot_xau_usd"] - 1.0) * 1e4

    if "ois_minus_lease_rate_bps" in out.columns and "fwd_implied_carry_bps" in out.columns:
        out["cip_basis_bps"] = out["ois_minus_lease_rate_bps"] - out["fwd_implied_carry_bps"]
    elif "ois_rate_diff_bps" in out.columns and "fwd_implied_rate_diff_bps" in out.columns:
        out["cip_basis_bps"] = out["ois_rate_diff_bps"] - out["fwd_implied_rate_diff_bps"]
    elif "cip_basis_bps" not in out.columns:
        out["cip_basis_bps"] = 0.0

    out["spot_eur_usd"] = out["spot_xau_usd"]
    out["implied_eur_usd"] = out["implied_xau_usd"]
    out["true_venue_basis_bps"] = out["true_cross_basis_bps"]
    out["regime_distortion_bps"] = out["naive_basis_bps"] - out["true_cross_basis_bps"]
    out["depeg_distortion_bps"] = out["regime_distortion_bps"]
    return out


decompose_crypto_fx_basis = decompose_fx_cross_and_cip_basis


def generate_fx_cross_rate_arbitrage_signals(
    df: pd.DataFrame,
    regime_monitor: FXPegAndCIPRegimeMonitor,
    z_entry: float = 2.0,
    lookback_bars: int = 60,
    ignore_regime_guardrail: bool = False,
    ignore_depeg_guardrail: bool = False,
) -> Dict[str, object]:
    """
    Estimate rolling cointegration / z-score spread between Direct `XAU/USD`
    and Synthetic / EFP-Implied `XAU/USD`, gated by `FXPegAndCIPRegimeMonitor`.
    """
    bypass_guard = bool(ignore_regime_guardrail or ignore_depeg_guardrail)
    decomposed = decompose_fx_cross_and_cip_basis(df)
    regime_status = regime_monitor.evaluate_fx_regime_health(
        decomposed["peg_reference_ratio"],
        decomposed["cip_basis_bps"],
    )
    decomposed["peg_deviation_bps"] = regime_status["peg_deviation_bps"]
    decomposed["cip_basis_abs_bps"] = regime_status["cip_basis_abs_bps"]
    decomposed["is_regime_breach"] = regime_status["is_regime_breach"]
    decomposed["is_depegged"] = regime_status["is_depegged"]

    if bypass_guard and bool(decomposed["is_regime_breach"].any()):
        max_dev = float(decomposed["peg_deviation_bps"].max())
        max_cip = float(decomposed["cip_basis_abs_bps"].max())
        min_ratio = float(decomposed["peg_reference_ratio"].min())
        raise FXRegimeOrPegBreakError(
            f"XAU/USD EFP & Gold Lease-Rate (CIP) Guardrail: Structural bullion regime breach detected "
            f"(min EFP parity ratio={min_ratio:.4f}, EFP shock={max_dev:.1f} bps > "
            f"{regime_monitor.max_peg_deviation_bps:.1f} bps limit, max Lease/CIP basis={max_cip:.1f} bps). "
            "Trading naive XAU/USD basis mean-reversion during a physical London-to-COMEX EFP squeeze "
            "or Gold Lease Rate spike confuses a physical delivery dislocation for stationary ECN noise."
        )

    clean_mask = ~decomposed["is_regime_breach"]
    x_spot = decomposed.loc[clean_mask, "spot_xau_usd"].to_numpy(dtype=float)
    y_impl = decomposed.loc[clean_mask, "implied_xau_usd"].to_numpy(dtype=float)
    clean_basis = decomposed.loc[clean_mask, "true_cross_basis_bps"].to_numpy(dtype=float)

    try:
        import statsmodels.api as sm
        from statsmodels.tsa.stattools import adfuller

        X = sm.add_constant(x_spot)
        ols_res = sm.OLS(y_impl, X).fit()
        hedge_beta = float(ols_res.params[1])
        try:
            adf_res = adfuller(clean_basis, result_object=False)
        except TypeError:
            adf_res = adfuller(clean_basis)
        adf_stat, adf_pvalue = float(adf_res[0]), float(adf_res[1])
    except ImportError:
        from scipy import stats

        hedge_beta = float(np.polyfit(x_spot, y_impl, 1)[0])
        dy = np.diff(clean_basis)
        y_lag = clean_basis[:-1]
        slope, _, _, p_val, std_err = stats.linregress(y_lag, dy)
        adf_stat = float(slope / max(std_err, 1e-12))
        adf_pvalue = float(p_val)

    rolling_mean = decomposed["true_cross_basis_bps"].rolling(lookback_bars, min_periods=20).mean()
    rolling_std = (
        decomposed["true_cross_basis_bps"]
        .rolling(lookback_bars, min_periods=20)
        .std()
        .replace(0.0, np.nan)
    )
    decomposed["z_score"] = (
        (decomposed["true_cross_basis_bps"] - rolling_mean) / rolling_std
    ).fillna(0.0)

    raw_pos = np.where(
        decomposed["z_score"] > z_entry,
        -1.0,
        np.where(decomposed["z_score"] < -z_entry, 1.0, 0.0),
    )
    decomposed["position"] = np.where(decomposed["is_regime_breach"], 0.0, raw_pos)

    basis_change_bps = decomposed["true_cross_basis_bps"].diff().fillna(0.0)
    turnover = pd.Series(decomposed["position"], index=decomposed.index).diff().abs().fillna(0.0)
    decomposed["strategy_net_pnl_bps"] = (
        pd.Series(decomposed["position"], index=decomposed.index).shift(1).fillna(0.0) * basis_change_bps
        - 0.45 * turnover
    )

    return {
        "frame": decomposed,
        "hedge_ratio_beta": hedge_beta,
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "regime_bars_halted": int(decomposed["is_regime_breach"].sum()),
        "depeg_bars_halted": int(decomposed["is_regime_breach"].sum()),
        "max_regime_distortion_bps": float(decomposed["regime_distortion_bps"].abs().max()),
        "max_depeg_distortion_bps": float(decomposed["regime_distortion_bps"].abs().max()),
        "cumulative_net_pnl_bps": float(decomposed["strategy_net_pnl_bps"].sum()),
    }


generate_fx_cross_arbitrage_signals = generate_fx_cross_rate_arbitrage_signals
generate_cross_venue_arbitrage_signals = generate_fx_cross_rate_arbitrage_signals


def load_xauusd_cross_efp_bars_from_csv(
    csv_path: str | Path = XAUUSD_CROSS_EFP_5M_CSV,
    n_bars: int | None = None,
) -> pd.DataFrame:
    """
    Load real historical 5-minute `XAU/USD` Spot vs. COMEX EFP & OIS/Lease-Rate bars
    from `data/xauusd_cross_efp_5m_real.csv`.
    """
    df = load_real_xauusd_cross_efp_csv(csv_path)
    if n_bars is not None and n_bars < len(df):
        # Slice around index 1550 where the real September FOMC & quarterly EFP transition occurs
        start = max(0, min(len(df) - n_bars, 1550 - n_bars // 2))
        df = df.iloc[start : start + n_bars].reset_index(drop=True)
    return df


def _generate_synthetic_fx_cross_cip_bars(
    n_bars: int = 400,
    seed: int = 42,
    csv_path: str | Path = XAUUSD_CROSS_EFP_5M_CSV,
) -> pd.DataFrame:
    """
    Load the real historical 5-minute `XAU/USD` Cross/EFP dataset from `data/xauusd_cross_efp_5m_real.csv`
    when available, or fall back to calibrated `XAU/USD` simulation if absent.
    """
    p = Path(csv_path)
    if p.exists():
        return load_xauusd_cross_efp_bars_from_csv(p, n_bars=n_bars)

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2023-03-10 00:00:00", periods=n_bars, freq="5min")
    base_xau = get_fx_default_spot_price(DEFAULT_PAIR)
    spot_xau_usd = base_xau + np.cumsum(rng.normal(0.0, 0.55, size=n_bars))
    peg_reference_ratio = np.ones(n_bars, dtype=float) + rng.normal(0.0, 0.0002, size=n_bars)
    ois_minus_lease_rate_bps = np.full(n_bars, 210.0, dtype=float) + rng.normal(0.0, 1.0, size=n_bars)
    cip_basis_shock_bps = rng.normal(-9.0, 2.8, size=n_bars)

    for i in range(220, 270):
        depth = np.sin(np.pi * (i - 220) / 50.0)
        peg_reference_ratio[i] = 1.0 - 0.048 * depth + rng.normal(0.0, 0.0005)
        cip_basis_shock_bps[i] = -9.0 - 85.0 * depth + rng.normal(0.0, 2.2)

    fwd_implied_carry_bps = ois_minus_lease_rate_bps - cip_basis_shock_bps
    ecn_xau_basis = np.zeros(n_bars, dtype=float)
    for i in range(1, n_bars):
        ecn_xau_basis[i] = 0.74 * ecn_xau_basis[i - 1] + rng.normal(0.0, 0.00042)
    quoted_synthetic_xau_usd = (spot_xau_usd * (1.0 + ecn_xau_basis)) / peg_reference_ratio

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "pair": DEFAULT_PAIR,
            "spot_xau_usd": spot_xau_usd,
            "quoted_synthetic_xau_usd": quoted_synthetic_xau_usd,
            "peg_reference_ratio": peg_reference_ratio,
            "ois_minus_lease_rate_bps": ois_minus_lease_rate_bps,
            "fwd_implied_carry_bps": fwd_implied_carry_bps,
            "spot_eur_usd": spot_xau_usd,
            "cross_implied_eur_usd": quoted_synthetic_xau_usd,
            "ois_rate_diff_bps": ois_minus_lease_rate_bps,
            "fwd_implied_rate_diff_bps": fwd_implied_carry_bps,
            "stablecoin_usd": peg_reference_ratio,
            "eur_stablecoin": quoted_synthetic_xau_usd,
        }
    )


_generate_synthetic_xauusd_cross_efp_bars = _generate_synthetic_fx_cross_cip_bars
_generate_synthetic_fx_crypto_bars = _generate_synthetic_fx_cross_cip_bars


def _run_demo() -> None:
    """Self-contained demonstration of Topic 4 using real historical 5m XAU/USD Cross/EFP CSV data."""
    print("=== Topic 4: XAU/USD Cross-Venue & EFP Arbitrage (Real CSV `data/xauusd_cross_efp_5m_real.csv`) ===")
    bars_df = load_xauusd_cross_efp_bars_from_csv()
    monitor = FXPegAndCIPRegimeMonitor(max_peg_deviation_bps=25.0, max_cip_basis_bps=40.0)

    # 1. Verify trading unadjusted XAU/USD EFP basis during a delivery/roll squeeze raises FXRegimeOrPegBreakError
    try:
        generate_fx_cross_rate_arbitrage_signals(bars_df, monitor, ignore_regime_guardrail=True)
    except FXRegimeOrPegBreakError as exc:
        print(f"[GUARDRAIL VERIFIED] {exc}")

    # 2. Run regime-gated XAU/USD basis strategy on real 5-minute data
    result = generate_fx_cross_rate_arbitrage_signals(bars_df, monitor, ignore_regime_guardrail=False)
    print(
        f"Loaded {len(bars_df):,d} real 5m XAU/USD Cross/EFP bars from `{XAUUSD_CROSS_EFP_5M_CSV.name}`:\n"
        f"  Clean XAU/USD Inter-Contract Basis ADF Stat: {result['adf_stat']:.2f} (p={result['adf_pvalue']:.4e}) | "
        f"XAU/USD Hedge Beta: {result['hedge_ratio_beta']:.4f}\n"
        f"  Max Real EFP / Lease-Rate Distortion Avoided: {result['max_regime_distortion_bps']:.1f} bps | "
        f"Bars Halted by XAU/USD Circuit Breaker: {result['regime_bars_halted']}"
    )


if __name__ == "__main__":
    _run_demo()
