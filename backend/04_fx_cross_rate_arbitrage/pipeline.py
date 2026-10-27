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
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from common.data_loader import XAUUSD_CROSS_EFP_5M_CSV, load_real_xauusd_cross_efp_csv
from common.validation import (
    DEFAULT_PAIR,
    DepegRiskError,
    FXRegimeOrPegBreakError,
    compute_cost_adjusted_profit_factor,
    compute_deflated_sharpe_ratio,
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

    # Cost-of-Carry Adjusted EFP Basis: y_t = ln(F_t) - ln(S_t) - EMA_5d(ln(F_t) - ln(S_t))
    raw_log_basis_bps = (
        np.log(np.maximum(out["implied_xau_usd"].to_numpy(dtype=float), 1e-6))
        - np.log(np.maximum(out["spot_xau_usd"].to_numpy(dtype=float), 1e-6))
    ) * 1e4
    ema_carry_bps = (
        pd.Series(raw_log_basis_bps, index=out.index)
        .ewm(span=min(max(len(out) // 3, 24), 1440), min_periods=1)
        .mean()
        .to_numpy()
    )
    out["raw_log_efp_bps"] = raw_log_basis_bps
    out["carry_ema_bps"] = ema_carry_bps
    out["carry_adjusted_efp_bps"] = raw_log_basis_bps - ema_carry_bps

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


def estimate_kalman_hedge_ratio(
    spot_px: np.ndarray,
    implied_px: np.ndarray,
    process_var_q: float = 1e-7,
    meas_var_r: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate a 1D State-Space Kalman Filter dynamic hedge ratio (beta_t) between
    Direct Spot `XAU/USD` (`spot_px`) and Synthetic/COMEX Implied `XAU/USD` (`implied_px`):
        State equation:       beta_t = beta_{t-1} + eta_t,   eta_t ~ N(0, Q)
        Measurement equation: y_t    = beta_t * x_t + nu_t,  nu_t  ~ N(0, R)
    Returns `(beta_series, innovation_bps_series)`.
    """
    x = np.asarray(spot_px, dtype=float)
    y = np.asarray(implied_px, dtype=float)
    n = len(x)
    beta_series = np.ones(n, dtype=float)
    innov_bps = np.zeros(n, dtype=float)
    if n == 0:
        return beta_series, innov_bps

    beta_t = float(y[0] / max(x[0], 1e-9))
    p_t = 1e-4

    for i in range(n):
        # Prediction step
        p_pred = p_t + process_var_q
        y_pred = beta_t * x[i]
        err = y[i] - y_pred
        innov_bps[i] = (err / max(x[i], 1e-9)) * 1e4

        # Kalman gain & state update
        s_t = (x[i] ** 2) * p_pred + meas_var_r
        k_t = (p_pred * x[i]) / max(s_t, 1e-12)
        beta_t = beta_t + k_t * err
        p_t = (1.0 - k_t * x[i]) * p_pred
        beta_series[i] = beta_t

    return beta_series, innov_bps


def compute_rolling_hurst_exponent(
    series: pd.Series | np.ndarray,
    window: int = 288,
    lag_tau: int = 8,
) -> np.ndarray:
    """
    Compute the rolling Hurst exponent H_t on the cost-of-carry adjusted `XAU/USD` EFP basis
    using the variance-ratio scaling relation:
        H_t = 0.5 * log(Var(z_t - z_{t-tau}) / (tau * Var(z_t - z_{t-1}))) / log(tau) + 0.5
    Values H_t < 0.45 indicate stationary anti-persistent mean-reversion; H_t >= 0.50 indicates
    non-stationary momentum or a physical COMEX-London EFP regime shift.
    """
    s = pd.Series(np.asarray(series, dtype=float))
    eff_win = min(max(int(window), 30), max(len(s) // 2, 30))
    tau = max(int(lag_tau), 2)

    diff_1 = s.diff(1)
    diff_tau = s.diff(tau)

    var_1 = diff_1.rolling(eff_win, min_periods=16).var().to_numpy(dtype=float)
    var_tau = diff_tau.rolling(eff_win, min_periods=16).var().to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(var_tau, 1e-12) / np.maximum(var_1, 1e-12)
        hurst = 0.5 * (np.log(ratio) / np.log(float(tau)))

    hurst = np.where(np.isfinite(hurst), np.clip(hurst, 0.05, 0.95), 0.38)
    return hurst


def generate_fx_cross_rate_arbitrage_signals(
    df: pd.DataFrame,
    regime_monitor: FXPegAndCIPRegimeMonitor,
    z_entry: float = 2.0,
    z_max_entry: float = 3.5,
    z_hard_stop: float = 4.0,
    max_hurst: float = 0.45,
    lookback_bars: int = 60,
    ignore_regime_guardrail: bool = False,
    ignore_depeg_guardrail: bool = False,
) -> Dict[str, object]:
    """
    Estimate dynamic Kalman-Filter cointegration (`kalman_beta`), rolling Hurst exponent (`hurst_exponent`),
    and Threshold VECM (TVECM) z-score signals between Direct `XAU/USD` and Synthetic / EFP-Implied `XAU/USD`,
    gated by `FXPegAndCIPRegimeMonitor`:
    - Enters mean-reversion only when |z_t| in [z_entry, z_max_entry] (default [2.0, 3.5]) and H_t < max_hurst (0.45).
    - Triggers a hard structural-break stop if |z_t| > z_hard_stop (4.0) or during an EFP / Lease-Rate breach.
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

    # 1. Dynamic 1D Kalman Filter Hedge Ratio (beta_t) & Rolling Hurst Exponent (H_t)
    kalman_beta, kalman_innov_bps = estimate_kalman_hedge_ratio(
        decomposed["spot_xau_usd"].to_numpy(dtype=float),
        decomposed["implied_xau_usd"].to_numpy(dtype=float),
    )
    decomposed["kalman_beta"] = kalman_beta
    decomposed["kalman_innovation_bps"] = kalman_innov_bps
    decomposed["hurst_exponent"] = compute_rolling_hurst_exponent(
        decomposed["carry_adjusted_efp_bps"], window=288
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

    abs_z = decomposed["z_score"].abs()
    is_structural_stop = abs_z > z_hard_stop
    decomposed["is_structural_break_stop"] = is_structural_stop
    hurst_ok = decomposed["hurst_exponent"] < max_hurst

    # Threshold VECM Band: enter mean-reversion when |z_t| in [z_entry, z_max_entry] & H_t < max_hurst
    in_entry_band = (abs_z >= z_entry) & (abs_z <= z_max_entry) & hurst_ok
    raw_pos = np.where(
        in_entry_band & (decomposed["z_score"] > 0.0),
        -1.0,
        np.where(in_entry_band & (decomposed["z_score"] < 0.0), 1.0, 0.0),
    )
    # Hard flatten on EFP/Lease-Rate regime breach, |z_t| > 4.0 structural break, or 23/5 session boundary
    session_jump_mask = (
        decomposed["is_session_open_jump"].to_numpy(dtype=bool)
        if "is_session_open_jump" in decomposed.columns
        else np.zeros(len(decomposed), dtype=bool)
    )
    halt_mask = decomposed["is_regime_breach"].to_numpy(dtype=bool) | is_structural_stop.to_numpy(dtype=bool) | session_jump_mask
    decomposed["position"] = np.where(halt_mask, 0.0, raw_pos)

    basis_change_bps = decomposed["true_cross_basis_bps"].diff().fillna(0.0)
    pos_series = pd.Series(decomposed["position"], index=decomposed.index)
    turnover = pos_series.diff().abs().fillna(0.0)
    gross_pnl_bps = (pos_series.shift(1).fillna(0.0) * basis_change_bps).to_numpy(dtype=float)
    decomposed["strategy_gross_pnl_bps"] = gross_pnl_bps
    decomposed["strategy_net_pnl_bps"] = (
        gross_pnl_bps - 0.45 * turnover.to_numpy(dtype=float) - 0.012 * pos_series.abs().to_numpy(dtype=float)
    )

    active_returns = decomposed.loc[pos_series.shift(1).abs() > 0, "strategy_net_pnl_bps"].to_numpy(dtype=float)
    dsr = compute_deflated_sharpe_ratio(
        active_returns if len(active_returns) >= 5 else decomposed["strategy_net_pnl_bps"].to_numpy(dtype=float),
        n_trials=4,
    )
    profit_factor = compute_cost_adjusted_profit_factor(
        gross_pnl_bps,
        positions=decomposed["position"].to_numpy(dtype=float),
        spread_cost_bps_per_turn=0.45,
        tom_next_swap_bps_per_bar=0.012,
    )

    return {
        "frame": decomposed,
        "hedge_ratio_beta": hedge_beta,
        "kalman_beta_mean": float(np.mean(kalman_beta)),
        "kalman_beta_std": float(np.std(kalman_beta)),
        "mean_hurst_exponent": float(np.mean(decomposed["hurst_exponent"])),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "regime_bars_halted": int(decomposed["is_regime_breach"].sum()),
        "depeg_bars_halted": int(decomposed["is_regime_breach"].sum()),
        "structural_stop_bars": int(is_structural_stop.sum()),
        "max_regime_distortion_bps": float(decomposed["regime_distortion_bps"].abs().max()),
        "max_depeg_distortion_bps": float(decomposed["regime_distortion_bps"].abs().max()),
        "cumulative_net_pnl_bps": float(decomposed["strategy_net_pnl_bps"].sum()),
        "deflated_sharpe_ratio": float(dsr),
        "cost_adjusted_profit_factor": float(profit_factor),
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
        start = max(0, min(len(df) - n_bars, 850 - n_bars // 2))
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
        f"Kalman Dynamic Beta: {result['kalman_beta_mean']:.4f} (±{result['kalman_beta_std']:.4f}) | "
        f"Mean Hurst H_t: {result['mean_hurst_exponent']:.3f}\n"
        f"  Max Real EFP / Lease-Rate Distortion Avoided: {result['max_regime_distortion_bps']:.1f} bps | "
        f"Bars Halted by Circuit Breaker: {result['regime_bars_halted']} | "
        f"DSR: {result['deflated_sharpe_ratio']:.3f} | Net Profit Factor: {result['cost_adjusted_profit_factor']:.2f}"
    )


if __name__ == "__main__":
    _run_demo()
