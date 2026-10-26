"""
Topic 3: XAU/USD ML vs. GARCH Volatility — Real Historical 24-Hour Intraday XAU/USD RV Aggregation & Leak-Free Scaling.

Primary Libraries: arch (for GARCH), pytorch-forecasting (for TFT), scikit-learn
Granularity: Real Historical 24-hour XAU/USD intraday bars (`data/xauusd_intraday_2y_real.csv` & `data/xauusd_5m_real.csv`)
aggregated to Daily XAU/USD Realized Volatility (RV)
Pitfall Prevented: Data leakage from fitting StandardScaler across the full XAU/USD dataset rather than
strictly inside each walk-forward training window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from common.data_loader import (
    XAUUSD_5M_CSV,
    XAUUSD_INTRADAY_2Y_CSV,
    load_real_xauusd_intraday_bars_csv,
)
from common.validation import (
    DEFAULT_PAIR,
    DataLeakageError,
    PurgedEmbargoWalkForwardSplit,
    assert_xauusd_pair,
    classify_fx_trading_session,
    get_fx_default_spot_price,
)


def compute_daily_realized_variance_from_5m(intraday_5m_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 24-hour intraday `XAU/USD` spot prices into:
    - Daily Realized Variance (`rv_daily`) and Annualized Realized Volatility (`rvol_ann`),
    - Barndorff-Nielsen & Shephard Bipower Variation (`bv_daily`) & Jump Variation (`jump_component`),
    - Patton & Sheppard (2015) Signed `XAU/USD` Semivariances (`rv_pos_daily`, `rv_neg_daily`),
    - Global 24h Bullion Session Realized Variances (`rv_tokyo`, `rv_london`, `rv_london_ny_overlap`, `rv_ny_sydney`).
    """
    df = intraday_5m_df.sort_values("timestamp").copy()
    ts = pd.to_datetime(df["timestamp"])
    df["date"] = ts.dt.floor("D")
    df["session"] = [classify_fx_trading_session(t.to_pydatetime()) for t in ts]

    df["log_price"] = np.log(df["close"].astype(float))
    df["ret_5m"] = df.groupby("date")["log_price"].diff().fillna(0.0)
    df["abs_ret_5m"] = df["ret_5m"].abs()
    df["lag_abs_ret_5m"] = df.groupby("date")["abs_ret_5m"].shift(1).fillna(0.0)

    mu_1_sq_inv = np.pi / 2.0
    df["bv_term"] = mu_1_sq_inv * df["abs_ret_5m"] * df["lag_abs_ret_5m"]
    df["ret_5m_sq"] = df["ret_5m"] ** 2
    df["ret_pos_sq"] = np.where(df["ret_5m"] > 0.0, df["ret_5m_sq"], 0.0)
    df["ret_neg_sq"] = np.where(df["ret_5m"] < 0.0, df["ret_5m_sq"], 0.0)

    df["rv_tokyo_bar"] = np.where(df["session"] == "Tokyo_Asian", df["ret_5m_sq"], 0.0)
    df["rv_london_bar"] = np.where(df["session"] == "London_European", df["ret_5m_sq"], 0.0)
    df["rv_overlap_bar"] = np.where(df["session"] == "London_NY_Overlap", df["ret_5m_sq"], 0.0)
    df["rv_ny_sydney_bar"] = np.where(df["session"] == "NewYork_Sydney", df["ret_5m_sq"], 0.0)

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            rv_daily=("ret_5m_sq", "sum"),
            bv_daily=("bv_term", "sum"),
            rv_pos_daily=("ret_pos_sq", "sum"),
            rv_neg_daily=("ret_neg_sq", "sum"),
            rv_tokyo=("rv_tokyo_bar", "sum"),
            rv_london=("rv_london_bar", "sum"),
            rv_london_ny_overlap=("rv_overlap_bar", "sum"),
            rv_ny_sydney=("rv_ny_sydney_bar", "sum"),
            daily_log_ret=("ret_5m", "sum"),
            n_bars=("ret_5m", "count"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    # Filter out thin weekend/holiday rollover stubs when running on real historical CSVs
    median_bars = float(daily["n_bars"].median()) if len(daily) > 0 else 0.0
    if median_bars >= 15.0:
        daily = daily[daily["n_bars"] >= 0.70 * median_bars].reset_index(drop=True)

    daily["rvol_ann"] = np.sqrt(252.0 * np.maximum(daily["rv_daily"], 1e-8))
    daily["rvol_overlap_ann"] = np.sqrt(252.0 * np.maximum(daily["rv_london_ny_overlap"], 1e-8))
    daily["jump_component"] = np.maximum(0.0, daily["rv_daily"] - daily["bv_daily"])
    daily["semivar_asymmetry"] = daily["rv_neg_daily"] - daily["rv_pos_daily"]
    return daily


def build_har_rv_features(daily_rv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct Corsi (2009) Heterogeneous Autoregressive (HAR-RV) + COMEX/LBMA overlap session,
    signed semivariance, and jump features strictly lagged up to day t to predict
    next-day 24-hour `XAU/USD` realized volatility `target_rvol_t1` (day t+1).
    """
    feat = daily_rv_df.copy()
    feat["rvol_d"] = feat["rvol_ann"]
    feat["rvol_w"] = feat["rvol_ann"].rolling(5).mean()
    feat["rvol_m"] = feat["rvol_ann"].rolling(22).mean()
    feat["jump_d"] = feat["jump_component"]
    feat["neg_ret_d"] = np.minimum(0.0, feat["daily_log_ret"])
    feat["rvol_overlap_d"] = (
        feat["rvol_overlap_ann"]
        if "rvol_overlap_ann" in feat.columns
        else feat["rvol_ann"] * 0.5
    )
    feat["semivar_asym_d"] = (
        feat["semivar_asymmetry"]
        if "semivar_asymmetry" in feat.columns
        else np.zeros(len(feat), dtype=float)
    )

    feat["target_rvol_t1"] = feat["rvol_ann"].shift(-1)
    feat = feat.dropna().reset_index(drop=True)
    return feat


class TemporalLeakFreeScaler:
    """
    Guards `sklearn.preprocessing.StandardScaler` against look-ahead normalization leakage.
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted_max_idx: Optional[int] = None

    def fit_on_train_only(
        self,
        X_full: np.ndarray,
        train_indices: Sequence[int],
        test_indices: Sequence[int],
    ) -> "TemporalLeakFreeScaler":
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError("train_indices and test_indices must be non-empty.")

        max_train = int(np.max(train_indices))
        min_test = int(np.min(test_indices))

        if len(train_indices) >= len(X_full):
            raise DataLeakageError(
                "Data Leakage Guardrail: Attempted to fit StandardScaler on the entire XAU/USD dataset. "
                "Fit strictly on `X_full[train_indices]` inside each walk-forward fold."
            )

        if max_train >= min_test:
            raise DataLeakageError(
                f"Data Leakage Guardrail: Training window max index ({max_train}) overlaps or "
                f"exceeds test window start index ({min_test})."
            )

        self._scaler.fit(X_full[train_indices])
        self._fitted_max_idx = max_train
        return self

    def transform_split(
        self,
        X_full: np.ndarray,
        indices: Sequence[int],
    ) -> np.ndarray:
        if self._fitted_max_idx is None:
            raise RuntimeError("Call `fit_on_train_only` before transforming features.")
        return self._scaler.transform(X_full[indices])


def qlike_loss(realized_vol: np.ndarray, forecast_vol: np.ndarray) -> float:
    """
    Patton (2011) robust QLIKE loss function on `XAU/USD` variance proxy:
    L(sigma^2, h) = sigma^2 / h - log(sigma^2 / h) - 1.
    """
    rv_var = np.maximum(realized_vol**2, 1e-10)
    fc_var = np.maximum(forecast_vol**2, 1e-10)
    ratio = rv_var / fc_var
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _forecast_garch_one_step(train_returns_pct: np.ndarray) -> float:
    """
    Fit a GARCH(1,1) model using `arch` (or EWMA fallback if `arch` is not installed)
    strictly on `train_returns_pct` and return the 1-step-ahead annualized `XAU/USD` volatility forecast.
    """
    try:
        from arch import arch_model

        am = arch_model(train_returns_pct, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        next_var_pct_sq = float(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
        return float(np.sqrt(252.0 * (next_var_pct_sq * 1e-4)))
    except ImportError:
        lam = 0.94
        var_pct_sq = float(np.var(train_returns_pct[:20]))
        for r in train_returns_pct:
            var_pct_sq = lam * var_pct_sq + (1.0 - lam) * (r**2)
        return float(np.sqrt(252.0 * (var_pct_sq * 1e-4)))


def run_walk_forward_volatility_comparison(
    feat_df: pd.DataFrame,
    min_train_days: int = 120,
    test_step_days: int = 15,
    return_series: bool = False,
) -> Dict[str, object]:
    """
    Compare GARCH(1,1) vs. Leak-Free ML (Log-HAR-XAU Gradient Boosted Regressor with
    QLIKE-consistent log-normal variance expectation) using an expanding walk-forward split.
    """
    base_cols = ["rvol_d", "rvol_w", "rvol_m", "jump_d", "neg_ret_d"]
    extra_cols = [c for c in ["rvol_overlap_d", "semivar_asym_d"] if c in feat_df.columns]
    feature_cols = base_cols + extra_cols

    X_full = feat_df[feature_cols].to_numpy(dtype=float)
    y_full = feat_df["target_rvol_t1"].to_numpy(dtype=float)
    ret_pct_full = (feat_df["daily_log_ret"] * 100.0).to_numpy(dtype=float)

    splitter = PurgedEmbargoWalkForwardSplit(
        min_train_size=min_train_days,
        test_size=test_step_days,
        step_size=test_step_days,
        purge_size=1,
        expanding=True,
    )

    realized_list = []
    ml_forecasts = []
    garch_forecasts = []
    oos_indices = []

    for train_idx, test_idx in splitter.split(len(feat_df)):
        scaler = TemporalLeakFreeScaler().fit_on_train_only(X_full, train_idx, test_idx)
        X_train_scaled = scaler.transform_split(X_full, train_idx)
        X_test_scaled = scaler.transform_split(X_full, test_idx)

        ml_model = HistGradientBoostingRegressor(
            max_depth=2,
            learning_rate=0.04,
            max_iter=60,
            min_samples_leaf=15,
            random_state=42,
        )
        log_y_train = np.log(np.maximum(y_full[train_idx], 0.02))
        ml_model.fit(X_train_scaled, log_y_train)
        resid_var = float(np.var(log_y_train - ml_model.predict(X_train_scaled)))
        ml_pred = np.maximum(0.02, np.exp(ml_model.predict(X_test_scaled) + resid_var))

        garch_pred = np.array(
            [_forecast_garch_one_step(ret_pct_full[:idx]) for idx in test_idx],
            dtype=float,
        )

        realized_list.append(y_full[test_idx])
        ml_forecasts.append(ml_pred)
        garch_forecasts.append(garch_pred)
        oos_indices.append(test_idx)

    y_eval = np.concatenate(realized_list)
    ml_eval = np.concatenate(ml_forecasts)
    garch_eval = np.concatenate(garch_forecasts)
    eval_idx = np.concatenate(oos_indices)

    out: Dict[str, object] = {
        "n_oos_days": int(len(y_eval)),
        "ml_rmse": float(np.sqrt(np.mean((y_eval - ml_eval) ** 2))),
        "garch_rmse": float(np.sqrt(np.mean((y_eval - garch_eval) ** 2))),
        "ml_qlike": qlike_loss(y_eval, ml_eval),
        "garch_qlike": qlike_loss(y_eval, garch_eval),
    }
    if return_series:
        out["oos_dates"] = feat_df["date"].iloc[eval_idx].to_numpy()
        out["realized_rvol"] = y_eval
        out["ml_forecast_rvol"] = ml_eval
        out["garch_forecast_rvol"] = garch_eval
    return out


def load_xauusd_intraday_bars_from_csv(
    csv_path: str | Path = XAUUSD_INTRADAY_2Y_CSV,
    pair: str = DEFAULT_PAIR,
) -> pd.DataFrame:
    """Load real historical 24h intraday `XAU/USD` bars from `data/xauusd_intraday_2y_real.csv` (or `data/xauusd_5m_real.csv`)."""
    assert_xauusd_pair(pair)
    return load_real_xauusd_intraday_bars_csv(csv_path)


def _generate_synthetic_5m_bars(
    n_days: int = 200,
    bars_per_day: int = 288,
    seed: int = 42,
    pair: str = DEFAULT_PAIR,
    csv_path: str | Path = XAUUSD_INTRADAY_2Y_CSV,
) -> pd.DataFrame:
    """
    Load real historical 24-hour intraday `XAU/USD` bars from `data/xauusd_intraday_2y_real.csv`
    when available (and when `bars_per_day` is not explicitly testing a small synthetic 288-bar unit check),
    or fall back to calibrated `XAU/USD` 5m simulation.
    """
    pair = assert_xauusd_pair(pair)
    p = Path(csv_path)
    if p.exists() and n_days >= 50:
        return load_xauusd_intraday_bars_from_csv(p, pair=pair)

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)

    session_weights = np.ones(bars_per_day, dtype=float)
    for b in range(bars_per_day):
        utc_hour = (b * 5) / 60.0
        if 0.0 <= utc_hour < 8.0:
            session_weights[b] = 0.70
        elif 8.0 <= utc_hour < 13.0:
            session_weights[b] = 1.10
        elif 13.0 <= utc_hour < 17.0:
            session_weights[b] = 2.15 if 14.75 <= utc_hour <= 16.25 else 1.75
        else:
            session_weights[b] = 0.60
    session_weights /= np.mean(session_weights)

    records = []
    log_px = np.log(get_fx_default_spot_price(pair))
    daily_var = (0.155**2) / 252.0

    for d in dates:
        daily_var = 0.0000048 + 0.85 * daily_var + 0.10 * daily_var * (rng.standard_normal() ** 2)
        base_bar_var = daily_var / bars_per_day
        for b in range(bars_per_day):
            ts = d + pd.Timedelta(minutes=5 * b)
            bar_sigma = np.sqrt(base_bar_var * session_weights[b])
            ret = rng.normal(0.0, bar_sigma)
            log_px += ret
            records.append({"pair": pair, "timestamp": ts, "close": float(np.exp(log_px))})

    return pd.DataFrame(records)


def _run_demo() -> None:
    """Self-contained demonstration of Topic 3 using real historical 24h XAU/USD intraday CSVs."""
    print("=== Topic 3: XAU/USD ML vs. GARCH Volatility (Real Historical XAU/USD Intraday CSVs) ===")
    df_5m_real = load_xauusd_intraday_bars_from_csv(XAUUSD_5M_CSV)
    intraday_df = load_xauusd_intraday_bars_from_csv(XAUUSD_INTRADAY_2Y_CSV)
    daily_rv = compute_daily_realized_variance_from_5m(intraday_df)
    feat_df = build_har_rv_features(daily_rv)

    overlap_share_pct = 100.0 * float((daily_rv["rv_london_ny_overlap"] / daily_rv["rv_daily"]).mean())
    print(
        f"Loaded {len(df_5m_real):,d} real 5m XAU/USD bars (`{XAUUSD_5M_CSV.name}`) & "
        f"{len(intraday_df):,d} real 2Y 24h XAU/USD bars (`{XAUUSD_INTRADAY_2Y_CSV.name}`, {len(daily_rv)} trading days):\n"
        f"  Mean Annualized Real XAU/USD RVol: {daily_rv['rvol_ann'].mean() * 100:.2f}% | "
        f"London-NY COMEX/LBMA Overlap Variance Share: {overlap_share_pct:.1f}%"
    )

    X_all = feat_df[["rvol_d", "rvol_w", "rvol_m", "jump_d", "neg_ret_d"]].to_numpy()
    all_idx = np.arange(len(X_all))

    # 1. Verify full-dataset scaling raises DataLeakageError
    try:
        TemporalLeakFreeScaler().fit_on_train_only(X_all, all_idx, all_idx[-20:])
    except DataLeakageError as exc:
        print(f"[GUARDRAIL VERIFIED] {exc}")

    # 2. Run walk-forward comparison on real 2024–2026 XAU/USD volatility
    metrics = run_walk_forward_volatility_comparison(feat_df, min_train_days=120, test_step_days=15)
    print(
        f"Out-of-Sample Real XAU/USD Volatility Evaluation ({metrics['n_oos_days']} trading days):\n"
        f"  Leak-Free ML (HAR-XAU + COMEX/LBMA Session RV) -> RMSE: {metrics['ml_rmse']:.4f} | QLIKE: {metrics['ml_qlike']:.4f}\n"
        f"  GARCH(1,1) / EWMA Baseline                     -> RMSE: {metrics['garch_rmse']:.4f} | QLIKE: {metrics['garch_qlike']:.4f}"
    )


if __name__ == "__main__":
    _run_demo()
