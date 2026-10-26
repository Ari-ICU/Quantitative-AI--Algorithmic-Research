"""
Cross-pillar temporal validation, XAU/USD (Spot Gold FX) conventions, & methodological guardrails
for Quantitative XAU/USD Algorithmic Research.

Provides custom exceptions, XAU/USD pip/price/session helpers, and time-series splitting utilities
to guarantee zero look-ahead bias, zero feature-normalization leakage, realistic ECN spread execution,
and protection against physical bullion EFP / Gold Lease-Rate (CIP) funding squeezes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator, Sequence, Tuple
import numpy as np
from scipy import stats

DEFAULT_PAIR: str = "XAU/USD"
XAUUSD_PIP_SIZE: float = 0.01          # $0.01 per troy ounce (1 standard XAU/USD lot = 100 oz)
XAUUSD_DEFAULT_SPOT: float = 2350.00   # Representative institutional XAU/USD spot price ($/oz)


class LookAheadBiasError(ValueError):
    """Raised when an LLM checkpoint, knowledge cutoff, or feature timestamp postdates the XAU/USD evaluation event."""


class DataLeakageError(ValueError):
    """Raised when preprocessing (e.g. StandardScaler) is fit on out-of-sample or full-dataset XAU/USD observations."""


class ExecutionIllusionError(ValueError):
    """Raised when an XAU/USD backtest attempts to fill market orders at mid-price instead of crossing the ECN Bid/Ask spread."""


class FXRegimeOrPegBreakError(ValueError):
    """
    Raised when an XAU/USD cross-bullion or Exchange-for-Physical (EFP) / Covered Interest Parity (CIP)
    strategy ignores a physical gold lease-rate / COMEX-London EFP delivery squeeze or structural regime break.
    """


# Alias retained for backward compatibility
DepegRiskError = FXRegimeOrPegBreakError


def assert_xauusd_pair(pair: str = DEFAULT_PAIR) -> str:
    """Validate and normalize the instrument identifier to 'XAU/USD'."""
    clean = pair.upper().replace("/", "").replace("_", "").replace("-", "").strip()
    if clean != "XAUUSD":
        raise ValueError(
            f"This repository is configured exclusively for 'XAU/USD' (Spot Gold vs. US Dollar), got '{pair}'."
        )
    return DEFAULT_PAIR


def get_fx_pip_size(pair: str = DEFAULT_PAIR) -> float:
    """
    Return the standard institutional ECN pip increment for `XAU/USD` ($0.01 per troy ounce).
    """
    assert_xauusd_pair(pair)
    return XAUUSD_PIP_SIZE


def get_fx_default_spot_price(pair: str = DEFAULT_PAIR) -> float:
    """Return the representative institutional `XAU/USD` spot price ($2,350.00 / troy oz)."""
    assert_xauusd_pair(pair)
    return XAUUSD_DEFAULT_SPOT


def classify_fx_trading_session(utc_timestamp: datetime) -> str:
    """
    Classify a UTC timestamp into its active 24-hour global `XAU/USD` (Spot Gold FX) liquidity session:
    - `Tokyo_Asian`:        00:00 <= UTC hour < 08:00 (Tokyo TOCOM / Shanghai Gold Exchange SGE active)
    - `London_European`:    08:00 <= UTC hour < 13:00 (London OTC Bullion morning + 10:30 LBMA AM Fix)
    - `London_NY_Overlap`:  13:00 <= UTC hour < 17:00 (Peak XAU/USD liquidity: 13:20 COMEX Open + 15:00 LBMA PM Auction + 16:00 WM Fix)
    - `NewYork_Sydney`:     17:00 <= UTC hour < 24:00 (NY afternoon + 17:00 NY gold swap/lease rollover + Sydney open)
    """
    hr = utc_timestamp.hour
    if 0 <= hr < 8:
        return "Tokyo_Asian"
    if 8 <= hr < 13:
        return "London_European"
    if 13 <= hr < 17:
        return "London_NY_Overlap"
    return "NewYork_Sydney"


def assert_strictly_causal_timestamps(
    feature_timestamp: datetime,
    evaluation_timestamp: datetime,
    context: str = "",
) -> None:
    """
    Ensure that information used to form an XAU/USD signal or representation (`feature_timestamp`)
    was strictly available prior to or at `evaluation_timestamp`.
    """
    if feature_timestamp > evaluation_timestamp:
        prefix = f"[{context}] " if context else ""
        raise LookAheadBiasError(
            f"{prefix}Causal violation: feature/model timestamp ({feature_timestamp.isoformat()}) "
            f"is after evaluation timestamp ({evaluation_timestamp.isoformat()})."
        )


@dataclass(frozen=True)
class PurgedEmbargoWalkForwardSplit:
    """
    Walk-forward cross-validator with a purge buffer (for multi-step XAU/USD targets)
    and an embargo period to prevent serial correlation leakage between train and test sets.
    """

    min_train_size: int
    test_size: int
    step_size: int = 1
    purge_size: int = 0
    embargo_size: int = 0
    expanding: bool = True

    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yields (train_indices, test_indices) ensuring `max(train_indices) + purge_size < min(test_indices)`."""
        if self.min_train_size <= 0 or self.test_size <= 0:
            raise ValueError("min_train_size and test_size must be positive integers.")

        train_end = self.min_train_size
        while train_end + self.purge_size + self.test_size <= n_samples:
            train_start = 0 if self.expanding else max(0, train_end - self.min_train_size)
            test_start = train_end + self.purge_size
            test_end = test_start + self.test_size

            train_idx = np.arange(train_start, train_end, dtype=int)
            test_idx = np.arange(test_start, test_end, dtype=int)
            yield train_idx, test_idx

            train_end += self.step_size + self.embargo_size


def assert_no_target_feature_timestamp_leakage(
    train_target_timestamps: Sequence[datetime],
    test_feature_timestamps: Sequence[datetime],
    embargo_td: timedelta = timedelta(0),
    context: str = "",
) -> None:
    """
    Verify timestamp-level purge & embargo separation between training target horizons
    and out-of-sample test features:
        `max(train_target_timestamp) + embargo_td < min(test_feature_timestamp)`
    """
    if len(train_target_timestamps) == 0 or len(test_feature_timestamps) == 0:
        raise ValueError("train_target_timestamps and test_feature_timestamps must be non-empty.")

    max_train_target_ts = max(train_target_timestamps)
    min_test_feature_ts = min(test_feature_timestamps)
    embargo_cutoff = max_train_target_ts + embargo_td

    if embargo_cutoff >= min_test_feature_ts:
        prefix = f"[{context}] " if context else ""
        raise DataLeakageError(
            f"{prefix}Temporal target-to-feature leakage: max(train_target_timestamp) + embargo "
            f"({embargo_cutoff.isoformat()}) >= min(test_feature_timestamp) ({min_test_feature_ts.isoformat()})."
        )


def compute_deflated_sharpe_ratio(
    returns_bps: Sequence[float] | np.ndarray,
    n_trials: int = 4,
    annualization_factor: float = 252.0,
) -> float:
    """
    Compute the Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR) probability P(SR > SR_0)
    for an XAU/USD strategy return stream, adjusting for non-Normal skewness, fat-tailed kurtosis,
    finite sample length T, and multiple testing across `n_trials`.
    """
    r = np.asarray(returns_bps, dtype=float)
    r = r[np.isfinite(r)]
    n_obs = len(r)
    if n_obs < 5 or float(np.std(r, ddof=1)) <= 1e-12:
        return 0.0

    sr_per_bar = float(np.mean(r) / np.std(r, ddof=1))
    skew = float(stats.skew(r, bias=False)) if n_obs > 3 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False, bias=False)) if n_obs > 4 else 3.0

    # Expected maximum Sharpe Ratio under null across n_trials independent configurations
    euler_gamma = 0.5772156649
    n_eff = max(int(n_trials), 1)
    if n_eff == 1:
        sr_benchmark = 0.0
    else:
        z1 = float(stats.norm.ppf(1.0 - 1.0 / n_eff))
        z2 = float(stats.norm.ppf(1.0 - 1.0 / (n_eff * np.e)))
        sr_std_null = np.sqrt(1.0 / max(n_obs - 1, 1))
        sr_benchmark = sr_std_null * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)

    denom_var = max(
        1.0 - skew * sr_per_bar + ((kurt - 1.0) / 4.0) * (sr_per_bar**2),
        1e-9,
    )
    z_dsr = ((sr_per_bar - sr_benchmark) * np.sqrt(max(n_obs - 1, 1))) / np.sqrt(denom_var)
    _ = annualization_factor  # retained for signature clarity (DSR is scale-invariant in per-period units)
    return float(stats.norm.cdf(z_dsr))


def compute_cost_adjusted_profit_factor(
    gross_pnl_bps: Sequence[float] | np.ndarray,
    positions: Sequence[float] | np.ndarray | None = None,
    spread_cost_bps_per_turn: float = 0.45,
    tom_next_swap_bps_per_bar: float = 0.012,
) -> float:
    """
    Compute the Cost-Adjusted Net Profit Factor (Gross Gains / Gross Losses) after deducting
    ECN XAU/USD bid-ask spread turnover drag and overnight tom-next gold lease/SOFR swap drag.
    """
    gross = np.asarray(gross_pnl_bps, dtype=float)
    if positions is not None:
        pos = np.asarray(positions, dtype=float)
        turnover = np.abs(np.diff(pos, prepend=0.0))
        holding = np.abs(pos)
        net = gross - spread_cost_bps_per_turn * turnover - tom_next_swap_bps_per_bar * holding
    else:
        net = gross

    gains = float(np.sum(np.maximum(net, 0.0)))
    losses = float(np.sum(np.maximum(-net, 0.0)))
    if losses <= 1e-12:
        return float(gains) if gains > 0.0 else 1.0
    return float(gains / losses)
