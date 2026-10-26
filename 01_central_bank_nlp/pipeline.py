"""
Topic 1: Central Bank XAU/USD NLP — Intraday Spot Gold (XAU/USD) Event Study with LLM Vintage Guardrail.

Primary Libraries: transformers (HuggingFace), PyTorch, statsmodels
Granularity: Real Historical 1-minute XAU/USD spot bars (`data/xauusd_1m_real.csv`) + FOMC sentence timestamps (`data/xauusd_fomc_events_real.csv`)
Pitfall Prevented: Look-ahead bias from using an LLM pre-trained on corpora post-dating the FOMC speech.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from common.data_loader import (
    XAUUSD_1M_CSV,
    XAUUSD_FOMC_EVENTS_CSV,
    load_real_xauusd_1m_bars_csv,
    load_real_xauusd_fomc_events_csv,
)
from common.validation import (
    DEFAULT_PAIR,
    LookAheadBiasError,
    assert_strictly_causal_timestamps,
    assert_xauusd_pair,
    get_fx_pip_size,
)


@dataclass(frozen=True)
class ModelVintageSpec:
    """Metadata specifying the strict pre-training / fine-tuning knowledge cutoff of an NLP model."""

    model_id: str
    knowledge_cutoff: datetime
    description: str


KNOWN_MODEL_VINTAGES: Dict[str, ModelVintageSpec] = {
    "google-bert/bert-base-uncased": ModelVintageSpec(
        model_id="google-bert/bert-base-uncased",
        knowledge_cutoff=datetime(2018, 10, 31),
        description="Original BERT base (Wikipedia + BookCorpus up to Oct 2018)",
    ),
    "ProsusAI/finbert": ModelVintageSpec(
        model_id="ProsusAI/finbert",
        knowledge_cutoff=datetime(2019, 8, 31),
        description="FinBERT fine-tuned on TRC2-financial + Financial PhraseBank (<= Aug 2019)",
    ),
    "gtfintechlab/FOMC-RoBERTa": ModelVintageSpec(
        model_id="gtfintechlab/FOMC-RoBERTa",
        knowledge_cutoff=datetime(2023, 6, 30),
        description="RoBERTa fine-tuned on FOMC transcripts through 2022/2023",
    ),
}

CENTRAL_BANK_TO_CURRENCY: Dict[str, str] = {
    "FED": "USD",
    "FOMC": "USD",
}


def compute_fx_pair_hawkish_polarity(pair: str = DEFAULT_PAIR, central_bank: str = "FED") -> float:
    """
    Determine the directional multiplier (-1.0) of a Hawkish FOMC/Fed shock on `XAU/USD` (Spot Gold vs. USD).
    """
    assert_xauusd_pair(pair)
    if central_bank.upper().strip() not in CENTRAL_BANK_TO_CURRENCY:
        raise ValueError(f"Unsupported central bank '{central_bank}' for XAU/USD FOMC event study.")
    return -1.0


@dataclass(frozen=True)
class SpeechSentenceEvent:
    """A single timestamped sentence spoken during an FOMC press conference affecting XAU/USD."""

    speaker: str
    sentence_timestamp: datetime
    text: str
    central_bank: str = "FED"


@dataclass(frozen=True)
class IntradayBar:
    """1-minute intraday XAU/USD spot bar ($/troy oz) indexed by bar open and bar close."""

    bar_open_time: datetime
    bar_close_time: datetime
    open_price: float
    close_price: float
    volume: float
    pair: str = DEFAULT_PAIR


class VintageGuardNLPExtractor:
    """
    Wraps HuggingFace / PyTorch transformer inference with a strict point-in-time vintage check.

    Raises `LookAheadBiasError` if any evaluated FOMC speech sentence occurred prior to the
    model's pre-training / fine-tuning `knowledge_cutoff`.
    """

    def __init__(
        self,
        model_id: str,
        custom_vintage_spec: Optional[ModelVintageSpec] = None,
        use_hf_pipeline: bool = False,
    ) -> None:
        if custom_vintage_spec is not None:
            self.vintage = custom_vintage_spec
        elif model_id in KNOWN_MODEL_VINTAGES:
            self.vintage = KNOWN_MODEL_VINTAGES[model_id]
        else:
            raise ValueError(
                f"Unknown model vintage for '{model_id}'. Register its exact pre-training "
                "cutoff date via `custom_vintage_spec` to prevent look-ahead bias."
            )
        self.use_hf_pipeline = use_hf_pipeline
        self._hf_model = None
        self._hf_tokenizer = None

    def validate_event_timestamp(self, sentence_timestamp: datetime) -> None:
        """Verify that the model's pre-training corpus strictly predates the speech timestamp."""
        assert_strictly_causal_timestamps(
            feature_timestamp=self.vintage.knowledge_cutoff,
            evaluation_timestamp=sentence_timestamp,
            context=f"LLM Vintage Guard ({self.vintage.model_id})",
        )

    def score_hawkish_dovish(self, events: Sequence[SpeechSentenceEvent]) -> np.ndarray:
        """
        Compute net Hawkish (+1) vs. Dovish (-1) FOMC monetary policy sentiment scores.
        """
        for ev in events:
            self.validate_event_timestamp(ev.sentence_timestamp)

        if self.use_hf_pipeline:
            return self._score_with_transformers(events)

        hawkish_terms = {
            "inflation", "restrictive", "tightening", "hike", "elevated",
            "overheating", "vigilant", "upside", "persistent", "firming",
            "recalibration", "hawkish", "yields",
        }
        dovish_terms = {
            "accommodative", "slowdown", "downside", "cut", "easing",
            "softening", "patient", "headwinds", "moderating", "support",
            "dovish", "disinflation", "unemployment", "fragile",
        }

        scores = np.zeros(len(events), dtype=float)
        for i, ev in enumerate(events):
            tokens = [
                tok.strip(".,;:!?\"'()").lower()
                for tok in ev.text.split()
            ]
            if not tokens:
                continue
            h_count = sum(1 for t in tokens if t in hawkish_terms)
            d_count = sum(1 for t in tokens if t in dovish_terms)
            scores[i] = (h_count - d_count) / np.sqrt(len(tokens))
        return scores

    def _score_with_transformers(self, events: Sequence[SpeechSentenceEvent]) -> np.ndarray:
        """Run forward pass in PyTorch / HuggingFace transformers with no grad."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._hf_tokenizer is None or self._hf_model is None:
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.vintage.model_id)
            self._hf_model = AutoModelForSequenceClassification.from_pretrained(self.vintage.model_id)
            self._hf_model.eval()

        texts = [ev.text for ev in events]
        encoded = self._hf_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._hf_model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        return probs[:, 0] - probs[:, 1]


def load_real_xauusd_1m_bars_and_fomc_events(
    bars_csv: str | Path = XAUUSD_1M_CSV,
    events_csv: str | Path = XAUUSD_FOMC_EVENTS_CSV,
) -> Tuple[List[IntradayBar], List[SpeechSentenceEvent]]:
    """
    Load real historical 1-minute `XAU/USD` bars (`data/xauusd_1m_real.csv`) and aligned
    FOMC speech sentence events (`data/xauusd_fomc_events_real.csv`).
    """
    df_bars = load_real_xauusd_1m_bars_csv(bars_csv)
    df_ev = load_real_xauusd_fomc_events_csv(events_csv)

    bars = [
        IntradayBar(
            bar_open_time=pd.to_datetime(row["timestamp"]).to_pydatetime(),
            bar_close_time=(pd.to_datetime(row["timestamp"]) + pd.Timedelta(seconds=59)).to_pydatetime(),
            open_price=float(row["open"]),
            close_price=float(row["close"]),
            volume=float(row["volume"]),
            pair=DEFAULT_PAIR,
        )
        for _, row in df_bars.iterrows()
    ]
    events = [
        SpeechSentenceEvent(
            speaker=str(row["speaker"]),
            sentence_timestamp=pd.to_datetime(row["sentence_timestamp"]).to_pydatetime(),
            text=str(row["text"]),
            central_bank=str(row["central_bank"]),
        )
        for _, row in df_ev.iterrows()
    ]
    return bars, events


def align_sentences_to_bars(
    events: Sequence[SpeechSentenceEvent],
    hawkish_scores: np.ndarray,
    bars: Sequence[IntradayBar],
    horizon_minutes: int = 3,
) -> pd.DataFrame:
    """
    Align each FOMC sentence spoken at timestamp `t_s` to the first 1-minute `XAU/USD` bar whose
    `bar_open_time >= t_s` (strictly post-utterance XAU/USD reaction window).
    """
    if len(events) != len(hawkish_scores):
        raise ValueError("Length of events and hawkish_scores must match.")

    sorted_bars = sorted(bars, key=lambda b: b.bar_open_time)
    bar_opens = [b.bar_open_time for b in sorted_bars]

    records = []
    for ev, score in zip(events, hawkish_scores):
        start_idx = next(
            (idx for idx, t_open in enumerate(bar_opens) if t_open >= ev.sentence_timestamp),
            None,
        )
        if start_idx is None or start_idx + horizon_minutes - 1 >= len(sorted_bars):
            continue

        entry_bar = sorted_bars[start_idx]
        exit_bar = sorted_bars[start_idx + horizon_minutes - 1]
        pair = assert_xauusd_pair(entry_bar.pair)
        pip_size = get_fx_pip_size(pair)
        polarity = compute_fx_pair_hawkish_polarity(pair, ev.central_bank)

        reaction_bps = float(np.log(exit_bar.close_price / entry_bar.open_price) * 1e4)
        reaction_usd_oz = float(exit_bar.close_price - entry_bar.open_price)
        reaction_pips = float(reaction_usd_oz / pip_size)

        records.append(
            {
                "speaker": ev.speaker,
                "central_bank": ev.central_bank,
                "pair": pair,
                "sentence_timestamp": ev.sentence_timestamp,
                "reaction_bar_open": entry_bar.bar_open_time,
                "reaction_bar_close": exit_bar.bar_close_time,
                "hawkish_score": float(score),
                "xauusd_bullish_score": float(score * polarity),
                "fx_pair_signed_score": float(score * polarity),
                "reaction_bps": float(reaction_bps),
                "reaction_usd_oz": float(reaction_usd_oz),
                "reaction_pips": float(reaction_pips),
                "volume_sum": float(
                    sum(b.volume for b in sorted_bars[start_idx : start_idx + horizon_minutes])
                ),
            }
        )

    return pd.DataFrame(records)


def estimate_hawkish_market_impact(aligned_df: pd.DataFrame, maxlags: int = 2) -> Dict[str, float]:
    """
    Estimate the intraday `XAU/USD` spot reaction (in bps, $/oz, and pips) per unit of Hawkish FOMC shock
    using OLS with Newey-West HAC standard errors via `statsmodels` (or `scipy.stats` fallback).
    """
    if aligned_df.empty or len(aligned_df) < 5:
        raise ValueError("Need at least 5 aligned sentence-bar observations for regression.")

    x_raw = aligned_df["hawkish_score"].to_numpy(dtype=float)
    y_bps = aligned_df["reaction_bps"].to_numpy(dtype=float)
    y_pips = (
        aligned_df["reaction_pips"].to_numpy(dtype=float)
        if "reaction_pips" in aligned_df.columns
        else y_bps
    )
    y_usd = (
        aligned_df["reaction_usd_oz"].to_numpy(dtype=float)
        if "reaction_usd_oz" in aligned_df.columns
        else y_pips * 0.01
    )

    try:
        import statsmodels.api as sm

        X = sm.add_constant(x_raw)
        model = sm.OLS(y_bps, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        pip_model = sm.OLS(y_pips, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        usd_model = sm.OLS(y_usd, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        return {
            "intercept_bps": float(model.params[0]),
            "beta_hawkish_bps": float(model.params[1]),
            "beta_hawkish_pips": float(pip_model.params[1]),
            "beta_hawkish_usd_oz": float(usd_model.params[1]),
            "t_stat_hac": float(model.tvalues[1]),
            "p_value_hac": float(model.pvalues[1]),
            "r_squared": float(model.rsquared),
            "n_obs": int(model.nobs),
        }
    except ImportError:
        from scipy import stats

        res = stats.linregress(x_raw, y_bps)
        pip_res = stats.linregress(x_raw, y_pips)
        usd_res = stats.linregress(x_raw, y_usd)
        t_stat = float(res.slope / max(res.stderr, 1e-12))
        return {
            "intercept_bps": float(res.intercept),
            "beta_hawkish_bps": float(res.slope),
            "beta_hawkish_pips": float(pip_res.slope),
            "beta_hawkish_usd_oz": float(usd_res.slope),
            "t_stat_hac": t_stat,
            "p_value_hac": float(res.pvalue),
            "r_squared": float(res.rvalue**2),
            "n_obs": int(len(y_bps)),
        }


def _run_demo() -> None:
    """Self-contained demonstration of Topic 1 using real historical 1-minute XAU/USD CSV data."""
    print("=== Topic 1: Central Bank XAU/USD NLP (Real Historical 1m XAU/USD CSV + FOMC Timestamps) ===")
    extractor = VintageGuardNLPExtractor(model_id="ProsusAI/finbert")

    # 1. Demonstrate look-ahead bias blocker on a 2018 FOMC speech with a 2019-vintage model
    try:
        extractor.validate_event_timestamp(datetime(2018, 12, 19, 14, 30, 0))
    except LookAheadBiasError as exc:
        print(f"[GUARDRAIL VERIFIED] Blocked look-ahead model usage: {exc}")

    # 2. Load real historical 1-minute XAU/USD bars (`data/xauusd_1m_real.csv`) & FOMC events (`data/xauusd_fomc_events_real.csv`)
    real_bars, real_events = load_real_xauusd_1m_bars_and_fomc_events()
    scores = extractor.score_hawkish_dovish(real_events)

    aligned = align_sentences_to_bars(real_events, scores, real_bars, horizon_minutes=2)
    stats = estimate_hawkish_market_impact(aligned)
    print(
        f"Loaded {len(real_bars):,d} real 1m XAU/USD bars from `{XAUUSD_1M_CSV.name}` & "
        f"aligned {stats['n_obs']} FOMC sentences:"
    )
    print(
        f"  Beta (Hawkish Fed Shock -> XAU/USD Spot): {stats['beta_hawkish_bps']:+.2f} bps "
        f"({stats['beta_hawkish_usd_oz']:+.2f} $/oz | {stats['beta_hawkish_pips']:+.0f} pips/shock | "
        f"HAC t-stat: {stats['t_stat_hac']:.2f}, R^2: {stats['r_squared']:.3f})"
    )


if __name__ == "__main__":
    _run_demo()
