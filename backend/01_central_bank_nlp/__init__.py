"""Topic 1: Central Bank FX NLP with Point-in-Time Vintage Guardrails & Base/Quote FX Polarity."""

from .pipeline import (
    ModelVintageSpec,
    KNOWN_MODEL_VINTAGES,
    CENTRAL_BANK_TO_CURRENCY,
    compute_fx_pair_hawkish_polarity,
    VintageGuardNLPExtractor,
    SpeechSentenceEvent,
    IntradayBar,
    align_sentences_to_bars,
    estimate_hawkish_market_impact,
)

__all__ = [
    "ModelVintageSpec",
    "KNOWN_MODEL_VINTAGES",
    "CENTRAL_BANK_TO_CURRENCY",
    "compute_fx_pair_hawkish_polarity",
    "VintageGuardNLPExtractor",
    "SpeechSentenceEvent",
    "IntradayBar",
    "align_sentences_to_bars",
    "estimate_hawkish_market_impact",
]
