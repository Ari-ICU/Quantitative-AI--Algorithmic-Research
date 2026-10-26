"""Topic 4: G10/EM FX Cross-Rate & Covered Interest Parity (CIP) Arbitrage with Peg-Break & Funding Squeeze Guardrail."""

from .pipeline import (
    FXPegAndCIPRegimeMonitor,
    decompose_fx_cross_and_cip_basis,
    generate_fx_cross_rate_arbitrage_signals,
    generate_fx_cross_arbitrage_signals,
)

__all__ = [
    "FXPegAndCIPRegimeMonitor",
    "decompose_fx_cross_and_cip_basis",
    "generate_fx_cross_rate_arbitrage_signals",
    "generate_fx_cross_arbitrage_signals",
]
