"""Topic 2: News Microstructure with Bid/Ask Spread & Liquidity-Sweep Execution."""

from .pipeline import (
    OrderExecutionSpec,
    generate_synthetic_nfp_ticks,
    compute_microstructure_metrics_duckdb,
    simulate_spread_aware_execution,
)

__all__ = [
    "OrderExecutionSpec",
    "generate_synthetic_nfp_ticks",
    "compute_microstructure_metrics_duckdb",
    "simulate_spread_aware_execution",
]
