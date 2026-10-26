# Pillar 2: `XAU/USD` News Microstructure (`02_news_microstructure`)

## Overview
- **Primary Libraries:** `polars`, `duckdb` (for fast out-of-core ECN tick processing), `scipy`
- **Instrument & Granularity:** Millisecond ECN `XAU/USD` (Spot Gold, $\$/\text{troy oz}$) L1/L2 tick data (`bid_px`, `ask_px`, `bid_sz_lots`, `ask_sz_lots` where $1\text{ lot} = 100\text{ troy oz}$)
- **Key Technical Pitfall:** **Execution illusion** — Assuming you can execute an `XAU/USD` order at the mid-price during the 5 seconds after US Non-Farm Payrolls (NFP) or CPI when ECN bullion spreads widen $>12\times$ and L1 gold liquidity evaporates.

## Methodological Design
1. **Zero-Mid-Price `XAU/USD` Execution Guardrail (`simulate_spread_aware_execution`):**
   - Explicitly rejects `execution_price_mode="mid"` with an `ExecutionIllusionError`.
   - Models realistic **network + matching-engine latency** ($\Delta t_{\text{latency}} > 0$ ms) so a signal triggered at $t_0$ executes against the ECN `XAU/USD` limit order book prevailing at $t_0 + \Delta t_{\text{latency}}$.
   - Evaluates **Effective Half-Spread Crossing** ($P_{\text{ask}}(t) - P_{\text{mid}}(t)$ for buys, $P_{\text{mid}}(t) - P_{\text{bid}}(t)$ for sells) in basis points, $\$/\text{oz}$, and `XAU/USD` pips ($\$0.01/\text{oz}$), plus **Top-of-Book L1/L2 Depth Depletion** (walking the ECN bullion book when institutional clip size $Q$ in 100-oz lots exceeds depleted L1 liquidity during the 5-second post-release vacuum).
2. **Vectorized DuckDB + Polars `XAU/USD` Tick Engine:**
   - Queries millisecond-level `XAU/USD` order book states around macro announcements (`[-2000ms, +10000ms]`), computing quoted spread widening ratios (`~1.15 bps` / `$0.27/oz` pre-release to `~15.5 bps` / `$3.65/oz` post-NFP), volume-weighted micro-price (`micro_px`), Order Book Imbalance (`obi`), and `scipy.stats` Kolmogorov-Smirnov regime-shift tests.
