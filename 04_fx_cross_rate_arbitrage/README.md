# Pillar 4: `XAU/USD` Cross-Venue & EFP / Gold Lease-Rate Arbitrage (`04_fx_cross_rate_arbitrage`)

## Overview
- **Primary Libraries:** `vectorbt` (backtesting), `statsmodels` (Engle-Granger / ADF cointegration & OLS hedge estimation), `scipy`
- **Instrument & Granularity:** 1-minute to 15-minute `XAU/USD` (Spot Gold, $\$/\text{troy oz}$) & OTC Gold Forward / COMEX Exchange-for-Physical (EFP) Swap bars
- **Key Technical Pitfall:** **Physical EFP & Gold Lease-Rate (CIP) Squeeze Risk** — Mistaking a London-to-COMEX physical bullion delivery squeeze or Gold Lease Rate (`GOFO`) spike for a mean-reverting OTC `XAU/USD` cross-venue dislocation.

## Methodological Design
1. **`XAU/USD` Spot vs. Synthetic/EFP & Gold Lease-Rate (CIP) Basis Decomposition (`decompose_fx_cross_and_cip_basis`):**
   - **Covered Interest Parity for Spot Gold (`XAU/USD`):**
     $$F_{\text{XAU/USD}}(t, T) = S_{\text{XAU/USD}}(t) \cdot \exp\Big(\big(r_{\text{USD}}^{\text{OIS}} - r_{\text{XAU}}^{\text{lease}} + b_{\text{EFP}}\big)\tau\Big)$$
   - Separates stationary ECN `XAU/USD` inter-dealer basis (`true_cross_basis_bps`) from physical EFP delivery / lease-rate squeezes (`regime_distortion_bps`).
2. **Real-Time `XAU/USD` EFP & Lease-Rate Circuit Breaker (`FXPegAndCIPRegimeMonitor`):**
   - Monitors both physical EFP parity (`peg_deviation_bps > max_peg_deviation_bps`) and USD OIS minus Gold Lease Rate basis blowouts (`abs(cip_basis_bps) > max_cip_basis_bps`).
   - Raises `FXRegimeOrPegBreakError` if a backtest attempts to trade an unadjusted `XAU/USD` spread while ignoring the EFP/lease-rate circuit breaker, and automatically flattens positions (`position = 0.0`) during physical bullion regime breaches.
