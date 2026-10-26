# Pillar 3: `XAU/USD` ML vs. GARCH Volatility (`03_ml_vs_garch_volatility`)

## Overview
- **Primary Libraries:** `arch` (for GARCH), `pytorch-forecasting` (for Temporal Fusion Transformer / TFT), `scikit-learn`
- **Instrument & Granularity:** 5-minute 24-hour `XAU/USD` (Spot Gold) bars ($M=288$ bars/day across SGE/Tokyo, London Bullion, COMEX/LBMA Overlap, and NY/Sydney sessions) aggregated to Daily `XAU/USD` Realized Volatility ($RV_t$)
- **Key Technical Pitfall:** **Data leakage** — Normalizing features (`StandardScaler`) across the entire `XAU/USD` dataset instead of fitting the scaler strictly on the expanding training window.

## Methodological Design
1. **24-Hour Intraday 5-Minute `XAU/USD` to Session-Decomposed Daily RV Aggregation:**
   $$RV_t = \sum_{j=1}^{288} r_{t,j}^2 = RV_{t,\text{Tokyo/SGE}} + RV_{t,\text{London}} + RV_{t,\text{COMEX/LBMA Overlap}} + RV_{t,\text{NY/Sydney}}, \qquad RVol_t = \sqrt{252 \cdot RV_t}$$
   where $r_{t,j} = \ln(S_{t,j} / S_{t,j-1})$ are the $M=288$ intraday 5-minute `XAU/USD` log returns per 24-hour trading day. We also decompose **Bipower Variation ($BV_t$)** to isolate intraday macro news jumps and **Patton & Sheppard (2015) Signed Semivariances** ($RV_t^+$, $RV_t^-$) to capture asymmetric safe-haven vs. liquidation `XAU/USD` volatility spikes.
2. **Leak-Free Temporal Scaling (`TemporalLeakFreeScaler`):**
   - Wraps `sklearn.preprocessing.StandardScaler` with an explicit index ceiling `max_train_idx`.
   - Any attempt to call `.fit()` on the full `XAU/USD` dataset or on indices intersecting the test horizon raises `DataLeakageError`.
3. **Econometric vs. ML `XAU/USD` Volatility Benchmark:**
   - Compares GARCH(1,1) (`arch` library), Corsi (2009) HAR-RV + COMEX/LBMA Overlap Session RV (`13:00–17:00 UTC`), and non-linear ML/TFT regressors under the **QLIKE** (Quasi-Likelihood) loss function.
