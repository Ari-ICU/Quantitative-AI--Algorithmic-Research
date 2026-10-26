# Quantitative `XAU/USD` (Spot Gold FX) AI & Algorithmic Research

A structured multi-pillar quantitative research repository focused **exclusively on `XAU/USD` (Spot Gold vs. US Dollar)**: FOMC Real-Yield Event Studies on `XAU/USD`, High-Frequency Millisecond ECN `XAU/USD` News Microstructure, 24-Hour `XAU/USD` Realized Volatility Modeling, and `XAU/USD` Spot vs. Synthetic/EFP & Gold Lease-Rate (CIP) Basis Arbitrage.

---

## `XAU/USD` Research Matrix

| # | `XAU/USD` Research Pillar | Primary Python Libraries | Ideal Granularity | Key Technical Pitfall to Avoid |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Central Bank `XAU/USD` NLP** (`01_central_bank_nlp`) | `transformers` (HuggingFace), `torch`, `statsmodels` | 1-minute `XAU/USD` bars ($\$2,350/\text{oz}$) + FOMC sentence timestamps | **Look-ahead bias:** Using an LLM pre-trained on knowledge from after the FOMC speech date to predict how `XAU/USD` reacted on that date. |
| **2** | **`XAU/USD` News Microstructure** (`02_news_microstructure`) | `polars`, `duckdb`, `scipy` | Millisecond ECN `XAU/USD` tick data (Bid/Ask $\$/\text{oz}$ & 100-oz lots) | **Execution illusion:** Assuming you can execute an `XAU/USD` order at the mid-price during the 5 seconds after NFP/CPI when ECN bullion spreads widen $>12\times$ and L1 gold liquidity evaporates. |
| **3** | **`XAU/USD` ML vs. GARCH Volatility** (`03_ml_vs_garch_volatility`) | `arch`, `pytorch-forecasting`, `scikit-learn` | 5-minute 24h `XAU/USD` bars (288 bars/day across SGE/Tokyo, London Bullion, COMEX/LBMA Overlap, NY) aggregated to Daily `XAU/USD` RV | **Data leakage:** Normalizing features (`StandardScaler`) across the entire `XAU/USD` dataset instead of fitting the scaler strictly on the expanding training window. |
| **4** | **`XAU/USD` Cross & EFP/Lease-Rate Arbitrage** (`04_fx_cross_rate_arbitrage`) | `vectorbt`, `statsmodels`, `scipy` | 1-minute to 15-minute `XAU/USD` Spot & EFP / Gold Forward Swap Points | **Physical EFP & Lease-Rate Squeeze Risk:** Mistaking a London-to-COMEX Exchange-for-Physical (EFP) delivery squeeze or Gold Lease Rate (`GOFO`) spike for a mean-reverting OTC `XAU/USD` basis move. |

---

## Repository Architecture

```text
Quantitative-AI&-Algorithmic-Research/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── common/                              # Shared temporal, statistical, & XAU/USD validation utilities
│   ├── __init__.py
│   ├── validation.py                    # Strict XAU/USD enforcement, point-in-time & walk-forward splits
│   └── plotting.py                      # Publication-quality 4-pillar XAU/USD comparison chart generator
├── figures/                             # Generated XAU/USD PNG comparison charts & 2x2 master dashboard
├── 01_central_bank_nlp/                 # Pillar 1: FOMC Speech NLP -> 1m XAU/USD Spot Impact ($/oz & bps)
│   ├── __init__.py
│   ├── README.md
│   └── pipeline.py                      # Vintage-gated LLM embeddings + 1m XAU/USD inverse real-yield reactions
├── 02_news_microstructure/              # Pillar 2: NFP/CPI Millisecond ECN XAU/USD Book Microstructure
│   ├── __init__.py
│   ├── README.md
│   └── pipeline.py                      # DuckDB/Polars tick ingestion + Ask/Bid 100-oz lot L2 sweep simulator
├── 03_ml_vs_garch_volatility/           # Pillar 3: 24-Hour XAU/USD Realized Volatility (HAR/GARCH vs. ML)
│   ├── __init__.py
│   ├── README.md
│   └── pipeline.py                      # 288x 5m XAU/USD bars -> COMEX/LBMA Session RV + leak-free scaling
├── 04_fx_cross_rate_arbitrage/          # Pillar 4: XAU/USD Spot vs. Synthetic/EFP & Gold Lease-Rate Arbitrage
│   ├── __init__.py
│   ├── README.md
│   └── pipeline.py                      # EFP/Lease-Rate regime guard + XAU/USD cointegration basis RV
└── tests/                               # Guardrail unit tests verifying all 4 XAU/USD pitfalls
    └── test_guardrails.py
```

---

## Quickstart

### 1. Run the `XAU/USD` Guardrail Test Suite

```bash
.venv/bin/pytest tests/test_guardrails.py -v
```

### 2. Run Individual `XAU/USD` Research Pipelines

```bash
.venv/bin/python -m 01_central_bank_nlp.pipeline
.venv/bin/python -m 02_news_microstructure.pipeline
.venv/bin/python -m 03_ml_vs_garch_volatility.pipeline
.venv/bin/python -m 04_fx_cross_rate_arbitrage.pipeline
```

### 3. Generate All 4-Pillar `XAU/USD` Comparison Charts

```bash
.venv/bin/python -m common.plotting
```

Generates 5 high-resolution `XAU/USD` comparison figures in `figures/`:
- `figures/fx_research_comparison_dashboard.png` (2x2 Master `XAU/USD` Guardrail Dashboard)
- `figures/01_central_bank_nlp_comparison.png` (1m `XAU/USD` Spot Trajectory & FOMC Hawkish Shock OLS HAC Fit)
- `figures/02_news_microstructure_comparison.png` (Millisecond `XAU/USD` NFP Spread Widening & Mid-Price Illusion vs. ECN Sweep PnL)
- `figures/03_ml_vs_garch_volatility_comparison.png` (Walk-Forward `XAU/USD` HAR-XAU ML vs. GARCH(1,1) & 24h Bullion Session RV Share)
- `figures/04_fx_cross_rate_arbitrage_comparison.png` (`XAU/USD` Spot vs. Synthetic/EFP Implied Basis & Lease-Rate Circuit Breaker)
