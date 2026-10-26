# Pillar 1: Central Bank `XAU/USD` NLP (`01_central_bank_nlp`)

## Overview
- **Primary Libraries:** `transformers` (HuggingFace), `torch`, `statsmodels`
- **Instrument & Granularity:** 1-minute `XAU/USD` (Spot Gold, $\$/\text{troy oz}$) bars aligned with sentence-level FOMC press-conference timestamps
- **Key Technical Pitfall:** **Look-ahead bias** — Using an LLM pre-trained on knowledge from after the FOMC speech date to predict how `XAU/USD` reacted on that date.

## Methodological Design
1. **Model Vintage Registry (`KNOWN_MODEL_VINTAGES`):** Every pretrained transformer checkpoint has an explicit `knowledge_cutoff` date (the latest date present in its pre-training or fine-tuning corpus, plus release buffer).
2. **Hard Point-in-Time Enforcement (`VintageGuardNLPExtractor`):** Before encoding any sentence spoken at timestamp $t_{\text{speech}}$, the extractor verifies:
   $$t_{\text{model\_cutoff}} \le t_{\text{speech}}$$
   If a researcher attempts to score a 2021 FOMC press conference using a model trained on 2022–2023 macro transcripts (e.g., `gtfintechlab/FOMC-RoBERTa` released in 2023), `LookAheadBiasError` is raised immediately.
3. **Inverse Real-Yield & USD Polarity on `XAU/USD` (`compute_fx_pair_hawkish_polarity`):**
   Spot Gold (`XAU/USD`) is a zero-coupon reserve asset priced against the US Dollar and 10-Year US TIPS real yields:
   - A **Hawkish ($+1$)** FOMC shock raises US real yields and strengthens `USD` $\rightarrow$ `XAU/USD` drops sharply ($-1.0$ polarity).
   - A **Dovish ($-1$)** FOMC shock lowers US real yields and weakens `USD` $\rightarrow$ `XAU/USD` rallies sharply.
4. **Intraday `XAU/USD` Bar Alignment & HAC Event Study:** Sentences spoken at timestamp $\tau_i$ are mapped strictly to the subsequent completed 1-minute `XAU/USD` bar $[t_m, t_{m+1})$ where $t_m \ge \tau_i$, measuring post-utterance reaction in basis points (`reaction_bps`), dollars per ounce (`reaction_usd_oz`), and standard `XAU/USD` pips (`reaction_pips`, where $1\text{ pip} = \$0.01/\text{oz}$) with Newey-West HAC standard errors.
