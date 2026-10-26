"""
Unit tests verifying all 4 `XAU/USD` (Spot Gold FX) methodological guardrails and domain utilities
across the repository (compatible with both `pytest` and standard library `unittest`):
1. Central Bank XAU/USD NLP: LLM pre-training cutoff vs. FOMC speech timestamp look-ahead guard + inverse real-yield polarity.
2. XAU/USD News Microstructure: Mid-price ECN execution illusion blocker & >12x post-NFP spread widening ($/oz, pips, bps).
3. XAU/USD ML vs. GARCH Volatility: Full-dataset StandardScaler data leakage blocker & 24h COMEX/LBMA session RV aggregation.
4. XAU/USD Cross & EFP/Lease-Rate Arbitrage: Physical EFP delivery / Gold Lease-Rate squeeze circuit breaker & basis decomposition.
"""

from __future__ import annotations

from datetime import datetime
import importlib
from pathlib import Path
import tempfile
import unittest
import numpy as np

from common.validation import (
    DEFAULT_PAIR,
    DataLeakageError,
    DepegRiskError,
    ExecutionIllusionError,
    FXRegimeOrPegBreakError,
    LookAheadBiasError,
    assert_xauusd_pair,
    classify_fx_trading_session,
    get_fx_default_spot_price,
    get_fx_pip_size,
)

cb_nlp = importlib.import_module("01_central_bank_nlp.pipeline")
news_micro = importlib.import_module("02_news_microstructure.pipeline")
vol_model = importlib.import_module("03_ml_vs_garch_volatility.pipeline")
fx_cross = importlib.import_module("04_fx_cross_rate_arbitrage.pipeline")
fx_plotting = importlib.import_module("common.plotting")


class TestQuantitativeXAUUSDResearchGuardrails(unittest.TestCase):
    def test_common_xauusd_only_conventions_and_sessions(self) -> None:
        """Verify strict XAU/USD-only enforcement, $0.01/oz pip size, $2,350/oz spot, and 24-hour bullion sessions."""
        self.assertEqual(DEFAULT_PAIR, "XAU/USD")
        self.assertEqual(assert_xauusd_pair("XAU/USD"), "XAU/USD")
        self.assertEqual(assert_xauusd_pair("xauusd"), "XAU/USD")
        self.assertAlmostEqual(get_fx_pip_size("XAU/USD"), 0.01)
        self.assertAlmostEqual(get_fx_default_spot_price("XAU/USD"), 2350.00)

        # Non-XAU/USD pairs must be rejected
        with self.assertRaises(ValueError):
            assert_xauusd_pair("EUR/USD")
        with self.assertRaises(ValueError):
            get_fx_pip_size("USD/JPY")

        self.assertEqual(classify_fx_trading_session(datetime(2023, 6, 15, 3, 0)), "Tokyo_Asian")
        self.assertEqual(classify_fx_trading_session(datetime(2023, 6, 15, 9, 30)), "London_European")
        self.assertEqual(classify_fx_trading_session(datetime(2023, 6, 15, 15, 0)), "London_NY_Overlap")
        self.assertEqual(classify_fx_trading_session(datetime(2023, 6, 15, 21, 0)), "NewYork_Sydney")

    def test_topic1_llm_vintage_lookahead_and_xauusd_polarity_guardrail(self) -> None:
        """Using a 2023-vintage LLM on a 2021 FOMC speech must raise LookAheadBiasError."""
        extractor = cb_nlp.VintageGuardNLPExtractor(model_id="gtfintechlab/FOMC-RoBERTa")
        early_speech = [
            cb_nlp.SpeechSentenceEvent(
                speaker="Chair Powell",
                sentence_timestamp=datetime(2021, 11, 3, 14, 30, 0),
                text="Inflation is elevated due to supply bottlenecks.",
                central_bank="FED",
            )
        ]
        with self.assertRaises(LookAheadBiasError):
            extractor.score_hawkish_dovish(early_speech)

        # Post-cutoff speech (e.g. late 2023) must succeed
        valid_speech = [
            cb_nlp.SpeechSentenceEvent(
                speaker="Chair Powell",
                sentence_timestamp=datetime(2023, 11, 1, 14, 30, 0),
                text="Tightening and restrictive policy remain appropriate while inflation is elevated.",
                central_bank="FED",
            )
        ]
        scores = extractor.score_hawkish_dovish(valid_speech)
        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0], 0.0)

        # Hawkish FED is -1.0 polarity on XAU/USD (higher real yields & stronger USD depress XAU/USD)
        self.assertEqual(cb_nlp.compute_fx_pair_hawkish_polarity("XAU/USD", "FED"), -1.0)
        with self.assertRaises(ValueError):
            cb_nlp.compute_fx_pair_hawkish_polarity("EUR/USD", "FED")

    def test_topic2_nfp_mid_price_execution_illusion_guardrail_on_xauusd(self) -> None:
        """Attempting mid-price XAU/USD execution after NFP must raise ExecutionIllusionError."""
        ticks = news_micro.generate_synthetic_nfp_ticks(n_ticks=600, pair="XAU/USD")
        metrics = news_micro.compute_microstructure_metrics_duckdb(ticks)
        self.assertGreaterEqual(metrics["spread_widening_multiple"], 10.0)
        self.assertGreater(metrics["max_shock_spread_usd_oz"], metrics["pre_spread_usd_oz"] * 10.0)

        with self.assertRaises(ExecutionIllusionError):
            news_micro.simulate_spread_aware_execution(
                ticks,
                news_micro.OrderExecutionSpec(
                    side="buy",
                    signal_ms_from_release=25,
                    execution_price_mode="mid",
                    pair="XAU/USD",
                ),
            )

        realized = news_micro.simulate_spread_aware_execution(
            ticks,
            news_micro.OrderExecutionSpec(
                side="buy",
                signal_ms_from_release=25,
                latency_ms=15,
                order_size_lots=10.0,
                execution_price_mode="spread_and_depth_sweep",
                pair="XAU/USD",
            ),
        )
        self.assertGreater(realized["execution_cost_drag_bps"], 6.0)
        self.assertLess(
            realized["realized_spread_swept_pnl_bps"],
            realized["mid_price_illusion_pnl_bps"],
        )

    def test_topic3_standard_scaler_data_leakage_and_24h_xauusd_rv_guardrail(self) -> None:
        """Fitting StandardScaler across the full XAU/USD dataset or overlapping test indices must fail."""
        rng = np.random.default_rng(7)
        X_full = rng.normal(size=(100, 4))

        with self.assertRaises(DataLeakageError):
            vol_model.TemporalLeakFreeScaler().fit_on_train_only(
                X_full,
                train_indices=np.arange(100),
                test_indices=np.arange(80, 100),
            )

        with self.assertRaises(DataLeakageError):
            vol_model.TemporalLeakFreeScaler().fit_on_train_only(
                X_full,
                train_indices=np.arange(0, 85),
                test_indices=np.arange(80, 100),
            )

        scaler = vol_model.TemporalLeakFreeScaler().fit_on_train_only(
            X_full,
            train_indices=np.arange(0, 80),
            test_indices=np.arange(81, 100),
        )
        X_train_scaled = scaler.transform_split(X_full, np.arange(0, 80))
        np.testing.assert_allclose(X_train_scaled.mean(axis=0), np.zeros(4), atol=1e-12)

        # Verify 24h XAU/USD session RV decomposition sums exactly to total daily RV
        bars_5m = vol_model._generate_synthetic_5m_bars(n_days=10, bars_per_day=288, pair="XAU/USD")
        daily_rv = vol_model.compute_daily_realized_variance_from_5m(bars_5m)
        session_sum = (
            daily_rv["rv_tokyo"]
            + daily_rv["rv_london"]
            + daily_rv["rv_london_ny_overlap"]
            + daily_rv["rv_ny_sydney"]
        )
        np.testing.assert_allclose(session_sum.to_numpy(), daily_rv["rv_daily"].to_numpy(), rtol=1e-10)

    def test_topic4_xauusd_efp_and_lease_rate_squeeze_guardrail(self) -> None:
        """Ignoring a physical EFP delivery squeeze or Gold Lease-Rate spike on XAU/USD must raise FXRegimeOrPegBreakError."""
        bars = fx_cross._generate_synthetic_fx_cross_cip_bars(n_bars=300)
        monitor = fx_cross.FXPegAndCIPRegimeMonitor(
            max_peg_deviation_bps=25.0,
            max_cip_basis_bps=40.0,
        )

        with self.assertRaises(FXRegimeOrPegBreakError):
            fx_cross.generate_fx_cross_rate_arbitrage_signals(
                bars,
                monitor,
                ignore_regime_guardrail=True,
            )

        with self.assertRaises(DepegRiskError):
            fx_cross.generate_fx_cross_rate_arbitrage_signals(
                bars,
                monitor,
                ignore_depeg_guardrail=True,
            )

        res = fx_cross.generate_fx_cross_rate_arbitrage_signals(
            bars,
            monitor,
            ignore_regime_guardrail=False,
        )
        frame = res["frame"]
        self.assertGreater(res["regime_bars_halted"], 0)
        self.assertTrue(np.all(frame.loc[frame["is_regime_breach"], "position"] == 0.0))

    def test_xauusd_comparison_chart_generation(self) -> None:
        """Verify all 5 XAU/USD comparison charts and master dashboard generate cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_paths = fx_plotting.generate_all_fx_comparison_charts(Path(tmpdir))
            self.assertEqual(len(chart_paths), 5)
            for path in chart_paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
