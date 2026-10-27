"""
Unit tests verifying all 4 `XAU/USD` (Spot Gold FX) methodological guardrails and domain utilities
across the repository (compatible with both `pytest` and standard library `unittest`):
1. Central Bank XAU/USD NLP: LLM pre-training cutoff vs. FOMC speech timestamp look-ahead guard + inverse real-yield polarity.
2. XAU/USD News Microstructure: Mid-price ECN execution illusion blocker & >12x post-NFP spread widening ($/oz, pips, bps).
3. XAU/USD ML vs. GARCH Volatility: Full-dataset StandardScaler data leakage blocker & 24h COMEX/LBMA session RV aggregation.
4. XAU/USD Cross & EFP/Lease-Rate Arbitrage: Physical EFP delivery / Gold Lease-Rate squeeze circuit breaker & basis decomposition.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
for _p in (str(_BACKEND_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.data_loader import (
    XAUUSD_1M_CSV,
    XAUUSD_5M_CSV,
    XAUUSD_CROSS_EFP_5M_CSV,
    XAUUSD_FOMC_EVENTS_CSV,
    XAUUSD_INTRADAY_2Y_CSV,
    XAUUSD_NFP_TICKS_CSV,
    load_real_xauusd_1m_bars_csv,
    load_real_xauusd_cross_efp_csv,
    load_real_xauusd_intraday_bars_csv,
    load_real_xauusd_ticks_csv,
    tag_xauusd_session_boundaries,
)
from common.validation import (
    DEFAULT_PAIR,
    DataLeakageError,
    DepegRiskError,
    ExecutionIllusionError,
    FXRegimeOrPegBreakError,
    LookAheadBiasError,
    assert_no_target_feature_timestamp_leakage,
    assert_xauusd_pair,
    classify_fx_trading_session,
    compute_cost_adjusted_profit_factor,
    compute_deflated_sharpe_ratio,
    get_fx_default_spot_price,
    get_fx_pip_size,
)

cb_nlp = importlib.import_module("01_central_bank_nlp.pipeline")
news_micro = importlib.import_module("02_news_microstructure.pipeline")
vol_model = importlib.import_module("03_ml_vs_garch_volatility.pipeline")
fx_cross = importlib.import_module("04_fx_cross_rate_arbitrage.pipeline")
fx_plotting = importlib.import_module("common.plotting")
macos_mt5 = importlib.import_module("common.macos_mt5_bridge")
signal_engine = importlib.import_module("backend.signal_engine")


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
        """Verify all 6 XAU/USD comparison charts (including the paper benchmark chart) generate cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_paths = fx_plotting.generate_all_fx_comparison_charts(Path(tmpdir))
            self.assertEqual(len(chart_paths), 6)
            for path in chart_paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 10_000)

    def test_real_xauusd_csv_datasets_and_loaders(self) -> None:
        """Verify all real historical XAU/USD tick and bar CSV files in `data/` exist and load cleanly."""
        for csv_file in [
            XAUUSD_1M_CSV,
            XAUUSD_FOMC_EVENTS_CSV,
            XAUUSD_NFP_TICKS_CSV,
            XAUUSD_5M_CSV,
            XAUUSD_INTRADAY_2Y_CSV,
            XAUUSD_CROSS_EFP_5M_CSV,
        ]:
            self.assertTrue(csv_file.exists(), f"Missing real XAU/USD CSV: {csv_file}")
            self.assertGreater(csv_file.stat().st_size, 500)

        df_1m = load_real_xauusd_1m_bars_csv()
        self.assertGreater(len(df_1m), 5000)
        self.assertTrue((df_1m["pair"] == "XAU/USD").all())

        df_ticks = load_real_xauusd_ticks_csv()
        self.assertGreaterEqual(len(df_ticks), 2000)
        self.assertTrue((df_ticks["ask_px"] > df_ticks["bid_px"]).all())

        df_5m = load_real_xauusd_intraday_bars_csv(XAUUSD_5M_CSV)
        self.assertGreater(len(df_5m), 10000)

        df_2y = load_real_xauusd_intraday_bars_csv(XAUUSD_INTRADAY_2Y_CSV)
        self.assertGreater(len(df_2y), 10000)

        df_cross = load_real_xauusd_cross_efp_csv()
        self.assertGreater(len(df_cross), 3000)

    def test_xauusd_blueprint_algorithmic_upgrades(self) -> None:
        """
        Verify all XAU/USD-specific algorithmic upgrades from the blueprint:
        1. Timestamp-level purge/embargo assertion & 23/5 session boundary masking.
        2. Two-Factor FOMC Policy Surprise (S_rate, S_tail) & cosine novelty weighting.
        3. Cont-Kukanov-Stoikov OFI, Liquidity Resiliency Gate (tau_res), & adverse selection cost.
        4. Finite-sample Bipower Variation (BV_t), Jump (J_t), & HAR-RV-J benchmark.
        5. Cost-of-carry EFP basis, 1D Kalman Filter dynamic beta_t, Hurst H_t, DSR, & Profit Factor.
        """
        # 1. Timestamp purge & embargo assertion
        train_targets = [datetime(2025, 5, 1, 16, 0), datetime(2025, 5, 2, 16, 0)]
        overlapping_test = [datetime(2025, 5, 2, 16, 30)]
        valid_test = [datetime(2025, 5, 5, 9, 0)]
        with self.assertRaises(DataLeakageError):
            assert_no_target_feature_timestamp_leakage(
                train_targets, overlapping_test, embargo_td=timedelta(hours=2)
            )
        assert_no_target_feature_timestamp_leakage(
            train_targets, valid_test, embargo_td=timedelta(hours=24)
        )

        # 23/5 Session boundary tagging on real 1m XAU/USD bars
        df_1m = load_real_xauusd_1m_bars_csv()
        self.assertIn("is_session_open_jump", df_1m.columns)
        self.assertIn("is_daily_maintenance_jump", df_1m.columns)
        self.assertIn("is_weekend_open_jump", df_1m.columns)
        self.assertGreater(int(df_1m["is_session_open_jump"].sum()), 1)

        # 2. Pillar 1: Two-Factor FOMC Policy Surprise & Cosine Novelty Weights
        extractor = cb_nlp.VintageGuardNLPExtractor(model_id="ProsusAI/finbert")
        real_bars, real_events = cb_nlp.load_real_xauusd_1m_bars_and_fomc_events()
        tf = extractor.score_two_factor_fomc_surprise(real_events)
        self.assertEqual(len(tf["rate_shock_score"]), len(real_events))
        self.assertEqual(len(tf["tail_safe_haven_score"]), len(real_events))
        self.assertTrue(np.all((tf["novelty_weight"] >= 0.15) & (tf["novelty_weight"] <= 1.0)))
        aligned = cb_nlp.align_sentences_to_bars(
            real_events,
            tf["rate_shock_score"],
            real_bars,
            horizon_minutes=2,
            tail_scores=tf["tail_safe_haven_score"],
            novelty_weights=tf["novelty_weight"],
        )
        stats1 = cb_nlp.estimate_hawkish_market_impact(aligned)
        self.assertIn("beta_statement_bps", stats1)
        self.assertIn("beta_qa_rate_bps", stats1)
        self.assertIn("beta_qa_tail_safe_haven_bps", stats1)
        self.assertGreater(stats1["two_factor_r_squared"], 0.5)

        # 3. Pillar 2: Tick-Rule OFI, Liquidity Resiliency Gate (tau_res), & Adverse Selection
        ticks = news_micro.load_xauusd_nfp_ticks_from_csv()
        self.assertIn("ofi_tick", ticks.columns)
        self.assertIn("ofi_rolling_50t", ticks.columns)
        gate = news_micro.evaluate_liquidity_resiliency_gate(ticks)
        self.assertGreater(gate["resiliency_recovery_ms"], 250.0)
        sim_gated = news_micro.simulate_spread_aware_execution(
            ticks,
            news_micro.OrderExecutionSpec(
                side="buy",
                signal_ms_from_release=50,
                latency_ms=20,
                enforce_resiliency_gate=True,
                pair="XAU/USD",
            ),
        )
        self.assertGreaterEqual(sim_gated["arrival_ms"], gate["resiliency_recovery_ms"])
        self.assertGreater(sim_gated["adverse_selection_cost_bps"], 0.0)

        # 4. Pillar 3: Finite-sample BV_t, Jump J_t, Safe-Haven Leverage, & HAR-RV-J benchmark
        intraday_df = vol_model.load_xauusd_intraday_bars_from_csv()
        daily_rv = vol_model.compute_daily_realized_variance_from_5m(intraday_df)
        self.assertIn("bv_vol_ann", daily_rv.columns)
        feat_df = vol_model.build_har_rv_features(daily_rv)
        self.assertIn("pos_ret_d", feat_df.columns)
        self.assertIn("bv_vol_d", feat_df.columns)
        vol_res = vol_model.run_walk_forward_volatility_comparison(feat_df, min_train_days=120, test_step_days=15)
        self.assertIn("har_rv_j_qlike", vol_res)
        self.assertIn("har_rv_j_rmse", vol_res)
        self.assertLess(vol_res["har_rv_j_qlike"], vol_res["garch_qlike"])

        # 5. Pillar 4: Cost-of-Carry EFP Basis, 1D Kalman Filter Beta, Hurst H_t, DSR, & Profit Factor
        bars_efp = fx_cross.load_xauusd_cross_efp_bars_from_csv(n_bars=400)
        monitor = fx_cross.FXPegAndCIPRegimeMonitor(max_peg_deviation_bps=25.0, max_cip_basis_bps=40.0)
        arb_res = fx_cross.generate_fx_cross_rate_arbitrage_signals(bars_efp, monitor, ignore_regime_guardrail=False)
        f_arb = arb_res["frame"]
        self.assertIn("carry_adjusted_efp_bps", f_arb.columns)
        self.assertIn("kalman_beta", f_arb.columns)
        self.assertIn("hurst_exponent", f_arb.columns)
        self.assertIn("is_structural_break_stop", f_arb.columns)
        self.assertGreaterEqual(arb_res["deflated_sharpe_ratio"], 0.0)
        self.assertLessEqual(arb_res["deflated_sharpe_ratio"], 1.0)
        self.assertGreater(arb_res["cost_adjusted_profit_factor"], 0.0)

    def test_live_4pillar_xauusd_signal_engine(self) -> None:
        """Verify the real-time 4-Pillar XAU/USD signal engine (`backend/signal_engine.py`)."""
        bridge = macos_mt5.MacOSMetaTraderBridge(mode="live_yahoo_ecn")
        bridge.initialize()
        tick = bridge.symbol_info_tick("XAU/USD")
        bars = bridge.copy_rates_from_pos("XAU/USD", macos_mt5.TIMEFRAME_M1, 0, 90, prefer_live=False)
        sig = signal_engine.compute_live_xauusd_4pillar_signal(tick=tick, bars_df=bars)

        self.assertEqual(sig["symbol"], "XAU/USD")
        self.assertIn(sig["direction"], ("BUY", "SELL"))
        self.assertGreater(float(sig["entry_market_price"]), 0.0)
        self.assertGreater(float(sig["risk_usd_oz"]), 0.0)
        self.assertIn("pillar1_macro", sig["pillars"])
        self.assertIn("pillar2_microstructure", sig["pillars"])
        self.assertIn("pillar3_volatility", sig["pillars"])
        self.assertIn("pillar4_arbitrage", sig["pillars"])

    def test_macos_mt5_bridge_and_live_xauusd_api(self) -> None:
        """Verify the macOS-native MetaTrader5-compatible bridge (`MacOSMetaTraderBridge`)."""
        bridge = macos_mt5.MacOSMetaTraderBridge(mode="live_yahoo_ecn")
        self.assertTrue(bridge.initialize())

        tick = bridge.symbol_info_tick("XAU/USD")
        self.assertEqual(tick.symbol, "XAU/USD")
        self.assertGreater(tick.ask, tick.bid)
        self.assertGreater(tick.spread_bps, 0.0)

        bars = bridge.copy_rates_from_pos("XAU/USD", macos_mt5.TIMEFRAME_M1, 0, 50, prefer_live=False)
        self.assertEqual(len(bars), 50)
        self.assertIn("is_session_open_jump", bars.columns)

        with self.assertRaises(ExecutionIllusionError):
            bridge.order_check(side="buy", volume_lots=5.0, execution_price_mode="mid")

        chk = bridge.order_check(side="buy", volume_lots=5.0)
        self.assertEqual(chk["symbol"], "XAU/USD")
        self.assertGreaterEqual(float(chk["estimated_fill_price"]), float(chk["quoted_ask"]))


if __name__ == "__main__":
    unittest.main()
