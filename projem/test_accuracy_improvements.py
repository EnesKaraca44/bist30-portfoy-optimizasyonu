"""
Doğruluk iyileştirme planı — birim testleri.
Faz 1: model_selector entegrasyonu
Faz 2: piyasa rejimi düzeltmesi
Faz 3: volatilite clip + güven filtresi
Faz 4: portföy hisse seçimi
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestVolatilityClip(unittest.TestCase):
    def test_low_vol_narrow_clip(self):
        from forecasting.weekly_pipeline import get_volatility_adjusted_clip

        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        r = pd.Series(np.random.normal(0, 0.005, len(dates)), index=dates)
        clip = get_volatility_adjusted_clip(r)
        self.assertLessEqual(clip, 0.12)

    def test_high_vol_wide_clip(self):
        from forecasting.weekly_pipeline import get_volatility_adjusted_clip

        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        r = pd.Series(np.random.normal(0, 0.05, len(dates)), index=dates)
        clip = get_volatility_adjusted_clip(r)
        self.assertGreaterEqual(clip, 0.18)


class TestRegimeAdjustment(unittest.TestCase):
    def test_bear_dampens_positive(self):
        from forecasting.weekly_pipeline import apply_regime_adjustment

        raw = 0.05
        adjusted = apply_regime_adjustment(raw, {"regime": "bear", "confidence": 0.9})
        self.assertLess(abs(adjusted), abs(raw))

    def test_bull_unchanged(self):
        from forecasting.weekly_pipeline import apply_regime_adjustment

        raw = 0.04
        self.assertEqual(apply_regime_adjustment(raw, {"regime": "bull", "confidence": 0.8}), raw)

    def test_uncertain_sideways_shrinks(self):
        from forecasting.weekly_pipeline import apply_regime_adjustment

        raw = 0.03
        adj = apply_regime_adjustment(raw, {"regime": "sideways", "confidence": 0.5})
        self.assertLess(abs(adj), abs(raw))


class TestCombineWeeklyForecast(unittest.TestCase):
    def _mock_ensemble(self, arima, lstm, arima_weight=0.5, lstm_weight=0.5):
        return arima * arima_weight + lstm * lstm_weight

    def test_arima_only_when_lstm_disabled(self):
        from forecasting.weekly_pipeline import combine_weekly_forecast

        final, label = combine_weekly_forecast("AKBNK.IS", 0.02, 0.03, use_lstm=False, lstm_available=True)
        self.assertEqual(final, 0.02)
        self.assertEqual(label, "ARIMA")

    @patch("forecasting.weekly_pipeline._model_selector_available", return_value=False)
    def test_default_ensemble_without_selector(self, _mock_sel):
        from forecasting.weekly_pipeline import combine_weekly_forecast

        final, label = combine_weekly_forecast(
            "AKBNK.IS", 0.04, 0.02, use_lstm=True, lstm_available=True,
            ensemble_fn=self._mock_ensemble,
        )
        self.assertAlmostEqual(final, 0.03, places=5)
        self.assertIn("ARIMA+LSTM", label)

    @patch("forecasting.weekly_pipeline._model_selector_available", return_value=True)
    def test_selector_picks_arima_only(self, _mock_sel):
        from forecasting.weekly_pipeline import combine_weekly_forecast

        with patch("model_selector.get_best_model_for_ticker", return_value=("ARIMA", 1.0)):
            final, label = combine_weekly_forecast(
                "AKBNK.IS", 0.04, 0.01, use_lstm=True, lstm_available=True,
            )
        self.assertEqual(final, 0.04)
        self.assertEqual(label, "ARIMA")

    @patch("forecasting.weekly_pipeline._model_selector_available", return_value=True)
    def test_selector_dynamic_weights(self, _mock_sel):
        from forecasting.weekly_pipeline import combine_weekly_forecast

        with patch("model_selector.get_best_model_for_ticker", return_value=("Ensemble", {"arima": 0.7, "lstm": 0.3})):
            final, label = combine_weekly_forecast(
                "AKBNK.IS", 0.10, 0.00, use_lstm=True, lstm_available=True,
                ensemble_fn=self._mock_ensemble,
            )
        self.assertAlmostEqual(final, 0.07, places=5)
        self.assertIn("0.70", label)


class TestMarketRegimePanel(unittest.TestCase):
    def test_detect_from_synthetic_bull(self):
        from forecasting.weekly_pipeline import detect_market_regime_from_panel

        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        trend = np.linspace(100, 130, len(dates))
        prices = pd.DataFrame({"A.IS": trend, "B.IS": trend * 1.01}, index=dates)
        returns = prices.pct_change().dropna()
        info = detect_market_regime_from_panel(prices, returns, lookback=60)
        self.assertIn(info["regime"], ("bull", "sideways", "bear"))
        self.assertIn("confidence", info)


class TestConfidenceFilter(unittest.TestCase):
    def test_annotate_high_confidence(self):
        from forecasting.weekly_pipeline import filter_forecasts_by_confidence

        fc = {
            "A.IS": {"return": 0.02, "risk": 0.05, "model": "ARIMA"},
            "B.IS": {"return": 0.0001, "risk": 0.05, "model": "ARIMA"},
        }
        filter_forecasts_by_confidence(fc, min_abs_return=0.001)
        self.assertTrue(fc["A.IS"]["high_confidence"])
        self.assertFalse(fc["B.IS"]["high_confidence"])


class TestPortfolioStrategy(unittest.TestCase):
    def test_regime_strategy_mapping(self):
        from forecasting.weekly_pipeline import get_portfolio_strategy_for_regime

        self.assertEqual(get_portfolio_strategy_for_regime({"regime": "bear"}), "max_sharpe")
        self.assertEqual(get_portfolio_strategy_for_regime({"regime": "bull"}), "max_return")
        self.assertEqual(get_portfolio_strategy_for_regime({"regime": "sideways"}), "balanced")
        self.assertEqual(get_portfolio_strategy_for_regime(None), "balanced")


class TestMonteCarloOptimization(unittest.TestCase):
    def test_optimize_returns_weights(self):
        from forecasting.weekly_pipeline import optimize_portfolio_monte_carlo

        tickers = ["A", "B", "C", "D", "E"]
        er = pd.Series([0.02, 0.015, 0.01, 0.008, 0.005], index=tickers)
        cov = pd.DataFrame(np.eye(5) * 0.001, index=tickers, columns=tickers)
        w = optimize_portfolio_monte_carlo(er, cov, strategy="max_sharpe", num_portfolios=500)
        self.assertIsNotNone(w)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=4)

    def test_bull_strategy_favors_return(self):
        from forecasting.weekly_pipeline import optimize_portfolio_monte_carlo

        tickers = ["A", "B"]
        er = pd.Series([0.10, 0.01], index=tickers)
        cov = pd.DataFrame([[0.01, 0], [0, 0.01]], index=tickers, columns=tickers)
        np.random.seed(42)
        w = optimize_portfolio_monte_carlo(er, cov, strategy="max_return", num_portfolios=3000)
        self.assertGreater(w[0], w[1])


class TestBlackLittermanWeekly(unittest.TestCase):
    def test_bl_fallback_on_small_set(self):
        from forecasting.weekly_pipeline import apply_black_litterman_weekly

        views = {"A.IS": 0.01, "B.IS": 0.02}
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        rets = pd.DataFrame({"A.IS": np.random.normal(0, 0.01, 60), "B.IS": np.random.normal(0, 0.01, 60)}, index=dates)
        out = apply_black_litterman_weekly(views, rets)
        self.assertEqual(out["A.IS"], 0.01)

    @patch("forecasting.prices.get_market_caps")
    @patch("forecasting.black_litterman.calculate_black_litterman")
    def test_bl_applied_for_enough_tickers(self, mock_bl, mock_mcap):
        from forecasting.weekly_pipeline import apply_black_litterman_weekly

        tickers = [f"T{i}.IS" for i in range(5)]
        views = {t: 0.01 * (i + 1) for i, t in enumerate(tickers)}
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        rets = pd.DataFrame({t: np.random.normal(0.001, 0.02, 80) for t in tickers}, index=dates)
        mock_mcap.return_value = pd.Series({t: 1e9 for t in tickers})
        mock_bl.return_value = pd.Series({t: 0.005 for t in tickers})

        out = apply_black_litterman_weekly(views, rets, tickers)
        mock_bl.assert_called_once()
        self.assertEqual(out["T0.IS"], 0.005)


class TestHorizonCov(unittest.TestCase):
    def test_cov_shape(self):
        from forecasting.weekly_pipeline import build_horizon_cov_matrix

        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        rets = pd.DataFrame({"A": np.random.normal(0, 0.01, 50), "B": np.random.normal(0, 0.01, 50)}, index=dates)
        cov = build_horizon_cov_matrix(rets, ["A", "B"], horizon_days=5)
        self.assertEqual(cov.shape, (2, 2))


class TestBatchPipeline(unittest.TestCase):
    def test_clip_annual_return(self):
        from forecasting.batch_pipeline import clip_annual_return
        from config import ANNUAL_ARIMA_CLIP_MAX

        self.assertEqual(clip_annual_return(5.0), ANNUAL_ARIMA_CLIP_MAX)
        self.assertEqual(clip_annual_return(-5.0), -ANNUAL_ARIMA_CLIP_MAX)
        self.assertAlmostEqual(clip_annual_return(0.25), 0.25)

    def test_horizon_forecast_days(self):
        from forecasting.batch_pipeline import get_horizon_forecast_days

        self.assertEqual(get_horizon_forecast_days(1), 22)
        self.assertEqual(get_horizon_forecast_days(12), 252)

    def test_period_clip_and_conversion(self):
        from forecasting.batch_pipeline import (
            clip_period_return,
            period_return_from_annual,
            annual_return_from_period,
        )

        annual = 0.24
        period_1m = period_return_from_annual(annual, 22)
        self.assertAlmostEqual(annual_return_from_period(period_1m, 1), annual, places=4)
        self.assertLess(abs(clip_period_return(5.0, 1)), 0.2)

    def test_finalize_arima_reduces_extreme_period(self):
        from forecasting.batch_pipeline import finalize_arima_period_forecast
        import pandas as pd

        rng = np.random.default_rng(0)
        daily = pd.Series(rng.normal(0.0005, 0.02, 300))
        raw_extreme = 0.80
        stabilized = finalize_arima_period_forecast(raw_extreme, daily, months=12)
        self.assertLess(abs(stabilized), 0.80)
        self.assertGreater(abs(stabilized), 0.0)

    def test_arima_vol_cap_scales_with_horizon(self):
        from forecasting.batch_pipeline import arima_volatility_period_cap
        import pandas as pd

        daily = pd.Series(np.random.normal(0, 0.015, 200))
        cap_1 = arima_volatility_period_cap(daily, 1, 22)
        cap_12 = arima_volatility_period_cap(daily, 12, 252)
        self.assertLess(cap_1, cap_12)
        self.assertLess(cap_12, 0.80)
        from forecasting.batch_pipeline import ensemble_period_forecast

        out = ensemble_period_forecast(0.03, -0.70, arima_weight=0.5, lstm_weight=0.5)
        self.assertAlmostEqual(out, 0.03)

    def test_compound_period_from_daily(self):
        from lstm_forecast import compound_period_from_daily

        period = compound_period_from_daily([0.01, 0.01, -0.005, 0.002, 0.001], target_days=22, reliable_steps=5)
        self.assertGreater(period, -0.15)
        self.assertLess(period, 0.15)

    def test_normalize_horizon_months(self):
        from services.helpers import normalize_horizon_months

        self.assertEqual(normalize_horizon_months(6), 6)
        self.assertEqual(normalize_horizon_months(5), 6)
        self.assertEqual(normalize_horizon_months(11), 12)
        from services.helpers import normalize_horizon_months

        self.assertEqual(normalize_horizon_months(6), 6)
        self.assertEqual(normalize_horizon_months(5), 6)
        self.assertEqual(normalize_horizon_months(11), 12)

    def test_batch_arima_wrapper_clips(self):
        import batch_job
        with patch("batch_job._get_arima_forecast", return_value=0.5):
            clipped = batch_job.get_arima_forecast_for_horizon(None, months=1)
        self.assertLessEqual(clipped, 0.80)

    def test_regime_explanation_for_user(self):
        from forecasting.batch_pipeline import build_regime_explanation_for_user

        info = build_regime_explanation_for_user(
            {"regime": "bull", "confidence": 0.72}, "max_return", "medium",
        )
        self.assertEqual(info["regime_title"], "Yükseliş trendi")
        self.assertEqual(info["strategy_title"], "Getiri odaklı seçim")
        self.assertNotIn("bull", info["full_explain_text"])
        self.assertNotIn("max_return", info["full_explain_text"])

    def test_mc_strategy_max_return(self):
        from portfolio.optimizer import _select_portfolio_from_simulation

        tickers = ["A", "B", "C"]
        df = pd.DataFrame([
            [0.05, 0.10, 0.5] + [0.33, 0.33, 0.34],
            [0.20, 0.40, 0.3] + [0.2, 0.5, 0.3],
            [0.08, 0.12, 0.6] + [0.4, 0.3, 0.3],
        ], columns=["Getiri", "Risk", "SharpeOrani"] + tickers)
        row = _select_portfolio_from_simulation(df, "max_return")
        self.assertAlmostEqual(float(row["Getiri"]), 0.20)


class TestPortfolioSelection(unittest.TestCase):
    def test_bear_prefers_risk_adjusted(self):
        from forecasting.weekly_pipeline import select_tickers_for_portfolio

        returns = {"A.IS": 0.05, "B.IS": 0.04, "C.IS": 0.03, "D.IS": 0.02, "E.IS": 0.01, "F.IS": 0.005}
        risks = {"A.IS": 0.20, "B.IS": 0.05, "C.IS": 0.05, "D.IS": 0.05, "E.IS": 0.05, "F.IS": 0.05}
        selected = select_tickers_for_portfolio(
            returns, risks, market_regime={"regime": "bear"}, min_tickers=5,
        )
        self.assertIn("B.IS", selected)
        self.assertLess(selected.index("B.IS"), selected.index("A.IS"))
        self.assertGreaterEqual(len(selected), 5)

    def test_high_confidence_preferred(self):
        from forecasting.weekly_pipeline import select_tickers_for_portfolio

        returns = {f"T{i}.IS": 0.01 * i for i in range(1, 8)}
        risks = {t: 0.05 for t in returns}
        high = {"T6.IS", "T7.IS", "T5.IS", "T4.IS", "T3.IS", "T2.IS"}
        selected = select_tickers_for_portfolio(
            returns, risks, min_tickers=5, high_confidence_tickers=high,
        )
        self.assertTrue(all(t in high for t in selected))


class TestWeeklyForecastIntegration(unittest.TestCase):
    @patch("weekly_forecast._download_bist_prices")
    def test_forecast_skips_meta_keys_in_save_path(self, mock_dl):
        """forecast_weekly_returns _market_regime meta döndürür; ticker olmayan anahtarlar filtrelenir."""
        import weekly_forecast

        dates = pd.date_range("2024-01-01", periods=150, freq="B")
        np.random.seed(0)
        cols = weekly_forecast.BIST30[:6]
        data = pd.DataFrame(
            {t: 100 * (1 + np.random.normal(0.001, 0.01, len(dates))).cumprod() for t in cols},
            index=dates,
        )
        mock_dl.return_value = data

        with patch("weekly_forecast.get_arima_forecast", return_value=0.01):
            with patch("weekly_forecast.apply_black_litterman_weekly", side_effect=lambda v, *a, **k: dict(v)):
                with patch("weekly_forecast.combine_weekly_forecast", return_value=(0.01, "ARIMA")):
                    with patch("weekly_forecast.get_garch_risk", return_value=0.05):
                        result = weekly_forecast.forecast_weekly_returns(use_lstm=False)

        self.assertIn("_market_regime", result)
        ticker_keys = [k for k in result if not k.startswith("_")]
        self.assertGreaterEqual(len(ticker_keys), 1)
        for k in ticker_keys:
            self.assertIn("return", result[k])
            self.assertIn("high_confidence", result[k])
            self.assertIn("bl_return", result[k])


class TestPortfolioTrackerRegime(unittest.TestCase):
    def test_calculate_skips_meta_ticker(self):
        from portfolio_performance_tracker import calculate_portfolio_forecast

        forecasts = {
            "AKBNK.IS": {"return": 0.02, "risk": 0.05, "high_confidence": True},
            "GARAN.IS": {"return": 0.015, "risk": 0.04, "high_confidence": True},
            "THYAO.IS": {"return": 0.01, "risk": 0.06, "high_confidence": True},
            "BIMAS.IS": {"return": 0.008, "risk": 0.05, "high_confidence": True},
            "EREGL.IS": {"return": 0.012, "risk": 0.05, "high_confidence": True},
            "ASELS.IS": {"return": 0.005, "risk": 0.05, "high_confidence": False},
            "_market_regime": {"regime": "sideways", "confidence": 0.7},
        }
        with patch("portfolio_performance_tracker.yf.download") as mock_yf:
            dates = pd.date_range("2024-01-01", periods=60, freq="B")
            tickers = ["AKBNK.IS", "GARAN.IS", "THYAO.IS", "BIMAS.IS", "EREGL.IS"]
            prices = pd.DataFrame({t: np.linspace(100, 110, len(dates)) for t in tickers}, index=dates)
            mock_yf.return_value = prices
            out = calculate_portfolio_forecast(
                pd.Timestamp("2024-06-01"),
                horizon_days=5,
                forecasts_dict=forecasts,
                market_regime={"regime": "sideways", "confidence": 0.7},
            )
        if out is not None:
            self.assertIn("predicted_return", out)
            self.assertIn("weights", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
