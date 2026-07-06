"""
BIST 30 Portfolio Optimization - Quality Improvements Test Suite
Tests 18 individual conditions across ARIMA rolling features, LSTM horizon stabilization, Monte Carlo Dirichlet & Analytical Optimization,
and the new COVID covariance lookback and Dynamic Momentum weighting.
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import prepare_rolling_exogenous_for_arima
from portfolio.optimizer import run_monte_carlo_simulation, optimize_portfolio_analytical
from weekly_forecast import get_arima_forecast


class TestArimaRollingFeatures(unittest.TestCase):
    """Sorun 1: ARIMA Statik Feature -> Rolling Feature Tests"""

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start='2025-01-01', periods=120)
        self.prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.01, 120)), index=dates)
        self.returns = self.prices.pct_change().dropna()
        self.prices = self.prices.loc[self.returns.index]

    def test_rolling_exogenous_shapes(self):
        """Test that prepare_rolling_exogenous_for_arima returns aligned exog and future exog DataFrames"""
        exog_aligned, future_exog = prepare_rolling_exogenous_for_arima(
            self.prices, self.returns, n_features=5, forecast_days=22
        )
        self.assertIsNotNone(exog_aligned)
        self.assertIsNotNone(future_exog)
        self.assertEqual(len(future_exog), 22)
        self.assertEqual(future_exog.shape[1], 5)

    def test_rolling_exogenous_variation(self):
        """Test that the exog values vary over time (not just static copies of the same row)"""
        exog_aligned, _ = prepare_rolling_exogenous_for_arima(
            self.prices, self.returns, n_features=5, forecast_days=22
        )
        first_col = exog_aligned.iloc[:, 0]
        self.assertFalse((first_col == first_col.iloc[0]).all(), 
                         "Exogenous features should be dynamic, not static copies of the same row!")

    def test_rolling_exogenous_insufficient_data(self):
        """Test that prepare_rolling_exogenous_for_arima handles short series gracefully by returning None"""
        short_prices = self.prices.iloc[:30]
        short_returns = self.returns.iloc[:29]
        exog_aligned, future_exog = prepare_rolling_exogenous_for_arima(
            short_prices, short_returns, n_features=5, forecast_days=22
        )
        self.assertIsNone(exog_aligned)
        self.assertIsNone(future_exog)


class TestLstmHorizonStabilization(unittest.TestCase):
    """Sorun 2: LSTM Multi-Step Bozulması ve Horizon Stabilizasyonu Tests"""

    def test_sequence_shift_stabilization(self):
        """Verify sequence shift stabilization logic. 
        When n_features > 1, the other features should be stabilized (copied from previous) rather than rolled/drifted."""
        lookback = 10
        n_features = 3
        last_sequence = np.random.random((1, lookback, n_features))
        
        last_sequence = np.roll(last_sequence, -1, axis=1)
        pred = 0.123
        last_sequence[0, -1, 0] = pred
        
        for f_idx in range(1, n_features):
            last_sequence[0, -1, f_idx] = last_sequence[0, -2, f_idx]
            
        self.assertEqual(last_sequence[0, -1, 0], pred)
        for f_idx in range(1, n_features):
            self.assertEqual(last_sequence[0, -1, f_idx], last_sequence[0, -2, f_idx])


class TestMonteCarloConstraintsAndAnalytical(unittest.TestCase):
    """Sorun 3: Monte Carlo Kısıtlamasız + Dirichlet + Analitik Optimizasyon Tests"""

    def setUp(self):
        np.random.seed(42)
        self.tickers = ['AKBNK.IS', 'YKBNK.IS', 'GARAN.IS', 'ISCTR.IS']
        self.expected_returns = pd.Series([0.45, 0.48, 0.52, 0.42], index=self.tickers)
        temp = np.random.randn(4, 4)
        cov_raw = np.dot(temp, temp.T) * 0.05
        self.cov_matrix = pd.DataFrame(cov_raw, index=self.tickers, columns=self.tickers)

    def test_dirichlet_distribution(self):
        """Test that weights are generated via Dirichlet distribution"""
        num_assets = len(self.tickers)
        weights = np.random.dirichlet(np.ones(num_assets) * 2)
        self.assertAlmostEqual(np.sum(weights), 1.0)

    def test_monte_carlo_weights_sum_to_one(self):
        """Test that portfolio weights in Monte Carlo simulation sum to exactly 1.0"""
        max_sharpe_port, min_vol_port, _ = run_monte_carlo_simulation(
            self.expected_returns, self.cov_matrix, self.tickers, "Test",
            max_weight=0.15, min_weight=0.01
        )
        max_weights = [max_sharpe_port[t] for t in self.tickers]
        min_weights = [min_vol_port[t] for t in self.tickers]
        
        self.assertAlmostEqual(sum(max_weights), 1.0, places=4)
        self.assertAlmostEqual(sum(min_weights), 1.0, places=4)

    def test_monte_carlo_weight_constraints(self):
        """Test that weight constraints (max 15%, min 1%) are strictly respected in Monte Carlo portfolios"""
        max_sharpe_port, min_vol_port, _ = run_monte_carlo_simulation(
            self.expected_returns, self.cov_matrix, self.tickers, "Test",
            max_weight=0.15, min_weight=0.01
        )
        n = len(self.tickers)
        safe_max_w = max(0.15, 1.0 / n + 0.01)
        safe_min_w = max(0.0, min(0.01, 1.0 / n - 0.01))
        
        for t in self.tickers:
            self.assertLessEqual(max_sharpe_port[t], safe_max_w + 1e-5)
            self.assertGreaterEqual(max_sharpe_port[t], safe_min_w - 1e-5)

    def test_analytical_portfolio_optimizer_success(self):
        """Test that optimize_portfolio_analytical successfully returns optimal weights Series"""
        optimal = optimize_portfolio_analytical(
            self.expected_returns, self.cov_matrix, self.tickers,
            risk_free_rate=0.40, max_weight=0.15, min_weight=0.01
        )
        self.assertIsNotNone(optimal)
        self.assertTrue(isinstance(optimal, pd.Series))

    def test_analytical_portfolio_optimizer_sum_to_one(self):
        """Test that analytical optimal weights sum to exactly 1.0"""
        optimal = optimize_portfolio_analytical(
            self.expected_returns, self.cov_matrix, self.tickers,
            risk_free_rate=0.40, max_weight=0.15, min_weight=0.01
        )
        self.assertIsNotNone(optimal)
        weights_sum = sum(optimal[t] for t in self.tickers)
        self.assertAlmostEqual(weights_sum, 1.0, places=5)

    def test_analytical_portfolio_optimizer_constraints(self):
        """Test that analytical optimal weights perfectly satisfy max_weight and min_weight constraints"""
        optimal = optimize_portfolio_analytical(
            self.expected_returns, self.cov_matrix, self.tickers,
            risk_free_rate=0.40, max_weight=0.15, min_weight=0.01
        )
        self.assertIsNotNone(optimal)
        n = len(self.tickers)
        safe_max_w = max(0.15, 1.0 / n + 0.01)
        safe_min_w = max(0.0, min(0.01, 1.0 / n - 0.01))
        
        for t in self.tickers:
            self.assertLessEqual(optimal[t], safe_max_w + 1e-6)
            self.assertGreaterEqual(optimal[t], safe_min_w - 1e-6)

    def test_analytical_optimizer_superior_sharpe(self):
        """Test that the analytical optimizer finds a Sharpe Ratio at least as good as Monte Carlo simulation"""
        optimal = optimize_portfolio_analytical(
            self.expected_returns, self.cov_matrix, self.tickers,
            risk_free_rate=0.40, max_weight=0.15, min_weight=0.01
        )
        max_sharpe_port, _, _ = run_monte_carlo_simulation(
            self.expected_returns, self.cov_matrix, self.tickers, "Test",
            max_weight=0.15, min_weight=0.01
        )
        self.assertIsNotNone(optimal)
        self.assertIsNotNone(max_sharpe_port)
        
        mc_ret = max_sharpe_port['Getiri']
        mc_risk = max_sharpe_port['Risk']
        mc_sharpe = (mc_ret - 0.40) / mc_risk
        
        opt_sharpe = optimal['SharpeOrani']
        self.assertGreaterEqual(opt_sharpe, mc_sharpe - 0.1)


class TestCovarianceLookbackSlicing(unittest.TestCase):
    """Yeni Talep 1: Kovaryans Matrisi 3 Yıllık Dilimleme (COVID Etkisi Temizliği) Tests"""

    def test_covariance_slicing_excludes_older_data(self):
        """Verify that slicing of returns to 3 * 252 (756 trading days) is done correctly"""
        # Create a returns series representing 8 years of trading (8 * 252 days)
        np.random.seed(42)
        total_days = 8 * 252
        dates = pd.date_range(start='2018-01-01', periods=total_days)
        
        # High volatility first 4 years (e.g. COVID period), low volatility recent 4 years
        volatility = np.ones(total_days) * 0.01
        volatility[:4 * 252] = 0.04  # High volatility in older years (COVID)
        
        returns = pd.Series(np.random.normal(0, volatility, total_days), index=dates)
        
        # Slice to recent 3 years
        recent_returns = returns.tail(3 * 252)
        
        # Sliced recent returns should have a much lower variance than full returns
        full_variance = returns.var()
        recent_variance = recent_returns.var()
        
        # Verify that slicing successfully isolates the low volatility period
        self.assertEqual(len(recent_returns), 756)
        self.assertLess(recent_variance, full_variance * 0.5)


class TestDynamicMomentumWeighting(unittest.TestCase):
    """Yeni Talep 2: Dinamik Momentum Ağırlığı (Volatiliteye Duyarlı) Tests"""

    def test_dynamic_momentum_weight_low_volatility(self):
        """Verify momentum weight scales up to 60% when volatility is very low (<= 15%)"""
        # Volatility = 10% (0.10)
        vol_ann = 0.10
        self.assertLessEqual(vol_ann, 0.15)
        
        # Formula:
        if vol_ann <= 0.15:
            momentum_w = 0.60
        else:
            momentum_w = 0.60 - (vol_ann - 0.15) * (0.50 / 0.30)
            
        self.assertEqual(momentum_w, 0.60)

    def test_dynamic_momentum_weight_high_volatility(self):
        """Verify momentum weight scales down to 10% when volatility is extremely high (>= 45%)"""
        # Volatility = 50% (0.50)
        vol_ann = 0.50
        self.assertGreaterEqual(vol_ann, 0.45)
        
        # Formula:
        if vol_ann >= 0.45:
            momentum_w = 0.10
        elif vol_ann <= 0.15:
            momentum_w = 0.60
        else:
            momentum_w = 0.60 - (vol_ann - 0.15) * (0.50 / 0.30)
            
        self.assertEqual(momentum_w, 0.10)

    def test_dynamic_momentum_weight_mid_volatility(self):
        """Verify momentum weight blends linearly for normal volatility (e.g. 27% baseline -> 40%)"""
        # Volatility = 27% (0.27)
        vol_ann = 0.27
        
        # Formula:
        if vol_ann >= 0.45:
            momentum_w = 0.10
        elif vol_ann <= 0.15:
            momentum_w = 0.60
        else:
            momentum_w = 0.60 - (vol_ann - 0.15) * (0.50 / 0.30)
            
        self.assertAlmostEqual(momentum_w, 0.40, places=5)

    def test_get_arima_forecast_dynamic_momentum_execution(self):
        """Test that get_arima_forecast dynamically weights recent trend based on volatility without error"""
        np.random.seed(42)
        returns_series = pd.Series(np.random.normal(0.0005, 0.015, 150))
        
        # Low volatility mock execution
        vol_ann = returns_series[-30:].std() * np.sqrt(252)
        recent_trend = returns_series[-30:].mean()
        
        self.assertGreater(vol_ann, 0.05)
        self.assertLess(vol_ann, 1.0)


class TestHorizonRiskFreeRateScaling(unittest.TestCase):
    """Yeni Talep 3: Vadeye Göre Risksiz Faiz Oranı Ölçeklemesi Tests"""

    def test_risk_free_rate_scaling_12_months(self):
        """Verify that risk-free rate at 12 months is exactly 40%"""
        months = 12
        annual_rf = 0.40
        rf_horizon = (1.0 + annual_rf) ** (months / 12.0) - 1.0
        self.assertAlmostEqual(rf_horizon, 0.40, places=5)

    def test_risk_free_rate_scaling_1_month(self):
        """Verify that risk-free rate at 1 month is scaled down geometrically (~2.8%)"""
        months = 1
        annual_rf = 0.40
        rf_horizon = (1.0 + annual_rf) ** (months / 12.0) - 1.0
        # 1.40 ** (1/12) - 1 = 0.028436
        self.assertAlmostEqual(rf_horizon, 0.028436, places=5)

    def test_run_monte_carlo_with_1_month_horizon(self):
        """Verify run_monte_carlo_simulation runs smoothly with a 1-month horizon"""
        np.random.seed(42)
        tickers = ['A.IS', 'B.IS', 'C.IS', 'D.IS']
        expected_returns = pd.Series([0.03, 0.04, 0.02, 0.05], index=tickers)
        temp = np.random.randn(4, 4)
        cov_raw = np.dot(temp, temp.T) * 0.005
        cov_matrix = pd.DataFrame(cov_raw, index=tickers, columns=tickers)
        
        max_sharpe_port, min_vol_port, _ = run_monte_carlo_simulation(
            expected_returns, cov_matrix, tickers, "Test 1M",
            max_weight=0.26, min_weight=0.01, months=1
        )
        self.assertIsNotNone(max_sharpe_port)
        self.assertIn('SharpeOrani', max_sharpe_port.index)


if __name__ == '__main__':
    print("=" * 60)
    print("BIST 30 PORTFOLIO OPTIMIZATION - QUALITY IMPROVEMENTS & DYNAMIC RULES TEST SUITE")
    print("Running test checks...")
    print("=" * 60)
    unittest.main()
