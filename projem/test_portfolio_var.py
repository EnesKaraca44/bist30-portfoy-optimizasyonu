"""Portföy VaR unit testleri."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestPortfolioVar(unittest.TestCase):
    def test_known_three_asset_portfolio(self):
        from forecasting.risk_metrics import calculate_portfolio_var

        tickers = ["A", "B", "C"]
        w = np.array([0.4, 0.35, 0.25])
        er = pd.Series([0.02, 0.015, 0.01], index=tickers)
        cov = pd.DataFrame(
            np.diag([0.0025, 0.0036, 0.0049]),
            index=tickers,
            columns=tickers,
        )
        result = calculate_portfolio_var(w, er, cov, confidence=0.95)

        mu_p = float(np.dot(w, er.values))
        sigma_p = float(np.sqrt(np.dot(w.T, np.dot(cov.values, w))))
        expected_var = max(-(mu_p - 1.645 * sigma_p), 0.0)

        self.assertAlmostEqual(result["expected_return"], mu_p, places=6)
        self.assertAlmostEqual(result["portfolio_vol"], sigma_p, places=6)
        self.assertAlmostEqual(result["var_95"], expected_var, places=6)
        self.assertGreater(result["probability_up"], 0.5)

    def test_weights_normalized(self):
        from forecasting.risk_metrics import calculate_portfolio_var

        w = np.array([2.0, 2.0])
        er = np.array([0.01, 0.01])
        cov = np.eye(2) * 0.01
        r1 = calculate_portfolio_var(w, er, cov)
        r2 = calculate_portfolio_var([0.5, 0.5], er, cov)
        self.assertAlmostEqual(r1["var_95"], r2["var_95"], places=6)

    def test_dynamic_stop_loss(self):
        from forecasting.risk_metrics import calculate_dynamic_stop_loss

        low = calculate_dynamic_stop_loss(0.01)
        high = calculate_dynamic_stop_loss(0.08)
        self.assertLess(low, high)
        self.assertLessEqual(high, 0.12)


if __name__ == "__main__":
    unittest.main()
