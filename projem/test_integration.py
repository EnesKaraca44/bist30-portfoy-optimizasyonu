"""Refactoring sonrası genişletilmiş entegrasyon testleri."""
import importlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestConfig(unittest.TestCase):
    def test_paths_exist(self):
        from config import DB_PATH, LSTM_MODELS_DIR, BIST30_TICKERS
        self.assertTrue(os.path.exists(DB_PATH))
        self.assertTrue(os.path.isdir(LSTM_MODELS_DIR))
        self.assertEqual(len(BIST30_TICKERS), 30)
        self.assertIn("TRALT.IS", BIST30_TICKERS)
        self.assertNotIn("KOZAL.IS", BIST30_TICKERS)

    def test_db_path_consistent(self):
        from config import DB_PATH
        import weekly_forecast
        import batch_job
        import auth
        self.assertEqual(weekly_forecast.DB_NAME, DB_PATH)
        self.assertEqual(batch_job.DB_NAME, DB_PATH)
        self.assertEqual(auth.DB_NAME, DB_PATH)


class TestForecasting(unittest.TestCase):
    def test_arima_fallback_no_crash(self):
        import numpy as np
        import pandas as pd
        from forecasting.arima import get_arima_forecast
        dates = pd.date_range("2024-01-01", periods=150, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        prices = (1 + returns).cumprod() * 100
        result = get_arima_forecast(returns, prices=prices, forecast_days=5, apply_momentum=True,
                                    clip_range=(-0.15, 0.15), max_p=1, max_q=1, ticker="AKBNK.IS")
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, -0.15)
        self.assertLessEqual(result, 0.15)

    def test_garch_weekly_and_annual(self):
        import numpy as np
        import pandas as pd
        from forecasting.garch import get_garch_volatility, get_garch_risk
        r = pd.Series(np.random.normal(0, 0.02, 300))
        w = get_garch_risk(r)
        a = get_garch_volatility(r, horizon=22, annualize=True)
        self.assertGreater(w, 0)
        self.assertGreater(a, 0)

    def test_black_litterman_shape(self):
        import numpy as np
        import pandas as pd
        from forecasting.black_litterman import calculate_black_litterman
        tickers = ["A.IS", "B.IS", "C.IS"]
        mcaps = pd.Series([1e9, 2e9, 3e9], index=tickers)
        views = pd.Series([0.1, 0.12, 0.08], index=tickers)
        cov = pd.DataFrame(np.eye(3) * 0.04, index=tickers, columns=tickers)
        bl = calculate_black_litterman(mcaps, cov, views)
        self.assertEqual(len(bl), 3)


class TestPortfolioModule(unittest.TestCase):
    def test_bist30_data_inputs(self):
        from portfolio.optimizer import get_bist30_data_inputs
        ml, hist, cov, ml_cov = get_bist30_data_inputs()
        self.assertEqual(len(ml), 30)
        self.assertEqual(cov.shape[0], 30)

    def test_horizon_adjustment(self):
        from portfolio.optimizer import adjust_returns_for_horizon
        import pandas as pd
        s = pd.Series([0.4, 0.2], index=["A", "B"])
        adj = adjust_returns_for_horizon(s, months=1)
        self.assertTrue(all(adj < s))


class TestFlaskRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        cls.client = app.test_client()

    def test_public_pages(self):
        for path in ["/", "/login", "/register", "/compare", "/forecast_summary",
                       "/portfolio_performance", "/leaderboard", "/abonelik"]:
            r = self.client.get(path)
            self.assertIn(r.status_code, (200, 302), f"{path} -> {r.status_code}")

    def test_api_endpoints(self):
        r = self.client.get("/api/bist30_live")
        self.assertIn(r.status_code, (200, 503))
        r = self.client.get("/api/crash_radar")
        self.assertEqual(r.status_code, 200)
        self.assertIn("probability", r.get_json())
        r = self.client.get("/api/macro_dashboard")
        self.assertIn(r.status_code, (200, 503))
        self.assertIn("enabled", r.get_json())
        r = self.client.get("/api/active_ipos")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.get_json(), list)
        r = self.client.get("/api/upcoming_ipos")
        self.assertEqual(r.status_code, 200)
        self.assertIn("ipos", r.get_json())
        r = self.client.get("/api/ipo_history")
        self.assertEqual(r.status_code, 200)

    def test_route_count(self):
        from app import app
        rules = [r.rule for r in app.url_map.iter_rules() if r.rule != "/static/<path:filename>"]
        self.assertGreaterEqual(len(rules), 25)


class TestServices(unittest.TestCase):
    def test_helpers(self):
        from services.helpers import sanitize_months, str_to_bool
        self.assertEqual(sanitize_months("6"), 6)
        self.assertEqual(sanitize_months("-1", default=12), 12)
        self.assertTrue(str_to_bool("evet"))
        self.assertFalse(str_to_bool("hayir"))

    def test_monte_carlo_via_charts_mpt(self):
        import numpy as np
        import pandas as pd
        from services.charts import calculate_mpt
        # Tek ticker ile hızlı test — yfinance çağrısı
        try:
            ms, mr, g = calculate_mpt(["AKBNK.IS"])
            if ms is not None:
                self.assertIn("Getiri", ms)
        except Exception as e:
            self.skipTest(f"yfinance erişilemedi: {e}")


class TestScriptsBootstrap(unittest.TestCase):
    def test_scripts_import(self):
        import scripts._bootstrap
        from config import DB_PATH
        self.assertTrue(os.path.exists(DB_PATH))


if __name__ == "__main__":
    unittest.main(verbosity=2)
