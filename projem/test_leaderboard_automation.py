"""Lider tablosu otomasyon testleri."""
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from portfolio_vs_ai_scorer import (
    compute_weighted_return,
    get_ai_benchmark_return,
    run_leaderboard_scoring,
)


class TestComputeWeightedReturn(unittest.TestCase):
    def test_equal_weights(self):
        returns = pd.Series({"A.IS": 0.10, "B.IS": -0.05})
        weights = {"A.IS": 0.5, "B.IS": 0.5}
        result = compute_weighted_return(weights, returns)
        self.assertAlmostEqual(result, 0.025)

    def test_missing_ticker_skipped(self):
        returns = pd.Series({"A.IS": 0.10})
        weights = {"A.IS": 0.5, "MISSING.IS": 0.5}
        self.assertAlmostEqual(compute_weighted_return(weights, returns), 0.05)


class TestLeaderboardScoring(unittest.TestCase):
    def _make_db(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        path = tmp.name
        tmp.close()
        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE beklenen_getiriler (
                ticker TEXT PRIMARY KEY,
                tahmini_yillik_getiri REAL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE user_portfolios (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                weights_json TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                week_start_date TEXT,
                week_end_date TEXT,
                user_return REAL,
                ai_return REAL,
                is_winner INTEGER
            )
            """
        )
        for t, r in [("A.IS", 0.5), ("B.IS", 0.4), ("C.IS", 0.3), ("D.IS", 0.2), ("E.IS", 0.1)]:
            c.execute("INSERT INTO beklenen_getiriler VALUES (?, ?)", (t, r))
        weights = json.dumps({"A.IS": 0.5, "B.IS": 0.5})
        c.execute("INSERT INTO user_portfolios (user_id, weights_json) VALUES (?, ?)", (1, weights))
        conn.commit()
        return conn, path

    def test_scores_user_and_writes_leaderboard(self):
        conn, path = self._make_db()
        returns = pd.Series({"A.IS": 0.08, "B.IS": 0.02, "C.IS": 0.01, "D.IS": -0.01, "E.IS": 0.0})
        try:
            result = run_leaderboard_scoring(
                end_date=pd.Timestamp("2026-07-04"),
                conn=conn,
                actual_returns=returns,
            )
            self.assertTrue(result["success"])
            self.assertEqual(result["users_scored"], 1)
            self.assertIsNotNone(result["ai_return"])

            c = conn.cursor()
            c.execute("SELECT user_return, ai_return, is_winner FROM leaderboard")
            row = c.fetchone()
            self.assertIsNotNone(row)
            self.assertAlmostEqual(row[0], 0.05)  # 0.5*0.08 + 0.5*0.02
        finally:
            conn.close()
            import os
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_no_users_returns_success_zero(self):
        conn, path = self._make_db()
        c = conn.cursor()
        c.execute("DELETE FROM user_portfolios")
        conn.commit()
        returns = pd.Series({"A.IS": 0.05})
        try:
            result = run_leaderboard_scoring(conn=conn, actual_returns=returns)
            self.assertTrue(result["success"])
            self.assertEqual(result["users_scored"], 0)
        finally:
            conn.close()
            import os
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_no_price_data_fails_gracefully(self):
        with patch("portfolio_vs_ai_scorer.fetch_weekly_actual_returns", return_value=None):
            result = run_leaderboard_scoring()
        self.assertFalse(result["success"])


class TestWeeklyForecastIntegration(unittest.TestCase):
    def test_weekly_job_calls_leaderboard_scorer(self):
        mock_result = {"success": True, "users_scored": 0, "message": "test"}
        with patch("portfolio_vs_ai_scorer.run_leaderboard_scoring", return_value=mock_result) as mock_lb, \
             patch("weekly_forecast.forecast_weekly_returns", return_value={"AKBNK.IS": {"return": 0.01, "risk": 0.05, "model": "ARIMA"}}), \
             patch("weekly_forecast.save_weekly_forecasts"), \
             patch("portfolio_performance_tracker.calculate_portfolio_forecast", return_value=None):
            from weekly_forecast import run_weekly_forecast_job
            run_weekly_forecast_job(use_lstm=False)
            mock_lb.assert_called_once()


if __name__ == "__main__":
    unittest.main()
