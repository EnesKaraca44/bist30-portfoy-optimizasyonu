"""Tahmin hata düzeltme katmanı unit testleri."""
import os
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _seed_forecasts(db_path: str, ticker: str, n: int = 10, optimistic: float = 0.01):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS tahminler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            tahmin_tarihi TEXT NOT NULL,
            tahmin_horizon INTEGER NOT NULL,
            tahmini_getiri REAL,
            gerceklesen_getiri REAL,
            tahmin_hatasi REAL,
            model_tipi TEXT,
            yukselis_olasiligi REAL,
            UNIQUE(ticker, tahmin_tarihi, tahmin_horizon)
        )
        """
    )
    for i in range(n):
        pred = 0.02
        actual = pred - optimistic
        c.execute(
            """
            INSERT INTO tahminler
            (ticker, tahmin_tarihi, tahmin_horizon, tahmini_getiri, gerceklesen_getiri, tahmin_hatasi, model_tipi)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ticker, f"2025-01-{i+1:02d}", 5, pred, actual, abs(pred - actual), "ARIMA+BL"),
        )
    conn.commit()
    conn.close()


class TestForecastBias(unittest.TestCase):
    def test_insufficient_data_no_correction(self):
        import forecast_bias_correction as fbc

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            _seed_forecasts(db_path, "THYAO.IS", n=3)
            conn = sqlite3.connect(db_path)
            corrected, applied = fbc.apply_bias_correction("THYAO.IS", 0.03, conn=conn)
            conn.close()
            self.assertAlmostEqual(corrected, 0.03)
            self.assertIsNone(applied)
        finally:
            os.unlink(db_path)

    def test_optimistic_bias_reduces_forecast(self):
        import forecast_bias_correction as fbc

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            _seed_forecasts(db_path, "THYAO.IS", n=12, optimistic=0.01)
            conn = sqlite3.connect(db_path)
            bias = fbc.get_ticker_bias("THYAO.IS", conn=conn)
            corrected, applied = fbc.apply_bias_correction("THYAO.IS", 0.03, shrinkage=0.5, conn=conn)
            conn.close()
            self.assertAlmostEqual(bias, 0.01, places=4)
            self.assertAlmostEqual(applied, 0.005, places=4)
            self.assertLess(corrected, 0.03)
        finally:
            os.unlink(db_path)

    def test_bias_clipped(self):
        import forecast_bias_correction as fbc

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE tahminler (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT, tahmin_tarihi TEXT, tahmin_horizon INTEGER,
                    tahmini_getiri REAL, gerceklesen_getiri REAL,
                    tahmin_hatasi REAL, model_tipi TEXT, yukselis_olasiligi REAL
                )
                """
            )
            for i in range(10):
                c.execute(
                    "INSERT INTO tahminler (ticker, tahmin_tarihi, tahmin_horizon, tahmini_getiri, gerceklesen_getiri) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("X.IS", f"2025-02-{i+1:02d}", 5, 0.10, -0.05),
                )
            conn.commit()
            bias = fbc.get_ticker_bias("X.IS", conn=conn)
            conn.close()
            self.assertLessEqual(abs(bias), fbc.MAX_BIAS)
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    unittest.main()
