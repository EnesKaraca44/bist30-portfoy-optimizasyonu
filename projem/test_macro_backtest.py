"""Makro backtest raporu testleri."""
import sqlite3
import tempfile
import unittest

from macro_backtest import (
    analyze_direction_accuracy,
    build_macro_backtest_report,
    compare_forecast_snapshots,
    format_report_text,
    get_ticker_category,
)


class TestMacroBacktest(unittest.TestCase):
    def test_ticker_category(self):
        self.assertEqual(get_ticker_category("FROTO.IS"), "ihracatci")
        self.assertEqual(get_ticker_category("AKBNK.IS"), "banka")
        self.assertEqual(get_ticker_category("THYAO.IS"), "havacilik_petrol")
        self.assertEqual(get_ticker_category("KCHOL.IS"), "diger")

    def test_compare_snapshots(self):
        old = {"FROTO.IS": -0.02, "AKBNK.IS": 0.01, "BIMAS.IS": -0.05}
        new = {"FROTO.IS": 0.01, "AKBNK.IS": 0.01, "BIMAS.IS": 0.02}
        result = compare_forecast_snapshots(old, new)
        self.assertEqual(result["compared_tickers"], 3)
        self.assertEqual(result["direction_flips"], 2)
        self.assertIn("ihracatci", result["by_category"])

    def test_report_with_temp_db(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        path = tmp.name
        tmp.close()
        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE tahminler (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                tahmin_tarihi TEXT,
                tahmini_getiri REAL,
                gerceklesen_getiri REAL,
                tahmin_hatasi REAL,
                model_tipi TEXT,
                tahmin_horizon INTEGER
            )
            """
        )
        rows = [
            ("FROTO.IS", "2026-07-03", -0.019, None),
            ("AKBNK.IS", "2026-07-03", 0.012, None),
            ("FROTO.IS", "2026-07-04", -0.013, None),
            ("AKBNK.IS", "2026-07-04", 0.007, None),
            ("THYAO.IS", "2026-06-01", 0.02, 0.03),
            ("THYAO.IS", "2026-05-01", -0.01, 0.005),
        ]
        for ticker, dt, pred, actual in rows:
            c.execute(
                "INSERT INTO tahminler (ticker, tahmin_tarihi, tahmini_getiri, gerceklesen_getiri) VALUES (?,?,?,?)",
                (ticker, dt, pred, actual),
            )
        conn.commit()

        report = build_macro_backtest_report(conn, "2026-07-03", "2026-07-04")
        self.assertTrue(report["success"])
        self.assertEqual(report["comparison"]["compared_tickers"], 2)
        text = format_report_text(report)
        self.assertIn("MAKRO BACKTEST", text)

        acc = analyze_direction_accuracy(conn)
        self.assertEqual(acc["total"], 2)

        conn.close()
        import os
        try:
            os.unlink(path)
        except OSError:
            pass

    def test_insufficient_dates(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        path = tmp.name
        tmp.close()
        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute("CREATE TABLE tahminler (ticker TEXT, tahmin_tarihi TEXT, tahmini_getiri REAL, gerceklesen_getiri REAL)")
        c.execute("INSERT INTO tahminler VALUES ('A.IS', '2026-07-04', 0.01, NULL)")
        conn.commit()
        report = build_macro_backtest_report(conn)
        self.assertFalse(report["success"])
        conn.close()
        import os
        try:
            os.unlink(path)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
