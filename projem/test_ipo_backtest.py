"""IPO geçmiş veri doğrulama ve backtest testleri."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestIpoBacktest(unittest.TestCase):
    def test_validate_history_structure(self):
        from ipo_predictor import validate_ipo_history
        v = validate_ipo_history()
        self.assertGreaterEqual(v["kayit_sayisi"], 25)
        self.assertIn("temiz", v)
        self.assertIn("sorunlar", v)

    def test_backtest_runs_on_all_records(self):
        from ipo_predictor import run_ipo_backtest
        result = run_ipo_backtest()
        self.assertEqual(result["test_edilen"], result["kayit_sayisi"])
        self.assertGreater(result["ortalama_hata_gun"], 0)
        self.assertLess(result["ortalama_hata_gun"], 5)
        self.assertGreaterEqual(result["aralik_isabet_yuzde"], 40)

    def test_leave_one_out_excludes_self(self):
        from ipo_predictor import load_ipo_history, run_ipo_backtest
        history = load_ipo_history()
        result = run_ipo_backtest()
        by_ticker = {r["borsa_kodu"]: r for r in result["detaylar"]}
        for rec in history[:3]:
            row = by_ticker[rec["ticker"]]
            self.assertNotEqual(row.get("en_benzer"), rec["ticker"])

    def test_optimize_returns_valid_weight(self):
        from ipo_predictor import optimize_backtest_weights
        best = optimize_backtest_weights()
        self.assertGreaterEqual(best["hist_weight"], 0.5)
        self.assertLessEqual(best["hist_weight"], 0.85)


class TestIpoBacktestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        cls.client = app.test_client()

    def test_ipo_backtest_endpoint(self):
        r = self.client.get("/api/ipo_backtest")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("ortalama_hata_gun", data)
        self.assertIn("dogrulama", data)
        self.assertNotIn("detaylar", data)

    def test_get_ipo_user_tasks(self):
        from ipo_predictor import get_ipo_user_tasks
        tasks = get_ipo_user_tasks()
        self.assertGreaterEqual(tasks["ozet"]["gecmis_dogrulanmamis"], 0)
        self.assertGreaterEqual(tasks["ozet"]["yaklasan_eksik"], 2)

    def test_ipo_user_tasks_endpoint(self):
        r = self.client.get("/api/ipo_user_tasks")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("yaklasan_guncelleme", data)
        self.assertIn("ozet", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
