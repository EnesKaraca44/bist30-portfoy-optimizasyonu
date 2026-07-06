"""Faz 1 — IPO tahmin motoru birim testleri."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestIpoPredictor(unittest.TestCase):
    def test_load_history_count(self):
        from ipo_predictor import load_ipo_history
        self.assertGreaterEqual(len(load_ipo_history()), 25)

    def test_koton_similarity_top(self):
        from ipo_predictor import find_similar_ipos
        similar = find_similar_ipos(4824, 15, 20, "Perakende", top_n=5)
        tickers = [s["ticker"] for s in similar]
        self.assertIn("KOTON", tickers)
        self.assertEqual(similar[0]["ticker"], "KOTON")

    def test_large_ipo_short_ceiling(self):
        from ipo_predictor import predict_ceiling_days
        result = predict_ceiling_days(size_mln=15000, inst_ratio=70, discount=12, sector="Enerji")
        self.assertTrue(result["success"])
        self.assertLessEqual(result["expected_days_avg"], 6)

    def test_small_ipo_longer_ceiling(self):
        from ipo_predictor import predict_ceiling_days
        result = predict_ceiling_days(size_mln=500, inst_ratio=10, discount=28, sector="Perakende")
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["expected_days_avg"], 2)

    def test_predict_by_ipo_id(self):
        from ipo_predictor import predict_ipo_performance
        result = predict_ipo_performance(ipo_id="SSAAT")
        self.assertTrue(result.get("success"))
        self.assertIn("similar_ipos", result)
        self.assertIn("factors", result)
        self.assertIn("disclaimer", result)
        self.assertEqual(result["ipo_info"]["id"], "SSAAT")

    def test_missing_params_error(self):
        from ipo_predictor import predict_ipo_performance
        result = predict_ipo_performance(size_mln=1000)
        self.assertIn("error", result)

    def test_response_has_range_not_single(self):
        from ipo_predictor import predict_ceiling_days
        result = predict_ceiling_days(size_mln=3200, inst_ratio=35, discount=17, sector="Enerji")
        self.assertLessEqual(result["min_days"], result["max_days"])
        self.assertIn("iş günü", result["expected_days_str"])

    def test_ceiling_strength_bounds(self):
        from ipo_predictor import predict_ceiling_days
        result = predict_ceiling_days(size_mln=3200, inst_ratio=35, discount=17)
        self.assertGreaterEqual(result["ceiling_strength"], 5)
        self.assertLessEqual(result["ceiling_strength"], 95)

    def test_get_active_ipos_has_fields(self):
        from ipo_predictor import get_active_ipos
        ipos = get_active_ipos()
        self.assertGreater(len(ipos), 0)
        first = ipos[0]
        for key in ("id", "name", "size_mln", "inst_ratio", "discount", "sector"):
            self.assertIn(key, first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
