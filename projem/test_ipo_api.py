"""Faz 2 — IPO API entegrasyon testleri."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestIpoApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        cls.client = app.test_client()

    def test_upcoming_ipos(self):
        r = self.client.get("/api/upcoming_ipos")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("ipos", data)
        self.assertIn("updated_at", data)
        if data["ipos"]:
            ipo = data["ipos"][0]
            self.assertIn("size_mln", ipo)
            self.assertIn("sector", ipo)

    def test_active_ipos_alias(self):
        r = self.client.get("/api/active_ipos")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.get_json(), list)

    def test_ipo_history(self):
        r = self.client.get("/api/ipo_history?limit=5")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertGreaterEqual(len(data["records"]), 5)

    def test_predict_ipo_manual(self):
        r = self.client.post(
            "/api/predict_ipo",
            json={"size_mln": 4824, "inst_ratio": 15, "discount": 20, "sector": "Perakende"},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue(data.get("success"))
        for key in ("expected_days_str", "similar_ipos", "factors", "disclaimer", "ceiling_strength"):
            self.assertIn(key, data)

    def test_predict_ipo_by_id(self):
        r = self.client.post(
            "/api/predict_ipo",
            json={"ipo_id": "SSAAT"},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data["ipo_info"]["id"], "SSAAT")

    def test_predict_ipo_missing_params(self):
        r = self.client.post(
            "/api/predict_ipo",
            json={"size_mln": 1000},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
