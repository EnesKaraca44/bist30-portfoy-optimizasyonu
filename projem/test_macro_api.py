"""Makro dashboard API ve UI payload testleri."""
import unittest
from unittest.mock import patch

from macro_features import get_macro_dashboard, compute_macro_stress_score


class TestMacroDashboard(unittest.TestCase):
    def test_disabled_when_config_off(self):
        with patch("config.USE_MACRO_FEATURES", False):
            data = get_macro_dashboard()
        self.assertFalse(data["enabled"])
        self.assertFalse(data["available"])

    def test_unavailable_when_no_metrics(self):
        with patch("config.USE_MACRO_FEATURES", True), \
             patch("macro_features.get_macro_regime_metrics", return_value=None):
            data = get_macro_dashboard()
        self.assertTrue(data["enabled"])
        self.assertFalse(data["available"])

    def test_calm_market_payload(self):
        metrics = {
            "usdtry_vol_10d": 0.01,
            "usdtry_mom_20d": 0.01,
            "xu100_trend": 0.02,
            "xu100_mom_20d": 0.03,
        }
        with patch("config.USE_MACRO_FEATURES", True), \
             patch("macro_features.get_macro_regime_metrics", return_value=metrics):
            data = get_macro_dashboard()
        self.assertTrue(data["available"])
        self.assertEqual(data["regime_level"], "positive")
        self.assertLess(data["stress_score"], 0.5)
        self.assertEqual(data["stress_label"], "Düşük")
        self.assertIn("metrics", data)
        self.assertIn("message", data)

    def test_stress_market_payload(self):
        metrics = {
            "usdtry_vol_10d": 0.40,
            "usdtry_mom_20d": 0.10,
            "xu100_trend": -0.05,
            "xu100_mom_20d": -0.04,
        }
        with patch("config.USE_MACRO_FEATURES", True), \
             patch("macro_features.get_macro_regime_metrics", return_value=metrics):
            data = get_macro_dashboard()
        self.assertEqual(data["regime_level"], "stress")


class TestMacroDashboardAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        cls.client = app.test_client()

    def test_macro_dashboard_endpoint(self):
        r = self.client.get("/api/macro_dashboard")
        self.assertIn(r.status_code, (200, 503))
        data = r.get_json()
        self.assertIn("enabled", data)
        self.assertIn("message", data)

    def test_macro_dashboard_structure_when_ok(self):
        mock = {
            "enabled": True,
            "available": True,
            "stress_score": 0.0,
            "stress_percent": 0,
            "regime_label": "Normal Seyir",
            "regime_level": "neutral",
            "stress_label": "Düşük",
            "metrics": {},
            "message": "test",
        }
        with patch("macro_features.get_macro_dashboard", return_value=mock):
            r = self.client.get("/api/macro_dashboard")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue(data["available"])
        self.assertEqual(data["regime_label"], "Normal Seyir")


if __name__ == "__main__":
    unittest.main()
