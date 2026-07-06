"""EVDS faiz verisi testleri."""
import json
import unittest
from unittest.mock import patch

import pandas as pd

from evds_client import compute_policy_rate_features, fetch_evds_series, get_policy_rate_series


class TestEvdsClient(unittest.TestCase):
    def test_fetch_without_api_key_returns_none(self):
        self.assertIsNone(fetch_evds_series("TP.MB.AO.A12", "2024-01-01", "2024-06-01", api_key=None))

    def test_fetch_parses_json(self):
        payload = {
            "items": [
                {"Tarih": "01-01-2024", "TP_APIFON4": "42.5"},
                {"Tarih": "02-01-2024", "TP_APIFON4": "42.5"},
                {"Tarih": "03-01-2024", "TP_APIFON4": "43.0"},
            ]
        }

        class FakeResp:
            status_code = 200
            text = json.dumps(payload)
            content = text.encode("utf-8")

        with patch("requests.get", return_value=FakeResp()):
            series = fetch_evds_series("TP.APIFON4", "2024-01-01", "2024-01-10", api_key="test-key")

        self.assertIsNotNone(series)
        self.assertEqual(len(series), 3)
        self.assertAlmostEqual(series.iloc[-1], 43.0)

    def test_policy_rate_features(self):
        idx = pd.bdate_range("2024-01-01", periods=30)
        rates = pd.Series([40.0 + i * 0.1 for i in range(30)], index=idx)
        feats = compute_policy_rate_features(rates)
        self.assertIsNotNone(feats)
        self.assertIn("policy_rate_chg_20d", feats.columns)
        self.assertIn("policy_rate_level", feats.columns)

    def test_get_policy_rate_disabled(self):
        with patch("config.USE_EVDS_INTEREST", False):
            self.assertIsNone(get_policy_rate_series())


class TestMacroPolicyIntegration(unittest.TestCase):
    def test_policy_features_attached_to_macro_frame(self):
        import macro_features as mf

        mf.clear_macro_cache()
        prices = mf.download_macro_data(
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2026-07-01"),
        )
        if prices is None:
            self.skipTest("yfinance makro verisi yok")

        feats = mf.compute_macro_feature_frame(prices)
        idx = pd.bdate_range("2025-06-01", "2026-06-30", freq="B")
        policy_idx = pd.bdate_range("2025-01-01", periods=200, freq="B")
        policy = pd.Series([45.0] * len(policy_idx), index=policy_idx)

        with patch("evds_client.get_policy_rate_series", return_value=policy):
            merged = mf._attach_policy_features(feats.copy(), idx[0], idx[-1])

        self.assertIn("policy_rate_chg_20d", merged.columns)

    def test_stress_includes_policy_hike(self):
        from macro_features import compute_macro_stress_score

        calm = {"usdtry_vol_10d": 0.05, "usdtry_mom_20d": 0.0, "xu100_trend": 0.0}
        with_hike = {**calm, "policy_rate_chg_20d": 0.025}
        self.assertGreater(compute_macro_stress_score(with_hike), compute_macro_stress_score(calm))


if __name__ == "__main__":
    unittest.main()
