"""Yükseliş olasılığı (P(getiri > 0)) unit testleri."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestUpProbability(unittest.TestCase):
    def test_positive_return_high_prob(self):
        from forecasting.weekly_pipeline import calculate_up_probability

        prob = calculate_up_probability(0.02, 0.05)
        self.assertGreater(prob, 0.60)
        self.assertLessEqual(prob, 0.95)

    def test_zero_return_neutral(self):
        from forecasting.weekly_pipeline import calculate_up_probability

        prob = calculate_up_probability(0.0, 0.05)
        self.assertAlmostEqual(prob, 0.5, places=2)

    def test_negative_return_low_prob(self):
        from forecasting.weekly_pipeline import calculate_up_probability

        prob = calculate_up_probability(-0.02, 0.05)
        self.assertLess(prob, 0.40)
        self.assertGreaterEqual(prob, 0.05)

    def test_zero_vol_clip(self):
        from forecasting.weekly_pipeline import calculate_up_probability

        self.assertAlmostEqual(calculate_up_probability(0.05, 0.0), 0.95, places=2)
        self.assertAlmostEqual(calculate_up_probability(-0.05, 0.0), 0.05, places=2)

    def test_invalid_inputs(self):
        from forecasting.weekly_pipeline import calculate_up_probability

        self.assertEqual(calculate_up_probability(float("nan"), 0.05), 0.5)


if __name__ == "__main__":
    unittest.main()
