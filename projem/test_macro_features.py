"""
Makro özellikler testleri.

Çoğu test sentetik veriyle çalışır (internet gerekmez).
Ağ gerektiren indirme testi, bağlantı yoksa otomatik atlanır.

Çalıştırma:
    python -m unittest test_macro_features -v
"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from macro_features import (
    MACRO_FEATURE_COLUMNS,
    apply_macro_regime_overlay,
    compute_macro_feature_frame,
    compute_macro_stress_score,
    download_macro_data,
    get_macro_features_for_index,
    get_macro_regime_metrics,
)


def _make_macro_prices(days=120, seed=42):
    """Sentetik makro fiyat paneli (usdtry, xu100, brent)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp('2026-06-30'), periods=days)
    usd = 35 * np.cumprod(1 + rng.normal(0.0005, 0.005, days))
    xu = 10000 * np.cumprod(1 + rng.normal(0.001, 0.015, days))
    brent = 75 * np.cumprod(1 + rng.normal(0.0, 0.02, days))
    return pd.DataFrame({'usdtry': usd, 'xu100': xu, 'brent': brent}, index=idx)


class TestMacroFeatureFrame(unittest.TestCase):
    def test_feature_columns_present(self):
        prices = _make_macro_prices()
        feats = compute_macro_feature_frame(prices)
        self.assertIsNotNone(feats)
        core_cols = ['usdtry_mom_5d', 'usdtry_vol_10d', 'xu100_mom_5d', 'xu100_trend', 'brent_mom_10d']
        for col in core_cols:
            self.assertIn(col, feats.columns)

    def test_no_inf_values(self):
        prices = _make_macro_prices()
        feats = compute_macro_feature_frame(prices)
        self.assertFalse(np.isinf(feats.fillna(0).values).any())

    def test_momentum_sign_matches_price_move(self):
        # Kur son 5 günde belirgin yükseliyorsa momentum pozitif olmalı
        prices = _make_macro_prices()
        prices.loc[prices.index[-5]:, 'usdtry'] *= np.linspace(1.0, 1.10, 5)
        feats = compute_macro_feature_frame(prices)
        self.assertGreater(feats['usdtry_mom_5d'].iloc[-1], 0)

    def test_empty_input_returns_none(self):
        self.assertIsNone(compute_macro_feature_frame(None))
        self.assertIsNone(compute_macro_feature_frame(pd.DataFrame()))

    def test_partial_columns_ok(self):
        # Sadece usdtry inebilmişse bile frame üretmeli
        prices = _make_macro_prices()[['usdtry']]
        feats = compute_macro_feature_frame(prices)
        self.assertIsNotNone(feats)
        self.assertIn('usdtry_mom_5d', feats.columns)
        self.assertNotIn('xu100_trend', feats.columns)


class TestIndexAlignment(unittest.TestCase):
    def test_alignment_to_stock_index(self):
        macro = compute_macro_feature_frame(_make_macro_prices())
        # BIST takvimi: makro tarihlerinin bir alt kümesi + farklı günler
        stock_idx = macro.index[20:80]
        aligned = get_macro_features_for_index(stock_idx, macro_features=macro)
        self.assertIsNotNone(aligned)
        self.assertEqual(len(aligned), len(stock_idx))
        self.assertTrue((aligned.index == stock_idx).all())

    def test_missing_dates_forward_filled(self):
        macro = compute_macro_feature_frame(_make_macro_prices())
        # Makroda olmayan bir tarih (hafta içi tatil senaryosu)
        extra = macro.index[50] + pd.Timedelta(days=1)
        stock_idx = pd.DatetimeIndex(list(macro.index[45:50]) + [extra])
        aligned = get_macro_features_for_index(stock_idx, macro_features=macro)
        self.assertFalse(aligned.isna().any().any())

    def test_empty_index_returns_none(self):
        macro = compute_macro_feature_frame(_make_macro_prices())
        self.assertIsNone(get_macro_features_for_index(pd.DatetimeIndex([]), macro_features=macro))


class TestStressScore(unittest.TestCase):
    def test_calm_market_low_stress(self):
        metrics = {'usdtry_vol_10d': 0.10, 'usdtry_mom_20d': 0.01, 'xu100_trend': 0.02}
        self.assertLess(compute_macro_stress_score(metrics), 0.5)

    def test_currency_shock_high_stress(self):
        metrics = {'usdtry_vol_10d': 0.40, 'usdtry_mom_20d': 0.10, 'xu100_trend': -0.05}
        self.assertGreaterEqual(compute_macro_stress_score(metrics), 0.5)

    def test_score_bounded_0_1(self):
        extreme = {'usdtry_vol_10d': 5.0, 'usdtry_mom_20d': 1.0, 'xu100_trend': -0.50}
        self.assertLessEqual(compute_macro_stress_score(extreme), 1.0)
        self.assertEqual(compute_macro_stress_score({}), 0.0)
        self.assertEqual(compute_macro_stress_score(None), 0.0)


class TestRegimeOverlay(unittest.TestCase):
    STRESS_METRICS = {'usdtry_vol_10d': 0.40, 'usdtry_mom_20d': 0.10, 'xu100_trend': -0.05}
    CALM_METRICS = {'usdtry_vol_10d': 0.10, 'usdtry_mom_20d': 0.00, 'xu100_trend': 0.02}

    def test_bull_downgraded_under_stress(self):
        base = {'regime': 'bull', 'confidence': 0.9, 'metrics': {}}
        result = apply_macro_regime_overlay(base, self.STRESS_METRICS)
        self.assertEqual(result['regime'], 'sideways')
        self.assertIn('macro_override', result)

    def test_sideways_to_bear_with_negative_trend(self):
        base = {'regime': 'sideways', 'confidence': 0.5, 'metrics': {}}
        result = apply_macro_regime_overlay(base, self.STRESS_METRICS)
        self.assertEqual(result['regime'], 'bear')

    def test_bear_confidence_boosted(self):
        base = {'regime': 'bear', 'confidence': 0.6, 'metrics': {}}
        result = apply_macro_regime_overlay(base, self.STRESS_METRICS)
        self.assertEqual(result['regime'], 'bear')
        self.assertGreater(result['confidence'], 0.6)

    def test_calm_market_no_change(self):
        base = {'regime': 'bull', 'confidence': 0.8, 'metrics': {}}
        result = apply_macro_regime_overlay(base, self.CALM_METRICS)
        self.assertEqual(result['regime'], 'bull')
        self.assertEqual(result['confidence'], 0.8)
        self.assertNotIn('macro_override', result)

    def test_metrics_attached_even_when_calm(self):
        base = {'regime': 'sideways', 'confidence': 0.5, 'metrics': {}}
        result = apply_macro_regime_overlay(base, self.CALM_METRICS)
        self.assertIn('macro', result['metrics'])
        self.assertIn('stress_score', result['metrics']['macro'])

    def test_none_metrics_no_crash(self):
        base = {'regime': 'bull', 'confidence': 0.8, 'metrics': {}}
        result = apply_macro_regime_overlay(base, None)
        self.assertEqual(result['regime'], 'bull')

    def test_regime_metrics_from_synthetic_prices(self):
        metrics = get_macro_regime_metrics(macro_prices=_make_macro_prices())
        self.assertIsNotNone(metrics)
        self.assertIn('usdtry_vol_10d', metrics)
        self.assertIn('xu100_trend', metrics)


class TestArimaExogIntegration(unittest.TestCase):
    """Makro kolonların ARIMA exog pipeline'ına girdiğini doğrular (ağ yok, mock)."""

    def _make_stock_series(self, days=200, seed=7):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range(end=pd.Timestamp('2026-06-30'), periods=days)
        prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.02, days)), index=idx)
        returns = prices.pct_change().dropna()
        return prices, returns

    def test_macro_columns_in_exog(self):
        import feature_engineering as fe

        if not fe.MACRO_AVAILABLE:
            self.skipTest('macro_features import edilemedi')

        prices, returns = self._make_stock_series()
        macro = compute_macro_feature_frame(_make_macro_prices(days=250))

        def fake_macro(target_index, macro_features=None):
            return get_macro_features_for_index(target_index, macro_features=macro)

        with patch.object(fe, 'USE_MACRO_FEATURES', True), \
             patch.object(fe, 'get_macro_features_for_index', side_effect=fake_macro):
            exog, future_exog = fe.prepare_rolling_exogenous_for_arima(
                prices, returns, n_features=5, forecast_days=5,
            )

        self.assertIsNotNone(exog)
        macro_cols = [c for c in MACRO_FEATURE_COLUMNS if c in exog.columns]
        self.assertGreater(len(macro_cols), 0, 'Exog icinde makro kolon yok!')
        # future_exog aynı kolonlara sahip olmalı (ARIMA predict için şart)
        self.assertListEqual(list(exog.columns), list(future_exog.columns))

    def test_macro_failure_falls_back_to_technical_only(self):
        import feature_engineering as fe

        if not fe.MACRO_AVAILABLE:
            self.skipTest('macro_features import edilemedi')

        prices, returns = self._make_stock_series()

        with patch.object(fe, 'USE_MACRO_FEATURES', True), \
             patch.object(fe, 'get_macro_features_for_index', side_effect=Exception('ag yok')):
            exog, future_exog = fe.prepare_rolling_exogenous_for_arima(
                prices, returns, n_features=5, forecast_days=5,
            )

        self.assertIsNotNone(exog)
        for col in MACRO_FEATURE_COLUMNS:
            self.assertNotIn(col, exog.columns)


class TestDownloadNetwork(unittest.TestCase):
    """Ağ gerektirir — bağlantı yoksa atlanır."""

    def test_download_real_macro_data(self):
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=90)
        try:
            prices = download_macro_data(start, end)
        except Exception:
            self.skipTest('Ag baglantisi yok')
        if prices is None:
            self.skipTest('yfinance veri donduremedi')
        self.assertGreater(len(prices), 20)
        self.assertIn('usdtry', prices.columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)
