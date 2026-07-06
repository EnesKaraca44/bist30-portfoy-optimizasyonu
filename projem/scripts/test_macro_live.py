"""Makro acik/kapali kisa karsilastirma (tek hisse + rejim)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

import feature_engineering as fe
from macro_features import get_macro_regime_metrics, apply_macro_regime_overlay, compute_macro_stress_score
from forecasting.weekly_pipeline import detect_market_regime_from_panel
from forecasting.arima import get_arima_forecast


def run():
    ticker = "AKBNK.IS"
    end = datetime.now()
    start = end - timedelta(days=730)
    print(f"[*] {ticker} verisi indiriliyor...")
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        print("[-] Veri indirilemedi")
        return 1
    prices = df["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    returns = prices.pct_change().dropna()
    print(f"[+] {len(returns)} gun veri")

    # Makro rejim
    metrics = get_macro_regime_metrics()
    stress = compute_macro_stress_score(metrics)
    print(f"[*] Makro metrikleri: {metrics}")
    print(f"[*] Stres skoru: {stress:.2f}")

    panel = pd.DataFrame({ticker: prices})
    panel_ret = panel.pct_change().dropna()
    regime = detect_market_regime_from_panel(panel, panel_ret)
    regime_macro = apply_macro_regime_overlay(regime, metrics)
    print(f"[*] Rejim (panel): {regime['regime']} -> makro sonrasi: {regime_macro['regime']}")
    if regime_macro.get("macro_override"):
        print(f"    Override: {regime_macro['macro_override']}")

    # ARIMA makro kapali
    fe.clear_macro_cache() if hasattr(fe, "clear_macro_cache") else None
    import macro_features as mf
    mf.clear_macro_cache()

    orig = fe.USE_MACRO_FEATURES
    fe.USE_MACRO_FEATURES = False
    pred_off = get_arima_forecast(returns, prices=prices, forecast_days=5, use_features=True, apply_momentum=True, clip_range=(-0.15, 0.15), ticker=ticker)
    print(f"[*] ARIMA (makro KAPALI): {pred_off*100:+.2f}% (5 gun)")

    # ARIMA makro acik
    mf.clear_macro_cache()
    fe.USE_MACRO_FEATURES = True
    pred_on = get_arima_forecast(returns, prices=prices, forecast_days=5, use_features=True, apply_momentum=True, clip_range=(-0.15, 0.15), ticker=ticker)
    print(f"[*] ARIMA (makro ACIK):   {pred_on*100:+.2f}% (5 gun)")
    print(f"[*] Fark: {(pred_on - pred_off)*100:+.2f} puan")

    fe.USE_MACRO_FEATURES = orig
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
