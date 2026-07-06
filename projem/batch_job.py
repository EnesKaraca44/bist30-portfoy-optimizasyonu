import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import sqlite3
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    from lstm_forecast import forecast_lstm
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    forecast_lstm = None
    ensemble_forecast = None

try:
    from feature_engineering import prepare_exogenous_features_for_arima, prepare_rolling_exogenous_for_arima
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    print("[!] Feature engineering modulu yuklenemedi, sadece basit ARIMA kullanilacak")

try:
    from model_selector import calculate_model_weights
    MODEL_SELECTOR_AVAILABLE = True
except ImportError:
    MODEL_SELECTOR_AVAILABLE = False
    print("[!] Model selector modulu yuklenemedi, sabit agirliklar kullanilacak")

from config import (
    DB_PATH as DB_NAME,
    BIST30_TICKERS,
    USE_LSTM,
    ENSEMBLE_ARIMA_W,
    ENSEMBLE_LSTM_W,
    ANNUAL_ARIMA_CLIP_MAX,
)
from forecasting.batch_pipeline import (
    HORIZON_MONTHS,
    clip_period_return,
    clip_period_series,
    annual_return_from_period,
    ensemble_period_forecast,
    period_clip_max,
    finalize_arima_period_forecast,
    apply_regime_to_period_forecasts,
    apply_black_litterman_for_horizon,
    get_horizon_forecast_days,
)
from forecasting.weekly_pipeline import detect_market_regime_from_panel

try:
    from config import USE_MACRO_FEATURES
except ImportError:
    USE_MACRO_FEATURES = False

try:
    from macro_features import get_macro_regime_metrics, apply_macro_regime_overlay
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False

from forecasting import (
    get_arima_forecast as _get_arima_forecast,
    get_garch_volatility,
    get_market_caps,
)


def migrate_beklenen_getiriler_schema(conn):
    """Vade bazli (ticker, horizon_ay) tablo yapisina gec."""
    from db.schema import ensure_beklenen_getiriler_horizon_schema

    ensure_beklenen_getiriler_horizon_schema(conn)


def setup_database():
    """Veritabanini ve vade bazli tahmin tablosunu hazirla."""
    conn = sqlite3.connect(DB_NAME)
    migrate_beklenen_getiriler_schema(conn)
    conn.close()
    print(f"Veritabani ({DB_NAME}) hazir.")


def get_arima_forecast_for_horizon(
    daily_returns,
    prices=None,
    months=1,
    use_features=True,
    ticker=None,
):
    forecast_days = get_horizon_forecast_days(months)
    raw = _get_arima_forecast(
        daily_returns,
        prices=prices,
        forecast_days=forecast_days,
        use_features=use_features and FEATURES_AVAILABLE,
        ticker=ticker,
        annualize=False,
        clip_range=None,
    )
    return finalize_arima_period_forecast(raw, daily_returns, months)


def run_bist30_ml_batch():
    print("[*] BIST 30 Black-Litterman Fabrikasi Basliyor (vade bazli)...")
    setup_database()

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    print("[*] Tarihsel veriler indiriliyor (Timeout hatalarina karsi tek tek)...")
    start_date = "2018-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    all_data_dict = {}
    for ticker in BIST30_TICKERS:
        basarili = False
        for _ in range(3):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty and "Close" in df.columns:
                    series = df["Close"]
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]
                    all_data_dict[ticker] = series
                    basarili = True
                    break
            except Exception:
                time.sleep(2)

        if not basarili:
            print(f"   [!] {ticker} verisi indirilemedi.")

    all_data = pd.DataFrame(all_data_dict)
    all_data = all_data.ffill().bfill()
    all_data = all_data.dropna(axis=1, how="all")

    valid_tickers = list(all_data.columns)
    daily_returns_all = all_data.pct_change().dropna()

    market_regime = detect_market_regime_from_panel(all_data, daily_returns_all)

    if USE_MACRO_FEATURES and MACRO_AVAILABLE:
        try:
            macro_metrics = get_macro_regime_metrics()
            market_regime = apply_macro_regime_overlay(market_regime, macro_metrics)
            if market_regime.get("macro_override"):
                print(f"[!] Makro override: {market_regime['macro_override']}")
        except Exception as e:
            print(f"[!] Makro rejim katmani atlandi: {e}")

    print(
        f"[*] Piyasa rejimi: {market_regime.get('regime', 'sideways')} "
        f"(guven={market_regime.get('confidence', 0.5):.2f})"
    )

    market_caps = get_market_caps()
    market_caps = market_caps[valid_tickers]

    garch_risks = {}
    print("\n[*] GARCH risk tahminleri hesaplaniyor...")
    for ticker in valid_tickers:
        try:
            series = daily_returns_all[ticker]
            garch_risks[ticker] = get_garch_volatility(series)
        except Exception:
            garch_risks[ticker] = series.std() * np.sqrt(252)

    model_type = "ARIMA+BL"
    guncelleme_zamani = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for months in HORIZON_MONTHS:
        forecast_days = get_horizon_forecast_days(months)
        print(f"\n[*] === {months} AY VADESI ({forecast_days} is gunu) ===")

        arima_forecasts = {}
        for ticker in valid_tickers:
            try:
                series = daily_returns_all[ticker]
                prices = all_data[ticker]
                arima_mu = get_arima_forecast_for_horizon(
                    series, prices=prices, months=months, use_features=True, ticker=ticker,
                )
                arima_forecasts[ticker] = arima_mu
                print(f"   [+] {ticker}: ARIMA %{arima_mu * 100:.1f} (donem)")
            except Exception as e:
                print(f"   [-] {ticker} ARIMA hatasi: {e}")
                series = daily_returns_all[ticker]
                fallback = clip_period_return(series.mean() * forecast_days, months)
                arima_forecasts[ticker] = fallback

        arima_series = clip_period_series(pd.Series(arima_forecasts), months)

        print(f"[*] Black-Litterman ({months} ay)...")
        bl_returns = apply_black_litterman_for_horizon(
            market_caps, daily_returns_all, arima_series, forecast_days,
        )
        bl_returns = clip_period_series(bl_returns, months)
        final_forecasts = bl_returns.to_dict()

        if USE_LSTM and LSTM_AVAILABLE:
            print(f"[*] LSTM ensemble ({months} ay)...")
            model_type = "ARIMA+BL+LSTM"
            lstm_cap = period_clip_max(months) * 0.5
            for ticker in valid_tickers:
                try:
                    series = daily_returns_all[ticker]
                    prices = all_data[ticker]
                    lstm_period = forecast_lstm(
                        series,
                        forecast_days=forecast_days,
                        ticker=ticker,
                        prices=prices,
                        use_features=True,
                        force_retrain=False,
                        return_period=True,
                    )
                    lstm_period = clip_period_return(lstm_period, months)
                    bl_value = final_forecasts.get(ticker, arima_forecasts.get(ticker, 0))

                    if MODEL_SELECTOR_AVAILABLE:
                        try:
                            weights = calculate_model_weights(
                                ticker, lookback_forecasts=20, method="direction_weighted",
                            )
                            arima_w = weights["arima"]
                            lstm_w = weights["lstm"]
                        except Exception:
                            arima_w = ENSEMBLE_ARIMA_W
                            lstm_w = ENSEMBLE_LSTM_W
                    else:
                        arima_w = ENSEMBLE_ARIMA_W
                        lstm_w = ENSEMBLE_LSTM_W

                    final_mu = ensemble_period_forecast(
                        bl_value,
                        lstm_period,
                        arima_weight=arima_w,
                        lstm_weight=lstm_w,
                        max_lstm_abs=lstm_cap,
                    )
                    final_forecasts[ticker] = clip_period_return(final_mu, months)
                    print(
                        f"   [+] {ticker}: BL %{bl_value * 100:.1f} | "
                        f"LSTM %{lstm_period * 100:.1f} | Final %{final_mu * 100:.1f}"
                    )
                except Exception as e:
                    print(f"   [!] {ticker} LSTM hatasi: {e}, sadece BL kullanilacak")

        final_forecasts = apply_regime_to_period_forecasts(final_forecasts, market_regime)
        final_forecasts = {
            t: clip_period_return(v, months) for t, v in final_forecasts.items()
        }

        for ticker in valid_tickers:
            period_final = final_forecasts.get(ticker, 0)
            period_bl = float(bl_returns.get(ticker, arima_forecasts.get(ticker, 0)))
            annual_final = annual_return_from_period(period_final, months)
            c.execute(
                """
                INSERT OR REPLACE INTO beklenen_getiriler
                (ticker, horizon_ay, tahmini_donem_getiri, tahmini_yillik_getiri,
                 tahmini_yillik_risk, tahmini_bl_getiri, son_guncelleme)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    months,
                    period_final,
                    annual_final,
                    garch_risks[ticker],
                    period_bl,
                    guncelleme_zamani,
                ),
            )

    placeholders = ",".join("?" * len(BIST30_TICKERS))
    c.execute(
        f"DELETE FROM beklenen_getiriler WHERE ticker NOT IN ({placeholders})",
        BIST30_TICKERS,
    )
    deleted_old = c.rowcount
    if deleted_old > 0:
        print(f"[*] Eski hisseler tablodan silindi: {deleted_old} kayit")

    conn.commit()
    conn.close()
    model_info = f" ({model_type} Ensemble)" if (USE_LSTM and LSTM_AVAILABLE) else ""
    print(f"[+] Islem Tamamlandi. {len(HORIZON_MONTHS)} vade icin tahminler hazir{model_info}!")


if __name__ == "__main__":
    run_bist30_ml_batch()
