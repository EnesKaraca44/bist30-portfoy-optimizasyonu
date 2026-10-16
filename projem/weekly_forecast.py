"""
Haftalık getiri tahminleri ve veritabanı kayıt modülü
"""
import yfinance as yf
import pandas as pd
import numpy as np
import pmdarima as pm
from arch import arch_model
from datetime import datetime, timedelta
import sqlite3
import time
import warnings
warnings.filterwarnings('ignore')

# LSTM modülünü yükle (opsiyonel)
try:
    from lstm_forecast import forecast_lstm, ensemble_forecast
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False

# Feature Engineering modülünü yükle
try:
    from feature_engineering import prepare_exogenous_features_for_arima, prepare_rolling_exogenous_for_arima
    FEATURES_AVAILABLE = True
except:
    FEATURES_AVAILABLE = False

from config import DB_PATH as DB_NAME, BIST30

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
    get_garch_risk,
    download_bist_prices,
)
from forecasting.weekly_pipeline import (
    apply_regime_adjustment,
    calculate_up_probability,
    apply_black_litterman_weekly,
    combine_weekly_forecast,
    detect_market_regime_from_panel,
    filter_forecasts_by_confidence,
    get_volatility_adjusted_clip,
)
from forecast_bias_correction import apply_bias_correction
from forecasting.risk_metrics import calculate_dynamic_stop_loss

DOWNLOAD_BATCH_SIZE = 8


def _download_bist_prices(start_date, end_date):
    return download_bist_prices(BIST30, start_date, end_date)


def get_arima_forecast(returns_series, prices=None, days_ahead=5, use_features=True, ticker=None, clip=None):
    clip_range = clip if clip is not None else (-0.15, 0.15)
    return _get_arima_forecast(
        returns_series,
        prices=prices,
        forecast_days=days_ahead,
        use_features=use_features,
        apply_momentum=True,
        clip_range=clip_range,
        max_p=2,
        max_q=2,
        ticker=ticker,
    )

def forecast_weekly_returns(use_lstm=False, force_refresh=False):
    """
    Tüm BIST30 hisseleri için haftalık tahmin oluştur
    
    Algoritma:
    1. Son 2 yıllık veriyi çek
    2. Her hisse için:
       a. ARIMA ile 5 günlük tahmin yap
       b. (Opsiyonel) LSTM ensemble
       c. GARCH ile risk tahmini
    3. Sonuçları döndür
    """
    print("[*] Haftalik Tahminler Olusturuluyor...")
    
    # Verileri indir (son 2 yıl yeterli)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    try:
        print(f"[*] Veriler indiriliyor: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
        data = _download_bist_prices(start_date, end_date)

        if data.empty or len(data.columns) < 5:
            print("[-] HATA: Indirilen veri bos veya cok az hisse!")
            print("    Ag baglantisini, DNS (ornegin 8.8.8.8) ve VPN/proxy kontrol edin.")
            return {}

        # Eksik kolonları temizle ve doldur
        data = data.dropna(axis=1, how='all')
        data = data.ffill().bfill()

        print(f"[+] {len(data.columns)} hisse icin veri bulundu")
        
        returns = data.pct_change().dropna()
        
        if returns.empty:
            print("[-] HATA: Getiri verisi hesaplanamadi!")
            return {}
        
    except Exception as e:
        print(f"[-] HATA: Veri indirme basarisiz: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    forecasts = {}
    valid_tickers = list(data.columns)  # Sadece veri bulunan hisseler

    market_regime = detect_market_regime_from_panel(data, returns)

    # Makro stres katmanı: kur şoku / endeks zayıflığı rejimi geçersiz kılabilir
    if USE_MACRO_FEATURES and MACRO_AVAILABLE:
        try:
            macro_metrics = get_macro_regime_metrics()
            market_regime = apply_macro_regime_overlay(market_regime, macro_metrics)
            if market_regime.get('macro_override'):
                print(f"[!] Makro override: {market_regime['macro_override']}")
            macro_info = (market_regime.get('metrics', {}) or {}).get('macro', {})
            if macro_info:
                print(f"[*] Makro stres skoru: {macro_info.get('stress_score', 0):.2f} "
                      f"(kur vol={macro_info.get('usdtry_vol_10d', 0):.2f}, "
                      f"endeks trend={macro_info.get('xu100_trend', 0)*100:.1f}%)")
        except Exception as e:
            print(f"[!] Makro rejim katmani atlandi: {e}")

    print(f"[*] Piyasa rejimi: {market_regime.get('regime', 'sideways')} "
          f"(guven={market_regime.get('confidence', 0.5):.2f})")

    raw_arima = {}
    raw_risk = {}

    for ticker in BIST30:
        try:
            if ticker not in valid_tickers:
                print(f"  [!] {ticker}: Veri bulunamadi, atlaniyor")
                continue

            series = returns[ticker].dropna()
            prices = data[ticker].dropna()

            if len(series) < 100:
                print(f"  [!] {ticker}: Yeterli veri yok ({len(series)} < 100), atlaniyor")
                continue

            vol_clip = get_volatility_adjusted_clip(series)
            arima_return = get_arima_forecast(
                series, prices=prices, days_ahead=5, use_features=True,
                ticker=ticker, clip=(-vol_clip, vol_clip),
            )
            raw_arima[ticker] = float(arima_return)
            raw_risk[ticker] = float(get_garch_risk(series))
        except Exception as e:
            print(f"  [-] {ticker}: ARIMA hatasi - {type(e).__name__}: {e}")
            continue

    if not raw_arima:
        print("[-] UYARI: Hicbir hisse icin ARIMA tahmini olusturulamadi!")
        return {}

    print(f"\n[*] Black-Litterman dengeleme ({len(raw_arima)} hisse)...")
    bl_views = apply_black_litterman_weekly(raw_arima, returns, list(raw_arima.keys()))

    for ticker in BIST30:
        try:
            if ticker not in bl_views:
                continue

            series = returns[ticker].dropna()
            prices = data[ticker].dropna()
            vol_clip = get_volatility_adjusted_clip(series)
            bl_return = bl_views[ticker]

            lstm_weekly = None
            if use_lstm and LSTM_AVAILABLE:
                lstm_annual = forecast_lstm(
                    series, forecast_days=22, ticker=ticker,
                    prices=prices, use_features=True, force_retrain=False,
                )
                lstm_weekly = (1 + lstm_annual) ** (5 / 252) - 1
                lstm_weekly = float(np.clip(lstm_weekly, -vol_clip, vol_clip))

            final_return, model_type = combine_weekly_forecast(
                ticker,
                bl_return,
                lstm_weekly if lstm_weekly is not None else bl_return,
                use_lstm=use_lstm,
                lstm_available=LSTM_AVAILABLE,
            )
            if model_type == "ARIMA":
                model_type = "ARIMA+BL"
            elif model_type.startswith("ARIMA+LSTM"):
                model_type = model_type.replace("ARIMA", "ARIMA+BL", 1)
            elif model_type == "LSTM":
                model_type = "BL+LSTM"

            final_return = apply_regime_adjustment(final_return, market_regime)
            final_return = float(np.clip(final_return, -vol_clip, vol_clip))

            raw_return = final_return
            final_return, bias_correction = apply_bias_correction(ticker, raw_return)
            final_return = float(np.clip(final_return, -vol_clip, vol_clip))

            risk = raw_risk.get(ticker, 0.05)
            prob_up = calculate_up_probability(final_return, risk)
            stop_loss = calculate_dynamic_stop_loss(risk)

            forecasts[ticker] = {
                'return': float(final_return),
                'raw_return': float(raw_return),
                'bias_correction': float(bias_correction) if bias_correction is not None else None,
                'risk': risk,
                'model': model_type,
                'bl_return': float(bl_return),
                'arima_return': raw_arima.get(ticker),
                'prob_up': prob_up,
                'stop_loss': stop_loss,
            }

            bias_note = f", bias={bias_correction*100:+.2f}%" if bias_correction else ""
            print(f"  [+] {ticker}: Getiri={final_return*100:.2f}%{bias_note}, "
                  f"BL={bl_return*100:.2f}%, P(yukselis)={prob_up*100:.0f}% [{model_type}]")
        
        except Exception as e:
            print(f"  [-] {ticker}: Hata - {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not forecasts:
        print("[-] UYARI: Hicbir hisse icin tahmin olusturulamadi!")
        print(f"    - Toplam {len(BIST30)} hisse kontrol edildi")
        print(f"    - {len(valid_tickers)} hisse icin veri bulundu")
        return {}

    filter_forecasts_by_confidence(forecasts)
    forecasts["_market_regime"] = market_regime
    return forecasts

def save_weekly_forecasts(forecasts):
    """
    Haftalık tahminleri veritabanına kaydet
    
    Duplicate kontrolü:
    - Bugün için bu hisse için tahmin varsa güncelle
    - Yoksa yeni kayıt ekle
    """
    from performance_tracker import save_forecast, setup_performance_tables
    
    # Tablolari olustur
    setup_performance_tables()
    
    today = datetime.now()
    
    # Bu haftanın kaç iş günü kaldığını hesapla
    weekday = today.weekday()  # 0=Pazartesi, 4=Cuma
    if weekday < 4:
        days_until_friday = 4 - weekday
    else:
        days_until_friday = 0
    
    horizon_days = max(days_until_friday, 5)  # Haftalık tahmin = 5 iş günü (Pazartesi-Cuma)
    
    print(f"\n[*] Veritabanina kaydediliyor... (horizon={horizon_days} gun)")
    
    # Bugünün tarihini string olarak al (duplicate kontrolü için)
    today_str = today.strftime('%Y-%m-%d')
    
    # Tek bağlantı ile tüm kayıtları yap
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries:
        try:
            conn = sqlite3.connect(DB_NAME, timeout=30.0)
            conn.execute('PRAGMA journal_mode=WAL')
            c = conn.cursor()
            
            # Bugün için mevcut tahminleri kontrol et
            c.execute('''
                SELECT ticker FROM tahminler 
                WHERE DATE(tahmin_tarihi) = ? 
                AND tahmin_horizon = ?
            ''', (today_str, horizon_days))
            existing_tickers = {row[0] for row in c.fetchall()}
            
            saved_count = 0
            skipped_count = 0
            
            for ticker, pred in forecasts.items():
                if ticker.startswith("_"):
                    continue
                try:
                    # Eğer bugün için bu hisse için tahmin varsa, güncelle (yeni kayıt ekleme)
                    prob_up = pred.get('prob_up')
                    raw_return = pred.get('raw_return', pred['return'])
                    bias_corr = pred.get('bias_correction')
                    if ticker in existing_tickers:
                        # Mevcut kaydı güncelle
                        c.execute('''
                            UPDATE tahminler 
                            SET tahmini_getiri = ?, model_tipi = ?, yukselis_olasiligi = ?,
                                ham_tahmini_getiri = ?, bias_duzeltme = ?
                            WHERE ticker = ? 
                            AND DATE(tahmin_tarihi) = ?
                            AND tahmin_horizon = ?
                        ''', (pred['return'], pred['model'], prob_up, raw_return, bias_corr,
                              ticker, today_str, horizon_days))
                        skipped_count += 1
                    else:
                        # Yeni kayıt ekle
                        timestamp = datetime.now().isoformat()
                        c.execute('''
                            INSERT INTO tahminler 
                            (ticker, tahmin_tarihi, tahmin_horizon, tahmini_getiri, model_tipi,
                             yukselis_olasiligi, ham_tahmini_getiri, bias_duzeltme)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (ticker, timestamp, horizon_days, pred['return'], pred['model'],
                              prob_up, raw_return, bias_corr))
                        saved_count += 1
                    
                except Exception as e:
                    print(f"    [!] {ticker} kaydedilemedi: {e}")
                    continue
            
            conn.commit()
            conn.close()
            if saved_count > 0:
                print(f"[+] {saved_count} yeni tahmin kaydedildi!")
            if skipped_count > 0:
                print(f"[*] {skipped_count} mevcut tahmin guncellendi (duplicate onlendi)")
            return
            
        except sqlite3.OperationalError as e:
            if 'locked' in str(e):
                retry_count += 1
                print(f"[!] Veritabani kilitli, deneme {retry_count}/{max_retries}...")
                time.sleep(0.5)
                continue
            else:
                raise
        except Exception as e:
            print(f"[-] Veritabani hatasi: {e}")
            return
    
    print("[-] Veritabani kaydedilemedi (cok fazla deneme)")

def run_weekly_forecast_job(use_lstm=False, force_refresh=False):
    """
    Haftalık tahmin işini başlat ve kaydet
    """
    print("[*] Haftalik Tahmin Isi Baslatiliyor...")
    print("=" * 60)

    # 0. Önce geçmiş haftaların gerçekleşen getirilerini değerlendir ve karneleri güncelle
    try:
        print("[*] Gecmis tahminlerin gerceklesen performanslari kontrol ediliyor...")
        from portfolio_performance_tracker import backfill_portfolio_performance
        from performance_tracker import backfill_actuals_for_ticker
        
        updated_port = backfill_portfolio_performance(lookback_days=30)
        if updated_port > 0:
            print(f"[+] {updated_port} haftalik portfoy tahmini gerceklesen degerlerle guncellendi.")

        conn_check = sqlite3.connect(DB_NAME)
        c_check = conn_check.cursor()
        c_check.execute("SELECT DISTINCT ticker FROM tahminler WHERE gerceklesen_getiri IS NULL")
        pending_tickers = [row[0] for row in c_check.fetchall()]
        conn_check.close()

        updated_stocks = 0
        for t in pending_tickers:
            updated_stocks += backfill_actuals_for_ticker(t, horizon_days=5, lookback_days=30)
        if updated_stocks > 0:
            print(f"[+] {updated_stocks} bireysel hisse tahmini gerceklesen degerlerle guncellendi.")
    except Exception as e:
        print(f"[!] Gecmis tahminleri guncelleme hatasi: {e}")

    today = datetime.now()
    friday = today + timedelta(days=(4 - today.weekday()))
    
    print(f"[*] Haftalik Tahmin: {today.strftime('%Y-%m-%d')} -> {friday.strftime('%Y-%m-%d')} ({(4-today.weekday())} gun)")
    
    if today.weekday() <= 4:
        print("[+] Bu hafta: Bugunden Cuma'ya kadar")
    else:
        print("[!] Hafta sonu - Tahmin Pazartesi icin gecerli olacak")
    
    forecasts = forecast_weekly_returns(use_lstm=use_lstm, force_refresh=force_refresh)
    
    market_regime = forecasts.pop("_market_regime", None) if forecasts else None
    
    if forecasts:
        save_weekly_forecasts(forecasts)
        
        # Portföy seviyesinde tahmin kaydet
        try:
            from portfolio_performance_tracker import (
                calculate_portfolio_forecast, 
                save_portfolio_forecast,
                setup_portfolio_performance_table
            )
            
            setup_portfolio_performance_table()
            # Portföy seviyesinde tahmin hesapla (Forecasts_dict parametresini kullan!)
            portfolio_forecast = calculate_portfolio_forecast(
                datetime.now(), 
                horizon_days=5, 
                forecasts_dict=forecasts,
                market_regime=market_regime,
            )
            
            if portfolio_forecast:
                save_portfolio_forecast(
                    datetime.now(),
                    horizon_days=5,
                    predicted_return=portfolio_forecast['predicted_return'],
                    weights_dict=portfolio_forecast['weights'],
                    var_95=portfolio_forecast.get('var_95'),
                    portfolio_vol=portfolio_forecast.get('portfolio_vol'),
                    probability_up=portfolio_forecast.get('probability_up'),
                )
                var_pct = portfolio_forecast.get('var_95', 0) * 100
                prob_pct = portfolio_forecast.get('probability_up', 0.5) * 100
                print(f"[+] Portfoy tahmini kaydedildi: %{portfolio_forecast['predicted_return']*100:.2f} "
                      f"| VaR95=%{var_pct:.2f} | P(yukselis)=%{prob_pct:.0f}")
        except Exception as e:
            print(f"[!] Portfoy tahmini kaydedilemedi: {e}")

        try:
            from model_selector import generate_model_performance_report
            report = generate_model_performance_report(lookback_forecasts=20)
            if report is not None and not report.empty:
                weak = report[report["arima_direction_acc"].fillna(0) < 0.45]
                if len(weak) > 0:
                    print(f"\n[!] Dusuk yon dogrulugu ({len(weak)} hisse):")
                    for _, row in weak.head(5).iterrows():
                        print(f"    {row['ticker']}: en iyi={row['best_model']}, "
                              f"arima_acc={row['arima_direction_acc']:.0%}")
        except Exception:
            pass

        try:
            from portfolio_vs_ai_scorer import run_leaderboard_scoring
            lb_result = run_leaderboard_scoring()
            if lb_result.get("success") and lb_result.get("users_scored", 0) > 0:
                print(f"[+] Liderlik tablosu otomatik guncellendi ({lb_result['users_scored']} kullanici)")
            elif not lb_result.get("success"):
                print(f"[!] Liderlik tablosu: {lb_result.get('message', 'atlandi')}")
        except Exception as e:
            print(f"[!] Liderlik tablosu guncellenemedi: {e}")
    else:
        print("[-] Hicbir tahmin olusturulamadi.")
    
    print("=" * 60)
    return forecasts

if __name__ == "__main__":
    # LSTM'yi test etmek için True yapabilirsiniz
    # NOT: Multi-feature LSTM + ARIMA Ensemble (Deep Learning)
    run_weekly_forecast_job(use_lstm=True)

