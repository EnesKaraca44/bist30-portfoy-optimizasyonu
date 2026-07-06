"""
Portföy Seviyesinde Performans Takibi
Bireysel hisse tahminleri yerine, portföy tahminlerini takip eder
"""
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import DB_PATH as DB_NAME, BIST30
from utils.dates import add_business_days

DEBUG = False

def setup_portfolio_performance_table():
    """Portföy performans tablosunu oluşturur"""
    conn = sqlite3.connect(DB_NAME, timeout=30.0)
    conn.execute('PRAGMA journal_mode=WAL')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfoy_performans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tahmin_tarihi TEXT NOT NULL,
            tahmin_horizon INTEGER NOT NULL,
            portfoy_tahmini_getiri REAL,
            portfoy_gerceklesen_getiri REAL,
            portfoy_hatasi REAL,
            portfoy_agirliklari TEXT,
            benchmark_getiri REAL,
            alpha REAL,
            var_95 REAL,
            portfoy_vol REAL,
            yukselis_olasiligi REAL,
            UNIQUE(tahmin_tarihi, tahmin_horizon)
        )
    ''')

    c.execute("PRAGMA table_info(portfoy_performans)")
    existing_cols = {row[1] for row in c.fetchall()}
    for col, ddl in (
        ("var_95", "ALTER TABLE portfoy_performans ADD COLUMN var_95 REAL"),
        ("portfoy_vol", "ALTER TABLE portfoy_performans ADD COLUMN portfoy_vol REAL"),
        ("yukselis_olasiligi", "ALTER TABLE portfoy_performans ADD COLUMN yukselis_olasiligi REAL"),
    ):
        if col not in existing_cols:
            c.execute(ddl)
    
    conn.commit()
    conn.close()

def calculate_portfolio_forecast(forecast_date, horizon_days=5, forecasts_dict=None, market_regime=None):
    """
    Belirli bir tarih için optimal portföyü ve beklenen getirisini hesapla.
    
    Args:
        forecast_date: Tahmin tarihi
        horizon_days: Tahmin ufku (gün)
        forecasts_dict: Opsiyonel, {ticker: {'return': float, 'risk': float}} formatında veri.
                        Verilirse DB'den ('beklenen_getiriler') veri çekmek yerine bunu kullanır.
    """
    try:
        # Getiri ve risk verilerini hazırla
        returns_dict = {}
        risk_dict = {}

        if forecasts_dict:
            # Doğrudan verilen veriyi kullan
            if market_regime is None and "_market_regime" in forecasts_dict:
                market_regime = forecasts_dict.get("_market_regime")
            for ticker, data in forecasts_dict.items():
                if ticker.startswith("_") or not isinstance(data, dict):
                    continue
                returns_dict[ticker] = data['return']
                risk_dict[ticker] = data.get('risk', 0.05) # Varsayılan risk %5
        else:
            from db.schema import ensure_beklenen_getiriler_horizon_schema
            from forecasting.batch_pipeline import get_horizon_forecast_days
            from portfolio.optimizer import _horizon_days_to_months

            months = _horizon_days_to_months(horizon_days)
            conn = sqlite3.connect(DB_NAME)
            ensure_beklenen_getiriler_horizon_schema(conn)
            c = conn.cursor()
            c.execute(
                '''
                SELECT ticker, tahmini_donem_getiri, tahmini_yillik_getiri, tahmini_yillik_risk
                FROM beklenen_getiriler
                WHERE ticker IN ({}) AND horizon_ay = ?
                '''.format(','.join('?' * len(BIST30))),
                (*BIST30, months),
            )
            
            rows = c.fetchall()
            conn.close()
            
            if not rows:
                return None
            
            for row in rows:
                ticker, period_return, annual_return, risk = row
                if period_return is not None and not np.isnan(period_return):
                    returns_dict[ticker] = float(period_return)
                elif annual_return is not None and not np.isnan(annual_return):
                    from portfolio.optimizer import adjust_returns_for_horizon
                    returns_dict[ticker] = float(adjust_returns_for_horizon(annual_return, months))
                else:
                    returns_dict[ticker] = 0.0
                risk_dict[ticker] = risk if risk and not np.isnan(risk) else 0.0001
        
        # Portföy optimizasyonu (basit Monte Carlo)
        valid_tickers = list(returns_dict.keys())
        if len(valid_tickers) < 5:
            return None

        # --- POZİTİF GETİRİ + REJİM/GÜVEN FİLTRESİ ---
        from forecasting.weekly_pipeline import select_tickers_for_portfolio

        high_conf = {
            t for t in valid_tickers
            if forecasts_dict and isinstance(forecasts_dict.get(t), dict)
            and forecasts_dict[t].get("high_confidence", True)
        } if forecasts_dict else None

        selected = select_tickers_for_portfolio(
            returns_dict,
            risk_dict,
            market_regime=market_regime,
            min_tickers=5,
            high_confidence_tickers=high_conf if high_conf else None,
        )
        valid_tickers = selected
        regime_label = (market_regime or {}).get("regime", "")
        print(f"[+] Portfoy hisse secimi ({regime_label or 'n/a'}): {len(valid_tickers)} hisse")
        # ---------------------------------

        returns_series = pd.Series([returns_dict[t] for t in valid_tickers], index=valid_tickers)
        
        # Kovaryans matrisi (basit korelasyon + volatilite)
        # Tarihsel veri çek
        end_date = forecast_date
        start_date = end_date - timedelta(days=252)  # 1 yıl
        
        prices = yf.download(valid_tickers, start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'), progress=False)['Close']
        
        if prices.empty:
            return None
        
        # MultiIndex kontrolü
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices.droplevel(0, axis=1)
        
        # Eksik verileri temizle
        prices = prices.ffill().bfill()
        prices = prices.dropna(axis=1, how='any')  # Tüm NaN'lı kolonları sil
        
        # Geçerli tickerları güncelle
        valid_tickers = [t for t in valid_tickers if t in prices.columns]
        if len(valid_tickers) < 5:
            if DEBUG:
                print(f"[HATA] Yeterli veri yok, sadece {len(valid_tickers)} hisse")
            return None
        
        returns_series = returns_series[valid_tickers]
        
        daily_returns = prices[valid_tickers].pct_change().dropna()
        corr_matrix = daily_returns.corr()
        
        # NaN kontrolü
        corr_matrix = corr_matrix.fillna(0)  # NaN'ları 0'la doldur (korelasyon yok)
        np.fill_diagonal(corr_matrix.values, 1.0)  # Diagonali 1 yap
        
        # GARCH riskleri ile kovaryans
        D = np.diag([risk_dict[t] for t in valid_tickers])
        cov_matrix = np.dot(D, np.dot(corr_matrix.values, D))
        cov_matrix = pd.DataFrame(cov_matrix, index=valid_tickers, columns=valid_tickers)
        
        # NaN kontrolü
        cov_matrix = cov_matrix.fillna(0)
        
        if DEBUG:
            print(f"[Kovaryans] Temizlendi: {len(valid_tickers)} hisse, NaN sayisi: {cov_matrix.isna().sum().sum()}")
        
        from forecasting.weekly_pipeline import (
            get_portfolio_strategy_for_regime,
            optimize_portfolio_monte_carlo,
        )
        strategy = get_portfolio_strategy_for_regime(market_regime)
        print(f"[+] Portfoy optimizasyon stratejisi: {strategy} (rejim={((market_regime or {}).get('regime', 'n/a'))})")

        best_weights = optimize_portfolio_monte_carlo(
            returns_series, cov_matrix, strategy=strategy, num_portfolios=25000,
        )
        
        if best_weights is None:
            best_weights = np.array([1 / len(valid_tickers)] * len(valid_tickers))
        
        if DEBUG:
            print(f"[MC] Strateji={strategy}, agirlik ornegi: {dict(list(zip(valid_tickers[:3], best_weights[:3])))}...")
        
        # Portföy tahmini getiri
        portfolio_return = np.dot(best_weights, returns_series.values)
        
        if forecasts_dict:
            # Zaten haftalık/horizon'a göre hesaplanmış değerler
            portfolio_return_scaled = portfolio_return
        else:
            months = _horizon_days_to_months(horizon_days)
            target_days = get_horizon_forecast_days(months)
            if int(horizon_days) == target_days:
                portfolio_return_scaled = portfolio_return
            else:
                portfolio_return_scaled = (1 + portfolio_return) ** (horizon_days / target_days) - 1
        
        weights_dict = {ticker: float(weight) for ticker, weight in zip(valid_tickers, best_weights)}

        from forecasting.risk_metrics import calculate_portfolio_var

        risk_metrics = calculate_portfolio_var(
            best_weights,
            returns_series,
            cov_matrix.loc[valid_tickers, valid_tickers],
        )
        
        return {
            'predicted_return': float(portfolio_return_scaled),
            'weights': weights_dict,
            'tickers': valid_tickers,
            'var_95': risk_metrics['var_95'],
            'portfolio_vol': risk_metrics['portfolio_vol'],
            'probability_up': risk_metrics['probability_up'],
        }
        
    except Exception as e:
        if DEBUG:
            print(f"Portföy tahmin hatası: {e}")
        return None

def calculate_portfolio_actual(forecast_date, horizon_days, weights_dict):
    """
    Portföyün gerçekleşen getirisini hesapla
    
    Algoritma:
    1. Tahmin tarihi + horizon (iş günü) = hedef tarih
    2. Her hisse için:
       a. Tahmin tarihindeki fiyatı bul
       b. Hedef tarihteki fiyatı bul
       c. Getiri = (hedef_fiyat / tahmin_fiyat) - 1
    3. Portföy getirisi = sum(weight[ticker] × getiri[ticker])
    """
    try:
        # DÜZELTME: İş günü hesaplama
        forecast_date_obj = forecast_date.date() if isinstance(forecast_date, pd.Timestamp) else forecast_date
        target_date = add_business_days(forecast_date_obj, horizon_days)
        
        # Fiyat verilerini çek
        start = (forecast_date - timedelta(days=5)).strftime('%Y-%m-%d')
        end = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')
        
        if DEBUG:
            print(f"[Actual] Tahmin: {forecast_date}, Hedef: {target_date}, Veri aralığı: {start} - {end}")
        
        tickers = list(weights_dict.keys())
        if DEBUG:
            print(f"[Actual] {len(tickers)} hisse için veri çekiliyor...")
        
        prices = yf.download(tickers, start=start, end=end, progress=False)['Close']
        
        if prices.empty:
            if DEBUG:
                print(f"[Actual] HATA: Fiyat verisi boş")
            return None
        
        if DEBUG:
            print(f"[Actual] Çekilen veri: {len(prices)} satır")
            print(f"[Actual] İlk tarih: {prices.index[0]}, Son tarih: {prices.index[-1]}")
        
        # Tek hisse varsa Series olur, DataFrame'e çevir
        if len(tickers) == 1:
            prices = prices.to_frame(name=tickers[0])
        
        # NaN temizleme
        prices = prices.ffill().bfill()
        
        if len(prices) < 2:
            if DEBUG:
                print(f"[Actual] HATA: Yeterli veri yok ({len(prices)} satır)")
            return None
        
        # Başlangıç fiyatları
        forecast_ts = pd.Timestamp(forecast_date)
        start_prices = {}
        for ticker in tickers:
            if forecast_ts in prices.index:
                start_prices[ticker] = float(prices.loc[forecast_ts, ticker])
            else:
                past_prices = prices[prices.index <= forecast_ts]
                if len(past_prices) > 0:
                    start_prices[ticker] = float(past_prices.iloc[-1][ticker])
                    if DEBUG:
                        print(f"[Actual] {ticker} başlangıç: {past_prices.index[-1]} tarihinden alındı")
                else:
                    if DEBUG:
                        print(f"[Actual] HATA: {ticker} için başlangıç fiyatı bulunamadı")
                    return None
        
        # Bitiş fiyatları
        target_ts = pd.Timestamp(target_date)
        end_prices = {}
        for ticker in tickers:
            if target_ts in prices.index:
                end_prices[ticker] = float(prices.loc[target_ts, ticker])
            else:
                future_prices = prices[prices.index >= target_ts]
                if len(future_prices) > 0:
                    end_prices[ticker] = float(future_prices.iloc[0][ticker])
                    if DEBUG:
                        print(f"[Actual] {ticker} bitiş: {future_prices.index[0]} tarihinden alındı")
                else:
                    if DEBUG:
                        print(f"[Actual] HATA: {ticker} için bitiş fiyatı bulunamadı")
                    return None
        
        # Portföy getirisi
        portfolio_return = 0
        total_weight = 0
        valid_tickers = []
        
        for ticker, weight in weights_dict.items():
            if ticker in start_prices and ticker in end_prices:
                start_price = start_prices[ticker]
                end_price = end_prices[ticker]
                
                # NaN kontrolü
                if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0:
                    if DEBUG:
                        print(f"[Actual] {ticker}: Veri eksik (NaN veya geçersiz), atlanıyor")
                    continue
                
                ticker_return = (end_price / start_price) - 1
                portfolio_return += weight * ticker_return
                total_weight += weight
                valid_tickers.append(ticker)
                
                if DEBUG:
                    print(f"[Actual] {ticker}: başlangıç={start_price:.2f}, bitiş={end_price:.2f}, getiri={ticker_return:.4f}, ağırlık={weight:.4f}")
            else:
                if DEBUG:
                    print(f"[Actual] {ticker}: Fiyat verisi bulunamadı, atlanıyor")
        
        # Ağırlıkları normalize et (veri bulunamayan hisseler için)
        if total_weight > 0 and total_weight < 1.0:
            portfolio_return = portfolio_return / total_weight
            if DEBUG:
                print(f"[Actual] Ağırlıklar normalize edildi: {total_weight:.4f} -> 1.0")
        
        if len(valid_tickers) == 0:
            if DEBUG:
                print(f"[Actual] HATA: Hiçbir hisse için geçerli veri bulunamadı")
            return None
        
        if DEBUG:
            print(f"[Actual] Toplam portföy getirisi: {portfolio_return:.4f} ({len(valid_tickers)}/{len(weights_dict)} hisse kullanıldı)")
        
        return float(portfolio_return)
        
    except Exception as e:
        if DEBUG:
            print(f"[Actual] HATA: {e}")
            import traceback
            traceback.print_exc()
        return None


def get_portfolio_actual_breakdown(forecast_date, horizon_days, weights_dict):
    """
    Portföy gerçekleşen getirisini ve hisse bazlı detayı döndürür (doğrulama için).
    Returns: (total_return, target_date, list of {ticker, start_date, end_date, start_price, end_price, ret_pct, weight, contribution_pct})
    """
    try:
        forecast_date_obj = forecast_date.date() if hasattr(forecast_date, 'date') else forecast_date
        if hasattr(forecast_date_obj, 'isoformat'):
            pass
        else:
            forecast_date_obj = pd.to_datetime(forecast_date_obj).date()
        target_date = add_business_days(forecast_date_obj, horizon_days)
        start = (pd.Timestamp(forecast_date_obj) - timedelta(days=5)).strftime('%Y-%m-%d')
        end = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')
        tickers = list(weights_dict.keys())
        prices = yf.download(tickers, start=start, end=end, progress=False)['Close']
        if prices.empty or len(prices) < 2:
            return None, None, []
        if len(tickers) == 1:
            prices = prices.to_frame(name=tickers[0])
        prices = prices.ffill().bfill()
        forecast_ts = pd.Timestamp(forecast_date_obj)
        target_ts = pd.Timestamp(target_date)
        start_prices = {}
        start_dates = {}
        for t in tickers:
            if forecast_ts in prices.index:
                start_prices[t] = float(prices.loc[forecast_ts, t])
                start_dates[t] = forecast_ts
            else:
                past = prices[prices.index <= forecast_ts]
                if len(past) > 0:
                    start_prices[t] = float(past.iloc[-1][t])
                    start_dates[t] = past.index[-1]
                else:
                    start_prices[t] = start_dates[t] = None
        end_prices = {}
        end_dates = {}
        for t in tickers:
            if target_ts in prices.index:
                end_prices[t] = float(prices.loc[target_ts, t])
                end_dates[t] = target_ts
            else:
                fut = prices[prices.index >= target_ts]
                if len(fut) > 0:
                    end_prices[t] = float(fut.iloc[0][t])
                    end_dates[t] = fut.index[0]
                else:
                    end_prices[t] = end_dates[t] = None
        total_ret = 0.0
        total_weight = 0.0
        breakdown = []
        for t, w in weights_dict.items():
            if t not in start_prices or t not in end_prices or start_prices[t] is None or end_prices[t] is None:
                continue
            sp, ep = start_prices[t], end_prices[t]
            if sp <= 0 or pd.isna(sp) or pd.isna(ep):
                continue
            ret = (ep / sp) - 1
            contrib = w * ret
            total_ret += contrib
            total_weight += w
            breakdown.append({
                'ticker': t,
                'start_date': start_dates[t],
                'end_date': end_dates[t],
                'start_price': round(sp, 4),
                'end_price': round(ep, 4),
                'ret_pct': round(ret * 100, 4),
                'weight': round(w * 100, 2),
                'contribution_pct': round(contrib * 100, 4),
            })
        if total_weight > 0 and total_weight < 1.0:
            total_ret = total_ret / total_weight
        return float(total_ret), target_date, breakdown
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, []


def save_portfolio_forecast(forecast_date, horizon_days, predicted_return, actual_return=None,
                           weights_dict=None, benchmark_return=None,
                           var_95=None, portfolio_vol=None, probability_up=None):
    """Portföy tahminini kaydet (Aynı gün için duplicate kontrolü yapar)"""
    setup_portfolio_performance_table()
    
    conn = sqlite3.connect(DB_NAME, timeout=30.0)
    conn.execute('PRAGMA journal_mode=WAL')
    c = conn.cursor()
    
    try:
        # Aynı gün ve horizon için mevcut kaydı kontrol et
        forecast_date_obj = forecast_date.date() if isinstance(forecast_date, pd.Timestamp) else forecast_date
        if isinstance(forecast_date_obj, pd.Timestamp):
            forecast_date_obj = forecast_date_obj.date()
        date_str = forecast_date_obj.isoformat()
        
        c.execute('''
            SELECT tahmin_tarihi, portfoy_gerceklesen_getiri
            FROM portfoy_performans
            WHERE DATE(tahmin_tarihi) = ?
            AND tahmin_horizon = ?
        ''', (date_str, horizon_days))
        
        existing = c.fetchone()
        
        error = None
        alpha = None
        if actual_return is not None and predicted_return is not None:
            error = abs(actual_return - predicted_return)
            if benchmark_return is not None:
                alpha = actual_return - benchmark_return
        
        import json
        weights_json = json.dumps(weights_dict) if weights_dict else None
        
        if existing:
            # Mevcut kayıt var - güncelle
            existing_timestamp, existing_actual = existing
            # Eğer gerçekleşen değer varsa, onu koru (doğrulanmış veri)
            final_actual = actual_return if actual_return is not None else existing_actual
            final_error = error if error is not None else None
            final_alpha = alpha if alpha is not None else None
            final_benchmark = benchmark_return if benchmark_return is not None else None
            
            # Gerçekleşen değer varsa error ve alpha'yı yeniden hesapla
            if final_actual is not None and predicted_return is not None:
                final_error = abs(final_actual - predicted_return)
                if final_benchmark is not None:
                    final_alpha = final_actual - final_benchmark
            
            c.execute('''
                UPDATE portfoy_performans
                SET portfoy_tahmini_getiri = ?,
                    portfoy_gerceklesen_getiri = ?,
                    portfoy_hatasi = ?,
                    portfoy_agirliklari = ?,
                    benchmark_getiri = ?,
                    alpha = ?,
                    var_95 = COALESCE(?, var_95),
                    portfoy_vol = COALESCE(?, portfoy_vol),
                    yukselis_olasiligi = COALESCE(?, yukselis_olasiligi)
                WHERE DATE(tahmin_tarihi) = ?
                AND tahmin_horizon = ?
            ''', (predicted_return, final_actual, final_error, weights_json,
                  final_benchmark, final_alpha, var_95, portfolio_vol, probability_up,
                  date_str, horizon_days))
        else:
            # Yeni kayıt ekle
            c.execute('''
                INSERT INTO portfoy_performans
                (tahmin_tarihi, tahmin_horizon, portfoy_tahmini_getiri, 
                 portfoy_gerceklesen_getiri, portfoy_hatasi, portfoy_agirliklari,
                 benchmark_getiri, alpha, var_95, portfoy_vol, yukselis_olasiligi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (forecast_date.isoformat(), horizon_days, predicted_return,
                  actual_return, error, weights_json, benchmark_return, alpha,
                  var_95, portfolio_vol, probability_up))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        if DEBUG:
            print(f"Kayıt hatası: {e}")
        conn.close()
        return False

def backfill_portfolio_performance(lookback_days=30):
    """Eksik portföy performans verilerini doldur"""
    setup_portfolio_performance_table()
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    today = datetime.now().date()
    earliest_date = today - timedelta(days=lookback_days)
    
    if DEBUG:
        print(f"[Backfill] Bugün: {today}, En erken tarih: {earliest_date}")
    
    # Horizon dolmuş ama gerçekleşen getiri boş olan kayıtları bul
    c.execute('''
        SELECT tahmin_tarihi, tahmin_horizon, portfoy_tahmini_getiri, portfoy_agirliklari
        FROM portfoy_performans
        WHERE portfoy_gerceklesen_getiri IS NULL
        AND DATE(tahmin_tarihi) >= ?
        AND DATE(tahmin_tarihi) <= ?
    ''', (earliest_date.isoformat(), today.isoformat()))
    
    rows = c.fetchall()
    conn.close()
    
    if DEBUG:
        print(f"[Backfill] Bulunan kayıt sayısı: {len(rows)}")
    
    updated = 0
    for row in rows:
        date_str, horizon, predicted, weights_json = row
        # ORİJİNAL timestamp'i koru (sadece gün değil)
        forecast_timestamp = pd.to_datetime(date_str)
        forecast_date = forecast_timestamp.date()
        # DÜZELTME: İş günü hesaplama
        target_date = add_business_days(forecast_date, horizon)
        
        if DEBUG:
            print(f"[Backfill] İşleniyor: Tahmin={forecast_date}, Horizon={horizon}, Hedef={target_date}")
        
        if target_date > today:
            if DEBUG:
                print(f"[Backfill] Atlandı: Horizon henüz dolmadı (hedef {target_date} > bugün {today})")
            continue  # Henüz horizon dolmadı
        
        # Ağırlıkları parse et
        import json
        weights_dict = json.loads(weights_json) if weights_json else None
        
        if not weights_dict:
            if DEBUG:
                print(f"[Backfill] Atlandı: Ağırlıklar bulunamadı")
            continue
        
        if DEBUG:
            print(f"[Backfill] Gerçekleşen getiri hesaplanıyor...")
        # Gerçekleşen getiriyi hesapla
        actual_return = calculate_portfolio_actual(
            pd.Timestamp(forecast_date), horizon, weights_dict
        )
        
        if actual_return is not None:
            if DEBUG:
                print(f"[Backfill] Gerçekleşen getiri: {actual_return:.4f}")
            # Benchmark getirisi (BIST30 endeksi)
            benchmark_return = None
            try:
                start = (forecast_date - timedelta(days=5)).strftime('%Y-%m-%d')
                end = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')
                index_prices = yf.download('XU030.IS', start=start, end=end, progress=False)['Close']
                if index_prices is None or (hasattr(index_prices, 'empty') and index_prices.empty):
                    index_prices = pd.Series(dtype=float)
                if isinstance(index_prices, pd.DataFrame):
                    index_prices = index_prices.squeeze()
                if len(index_prices) < 2:
                    benchmark_return = None
                else:
                    # Tarih bazli eslestirme (yfinance timezone nedeniyle 'in' yanlis gun verebiliyordu)
                    idx = pd.DatetimeIndex(index_prices.index)
                    if idx.tz is not None:
                        idx = idx.tz_localize(None)
                    index_dates = pd.Series(idx.date, index=index_prices.index)
                    # Baslangic: forecast_date veya oncesi son is gunu
                    start_mask = index_dates.values <= forecast_date
                    if not start_mask.any():
                        start_idx = index_prices.iloc[0]
                    else:
                        start_idx = index_prices.iloc[np.where(start_mask)[0][-1]]
                    # Bitis: target_date veya sonrasi ilk is gunu
                    end_mask = index_dates.values >= target_date
                    if not end_mask.any():
                        end_idx = index_prices.iloc[-1]
                    else:
                        end_idx = index_prices.iloc[np.where(end_mask)[0][0]]
                    benchmark_return = float((end_idx / start_idx) - 1)
                    if DEBUG:
                        print(f"[Backfill] Benchmark: {forecast_date} ({start_idx:.2f}) -> {target_date} ({end_idx:.2f}) = {benchmark_return:.4f}")
            except Exception as e:
                if DEBUG:
                    print(f"[Backfill] Benchmark hesaplanamadı: {e}")
                pass
            
            # Güncelle - ORİJİNAL timestamp ile (kayıt eşleşmesi için)
            save_portfolio_forecast(
                forecast_timestamp, horizon, predicted, 
                actual_return, weights_dict, benchmark_return
            )
            print(f"[Backfill] OK: Portfoy performansi guncellendi: {forecast_date}")
            updated += 1
        else:
            if DEBUG:
                print(f"[Backfill] HATA: Gerceklesen getiri hesaplanamadi")
    
    if updated > 0:
        print(f"[Backfill] Toplam {updated} portföy performansı güncellendi")
    return updated


def get_benchmark_return(forecast_date, target_date, max_retries=3):
    """
    BIST30 (XU030.IS) getirisini tahmin tarihi -> hedef tarih aralığında hesaplar.
    Tarih eşleştirmesi tarih bazlı (timezone hatası önlenir).
    """
    import time
    try:
        if hasattr(forecast_date, 'date'):
            forecast_date = forecast_date.date()
        if hasattr(target_date, 'date'):
            target_date = target_date.date()
        start = (pd.Timestamp(forecast_date) - timedelta(days=5)).strftime('%Y-%m-%d')
        end = (pd.Timestamp(target_date) + timedelta(days=5)).strftime('%Y-%m-%d')
        index_prices = None
        for attempt in range(max_retries):
            try:
                index_prices = yf.download('XU030.IS', start=start, end=end, progress=False, timeout=15)['Close']
                if index_prices is not None and (not hasattr(index_prices, 'empty') or not index_prices.empty):
                    break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(8)
                else:
                    raise
        if index_prices is None or (hasattr(index_prices, 'empty') and index_prices.empty):
            return None
        if isinstance(index_prices, pd.DataFrame):
            index_prices = index_prices.squeeze()
        if len(index_prices) < 2:
            return None
        idx = pd.DatetimeIndex(index_prices.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        index_dates = pd.Series(idx.date, index=index_prices.index)
        start_mask = index_dates.values <= forecast_date
        if not start_mask.any():
            start_idx = index_prices.iloc[0]
        else:
            start_idx = index_prices.iloc[np.where(start_mask)[0][-1]]
        end_mask = index_dates.values >= target_date
        if not end_mask.any():
            end_idx = index_prices.iloc[-1]
        else:
            end_idx = index_prices.iloc[np.where(end_mask)[0][0]]
        return float((end_idx / start_idx) - 1)
    except Exception:
        return None


if __name__ == "__main__":
    # Test
    setup_portfolio_performance_table()
    print("Portföy performans tablosu oluşturuldu!")

