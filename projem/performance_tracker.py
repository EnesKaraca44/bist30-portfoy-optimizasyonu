"""
Performans Takip Modülü
Tahminlerin gerçekleşen performansını takip eder ve metrikler hesaplar.
"""
import sqlite3
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import DB_PATH as DB_NAME
from utils.dates import add_business_days

def setup_performance_tables():
    """Performans takip tablolarını oluşturur."""
    conn = sqlite3.connect(DB_NAME, timeout=30.0)
    conn.execute('PRAGMA journal_mode=WAL')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS tahminler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            tahmin_tarihi TEXT NOT NULL,
            tahmin_horizon INTEGER NOT NULL,
            tahmini_getiri REAL,
            gerceklesen_getiri REAL,
            tahmin_hatasi REAL,
            model_tipi TEXT,
            yukselis_olasiligi REAL,
            UNIQUE(ticker, tahmin_tarihi, tahmin_horizon)
        )
    ''')

    # Migrasyon: eski veritabanlarında yukselis_olasiligi kolonu olmayabilir
    c.execute("PRAGMA table_info(tahminler)")
    existing_cols = {row[1] for row in c.fetchall()}
    if 'yukselis_olasiligi' not in existing_cols:
        c.execute("ALTER TABLE tahminler ADD COLUMN yukselis_olasiligi REAL")
    if 'ham_tahmini_getiri' not in existing_cols:
        c.execute("ALTER TABLE tahminler ADD COLUMN ham_tahmini_getiri REAL")
    if 'bias_duzeltme' not in existing_cols:
        c.execute("ALTER TABLE tahminler ADD COLUMN bias_duzeltme REAL")

    conn.commit()
    conn.close()

def save_forecast(ticker, forecast_date, horizon_days, predicted_return, actual_return=None, model_type='ARIMA', conn=None):
    """
    Tahmin kaydını veritabanına kaydeder.
    
    Args:
        ticker: Hisse sembolü
        forecast_date: Tahmin tarihi (ISO format string)
        horizon_days: Tahmin ufku (gün)
        predicted_return: Tahmini getiri
        actual_return: Gerçekleşen getiri (opsiyonel)
        model_type: Model tipi (ARIMA, LSTM, Ensemble)
        conn: Mevcut veritabanı bağlantısı (opsiyonel)
    """
    should_close = False
    if conn is None:
        conn = sqlite3.connect(DB_NAME, timeout=30.0)
        conn.execute('PRAGMA journal_mode=WAL')
        should_close = True
    
    try:
        error = None
        if actual_return is not None and predicted_return is not None:
            error = abs(actual_return - predicted_return)
        
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO tahminler 
            (ticker, tahmin_tarihi, tahmin_horizon, tahmini_getiri, 
             gerceklesen_getiri, tahmin_hatasi, model_tipi)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (ticker, forecast_date, horizon_days, predicted_return, actual_return, error, model_type))
        conn.commit()
    except sqlite3.OperationalError as e:
        if 'locked' in str(e).lower():
            time.sleep(0.5)
            # Retry once
            try:
                c.execute('''
                    INSERT OR REPLACE INTO tahminler 
                    (ticker, tahmin_tarihi, tahmin_horizon, tahmini_getiri, 
                     gerceklesen_getiri, tahmin_hatasi, model_tipi)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (ticker, forecast_date, horizon_days, predicted_return, actual_return, error, model_type))
                conn.commit()
            except:
                pass
    except Exception as e:
        if should_close:
            conn.close()
        raise
    
    if should_close:
        conn.close()

def backfill_actuals_for_ticker(ticker, horizon_days=5, lookback_days=30):
    """
    Eksik gerçekleşen getirileri otomatik doldur
    
    Algoritma:
    1. Horizon dolmuş ama gerçekleşen getiri boş olan tahminleri bul
    2. Her tahmin için:
       a. Tahmin tarihi + horizon (iş günü) = hedef tarih
       b. Tahmin tarihindeki fiyatı bul
       c. Hedef tarihteki fiyatı bul
       d. Gerçekleşen getiri = (hedef_fiyat / tahmin_fiyat) - 1
    3. Veritabanını güncelle
    """
    import time
    from datetime import date
    
    today = datetime.now().date()
    # Tahmin bitiş tarihi (tahmin_tarihi + horizon) bugünden önce veya bugüne eşit olmalı
    # En erken tahmin tarihi: bugünden (horizon + lookback) gün önce
    earliest_date = today - timedelta(days=lookback_days + horizon_days)
    
    conn = sqlite3.connect(DB_NAME, timeout=30.0)
    conn.execute('PRAGMA journal_mode=WAL')
    
    try:
        # Horizon dolmuş, gerçekleşen getiri boş olan tahminleri bul
        # SQLite DATE() fonksiyonu ISO datetime'ı parse edemeyebilir,
        # bu yüzden tüm kayıtları çekip Python'da filtreleyeceğiz
        query = """
            SELECT id, tahmin_tarihi, tahmini_getiri, tahmin_horizon
            FROM tahminler 
            WHERE ticker = ? 
            AND gerceklesen_getiri IS NULL 
            AND tahmin_horizon IS NOT NULL 
            AND tahmin_horizon > 0
        """
        params = (ticker,)
        
        df = pd.read_sql_query(query, conn, params=params)
        
        # Python'da tarih filtreleme (ISO datetime formatını doğru parse eder)
        # Tahmin bitiş tarihi (tahmin_tarihi + horizon) bugünden önce veya bugüne eşit olmalı
        if not df.empty:
            df['tahmin_tarihi_parsed'] = pd.to_datetime(df['tahmin_tarihi']).dt.date
            # DÜZELTME: İş günü hesaplama (takvim günü değil!)
            df['tahmin_bitis_tarihi'] = df.apply(
                lambda row: add_business_days(row['tahmin_tarihi_parsed'], int(row['tahmin_horizon'])), 
                axis=1
            )
            # Tahmin bitiş tarihi bugünden önce/eşit ve tahmin tarihi earliest_date'den sonra olmalı
            df = df[
                (df['tahmin_bitis_tarihi'] <= today) & 
                (df['tahmin_tarihi_parsed'] >= earliest_date)
            ]
            df = df.drop(['tahmin_tarihi_parsed', 'tahmin_bitis_tarihi'], axis=1)
        
        if df.empty:
            conn.close()
            return 0
        
        # Tarih aralığında fiyat verilerini çek
        start = (earliest_date - timedelta(days=10)).isoformat()
        # DÜZELTME: Bugün+5 gün çek (bugünün fiyatını garantilemek için)
        end = (today + timedelta(days=5)).isoformat()
        
        try:
            prices = yf.download(ticker, start=start, end=end, progress=False)['Close']
            prices = prices.dropna()
            
            if len(prices) < 2:
                conn.close()
                return 0
            
            # Debug: Fiyat verisi çekildi mi?
            # print(f"[DEBUG] {ticker}: {len(prices)} fiyat verisi, {len(df)} tahmin güncellenecek")
            
            updates = []
            cur = conn.cursor()
            
            # print(f"[DEBUG] For loop basliyor, {len(df)} satir")
            for _, row in df.iterrows():
                # print(f"[DEBUG] Loop: ID={row['id']}")
                try:
                    # ISO datetime formatını parse et
                    try:
                        forecast_date = pd.to_datetime(row['tahmin_tarihi']).date()
                    except:
                        # Eğer parse edilemezse, string'den ilk 10 karakteri al (YYYY-MM-DD)
                        date_str = str(row['tahmin_tarihi'])[:10]
                        forecast_date = pd.to_datetime(date_str).date()
                    
                    # Horizon değerini al (SELECT'te ekledik)
                    horizon = int(row.get('tahmin_horizon') or horizon_days)
                    
                    # DÜZELTME: Tahmin tarihinden itibaren horizon kadar İŞ GÜNÜ sonraki fiyatı bul
                    target_date = add_business_days(forecast_date, horizon)
                    
                    if target_date > today:
                        continue  # Henüz horizon dolmadı
                    
                    # En yakın fiyatı bul (target_date'e en yakın işlem günü)
                    # DatetimeIndex ile date karşılaştırması
                    target_ts = pd.Timestamp(target_date)
                    forecast_ts = pd.Timestamp(forecast_date)
                    
                    # Target date için fiyat bul (tam eşleşme veya sonraki en yakın)
                    end_price = None
                    if target_ts in prices.index:
                        end_price = float(prices.loc[target_ts])
                    else:
                        # Sonraki en yakın tarihi al
                        future_prices = prices[prices.index >= target_ts]
                        if len(future_prices) > 0:
                            end_price = float(future_prices.iloc[0])
                    
                    if end_price is None or pd.isna(end_price) or end_price <= 0:
                        continue
                    
                    # Başlangıç fiyatını bul (tam eşleşme veya önceki en yakın)
                    start_price = None
                    if forecast_ts in prices.index:
                        start_price = float(prices.loc[forecast_ts])
                    else:
                        # Önceki en yakın tarihi al
                        past_prices = prices[prices.index <= forecast_ts]
                        if len(past_prices) > 0:
                            start_price = float(past_prices.iloc[-1])
                    
                    if start_price is None or pd.isna(start_price) or start_price <= 0:
                        continue
                    
                    # Gerçekleşen getiriyi hesapla
                    actual_return = float((end_price / start_price) - 1)
                    error = abs(actual_return - row['tahmini_getiri'])
                    
                    updates.append((actual_return, error, row['id']))
                except Exception as e:
                    # Hata olursa devam et (debug için print edebilirsiniz)
                    continue
            
            if updates:
                cur.executemany(
                    "UPDATE tahminler SET gerceklesen_getiri = ?, tahmin_hatasi = ? WHERE id = ?",
                    updates
                )
                conn.commit()
            
            conn.close()
            return len(updates)
            
        except Exception as e:
            conn.close()
            return 0
            
    except Exception as e:
        conn.close()
        return 0

def get_performance_metrics(ticker, start_date=None, end_date=None):
    """
    Belirli bir hisse için performans metriklerini getirir.
    
    Args:
        ticker: Hisse sembolü
        start_date: Başlangıç tarihi (opsiyonel)
        end_date: Bitiş tarihi (opsiyonel)
    
    Returns:
        dict: Performans metrikleri
    """
    conn = sqlite3.connect(DB_NAME, timeout=30.0)
    
    query = "SELECT * FROM tahminler WHERE ticker = ?"
    params = [ticker]
    
    if start_date:
        query += " AND tahmin_tarihi >= ?"
        params.append(start_date)
    if end_date:
        query += " AND tahmin_tarihi <= ?"
        params.append(end_date)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return {
            'mae': None,
            'rmse': None,
            'accuracy': None,
            'total_forecasts': 0,
            'valid_forecasts': 0
        }
    
    # Sadece gerçekleşen getirisi olan tahminleri kullan
    valid_df = df[df['gerceklesen_getiri'].notna() & df['tahmini_getiri'].notna()]
    
    if len(valid_df) == 0:
        return {
            'mae': None,
            'rmse': None,
            'accuracy': None,
            'total_forecasts': len(df),
            'valid_forecasts': 0
        }
    
    errors = abs(valid_df['gerceklesen_getiri'] - valid_df['tahmini_getiri'])
    mae = errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    
    # Yön doğruluğu (direction accuracy)
    direction_correct = (
        (valid_df['tahmini_getiri'] > 0) == (valid_df['gerceklesen_getiri'] > 0)
    ).sum()
    accuracy = direction_correct / len(valid_df)
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'accuracy': float(accuracy),
        'total_forecasts': len(df),
        'valid_forecasts': len(valid_df),
        'mean_absolute_error': float(mae),
        'root_mean_squared_error': float(rmse),
        'direction_accuracy': float(accuracy)
    }

def check_horizon_status(ticker, forecast_date, horizon_days):
    """
    Horizon doldu mu kontrol et.
    
    Returns:
        dict: {'ready': bool, 'days_remaining': int, 'message': str}
    """
    today = datetime.now().date()
    forecast_dt = pd.to_datetime(forecast_date).date()
    target_date = forecast_dt + timedelta(days=horizon_days)
    
    days_remaining = (target_date - today).days
    
    if days_remaining <= 0:
        return {
            'ready': True,
            'days_remaining': 0,
            'message': 'Horizon doldu, doğrulama yapılabilir.'
        }
    else:
        return {
            'ready': False,
            'days_remaining': days_remaining,
            'message': f'Horizon henüz dolmadı. {days_remaining} işlem günü kaldı.'
        }
