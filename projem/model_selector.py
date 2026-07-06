"""
Model Selector Modülü
Her hisse için en iyi modeli seçer ve dinamik ensemble ağırlıkları hesaplar.
Performans bazlı adaptif model seçimi.
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import DB_PATH as DB_NAME


def get_recent_forecasts(ticker, limit=20, model_type=None):
    """
    Bir hisse için son N tahminini getir
    
    Args:
        ticker: Hisse sembolü
        limit: Kaç tahmin getirileceği
        model_type: Model tipi filtresi ('ARIMA', 'LSTM', 'Ensemble')
    
    Returns:
        DataFrame: Tahminler
    """
    try:
        conn = sqlite3.connect(DB_NAME, timeout=30.0)
        
        query = """
            SELECT ticker, tahmin_tarihi, tahmini_getiri, gerceklesen_getiri, 
                   tahmin_hatasi, model_tipi
            FROM tahminler 
            WHERE ticker = ? 
            AND gerceklesen_getiri IS NOT NULL
            AND tahmin_hatasi IS NOT NULL
        """
        
        params = [ticker]
        
        if model_type:
            query += " AND model_tipi = ?"
            params.append(model_type)
        
        query += " ORDER BY tahmin_tarihi DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    except Exception as e:
        return pd.DataFrame()


def calculate_model_performance(ticker, lookback_forecasts=20):
    """
    Her model için performans metriklerini hesapla
    
    Args:
        ticker: Hisse sembolü
        lookback_forecasts: Son kaç tahmine bakılacağı
    
    Returns:
        dict: Her model için metrikler
    """
    results = {}
    
    # Her model tipi için metrik hesapla
    for model_type in ['ARIMA', 'LSTM', 'ARIMA+LSTM', 'ARIMA+BL', 'ARIMA+BL+LSTM']:
        df = get_recent_forecasts(ticker, limit=lookback_forecasts, model_type=model_type)
        
        if len(df) < 5:  # Minimum 5 tahmin gerekli
            results[model_type] = {
                'mae': np.inf,
                'rmse': np.inf,
                'direction_accuracy': 0.0,
                'count': len(df),
                'available': False
            }
            continue
        
        # Metrikler
        errors = np.abs(df['gerceklesen_getiri'] - df['tahmini_getiri'])
        mae = errors.mean()
        rmse = np.sqrt((errors ** 2).mean())
        
        # Direction accuracy
        correct_direction = (
            (df['tahmini_getiri'] > 0) == (df['gerceklesen_getiri'] > 0)
        ).sum()
        direction_acc = correct_direction / len(df)
        
        results[model_type] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'direction_accuracy': float(direction_acc),
            'count': len(df),
            'available': True
        }
    
    return results


def calculate_model_weights(ticker, lookback_forecasts=20, method='inverse_mae'):
    """
    Performans bazlı model ağırlıklarını hesapla
    
    Args:
        ticker: Hisse sembolü
        lookback_forecasts: Son kaç tahmine bakılacağı
        method: Ağırlık hesaplama yöntemi ('inverse_mae', 'inverse_rmse', 'direction_weighted')
    
    Returns:
        dict: Model ağırlıkları {'arima': 0.6, 'lstm': 0.4}
    """
    # Varsayılan ağırlıklar (yeterli veri yoksa)
    default_weights = {'arima': 0.5, 'lstm': 0.5}
    
    try:
        # Her model için performans al
        # ARIMA bazlı model (ARIMA veya ARIMA+BL)
        arima_df = get_recent_forecasts(ticker, limit=lookback_forecasts)
        arima_df = arima_df[arima_df['model_tipi'].str.contains('ARIMA', na=False)]
        
        # LSTM içeren model
        lstm_df = get_recent_forecasts(ticker, limit=lookback_forecasts)
        lstm_df = lstm_df[lstm_df['model_tipi'].str.contains('LSTM', na=False)]
        
        # Yeterli veri yoksa varsayılan
        if len(arima_df) < 5 or len(lstm_df) < 5:
            return default_weights
        
        # ARIMA metrikleri
        arima_errors = np.abs(arima_df['gerceklesen_getiri'] - arima_df['tahmini_getiri'])
        arima_mae = arima_errors.mean()
        arima_rmse = np.sqrt((arima_errors ** 2).mean())
        
        # LSTM metrikleri
        lstm_errors = np.abs(lstm_df['gerceklesen_getiri'] - lstm_df['tahmini_getiri'])
        lstm_mae = lstm_errors.mean()
        lstm_rmse = np.sqrt((lstm_errors ** 2).mean())
        
        # Direction accuracy
        arima_direction = ((arima_df['tahmini_getiri'] > 0) == (arima_df['gerceklesen_getiri'] > 0)).mean()
        lstm_direction = ((lstm_df['tahmini_getiri'] > 0) == (lstm_df['gerceklesen_getiri'] > 0)).mean()
        
        # Ağırlık hesaplama
        if method == 'inverse_mae':
            # Düşük hata = yüksek ağırlık
            arima_score = 1 / (arima_mae + 1e-6)
            lstm_score = 1 / (lstm_mae + 1e-6)
        elif method == 'inverse_rmse':
            arima_score = 1 / (arima_rmse + 1e-6)
            lstm_score = 1 / (lstm_rmse + 1e-6)
        elif method == 'direction_weighted':
            # Direction accuracy ve MAE kombinasyonu
            arima_score = arima_direction * (1 / (arima_mae + 1e-6))
            lstm_score = lstm_direction * (1 / (lstm_mae + 1e-6))
        else:
            arima_score = 1 / (arima_mae + 1e-6)
            lstm_score = 1 / (lstm_mae + 1e-6)
        
        # Normalize et
        total_score = arima_score + lstm_score
        
        if total_score > 0:
            weights = {
                'arima': arima_score / total_score,
                'lstm': lstm_score / total_score
            }
        else:
            weights = default_weights
        
        # Aşırı uç ağırlıkları sınırla (minimum %20, maksimum %80)
        weights['arima'] = np.clip(weights['arima'], 0.2, 0.8)
        weights['lstm'] = np.clip(weights['lstm'], 0.2, 0.8)
        
        # Tekrar normalize et
        total = weights['arima'] + weights['lstm']
        weights['arima'] /= total
        weights['lstm'] /= total
        
        return weights
        
    except Exception as e:
        return default_weights


def get_best_model_for_ticker(ticker, lookback_forecasts=20, threshold=0.7):
    """
    Bir hisse için en iyi modeli seç
    
    Args:
        ticker: Hisse sembolü
        lookback_forecasts: Son kaç tahmine bakılacağı
        threshold: Tek model kullanmak için minimum ağırlık eşiği
    
    Returns:
        tuple: (model_name, weight_or_weights)
            - Tek model: ('ARIMA', 1.0) veya ('LSTM', 1.0)
            - Ensemble: ('Ensemble', {'arima': 0.6, 'lstm': 0.4})
    """
    weights = calculate_model_weights(ticker, lookback_forecasts)
    
    # Eğer bir model çok dominant ise onu tek başına kullan
    if weights['arima'] >= threshold:
        return ('ARIMA', 1.0)
    elif weights['lstm'] >= threshold:
        return ('LSTM', 1.0)
    else:
        return ('Ensemble', weights)


def get_all_model_weights(tickers, lookback_forecasts=20, method='inverse_mae'):
    """
    Tüm hisseler için model ağırlıklarını hesapla
    
    Args:
        tickers: Hisse listesi
        lookback_forecasts: Son kaç tahmine bakılacağı
        method: Ağırlık hesaplama yöntemi
    
    Returns:
        dict: {ticker: {'arima': weight, 'lstm': weight}}
    """
    all_weights = {}
    
    for ticker in tickers:
        weights = calculate_model_weights(ticker, lookback_forecasts, method)
        all_weights[ticker] = weights
    
    return all_weights


def generate_model_performance_report(tickers=None, lookback_forecasts=20):
    """
    Model performans raporu oluştur
    
    Args:
        tickers: Hisse listesi (None ise tüm hisseler)
        lookback_forecasts: Son kaç tahmine bakılacağı
    
    Returns:
        DataFrame: Performans raporu
    """
    if tickers is None:
        # Veritabanından tüm ticker'ları al
        conn = sqlite3.connect(DB_NAME, timeout=30.0)
        c = conn.cursor()
        c.execute("SELECT DISTINCT ticker FROM tahminler ORDER BY ticker")
        tickers = [row[0] for row in c.fetchall()]
        conn.close()
    
    report_data = []
    
    for ticker in tickers:
        # Performans metrikleri
        performance = calculate_model_performance(ticker, lookback_forecasts)
        
        # Ağırlıklar
        weights = calculate_model_weights(ticker, lookback_forecasts)
        
        # En iyi model
        best_model, best_weight = get_best_model_for_ticker(ticker, lookback_forecasts)
        
        report_data.append({
            'ticker': ticker,
            'best_model': best_model,
            'arima_weight': weights['arima'],
            'lstm_weight': weights['lstm'],
            'arima_mae': performance.get('ARIMA', {}).get('mae', np.nan),
            'lstm_mae': performance.get('LSTM', {}).get('mae', np.nan),
            'arima_direction_acc': performance.get('ARIMA', {}).get('direction_accuracy', np.nan),
            'lstm_direction_acc': performance.get('LSTM', {}).get('direction_accuracy', np.nan),
            'arima_count': performance.get('ARIMA', {}).get('count', 0),
            'lstm_count': performance.get('LSTM', {}).get('count', 0)
        })
    
    df = pd.DataFrame(report_data)
    return df


if __name__ == "__main__":
    # Test
    print("[*] Model Selector Test")
    print("=" * 60)
    
    # Örnek hisse
    ticker = "AKBNK.IS"
    
    print(f"\n[*] {ticker} için model performansı:")
    performance = calculate_model_performance(ticker)
    
    for model_type, metrics in performance.items():
        if metrics['available']:
            print(f"\n{model_type}:")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
            print(f"  Tahmin Sayısı: {metrics['count']}")
    
    print(f"\n[*] {ticker} için önerilen ağırlıklar:")
    weights = calculate_model_weights(ticker)
    print(f"  ARIMA: {weights['arima']:.2%}")
    print(f"  LSTM: {weights['lstm']:.2%}")
    
    print(f"\n[*] {ticker} için en iyi model:")
    best_model, best_weight = get_best_model_for_ticker(ticker)
    print(f"  {best_model}: {best_weight}")
    
    print("\n[+] Test tamamlandı!")
