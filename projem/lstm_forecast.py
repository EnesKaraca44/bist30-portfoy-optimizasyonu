"""
LSTM tabanlı zaman serisi tahmini modülü
OPTİMİZE EDİLDİ: Model her tahminde sıfırdan eğitilmez, önceden eğitilmiş model kaydedilir ve yüklenir.
FEATURE ENGINEERING: Multi-feature input desteği (returns, rsi, macd, volatility, momentum)
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Feature Engineering modülünü yükle
try:
    from feature_engineering import prepare_lstm_features
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False

# Model cache (bellekte tutmak için - aynı ticker için tekrar yüklemeden kullan)
_MODEL_CACHE = {}
_SCALER_CACHE = {}

from config import LSTM_MODELS_DIR

# Model dosyaları için klasör
MODELS_DIR = LSTM_MODELS_DIR
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Model geçerlilik süresi (gün) - bu süreden eski modeller yeniden eğitilir
MODEL_MAX_AGE_DAYS = 30
# Multi-feature modda güvenilir iterasyon; sonrası geometrik extrapolasyon
MULTI_FEATURE_RELIABLE_DAYS = 5
# Günlük tahmin clip — aşırı bileşik getiriyi engeller
DAILY_PRED_CLIP = 0.06


def compound_period_from_daily(daily_preds, target_days=None, reliable_steps=None):
    """
  Günlük tahminlerden bileşik dönem getirisi.
  reliable_steps < target_days ise kalan günler geometrik extrapolasyonla tamamlanır.
    """
    preds = np.clip(np.asarray(daily_preds, dtype=float), -DAILY_PRED_CLIP, DAILY_PRED_CLIP)
    if len(preds) == 0:
        return 0.0
    steps = reliable_steps or len(preds)
    period = float(np.prod(1.0 + preds) - 1.0)
    if target_days and target_days > steps:
        period = float((1.0 + period) ** (target_days / steps) - 1.0)
    return period

def create_lstm_model(input_shape):
    """
    LSTM model mimarisi oluşturur
    
    Multi-feature için:
    - 3 katmanlı LSTM (128, 64, 32 nöron)
    - Dropout (%30)
    - Dense katmanlar (16, 1)
    
    Single feature için:
    - 2 katmanlı LSTM (64, 32 nöron)
    - Dropout (%20)
    """
    n_features = input_shape[1] if len(input_shape) > 1 else 1
    
    if n_features > 1:
        # Multi-feature model (daha derin)
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
    else:
        # Single feature model (orijinal)
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def prepare_lstm_data(returns_series, lookback=60, scaler=None):
    """
    LSTM için veri hazırlığı (Single feature - backward compatibility)
    
    Args:
        returns_series: Günlük getiri serisi
        lookback: Geriye bakış penceresi
        scaler: Eğer varsa mevcut scaler'ı kullan (inference için)
    
    Returns:
        X, y, scaler: Eğitim verisi ve scaler
    """
    if len(returns_series) < lookback + 50:
        return None, None, None
    
    # Son 500 günü kullan (bellek optimizasyonu)
    data = returns_series.values[-500:].reshape(-1, 1)
    
    # Eğer scaler yoksa yeni oluştur (eğitim), varsa kullan (inference)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        return None, None, None
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler


def prepare_lstm_data_multifeature(prices, returns_series, lookback=60, scaler=None):
    """
    LSTM için multi-feature veri hazırlığı
    
    Args:
        prices: Fiyat serisi (feature'lar için gerekli)
        returns_series: Günlük getiri serisi
        lookback: Geriye bakış penceresi
        scaler: Eğer varsa mevcut scaler'ı kullan (inference için)
    
    Returns:
        X, y, scaler: Eğitim verisi ve scaler
    """
    if not FEATURES_AVAILABLE or prices is None or len(prices) < 100:
        # Feature engineering yoksa basit veriyi kullan
        return prepare_lstm_data(returns_series, lookback, scaler)
    
    try:
        # Feature DataFrame'i oluştur
        feature_df = prepare_lstm_features(prices, returns_series, volume=None, n_features=5)
        
        if feature_df.empty or len(feature_df) < lookback + 50:
            # Feature oluşturma başarısız, basit veri kullan
            return prepare_lstm_data(returns_series, lookback, scaler)
        
        # Son 500 satırı kullan (bellek optimizasyonu)
        if len(feature_df) > 500:
            feature_df = feature_df[-500:]
        
        # Feature'ları numpy array'e çevir
        data = feature_df.values
        n_features = data.shape[1]
        
        # Eğer scaler yoksa yeni oluştur (eğitim), varsa kullan (inference)
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
        else:
            scaled_data = scaler.transform(data)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, :])  # Tüm feature'lar
            y.append(scaled_data[i, 0])  # Sadece returns'ü tahmin et (ilk kolon)
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            return None, None, None
        
        # Shape zaten (samples, lookback, features) formatında
        
        return X, y, scaler
        
    except Exception as e:
        # Hata durumunda basit veri kullan
        return prepare_lstm_data(returns_series, lookback, scaler)

def get_model_path(ticker):
    """Ticker için model dosya yolunu döndür (.keras formatı - modern)"""
    return os.path.join(MODELS_DIR, f"{ticker}_model.keras")

def get_scaler_path(ticker):
    """Ticker için scaler dosya yolunu döndür"""
    return os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")

def get_model_meta_path(ticker):
    """Ticker için model metadata dosya yolunu döndür"""
    return os.path.join(MODELS_DIR, f"{ticker}_meta.pkl")

def is_model_valid(ticker):
    """Model dosyası var mı ve geçerli mi kontrol et"""
    model_path = get_model_path(ticker)
    scaler_path = get_scaler_path(ticker)
    meta_path = get_model_meta_path(ticker)
    
    # Sadece .keras formatını kabul et (eski .h5 dosyalarını ignore et)
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(meta_path)):
        # Eski .h5 dosyası varsa geçersiz say
        old_model_path = os.path.join(MODELS_DIR, f"{ticker}_model.h5")
        if os.path.exists(old_model_path):
            return False  # Eski format geçersiz
        return False
    
    # Model yaşını kontrol et
    try:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        train_date = meta.get('train_date', datetime.min)
        age = (datetime.now() - train_date).days
        return age < MODEL_MAX_AGE_DAYS
    except:
        return False

def save_lstm_model(ticker, model, scaler):
    """LSTM modelini ve scaler'ı kaydet (.keras formatı kullan)"""
    try:
        model_path = get_model_path(ticker)
        scaler_path = get_scaler_path(ticker)
        meta_path = get_model_meta_path(ticker)
        
        # .keras formatında kaydet (modern format, metrics sorunu yok)
        # Eski .h5 dosyasını sil (varsa)
        old_model_path = model_path.replace('.keras', '.h5')
        if os.path.exists(old_model_path):
            try:
                os.remove(old_model_path)
            except:
                pass
        
        # Modern .keras formatında kaydet
        model.save(model_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        meta = {'train_date': datetime.now(), 'ticker': ticker}
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        # Cache'e ekle
        _MODEL_CACHE[ticker] = model
        _SCALER_CACHE[ticker] = scaler
        
    except Exception as e:
        print(f"[!] Model kaydetme hatasi ({ticker}): {e}")

def load_lstm_model(ticker):
    """Kaydedilmiş LSTM modelini ve scaler'ı yükle"""
    # Önce cache'den kontrol et
    if ticker in _MODEL_CACHE and ticker in _SCALER_CACHE:
        return _MODEL_CACHE[ticker], _SCALER_CACHE[ticker]
    
    try:
        model_path = get_model_path(ticker)
        scaler_path = get_scaler_path(ticker)
        
        # Eski .h5 formatı da kontrol et (backward compatibility)
        old_model_path = os.path.join(MODELS_DIR, f"{ticker}_model.h5")
        if os.path.exists(old_model_path) and not os.path.exists(model_path):
            # Eski format varsa yeniye dönüştür
            try:
                old_model = load_model(old_model_path)
                old_model.save(model_path, save_format='keras')
                os.remove(old_model_path)  # Eski dosyayı sil
            except:
                model_path = old_model_path  # Dönüşüm başarısızsa eskiyi kullan
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            return None, None
        
        # Modeli yükle
        model = load_model(model_path, compile=False)
        # Modeli compile et (metrics olmadan, sadece inference için)
        model.compile(optimizer='adam', loss='mse')
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Cache'e ekle
        _MODEL_CACHE[ticker] = model
        _SCALER_CACHE[ticker] = scaler
        
        return model, scaler
    except Exception as e:
        print(f"[!] Model yukleme hatasi ({ticker}): {e}")
        return None, None

def train_lstm_model(ticker, returns_series, prices=None, lookback=60, use_features=True):
    """
    LSTM modelini eğit ve kaydet (Multi-feature destekli)
    
    Args:
        ticker: Hisse ticker'ı
        returns_series: Getiri serisi
        prices: Fiyat serisi (feature'lar için, opsiyonel)
        lookback: Geriye bakış penceresi
        use_features: Multi-feature kullan mı?
    """
    try:
        # Multi-feature veya single feature
        if use_features and FEATURES_AVAILABLE and prices is not None:
            X, y, scaler = prepare_lstm_data_multifeature(prices, returns_series, lookback=lookback)
        else:
            X, y, scaler = prepare_lstm_data(returns_series, lookback=lookback)
        
        if X is None or len(X) < 50:
            return None, None
        
        # Input shape'i belirle
        input_shape = (X.shape[1], X.shape[2])  # (lookback, n_features)
        
        # Modeli oluştur ve eğit
        model = create_lstm_model(input_shape)
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='loss', patience=3, restore_best_weights=True
        )
        
        # Hızlı eğitim için epoch sayısını azalttık (30 -> 15)
        model.fit(X, y, epochs=15, batch_size=32, verbose=0, 
                  callbacks=[early_stop], validation_split=0.1)
        
        # Modeli kaydet
        save_lstm_model(ticker, model, scaler)
        
        return model, scaler
    except Exception as e:
        print(f"[!] Model egitim hatasi ({ticker}): {e}")
        return None, None

def forecast_lstm(returns_series, forecast_days=22, ticker=None, prices=None, force_retrain=False, use_features=True, return_period=False):
    """
    LSTM ile tahmin yap (OPTİMİZE: Model cache sistemi)
    
    Algoritma:
    1. Model cache'den kontrol et (30 gün geçerli)
    2. Yoksa veya eskiyse eğit
    3. Son 60 günlük veri ile iteratif günlük tahmin
    4. Bileşik dönem getirisi (gerekirse ufka extrapolasyon)
    5. Varsayılan: yıllıklandırılmış getiri döner; return_period=True ise dönem getirisi
    """
    try:
        lookback = 60
        
        # Ticker yoksa seriden çıkarmaya çalış (pandas Series ise)
        if ticker is None and hasattr(returns_series, 'name'):
            ticker = returns_series.name
        if ticker is None:
            ticker = 'unknown'
        
        # Model yükleme veya eğitme
        model = None
        scaler = None
        
        if not force_retrain and is_model_valid(ticker):
            # Geçerli model varsa yükle
            model, scaler = load_lstm_model(ticker)
        
        if model is None or scaler is None:
            # Model yoksa veya geçersizse eğit
            model, scaler = train_lstm_model(ticker, returns_series, prices=prices, 
                                            lookback=lookback, use_features=use_features)
        
        if model is None or scaler is None:
            fallback_daily = float(returns_series.mean())
            if return_period:
                return compound_period_from_daily(
                    [fallback_daily] * min(forecast_days, MULTI_FEATURE_RELIABLE_DAYS),
                    target_days=forecast_days,
                    reliable_steps=min(forecast_days, MULTI_FEATURE_RELIABLE_DAYS),
                )
            return fallback_daily * 252
        
        # Inference için veri hazırla (scaler ile birlikte)
        if use_features and FEATURES_AVAILABLE and prices is not None:
            X, _, _ = prepare_lstm_data_multifeature(prices, returns_series, lookback=lookback, scaler=scaler)
        else:
            X, _, _ = prepare_lstm_data(returns_series, lookback=lookback, scaler=scaler)
        
        if X is None or len(X) < 1:
            fallback_daily = float(returns_series.mean())
            if return_period:
                return compound_period_from_daily(
                    [fallback_daily] * min(forecast_days, MULTI_FEATURE_RELIABLE_DAYS),
                    target_days=forecast_days,
                    reliable_steps=min(forecast_days, MULTI_FEATURE_RELIABLE_DAYS),
                )
            return fallback_daily * 252
        
        # Input shape'i belirle
        n_features = X.shape[2]
        
        # Tahmin yap
        last_sequence = X[-1].reshape(1, lookback, n_features)
        predictions = []
        
        # Multi-feature: kısa güvenilir pencere + ufka extrapolasyon
        reliable_horizon = (
            min(MULTI_FEATURE_RELIABLE_DAYS, forecast_days) if n_features > 1 else forecast_days
        )
        
        for step in range(reliable_horizon):
            pred = model.predict(last_sequence, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Sekansı kaydır
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = pred  # Returns'ü güncelle (ilk feature)
            # Diğer feature'ları son bilinen değerde sabitle (kayma yerine)
            if n_features > 1:
                for f_idx in range(1, n_features):
                    last_sequence[0, -1, f_idx] = last_sequence[0, -2, f_idx]
        
        # Tahminleri ölçeklendir
        predictions_reshaped = np.array(predictions).reshape(-1, 1)
        
        # Multi-feature ise sadece returns kolonunu inverse transform et
        if n_features > 1:
            # Scaler tüm feature'ları içeriyor, sadece ilk kolonu (returns) çıkar
            # Dummy feature'lar oluştur
            dummy_features = np.zeros((len(predictions), n_features))
            dummy_features[:, 0] = predictions  # İlk kolona predictions'ı koy
            
            # Inverse transform
            full_inverse = scaler.inverse_transform(dummy_features)
            predictions_unscaled = full_inverse[:, 0]  # Sadece returns kolonunu al
        else:
            predictions_unscaled = scaler.inverse_transform(predictions_reshaped).flatten()

        period_return = compound_period_from_daily(
            predictions_unscaled,
            target_days=forecast_days if n_features > 1 else None,
            reliable_steps=reliable_horizon if n_features > 1 else None,
        )

        if return_period:
            return float(period_return)

        days = max(int(forecast_days), 1)
        annualized = float((1.0 + period_return) ** (252.0 / days) - 1.0)
        annualized = np.clip(annualized, -1.0, 1.0)
        return annualized
        
    except Exception as e:
        fallback_daily = float(returns_series.mean())
        if return_period:
            return compound_period_from_daily(
                [fallback_daily] * min(forecast_days, MULTI_FEATURE_RELIABLE_DAYS),
                target_days=forecast_days,
                reliable_steps=min(forecast_days, MULTI_FEATURE_RELIABLE_DAYS),
            )
        return fallback_daily * 252

def ensemble_forecast(arima_pred, lstm_pred, arima_weight=0.6, lstm_weight=0.4):
    """
    ARIMA ve LSTM tahminlerini birleştir
    
    Args:
        arima_pred: ARIMA tahmini
        lstm_pred: LSTM tahmini
        arima_weight: ARIMA ağırlığı
        lstm_weight: LSTM ağırlığı
    
    Returns:
        float: Birleştirilmiş tahmin
    """
    if arima_weight + lstm_weight != 1.0:
        total = arima_weight + lstm_weight
        arima_weight /= total
        lstm_weight /= total
    
    return arima_weight * arima_pred + lstm_weight * lstm_pred

