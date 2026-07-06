"""
Feature Engineering Modülü
Fiyat ve getiri verilerinden zengin feature'lar çıkarır.
ML modellerinin performansını artırmak için tasarlanmıştır.
"""
import pandas as pd
import numpy as np
from technical_indicators import (
    calculate_rsi, 
    calculate_macd, 
    calculate_bollinger_bands,
    calculate_momentum_score
)
import warnings
warnings.filterwarnings('ignore')

try:
    from config import USE_MACRO_FEATURES, MACRO_N_FEATURES
except ImportError:
    USE_MACRO_FEATURES = False
    MACRO_N_FEATURES = 3

try:
    from macro_features import get_macro_features_for_index, MACRO_FEATURE_COLUMNS
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False


def calculate_volatility_features(returns, windows=[5, 10, 20, 30]):
    """
    Farklı periyotlarda volatilite özellikleri hesapla
    
    Args:
        returns: Günlük getiri serisi
        windows: Volatilite pencere boyutları
    
    Returns:
        dict: Volatilite feature'ları
    """
    features = {}
    
    for window in windows:
        if len(returns) >= window:
            try:
                # Rolling volatility
                vol = returns.rolling(window=window).std()
                vol_value = float(vol.iloc[-1]) if len(vol) > 0 else 0.0
                features[f'volatility_{window}d'] = vol_value if not np.isnan(vol_value) else 0.0
                
                # Volatility ratio (son dönem / genel)
                recent_vol = float(returns[-window:].std())
                overall_vol = float(returns.std())
                features[f'vol_ratio_{window}d'] = recent_vol / overall_vol if overall_vol > 0 else 1.0
            except:
                features[f'volatility_{window}d'] = 0.0
                features[f'vol_ratio_{window}d'] = 1.0
    
    return features


def calculate_trend_features(prices, windows=[10, 20, 50, 100]):
    """
    Trend özellikleri hesapla (SMA, EMA, trend gücü)
    
    Args:
        prices: Fiyat serisi
        windows: Trend pencere boyutları
    
    Returns:
        dict: Trend feature'ları
    """
    features = {}
    current_price = float(prices.iloc[-1])
    
    for window in windows:
        if len(prices) >= window:
            try:
                # Simple Moving Average
                sma = float(prices.rolling(window=window).mean().iloc[-1])
                features[f'sma_{window}'] = sma
                
                # Fiyat / SMA oranı (1'in üstü = yükseliş trendi)
                features[f'price_to_sma_{window}'] = current_price / sma if sma > 0 else 1.0
                
                # Exponential Moving Average
                ema = float(prices.ewm(span=window, adjust=False).mean().iloc[-1])
                features[f'ema_{window}'] = ema
                features[f'price_to_ema_{window}'] = current_price / ema if ema > 0 else 1.0
            except:
                features[f'sma_{window}'] = current_price
                features[f'price_to_sma_{window}'] = 1.0
                features[f'ema_{window}'] = current_price
                features[f'price_to_ema_{window}'] = 1.0
    
    # Trend gücü (kısa / uzun SMA)
    if len(prices) >= 50:
        try:
            sma_20 = float(prices.rolling(20).mean().iloc[-1])
            sma_50 = float(prices.rolling(50).mean().iloc[-1])
            features['trend_strength'] = (sma_20 / sma_50) - 1 if sma_50 > 0 else 0
        except:
            features['trend_strength'] = 0
    else:
        features['trend_strength'] = 0
    
    return features


def calculate_momentum_features(prices, returns, windows=[5, 10, 20, 60]):
    """
    Momentum özellikleri hesapla
    
    Args:
        prices: Fiyat serisi
        returns: Getiri serisi
        windows: Momentum pencere boyutları
    
    Returns:
        dict: Momentum feature'ları
    """
    features = {}
    current_price = float(prices.iloc[-1])
    
    for window in windows:
        if len(prices) >= window:
            try:
                # Price momentum (son N günlük getiri)
                past_price = float(prices.iloc[-window])
                momentum = (current_price / past_price) - 1 if past_price > 0 else 0
                features[f'momentum_{window}d'] = momentum
                
                # Rate of change
                roc = ((current_price - past_price) / past_price) * 100 if past_price > 0 else 0
                features[f'roc_{window}d'] = roc
            except:
                features[f'momentum_{window}d'] = 0
                features[f'roc_{window}d'] = 0
    
    # Momentum score (teknik indikatörlerden)
    if len(prices) >= 60:
        try:
            features['momentum_score'] = float(calculate_momentum_score(prices, short_period=20, long_period=60))
        except:
            features['momentum_score'] = 0
    else:
        features['momentum_score'] = 0
    
    return features


def calculate_volume_features(volume_series):
    """
    Hacim özellikleri hesapla (eğer hacim verisi varsa)
    
    Args:
        volume_series: Hacim serisi (None olabilir)
    
    Returns:
        dict: Hacim feature'ları
    """
    features = {}
    
    if volume_series is None or len(volume_series) == 0:
        # Hacim verisi yoksa sıfır değerler
        features['volume_sma_20'] = 0
        features['volume_ratio'] = 1.0
        features['volume_trend'] = 0
        return features
    
    try:
        # Volume moving average
        volume_sma = volume_series.rolling(20).mean()
        vol_sma_val = float(volume_sma.iloc[-1])
        features['volume_sma_20'] = vol_sma_val if not np.isnan(vol_sma_val) else 0
        
        # Current volume / SMA ratio
        current_vol = float(volume_series.iloc[-1])
        avg_vol = vol_sma_val
        features['volume_ratio'] = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Volume trend (son 10 gün artış/azalış)
        if len(volume_series) >= 10:
            recent_avg = float(volume_series[-10:].mean())
            older_avg = float(volume_series[-20:-10].mean()) if len(volume_series) >= 20 else recent_avg
            features['volume_trend'] = (recent_avg / older_avg) - 1 if older_avg > 0 else 0
        else:
            features['volume_trend'] = 0
    except:
        features['volume_sma_20'] = 0
        features['volume_ratio'] = 1.0
        features['volume_trend'] = 0
    
    return features


def extract_all_features(prices, returns, volume=None, lookback=252):
    """
    Tüm feature'ları hesapla (51 teknik gösterge)
    
    Kategoriler:
    1. Technical Indicators: RSI, MACD, Bollinger Bands
    2. Volatility Features: 5, 10, 20, 30 günlük volatilite
    3. Trend Features: SMA, EMA, trend gücü
    4. Momentum Features: 5, 10, 20, 60 günlük momentum
    5. Volume Features: Hacim ortalaması, hacim trendi
    6. Statistical Features: Mean, std, skew, kurtosis
    """
    # Son N günü kullan
    if len(prices) > lookback:
        prices = prices[-lookback:]
        returns = returns[-lookback:]
        if volume is not None:
            volume = volume[-lookback:]
    
    all_features = {}
    
    # 1. Technical Indicators
    try:
        rsi = calculate_rsi(prices, period=14)
        all_features['rsi'] = rsi
        all_features['rsi_norm'] = (rsi - 50) / 50  # -1 ile 1 arası normalize
    except:
        all_features['rsi'] = 50
        all_features['rsi_norm'] = 0
    
    try:
        macd = calculate_macd(prices)
        all_features['macd'] = macd['macd']
        all_features['macd_signal'] = macd['signal']
        all_features['macd_histogram'] = macd['histogram']
        # Normalize (histogram / price)
        current_price = prices.iloc[-1]
        all_features['macd_hist_norm'] = macd['histogram'] / current_price if current_price > 0 else 0
    except:
        all_features['macd'] = 0
        all_features['macd_signal'] = 0
        all_features['macd_histogram'] = 0
        all_features['macd_hist_norm'] = 0
    
    try:
        bb = calculate_bollinger_bands(prices)
        all_features['bb_upper'] = bb['upper']
        all_features['bb_middle'] = bb['middle']
        all_features['bb_lower'] = bb['lower']
        all_features['bb_position'] = bb['position']  # 0-1 arası (bantlar içindeki pozisyon)
    except:
        all_features['bb_upper'] = 0
        all_features['bb_middle'] = 0
        all_features['bb_lower'] = 0
        all_features['bb_position'] = 0.5
    
    # 2. Volatility Features
    vol_features = calculate_volatility_features(returns)
    all_features.update(vol_features)
    
    # 3. Trend Features
    trend_features = calculate_trend_features(prices)
    all_features.update(trend_features)
    
    # 4. Momentum Features
    momentum_features = calculate_momentum_features(prices, returns)
    all_features.update(momentum_features)
    
    # 5. Volume Features
    volume_features = calculate_volume_features(volume)
    all_features.update(volume_features)
    
    # 6. Statistical Features
    try:
        all_features['returns_mean_20d'] = float(returns[-20:].mean()) if len(returns) >= 20 else float(returns.mean())
        all_features['returns_std_20d'] = float(returns[-20:].std()) if len(returns) >= 20 else float(returns.std())
        all_features['returns_skew'] = float(returns.skew()) if len(returns) >= 30 else 0.0
        all_features['returns_kurt'] = float(returns.kurtosis()) if len(returns) >= 30 else 0.0
    except:
        all_features['returns_mean_20d'] = 0.0
        all_features['returns_std_20d'] = 0.0
        all_features['returns_skew'] = 0.0
        all_features['returns_kurt'] = 0.0
    
    return all_features


def create_feature_dataframe(prices, returns, volume=None, lookback=252):
    """
    Zaman serisi feature DataFrame'i oluştur (her gün için feature'lar)
    
    Args:
        prices: Fiyat serisi
        returns: Getiri serisi
        volume: Hacim serisi (opsiyonel)
        lookback: Minimum veri gereksinimi
    
    Returns:
        pandas.DataFrame: Feature'lar (index: tarih)
    """
    if len(prices) < lookback:
        return pd.DataFrame()
    
    # Her gün için feature hesapla (sliding window)
    feature_list = []
    dates = []
    
    # Son 60 günü hesapla (eğitim için yeterli)
    for i in range(lookback, len(prices)):
        window_prices = prices.iloc[:i+1]
        window_returns = returns.iloc[:i+1]
        window_volume = volume.iloc[:i+1] if volume is not None else None
        
        features = extract_all_features(window_prices, window_returns, window_volume, lookback=lookback)
        feature_list.append(features)
        dates.append(prices.index[i])
    
    df = pd.DataFrame(feature_list, index=dates)
    
    # NaN ve Inf değerleri temizle
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df


def select_top_features(feature_df, returns, n_features=10):
    """
    En önemli feature'ları seç (korelasyon bazlı)
    
    Args:
        feature_df: Feature DataFrame
        returns: Hedef değişken (gelecek getiriler)
        n_features: Seçilecek feature sayısı
    
    Returns:
        list: En önemli feature isimleri
    """
    if len(feature_df) == 0:
        return []
    
    # Her feature'ın getiri ile korelasyonunu hesapla
    correlations = {}
    
    for col in feature_df.columns:
        try:
            corr = abs(feature_df[col].corr(returns))
            if not np.isnan(corr):
                correlations[col] = corr
        except:
            continue
    
    # En yüksek korelasyonlu feature'ları seç
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:n_features]]
    
    return top_features


def prepare_exogenous_features_for_arima(prices, returns, volume=None, n_features=5):
    """
    ARIMA için exogenous features hazırla
    
    Öncelikli feature'lar:
    1. rsi_norm: RSI normalize (-1 ile 1 arası)
    2. macd_hist_norm: MACD histogram normalize
    3. bb_position: Bollinger Bands pozisyonu (0-1)
    4. momentum_20d: 20 günlük momentum
    5. trend_strength: Trend gücü
    """
    
    # Tüm feature'ları hesapla
    all_features = extract_all_features(prices, returns, volume)
    
    # En önemli feature'ları seç (öncelikli olanlar)
    priority_features = [
        'rsi_norm',           # Momentum
        'macd_hist_norm',     # Trend
        'bb_position',        # Volatility regime
        'momentum_20d',       # Short-term momentum
        'trend_strength',     # Trend gücü
        'volatility_20d',     # Volatility
        'vol_ratio_20d',      # Volatility değişimi
        'price_to_sma_50',   # Long-term trend
        'returns_mean_20d',   # Recent performance
    ]
    
    # Mevcut feature'lardan seç
    selected = {}
    for feat in priority_features[:n_features]:
        if feat in all_features:
            selected[feat] = all_features[feat]
    
    # DataFrame'e çevir (tek satır)
    df = pd.DataFrame([selected])
    
    # NaN ve Inf temizle
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df


def prepare_rolling_exogenous_for_arima(prices, returns, volume=None, n_features=5, forecast_days=22, ticker=None):
    """
    ARIMA için ROLLING exogenous features.
    Her gün için O GÜNE AİT gerçek feature değerlerini hesaplar.
    
    ESKİ FONKSİYON (prepare_exogenous_features_for_arima):
        Son günün değerini tüm günlere kopyalıyordu → ARIMA bilgi öğrenemiyordu.
    
    YENİ FONKSİYON (bu):
        Her gün için rolling hesaplama → ARIMA feature değişimlerinden sinyal alabilir.
    
    Args:
        prices: Fiyat serisi
        returns: Getiri serisi
        volume: Hacim serisi (opsiyonel)
        n_features: Kullanılacak feature sayısı
        forecast_days: Gelecek kaç gün tahmin edileceği
    
    Returns:
        tuple: (exog_aligned, future_exog)
            - exog_aligned: Returns ile aynı index'te feature DataFrame
            - future_exog: Gelecek N gün için feature tahmini
            Yetersiz veri varsa (None, None) döner
    """
    priority_features = [
        'rsi_norm', 'macd_hist_norm', 'bb_position',
        'momentum_20d', 'trend_strength',
        'volatility_20d', 'vol_ratio_20d',
        'price_to_sma_50', 'returns_mean_20d',
    ]
    priority_features = priority_features[:n_features]
    
    min_lookback = 60  # En az 60 gün geçmiş veri lazım
    
    if len(prices) < min_lookback + 20:
        return None, None

    # Her gün için feature hesapla (rolling window)
    feature_records = []
    feature_dates = []
    
    for i in range(min_lookback, len(prices)):
        window_prices = prices.iloc[:i+1]
        window_returns = returns.iloc[:min(i+1, len(returns))]
        
        feats = extract_all_features(window_prices, window_returns, volume, lookback=252)
        selected = {f: feats.get(f, 0.0) for f in priority_features}
        feature_records.append(selected)
        feature_dates.append(prices.index[i])
    
    if len(feature_records) == 0:
        return None, None
    
    df = pd.DataFrame(feature_records, index=feature_dates)

    # Makro kolonları ekle (USDTRY, XU100, Brent) — indirme başarısızsa teknik feature'larla devam
    if USE_MACRO_FEATURES and MACRO_AVAILABLE and MACRO_N_FEATURES > 0:
        try:
            macro_df = get_macro_features_for_index(df.index)
            if macro_df is not None and not macro_df.empty:
                macro_cols = [c for c in MACRO_FEATURE_COLUMNS if c in macro_df.columns]
                macro_cols = macro_cols[:MACRO_N_FEATURES]
                for col in macro_cols:
                    df[col] = macro_df[col].values
        except Exception:
            pass

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(0)
    
    # Returns ile hizala (ortak indeks)
    common_idx = df.index.intersection(returns.index)
    if len(common_idx) < 50:
        return None, None
    
    exog_aligned = df.loc[common_idx]
    
    # Gelecek tahmini: son 5 günün EMA'sını kullan (daha kararlı)
    last_5 = df.tail(5)
    future_values = last_5.ewm(span=3).mean().iloc[-1].values
    future_exog = pd.DataFrame(
        np.tile(future_values, (forecast_days, 1)),
        columns=df.columns
    )
    
    return exog_aligned, future_exog


def prepare_lstm_features(prices, returns, volume=None, n_features=5):
    """
    LSTM için multi-feature input hazırla
    
    Args:
        prices: Fiyat serisi
        returns: Getiri serisi
        volume: Hacim serisi (opsiyonel)
        n_features: Kullanılacak feature sayısı
    
    Returns:
        pandas.DataFrame: Feature time series (her gün için feature'lar)
    """
    # Zaman serisi feature DataFrame'i oluştur
    feature_df = create_feature_dataframe(prices, returns, volume, lookback=100)
    
    if len(feature_df) == 0:
        return pd.DataFrame()
    
    # Öncelikli feature'ları seç
    priority_features = [
        'rsi_norm',
        'macd_hist_norm',
        'bb_position',
        'momentum_20d',
        'volatility_20d',
        'trend_strength',
        'returns_mean_20d',
        'vol_ratio_20d',
    ]
    
    # Mevcut olanları al
    available_features = [f for f in priority_features if f in feature_df.columns][:n_features]
    
    # Returns'ü de ekle (target olarak gerekli)
    if 'returns' not in feature_df.columns:
        feature_df['returns'] = returns.reindex(feature_df.index)
    
    # Seçili feature'ları döndür
    selected_cols = ['returns'] + available_features
    result = feature_df[selected_cols]
    
    return result


if __name__ == "__main__":
    # Test scripti
    import yfinance as yf
    
    print("[*] Feature Engineering Test")
    print("=" * 60)
    
    # Örnek veri: AKBNK
    ticker = "AKBNK.IS"
    data = yf.download(ticker, start='2023-01-01', progress=False)
    
    prices = data['Close']
    returns = prices.pct_change().dropna()
    volume = data['Volume']
    
    # Feature'ları çıkar
    print(f"\n[*] {ticker} için feature'lar hesaplanıyor...")
    features = extract_all_features(prices, returns, volume)
    
    print(f"\n[+] Toplam {len(features)} feature hesaplandı:")
    for name, value in list(features.items())[:10]:
        print(f"  {name:25s}: {value:>10.4f}")
    
    # ARIMA için hazırlık
    print("\n[*] ARIMA için exogenous features:")
    arima_features = prepare_exogenous_features_for_arima(prices, returns, volume, n_features=5)
    print(arima_features.T)
    
    # LSTM için hazırlık
    print("\n[*] LSTM için time series features:")
    lstm_features = prepare_lstm_features(prices, returns, volume, n_features=5)
    print(f"\n[+] LSTM feature shape: {lstm_features.shape}")
    print(lstm_features.tail())
    
    print("\n[+] Test tamamlandı!")
