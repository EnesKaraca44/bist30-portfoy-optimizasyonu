"""
Teknik göstergeler ve piyasa rejimi tespiti
"""
import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    """
    Relative Strength Index hesapla
    
    Args:
        prices: Fiyat serisi
        period: RSI periyodu
    
    Returns:
        float: RSI değeri (0-100)
    """
    if len(prices) < period + 1:
        return 50.0
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    MACD (Moving Average Convergence Divergence) hesapla
    
    Args:
        prices: Fiyat serisi
        fast: Hızlı EMA periyodu
        slow: Yavaş EMA periyodu
        signal: Sinyal hattı periyodu
    
    Returns:
        dict: {'macd': float, 'signal': float, 'histogram': float}
    """
    if len(prices) < slow + signal:
        return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line.iloc[-1],
        'signal': signal_line.iloc[-1],
        'histogram': histogram.iloc[-1]
    }

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    Bollinger Bands hesapla
    
    Args:
        prices: Fiyat serisi
        period: Hareketli ortalama periyodu
        std_dev: Standart sapma çarpanı
    
    Returns:
        dict: {'upper': float, 'middle': float, 'lower': float, 'position': float}
    """
    if len(prices) < period:
        return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 0.5}
    
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    current_price = prices.iloc[-1]
    upper = upper_band.iloc[-1]
    middle = sma.iloc[-1]
    lower = lower_band.iloc[-1]
    
    # Bantlar içindeki pozisyon (0=alt, 0.5=orta, 1=üst)
    if upper != lower:
        position = (current_price - lower) / (upper - lower)
    else:
        position = 0.5
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'position': position
    }

def detect_market_regime(prices, returns, lookback=60):
    """
    Piyasa rejimini tespit et (bull/bear/sideways)
    
    Args:
        prices: Fiyat serisi
        returns: Getiri serisi
        lookback: Geriye dönük bakış periyodu
    
    Returns:
        dict: {'regime': str, 'confidence': float, 'metrics': dict}
    """
    if len(prices) < lookback:
        return {
            'regime': 'sideways',
            'confidence': 0.5,
            'metrics': {}
        }
    
    recent_prices = prices[-lookback:]
    recent_returns = returns[-lookback:]
    
    # Temel metrikler
    total_return = (recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1
    volatility = recent_returns.std() * np.sqrt(252)
    sharpe = (recent_returns.mean() * 252) / volatility if volatility > 0 else 0
    
    # Trend gücü (SMA eğimi)
    sma_20 = recent_prices.rolling(20).mean()
    sma_50 = recent_prices.rolling(50).mean() if len(recent_prices) >= 50 else sma_20
    
    trend_strength = (sma_20.iloc[-1] / sma_50.iloc[-1]) - 1 if len(sma_50) > 0 else 0
    
    # RSI
    rsi = calculate_rsi(recent_prices)
    
    # Rejim belirleme
    regime = 'sideways'
    confidence = 0.5
    
    if total_return > 0.10 and trend_strength > 0.02:
        regime = 'bull'
        confidence = min(0.5 + abs(total_return), 1.0)
    elif total_return < -0.10 and trend_strength < -0.02:
        regime = 'bear'
        confidence = min(0.5 + abs(total_return), 1.0)
    else:
        regime = 'sideways'
        confidence = 1.0 - abs(total_return) * 2  # Yatay piyasada küçük hareket = yüksek güven
        confidence = max(0.5, min(confidence, 1.0))
    
    return {
        'regime': regime,
        'confidence': confidence,
        'metrics': {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'trend_strength': trend_strength,
            'rsi': rsi
        }
    }

def calculate_momentum_score(prices, short_period=20, long_period=60):
    """
    Momentum skoru hesapla
    
    Args:
        prices: Fiyat serisi
        short_period: Kısa periyot
        long_period: Uzun periyot
    
    Returns:
        float: Momentum skoru (-1 ile 1 arası)
    """
    if len(prices) < long_period:
        return 0.0
    
    short_return = (prices.iloc[-1] / prices.iloc[-short_period]) - 1
    long_return = (prices.iloc[-1] / prices.iloc[-long_period]) - 1
    
    # Kısa vadeli momentuma daha fazla ağırlık
    momentum = 0.6 * short_return + 0.4 * long_return
    
    # -1 ile 1 arası normalize et
    return np.clip(momentum * 2, -1, 1)

