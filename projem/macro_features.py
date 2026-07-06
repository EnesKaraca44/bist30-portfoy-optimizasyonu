"""
Makro Özellikler Modülü
USDTRY, XU100 ve Brent verilerinden ARIMA exog feature'ları ve
rejim tespiti için makro stres metrikleri üretir.

Tüm veri yfinance'tan gelir; indirme başarısız olursa fonksiyonlar
None döndürür ve pipeline makrosuz devam eder (graceful fallback).
"""
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from config import MACRO_SYMBOLS
except ImportError:
    MACRO_SYMBOLS = {
        'usdtry': 'TRY=X',
        'xu100': 'XU100.IS',
        'brent': 'BZ=F',
    }

# ARIMA exog'a eklenecek makro kolonlar (oncelik sirasi)
MACRO_FEATURE_COLUMNS = [
    'usdtry_mom_5d',
    'usdtry_vol_10d',
    'policy_rate_chg_20d',  # TCMB politika faizi (EVDS) — bankalar
    'xu100_mom_5d',
    'xu100_trend',
    'brent_mom_10d',
    'policy_rate_level',
]

DOWNLOAD_RETRIES = 2
DOWNLOAD_RETRY_DELAY = 5
# Rolling pencerelerin ısınması için hedef aralıktan önce çekilecek pay
WARMUP_CALENDAR_DAYS = 60

# Aynı çalıştırma içinde 30 hisse için tekrar tekrar indirmemek için önbellek
_cache = {'prices': None, 'features': None, 'policy': None, 'start': None, 'end': None}


def clear_macro_cache():
    """Önbelleği temizle (testler ve force refresh için)."""
    _cache['prices'] = None
    _cache['features'] = None
    _cache['policy'] = None
    _cache['start'] = None
    _cache['end'] = None


def _attach_policy_features(feats, start_ts, end_ts):
    """EVDS politika faizi feature'larini makro frame'e ekler."""
    try:
        from evds_client import get_policy_rate_series, compute_policy_rate_features
    except ImportError:
        return feats

    policy = _cache.get('policy')
    if policy is None:
        policy = get_policy_rate_series(
            start_date=start_ts - timedelta(days=WARMUP_CALENDAR_DAYS),
            end_date=end_ts + timedelta(days=1),
        )
        _cache['policy'] = policy

    pf = compute_policy_rate_features(policy)
    if pf is None or pf.empty:
        return feats

    aligned = pf.reindex(feats.index, method='ffill').fillna(0.0)
    for col in aligned.columns:
        feats[col] = aligned[col].values
    return feats


def _to_timestamp(value):
    return pd.Timestamp(value).tz_localize(None) if value is not None else None


def download_macro_data(start_date, end_date, retries=DOWNLOAD_RETRIES):
    """
    Makro serileri (kapanış) indirir.

    Returns:
        pd.DataFrame: Kolonlar MACRO_SYMBOLS anahtarları, indeks tarih.
        Hiçbir seri indirilemezse None.
    """
    import yfinance as yf

    start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date
    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date

    series_map = {}
    for name, symbol in MACRO_SYMBOLS.items():
        for attempt in range(1, retries + 1):
            try:
                df = yf.download(
                    symbol,
                    start=start_str,
                    end=end_str,
                    progress=False,
                    auto_adjust=False,
                    timeout=15,
                )
                if not df.empty and 'Close' in df.columns:
                    close = df['Close']
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
                    series_map[name] = close.dropna()
                    break
            except Exception:
                pass
            if attempt < retries:
                time.sleep(DOWNLOAD_RETRY_DELAY)

    if not series_map:
        return None
    return pd.DataFrame(series_map).sort_index()


def compute_macro_feature_frame(macro_prices):
    """
    Makro fiyat serilerinden günlük feature DataFrame'i üretir.

    Args:
        macro_prices: download_macro_data çıktısı (kolonlar: usdtry, xu100, brent)

    Returns:
        pd.DataFrame: MACRO_FEATURE_COLUMNS alt kümesi, indeks tarih.
        Girdi boşsa None.
    """
    if macro_prices is None or macro_prices.empty:
        return None

    feats = pd.DataFrame(index=macro_prices.index)

    if 'usdtry' in macro_prices.columns:
        usd = macro_prices['usdtry'].ffill()
        feats['usdtry_mom_5d'] = usd.pct_change(5)
        feats['usdtry_vol_10d'] = usd.pct_change().rolling(10).std() * np.sqrt(252)

    if 'xu100' in macro_prices.columns:
        xu = macro_prices['xu100'].ffill()
        feats['xu100_mom_5d'] = xu.pct_change(5)
        sma20 = xu.rolling(20).mean()
        feats['xu100_trend'] = (xu / sma20) - 1

    if 'brent' in macro_prices.columns:
        brent = macro_prices['brent'].ffill()
        feats['brent_mom_10d'] = brent.pct_change(10)

    if feats.empty or len(feats.columns) == 0:
        return None

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.dropna(how='all')
    return feats


def _get_cached_features(start_ts, end_ts):
    """İstenen aralığı kapsayan önbellekli feature frame'i döndür (yoksa indir)."""
    cached_ok = (
        _cache['features'] is not None
        and _cache['start'] is not None
        and _cache['end'] is not None
        and _cache['start'] <= start_ts
        and _cache['end'] >= end_ts
    )
    if not cached_ok:
        dl_start = start_ts - timedelta(days=WARMUP_CALENDAR_DAYS)
        prices = download_macro_data(dl_start, end_ts + timedelta(days=1))
        features = compute_macro_feature_frame(prices)
        if features is None:
            return None
        features = _attach_policy_features(features, start_ts, end_ts)
        _cache['prices'] = prices
        _cache['features'] = features
        _cache['start'] = start_ts
        _cache['end'] = end_ts
    return _cache['features']


def get_macro_features_for_index(target_index, macro_features=None):
    """
    Makro feature'ları hedef tarih indeksine hizalar (ARIMA exog için).

    Args:
        target_index: Hisse getiri serisinin DatetimeIndex'i
        macro_features: Önceden hesaplanmış frame (testler için); None ise indirilir

    Returns:
        pd.DataFrame veya None: target_index ile aynı indekste makro kolonlar
    """
    if target_index is None or len(target_index) == 0:
        return None

    try:
        idx = pd.DatetimeIndex(target_index).tz_localize(None)
    except (TypeError, ValueError):
        idx = pd.DatetimeIndex(target_index)

    if macro_features is None:
        macro_features = _get_cached_features(idx[0], idx[-1])
    if macro_features is None or macro_features.empty:
        return None

    # Kur/emtia takvimi BIST'ten farklı: hisse tarihlerine ffill ile hizala
    aligned = macro_features.reindex(idx, method='ffill')
    aligned = aligned.fillna(0.0)
    return aligned


def get_macro_regime_metrics(macro_prices=None):
    """
    Rejim tespiti için güncel makro metrikleri hesaplar.

    Returns:
        dict veya None: usdtry_vol_10d, usdtry_mom_20d, xu100_trend, xu100_mom_20d
    """
    if macro_prices is None:
        end_ts = pd.Timestamp.now().normalize()
        start_ts = end_ts - timedelta(days=120)
        macro_prices = _cache['prices']
        cache_covers = (
            macro_prices is not None
            and not macro_prices.empty
            and macro_prices.index[-1] >= end_ts - timedelta(days=7)
        )
        if not cache_covers:
            macro_prices = download_macro_data(start_ts, end_ts + timedelta(days=1))

    if macro_prices is None or macro_prices.empty:
        return None

    metrics = {}
    try:
        if 'usdtry' in macro_prices.columns:
            usd = macro_prices['usdtry'].ffill().dropna()
            if len(usd) >= 21:
                metrics['usdtry_vol_10d'] = float(
                    usd.pct_change().rolling(10).std().iloc[-1] * np.sqrt(252)
                )
                metrics['usdtry_mom_20d'] = float(usd.iloc[-1] / usd.iloc[-21] - 1)

        if 'xu100' in macro_prices.columns:
            xu = macro_prices['xu100'].ffill().dropna()
            if len(xu) >= 21:
                metrics['xu100_trend'] = float(xu.iloc[-1] / xu.rolling(20).mean().iloc[-1] - 1)
                metrics['xu100_mom_20d'] = float(xu.iloc[-1] / xu.iloc[-21] - 1)

        try:
            from evds_client import get_policy_rate_series, compute_policy_rate_features
            policy = get_policy_rate_series()
            pf = compute_policy_rate_features(policy)
            if pf is not None and not pf.empty:
                metrics['policy_rate_pct'] = float(policy.iloc[-1])
                metrics['policy_rate_chg_20d'] = float(pf['policy_rate_chg_20d'].iloc[-1])
        except Exception:
            pass
    except Exception:
        return metrics if metrics else None

    return metrics if metrics else None


def compute_macro_stress_score(macro_metrics):
    """
    Makro metriklerden 0-1 arası stres skoru.

    Bileşenler:
    - Kur volatilitesi (yıllık %20+ = stres başlangıcı, %35+ = yüksek stres)
    - 20 günlük kur değer kaybı (%4+ = orta, %8+ = yüksek)
    - Endeksin SMA20 altına sarkması
    """
    if not macro_metrics:
        return 0.0

    stress = 0.0

    vol = macro_metrics.get('usdtry_vol_10d')
    if vol is not None:
        if vol >= 0.35:
            stress += 0.50
        elif vol >= 0.20:
            stress += 0.25

    dep = macro_metrics.get('usdtry_mom_20d')
    if dep is not None:
        if dep >= 0.08:
            stress += 0.35
        elif dep >= 0.04:
            stress += 0.15

    trend = macro_metrics.get('xu100_trend')
    if trend is not None:
        if trend <= -0.03:
            stress += 0.30
        elif trend <= -0.01:
            stress += 0.15

    policy_chg = macro_metrics.get('policy_rate_chg_20d')
    if policy_chg is not None and policy_chg >= 0.02:
        stress += 0.20
    elif policy_chg is not None and policy_chg >= 0.01:
        stress += 0.10

    return float(min(stress, 1.0))


def _stress_label(stress):
    """Kullaniciya gosterilecek risk seviyesi."""
    if stress >= 0.5:
        return "Yüksek"
    if stress >= 0.25:
        return "Orta"
    return "Düşük"


def _macro_dashboard_message(stress, metrics):
    """Kullaniciya gosterilecek kisa makro ozeti."""
    if stress >= 0.5:
        return "Dolar ve borsada hareketlilik var. Tahminler daha temkinli üretiliyor."
    xu_trend = metrics.get("xu100_trend")
    if xu_trend is not None and xu_trend > 0.02:
        return "Borsa yukarı yönlü. Makro ortam tahminlere destek veriyor."
    if xu_trend is not None and xu_trend < -0.02:
        return "Borsa zayıf seyrediyor. Dikkatli pozisyon önerilir."
    return "Piyasa sakin görünüyor. Tahminler normal modda üretiliyor."


def _macro_regime_label(stress, metrics):
    """Stres ve endeks trendine gore kisa rejim etiketi."""
    if stress >= 0.5:
        return "Dikkatli Olun", "stress"
    xu_trend = metrics.get("xu100_trend")
    if xu_trend is not None and xu_trend > 0.01:
        return "İyi Görünüm", "positive"
    if xu_trend is not None and xu_trend < -0.01:
        return "Zayıf Seyir", "weak"
    return "Normal Seyir", "neutral"


def get_macro_dashboard():
    """
    Arayüz makro kartı için JSON payload.

    Returns:
        dict: enabled, available, stress_score, regime_label, metrics, message
    """
    try:
        from config import USE_MACRO_FEATURES
    except ImportError:
        USE_MACRO_FEATURES = False

    if not USE_MACRO_FEATURES:
        return {
            "enabled": False,
            "available": False,
            "message": "Makro özellikler kapalı (config.USE_MACRO_FEATURES=False)",
        }

    metrics = get_macro_regime_metrics()
    if not metrics:
        return {
            "enabled": True,
            "available": False,
            "message": "Makro veri şu an alınamadı. Ağ bağlantısını kontrol edin.",
        }

    stress = compute_macro_stress_score(metrics)
    regime_label, regime_level = _macro_regime_label(stress, metrics)

    def _pct(key):
        val = metrics.get(key)
        return round(float(val) * 100, 2) if val is not None else None

    payload_metrics = {
        "usdtry_vol_10d": round(float(metrics["usdtry_vol_10d"]), 4) if metrics.get("usdtry_vol_10d") is not None else None,
        "usdtry_mom_20d_pct": _pct("usdtry_mom_20d"),
        "xu100_trend_pct": _pct("xu100_trend"),
        "xu100_mom_20d_pct": _pct("xu100_mom_20d"),
    }
    if metrics.get("policy_rate_pct") is not None:
        payload_metrics["policy_rate_pct"] = round(float(metrics["policy_rate_pct"]), 2)
    if metrics.get("policy_rate_chg_20d") is not None:
        payload_metrics["policy_rate_chg_20d_pp"] = round(float(metrics["policy_rate_chg_20d"]) * 100, 2)

    return {
        "enabled": True,
        "available": True,
        "stress_score": stress,
        "stress_percent": round(stress * 100),
        "stress_label": _stress_label(stress),
        "regime_label": regime_label,
        "regime_level": regime_level,
        "metrics": payload_metrics,
        "message": _macro_dashboard_message(stress, metrics),
    }


def apply_macro_regime_overlay(regime_info, macro_metrics):
    """
    Panel bazlı rejim tespitine makro stres katmanı uygular (saf fonksiyon).

    Kurallar (stres skoru >= 0.5 iken):
    - bull  -> sideways'e indir (kur şoku fiyatlara henüz yansımamış olabilir)
    - sideways + endeks trendi negatif -> bear'e indir
    - bear  -> güveni artır

    Stres düşükse rejim değişmez; makro metrikler her durumda
    metrics['macro'] altına eklenir.
    """
    result = dict(regime_info or {})
    result.setdefault('regime', 'sideways')
    result.setdefault('confidence', 0.5)

    if not macro_metrics:
        return result

    stress = compute_macro_stress_score(macro_metrics)

    metrics = dict(result.get('metrics', {}) or {})
    metrics['macro'] = {**macro_metrics, 'stress_score': stress}
    result['metrics'] = metrics

    if stress < 0.5:
        return result

    regime = result['regime']
    confidence = float(result['confidence'])
    xu_trend = macro_metrics.get('xu100_trend')

    if regime == 'bull':
        result['regime'] = 'sideways'
        result['confidence'] = min(confidence, 0.60)
        result['macro_override'] = 'bull->sideways (makro stres)'
    elif regime == 'sideways' and xu_trend is not None and xu_trend < 0:
        result['regime'] = 'bear'
        result['confidence'] = max(confidence, 0.60)
        result['macro_override'] = 'sideways->bear (makro stres + negatif endeks trendi)'
    elif regime == 'bear':
        result['confidence'] = min(confidence + 0.15, 1.0)
        result['macro_override'] = 'bear guven artisi (makro stres)'

    return result


if __name__ == '__main__':
    print('[*] Makro Features Test')
    print('=' * 60)

    end = pd.Timestamp.now()
    start = end - timedelta(days=180)

    print('[*] Makro veriler indiriliyor (USDTRY, XU100, Brent)...')
    prices = download_macro_data(start, end)
    if prices is None:
        print('[-] Indirme basarisiz! Ag baglantisini kontrol edin.')
    else:
        print(f'[+] {len(prices)} gun, kolonlar: {list(prices.columns)}')

        feats = compute_macro_feature_frame(prices)
        print(f'\n[+] Feature frame: {feats.shape}')
        print(feats.tail())

        metrics = get_macro_regime_metrics(macro_prices=prices)
        print(f'\n[+] Rejim metrikleri: {metrics}')
        print(f"[+] Stres skoru: {compute_macro_stress_score(metrics):.2f}")

        base = {'regime': 'bull', 'confidence': 0.8, 'metrics': {}}
        print(f"\n[+] Overlay ornegi (bull): {apply_macro_regime_overlay(base, metrics)}")
