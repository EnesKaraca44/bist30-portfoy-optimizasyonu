"""Yillik batch_job (Yapay Zeka Tahminleri) iyilestirmeleri."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ANNUAL_ARIMA_CLIP_MAX

HORIZON_MONTHS = (1, 3, 6, 12)
# Uzun vadede ARIMA tahminini tarihsel ortalamaya cekme agirligi
ARIMA_HORIZON_SHRINK = {1: 0.30, 3: 0.45, 6: 0.58, 12: 0.70}
# Volatilite bazli donem clip carpani (sigma)
ARIMA_VOL_SIGMA_MULT = {1: 2.2, 3: 2.0, 6: 1.75, 12: 1.5}


def clip_annual_return(value: float, clip_max: float = ANNUAL_ARIMA_CLIP_MAX) -> float:
    """Yillik getiri tahminini backtest ile ayni sinirlara kes."""
    return float(np.clip(value, -clip_max, clip_max))


def clip_annual_series(series: pd.Series, clip_max: float = ANNUAL_ARIMA_CLIP_MAX) -> pd.Series:
    return series.clip(-clip_max, clip_max)


def period_clip_max(months: int, clip_max: float = ANNUAL_ARIMA_CLIP_MAX) -> float:
    months = max(int(months), 1)
    return float((1 + clip_max) ** (months / 12.0) - 1)


def clip_period_return(value: float, months: int, clip_max: float = ANNUAL_ARIMA_CLIP_MAX) -> float:
    cap = period_clip_max(months, clip_max)
    return float(np.clip(value, -cap, cap))


def clip_period_series(series: pd.Series, months: int, clip_max: float = ANNUAL_ARIMA_CLIP_MAX) -> pd.Series:
    cap = period_clip_max(months, clip_max)
    return series.clip(-cap, cap)


def historical_period_benchmark(daily_returns, forecast_days: int) -> float:
    """Tarihsel ortalama gunluk getiriden bileşik donem getirisi."""
    s = pd.Series(daily_returns).dropna()
    if s.empty:
        return 0.0
    mu = float(np.clip(s.mean(), -0.04, 0.04))
    days = max(int(forecast_days), 1)
    return float((1.0 + mu) ** days - 1.0)


def arima_volatility_period_cap(daily_returns, months: int, forecast_days: int) -> float:
    """Volatiliteye gore yumusak donem ust siniri."""
    s = pd.Series(daily_returns).dropna()
    vol_ann = float(s.std() * np.sqrt(252)) if len(s) > 20 else 0.30
    vol_ann = max(vol_ann, 0.12)
    vol_period = vol_ann * np.sqrt(max(int(forecast_days), 1) / 252.0)
    sigma = ARIMA_VOL_SIGMA_MULT.get(int(months), 1.75)
    soft_cap = sigma * vol_period
    hard_cap = period_clip_max(months)
    return float(min(hard_cap, max(soft_cap, 0.025)))


def finalize_arima_period_forecast(
    raw_period: float,
    daily_returns,
    months: int,
) -> float:
    """
    ARIMA donem tahminini stabilize et:
    1) Tarihsel benchmark'a shrink
    2) Volatilite bazli yumusak clip
    """
    months = max(int(months), 1)
    forecast_days = get_horizon_forecast_days(months)
    hist = historical_period_benchmark(daily_returns, forecast_days)
    shrink = ARIMA_HORIZON_SHRINK.get(months, 0.5)
    blended = (1.0 - shrink) * float(raw_period) + shrink * hist
    cap = arima_volatility_period_cap(daily_returns, months, forecast_days)
    return float(np.clip(blended, -cap, cap))


def period_return_from_annual(annual_return: float, forecast_days: int) -> float:
    days = max(int(forecast_days), 1)
    base = max(1 + float(annual_return), 1e-6)
    return base ** (days / 252.0) - 1


def annual_return_from_period(period_return: float, months: int) -> float:
    days = max(get_horizon_forecast_days(months), 1)
    base = max(1 + float(period_return), 1e-6)
    return base ** (252.0 / days) - 1


def apply_regime_to_annual_forecasts(
    forecasts: dict[str, float],
    market_regime: dict | None,
) -> dict[str, float]:
    """Haftalik pipeline rejim mantigini yillik tahminlere uygula."""
    from forecasting.weekly_pipeline import apply_regime_adjustment

    if not market_regime:
        return forecasts
    return {t: apply_regime_adjustment(v, market_regime) for t, v in forecasts.items()}


def apply_regime_to_period_forecasts(
    forecasts: dict[str, float],
    market_regime: dict | None,
) -> dict[str, float]:
    """Dönem getirilerine rejim düzeltmesi uygula."""
    return apply_regime_to_annual_forecasts(forecasts, market_regime)


def ensemble_period_forecast(
    bl_period: float,
    lstm_period: float,
    arima_weight: float = 0.5,
    lstm_weight: float = 0.5,
    max_lstm_abs: float = 0.25,
    conflict_lstm_abs: float = 0.10,
) -> float:
    """BL ile LSTM donem getirilerini guvenli birlestir."""
    if bl_period * lstm_period < 0 and abs(lstm_period) > conflict_lstm_abs:
        return float(bl_period)
    lstm_capped = float(np.clip(lstm_period, -max_lstm_abs, max_lstm_abs))
    total = arima_weight + lstm_weight
    if total <= 0:
        return float(bl_period)
    aw, lw = arima_weight / total, lstm_weight / total
    return float(aw * bl_period + lw * lstm_capped)


def apply_black_litterman_for_horizon(
    market_caps: pd.Series,
    daily_returns: pd.DataFrame,
    arima_views: pd.Series,
    forecast_days: int,
    risk_aversion: float = 2.5,
    tau: float = 0.10,
) -> pd.Series:
    """Ufka uygun kovaryans ile Black-Litterman."""
    from forecasting.black_litterman import calculate_black_litterman

    tickers = [t for t in arima_views.index if t in daily_returns.columns]
    if len(tickers) < 3:
        return arima_views
    cov = daily_returns[tickers].cov() * max(int(forecast_days), 1)
    mcaps = market_caps.reindex(tickers).fillna(1e9)
    views = arima_views.reindex(tickers).fillna(0)
    return calculate_black_litterman(mcaps, cov, views, risk_aversion=risk_aversion, tau=tau)


def get_horizon_forecast_days(months: int) -> int:
    """Vade icin yaklasik is gunu (ARIMA ufku)."""
    mapping = {1: 22, 3: 66, 6: 126, 12: 252}
    return mapping.get(max(1, months), min(252, max(22, months * 21)))


REGIME_USER_LABELS = {
    "bull": (
        "Yükseliş trendi",
        "Son haftalarda BIST30 genel olarak yükseliş eğiliminde. Yapay zeka, momentumu güçlü "
        "hisselere daha fazla yer verebilir.",
    ),
    "bear": (
        "Düşüş trendi",
        "Son haftalarda piyasa baskı altında. Yapay zeka daha temkinli, savunmacı hisse "
        "seçimine yönelir.",
    ),
    "sideways": (
        "Yatay / belirsiz piyasa",
        "Piyasa net bir yön göstermiyor. Yapay zeka getiri ile risk arasında denge kurar.",
    ),
}

STRATEGY_USER_LABELS = {
    "max_return": (
        "Getiri odaklı seçim",
        "Bu ortamda öncelik yüksek getiri beklentisi olan hisselere verildi.",
    ),
    "max_sharpe": (
        "Risk-getiri dengesi",
        "Bu ortamda öncelik, birim risk başına en iyi getiriyi sunan hisselere verildi (savunmacı).",
    ),
    "balanced": (
        "Dengeli seçim",
        "Bu ortamda hem getiri hem güvenlik gözetilerek dengeli bir dağılım hedeflendi.",
    ),
}

RISK_PROFILE_USER_LABELS = {
    "low": "Risk profiliniz Düşük olduğu için dalgalanması en az portföy tercih edildi.",
    "medium": "Risk profiliniz Orta olduğu için getiri ve güvenlik harmanlandı.",
    "high": "Risk profiliniz Yüksek olduğu için getiri potansiyeli öne çıkarıldı.",
}


def build_regime_explanation_for_user(
    market_regime: dict | None,
    strategy: str,
    risk_profile: str = "medium",
) -> dict:
    """Arayuz icin Turkce, kullanici dostu piyasa rejimi aciklamasi."""
    regime = (market_regime or {}).get("regime", "sideways")
    confidence = float((market_regime or {}).get("confidence", 0.5))

    regime_title, regime_detail = REGIME_USER_LABELS.get(
        regime, REGIME_USER_LABELS["sideways"]
    )
    strategy_title, strategy_detail = STRATEGY_USER_LABELS.get(
        strategy, STRATEGY_USER_LABELS["balanced"]
    )
    risk_text = RISK_PROFILE_USER_LABELS.get(
        risk_profile, RISK_PROFILE_USER_LABELS["medium"]
    )

    confidence_pct = int(round(confidence * 100))
    confidence_note = (
        f"Model bu tabloyu %{confidence_pct} güvenle okudu."
        if confidence_pct >= 55
        else "Piyasa sinyali zayıf; tahminler daha temkinli yorumlanmalı."
    )

    full_text = (
        f"{risk_text} "
        f"Piyasa görünümü: {regime_title}. {regime_detail} "
        f"Hisse seçiminde: {strategy_title}. {strategy_detail} "
        f"{confidence_note}"
    )

    return {
        "regime": regime,
        "regime_title": regime_title,
        "regime_detail": regime_detail,
        "strategy": strategy,
        "strategy_title": strategy_title,
        "strategy_detail": strategy_detail,
        "confidence_pct": confidence_pct,
        "confidence_note": confidence_note,
        "risk_profile_text": risk_text,
        "full_explain_text": full_text,
    }
