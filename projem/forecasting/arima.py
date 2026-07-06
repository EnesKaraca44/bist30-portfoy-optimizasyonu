"""ARIMA tabanlı getiri tahmini."""
import numpy as np
import pandas as pd
import pmdarima as pm
import warnings

warnings.filterwarnings("ignore")

try:
    from feature_engineering import prepare_rolling_exogenous_for_arima
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False


def _dynamic_momentum_weight(returns_series):
    """Volatiliteye göre momentum ağırlığı (haftalık tahmin için)."""
    vol_ann = (
        returns_series[-30:].std() * np.sqrt(252)
        if len(returns_series) > 30
        else returns_series.std() * np.sqrt(252)
    )
    if vol_ann >= 0.45:
        return 0.10
    if vol_ann <= 0.15:
        return 0.60
    return 0.60 - (vol_ann - 0.15) * (0.50 / 0.30)


def get_arima_forecast(
    daily_returns,
    prices=None,
    forecast_days=22,
    use_features=True,
    apply_momentum=False,
    clip_range=None,
    max_p=3,
    max_q=3,
    ticker=None,
    annualize=True,
):
    """
    ARIMA/SARIMAX ile getiri tahmini.

    Args:
        daily_returns: Günlük getiri serisi
        prices: Fiyat serisi (feature engineering için)
        forecast_days: Tahmin ufku (gün)
        use_features: Rolling exogenous feature kullan
        apply_momentum: Haftalık mod — momentum ile birleştir
        clip_range: (min, max) tuple; None ise yıllıklandırılmış değer döner
        max_p, max_q: ARIMA parametre sınırları
        annualize: True ise yıllık getiri; False ise forecast_days dönem getirisi

    Returns:
        float: Yıllıklandırılmış veya dönem getirisi (ondalık)
    """
    try:
        returns_series = daily_returns.copy()
        exog_features = None
        future_exog = None

        if use_features and FEATURES_AVAILABLE and prices is not None and len(prices) >= 100:
            try:
                exog_features, future_exog = prepare_rolling_exogenous_for_arima(
                    prices, returns_series, volume=None, n_features=5, forecast_days=forecast_days,
                    ticker=ticker,
                )
                if exog_features is not None:
                    returns_series = returns_series.loc[exog_features.index]
            except Exception:
                exog_features = None
                future_exog = None

        arima_kwargs = dict(
            stepwise=True,
            suppress_warnings=True,
            seasonal=False,
            max_p=max_p,
            max_q=max_q,
            error_action="ignore",
        )

        if exog_features is not None and future_exog is not None:
            model = pm.auto_arima(returns_series, X=exog_features, **arima_kwargs)
            forecast = model.predict(n_periods=forecast_days, X=future_exog)
        else:
            model = pm.auto_arima(returns_series, **arima_kwargs)
            forecast = model.predict(n_periods=forecast_days)

        daily_preds = np.clip(np.asarray(forecast, dtype=float), -0.06, 0.06)
        combined_daily = float(daily_preds.mean()) if len(daily_preds) else 0.0
        compound_period = float(np.prod(1.0 + daily_preds) - 1.0) if len(daily_preds) else 0.0

        if apply_momentum:
            recent_trend = (
                returns_series[-30:].mean()
                if len(returns_series) > 30
                else returns_series.mean()
            )
            momentum_w = _dynamic_momentum_weight(returns_series)
            combined_daily = (1.0 - momentum_w) * combined_daily + momentum_w * recent_trend
            period_return = combined_daily * forecast_days
            if clip_range:
                return float(np.clip(period_return, clip_range[0], clip_range[1]))
            return float(period_return)

        period_return = compound_period
        if not annualize:
            if clip_range:
                return float(np.clip(period_return, clip_range[0], clip_range[1]))
            return float(period_return)
        days = max(int(forecast_days), 1)
        return float((1.0 + period_return) ** (252.0 / days) - 1.0)

    except Exception:
        if apply_momentum:
            fallback = daily_returns.mean() * forecast_days
            if clip_range:
                return float(np.clip(fallback, clip_range[0], clip_range[1]))
            return float(fallback)
        if not annualize:
            mu = float(daily_returns.mean())
            days = max(int(forecast_days), 1)
            fallback = float((1.0 + mu) ** days - 1.0)
            if clip_range:
                return float(np.clip(fallback, clip_range[0], clip_range[1]))
            return fallback
        return float(daily_returns.mean() * 252)
