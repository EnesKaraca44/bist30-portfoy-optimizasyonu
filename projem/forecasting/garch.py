"""GARCH tabanlı volatilite tahmini."""
import numpy as np
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")


def get_garch_volatility(daily_returns, horizon=22, annualize=True, maxiter=100):
    """
    GARCH(1,1) ile volatilite tahmini.

    Args:
        daily_returns: Günlük getiri serisi
        horizon: Tahmin ufku (gün)
        annualize: True ise yıllıklandırılmış volatilite döner
        maxiter: GARCH fit iterasyon limiti

    Returns:
        float: Volatilite (ondalık)
    """
    try:
        recent = daily_returns[-252:] if len(daily_returns) > 252 else daily_returns
        scaled = recent * 100
        model = arch_model(scaled, vol="Garch", p=1, q=1)
        res = model.fit(disp="off", maxiter=maxiter)
        forecast = res.forecast(horizon=horizon)
        vol_daily = np.sqrt(forecast.variance.values[-1, :]).mean() / 100

        if annualize:
            return float(vol_daily * np.sqrt(252))
        return float(vol_daily * np.sqrt(horizon))

    except Exception:
        std = daily_returns.std()
        if annualize:
            return float(std * np.sqrt(252))
        return float(std * np.sqrt(horizon))


def get_garch_risk(returns_series):
    """Haftalık GARCH risk tahmini (geriye dönük uyumluluk)."""
    return get_garch_volatility(returns_series, horizon=5, annualize=False, maxiter=50)
