"""
Haftalık tahmin doğruluğu iyileştirmeleri — test edilebilir saf fonksiyonlar.

- Hisse bazlı model seçimi (model_selector)
- Volatiliteye göre clip
- Piyasa rejimi düzeltmesi
- Düşük güvenilirlik filtresi
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ARIMA_WEIGHT = 0.5
DEFAULT_LSTM_WEIGHT = 0.5
ENSEMBLE_DOMINANCE_THRESHOLD = 0.7
LOOKBACK_FORECASTS = 20
WEIGHT_METHOD = "direction_weighted"
MIN_SIGNAL_ABS_RETURN = 0.001


def get_volatility_adjusted_clip(
    returns_series: pd.Series,
    base_clip: float = 0.15,
    min_clip: float = 0.08,
    max_clip: float = 0.25,
) -> float:
    """Yüksek volatilite = daha geniş clip; düşük volatilite = daha dar."""
    if returns_series is None or len(returns_series) < 20:
        return base_clip
    vol_weekly = float(returns_series.std() * np.sqrt(5))
    if vol_weekly <= 0.015:
        return min_clip
    if vol_weekly >= 0.06:
        return max_clip
    ratio = (vol_weekly - 0.015) / (0.06 - 0.015)
    return float(min_clip + ratio * (max_clip - min_clip))


def calculate_up_probability(
    expected_return: float,
    weekly_risk: float,
    min_prob: float = 0.05,
    max_prob: float = 0.95,
) -> float:
    """
    Haftalık getiri tahmini (mu) ve GARCH riski (sigma) ile
    P(getiri > 0) olasılığı — normal dağılım varsayımı.

    Aşırı kendine güveni önlemek için [min_prob, max_prob] aralığına kırpılır.
    """
    import math

    mu = float(expected_return)
    sigma = float(weekly_risk)

    if not np.isfinite(mu) or not np.isfinite(sigma):
        return 0.5
    if sigma <= 1e-9:
        prob = 1.0 if mu > 0 else (0.0 if mu < 0 else 0.5)
    else:
        prob = 0.5 * (1.0 + math.erf(mu / (sigma * math.sqrt(2.0))))

    return float(np.clip(prob, min_prob, max_prob))


def apply_regime_adjustment(forecast_return: float, regime_info: dict) -> float:
    """
    Bear: tahmin büyüklüğünü küçült.
    Belirsiz sideways: sıfıra doğru çek.
    Bull: değiştirme.
    """
    regime = regime_info.get("regime", "sideways")
    confidence = float(regime_info.get("confidence", 0.5))

    if regime == "bear":
        dampen = 0.55 + (1.0 - confidence) * 0.15
        return float(forecast_return * dampen)
    if regime == "sideways" and confidence < 0.65:
        return float(forecast_return * 0.80)
    return float(forecast_return)


def _model_selector_available() -> bool:
    try:
        import model_selector  # noqa: F401
        return True
    except ImportError:
        return False


def get_ensemble_weights(
    ticker: str,
    lookback: int = LOOKBACK_FORECASTS,
    method: str = WEIGHT_METHOD,
) -> tuple[float, float]:
    """Hisse bazlı ARIMA/LSTM ağırlıkları; yeterli veri yoksa varsayılan."""
    if not _model_selector_available():
        return DEFAULT_ARIMA_WEIGHT, DEFAULT_LSTM_WEIGHT
    try:
        from model_selector import calculate_model_weights

        weights = calculate_model_weights(ticker, lookback_forecasts=lookback, method=method)
        return float(weights["arima"]), float(weights["lstm"])
    except Exception:
        return DEFAULT_ARIMA_WEIGHT, DEFAULT_LSTM_WEIGHT


def combine_weekly_forecast(
    ticker: str,
    arima_return: float,
    lstm_weekly: float,
    use_lstm: bool,
    lstm_available: bool,
    ensemble_fn=None,
    lookback: int = LOOKBACK_FORECASTS,
    dominance_threshold: float = ENSEMBLE_DOMINANCE_THRESHOLD,
) -> tuple[float, str]:
    """
    model_selector ile tek model veya dinamik ensemble.
    Returns: (final_return, model_type_label)
    """
    if not use_lstm or not lstm_available:
        return float(arima_return), "ARIMA"

    if _model_selector_available():
        try:
            from model_selector import get_best_model_for_ticker

            best_model, best_weight = get_best_model_for_ticker(
                ticker, lookback_forecasts=lookback, threshold=dominance_threshold
            )
            if best_model == "ARIMA":
                return float(arima_return), "ARIMA"
            if best_model == "LSTM":
                return float(lstm_weekly), "LSTM"
            if isinstance(best_weight, dict):
                arima_w = float(best_weight["arima"])
                lstm_w = float(best_weight["lstm"])
            else:
                arima_w, lstm_w = get_ensemble_weights(ticker, lookback)
        except Exception:
            arima_w, lstm_w = DEFAULT_ARIMA_WEIGHT, DEFAULT_LSTM_WEIGHT
    else:
        arima_w, lstm_w = DEFAULT_ARIMA_WEIGHT, DEFAULT_LSTM_WEIGHT

    if ensemble_fn is None:
        from lstm_forecast import ensemble_forecast

        ensemble_fn = ensemble_forecast

    final = ensemble_fn(arima_return, lstm_weekly, arima_weight=arima_w, lstm_weight=lstm_w)
    label = "ARIMA+LSTM" if abs(arima_w - 0.5) < 0.05 else f"ARIMA+LSTM({arima_w:.2f}/{lstm_w:.2f})"
    return float(final), label


def detect_market_regime_from_panel(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    lookback: int = 60,
) -> dict:
    """BIST30 panel ortalamasından piyasa rejimi."""
    try:
        from technical_indicators import detect_market_regime
    except ImportError:
        return {"regime": "sideways", "confidence": 0.5, "metrics": {}}

    if prices_df.empty or returns_df.empty:
        return {"regime": "sideways", "confidence": 0.5, "metrics": {}}

    avg_prices = prices_df.mean(axis=1).dropna()
    avg_returns = returns_df.mean(axis=1).dropna()
    common = avg_prices.index.intersection(avg_returns.index)
    if len(common) < lookback:
        return {"regime": "sideways", "confidence": 0.5, "metrics": {}}

    return detect_market_regime(
        avg_prices.loc[common],
        avg_returns.loc[common],
        lookback=lookback,
    )


def annotate_forecast_confidence(
    forecasts_dict: dict,
    min_abs_return: float = MIN_SIGNAL_ABS_RETURN,
) -> dict:
    """Her tahmine high_confidence bayrağı ekle (in-place + return)."""
    for data in forecasts_dict.values():
        if not isinstance(data, dict):
            continue
        abs_ret = abs(float(data.get("return", 0)))
        data["high_confidence"] = abs_ret >= min_abs_return
    return forecasts_dict


def select_tickers_for_portfolio(
    returns_dict: dict[str, float],
    risk_dict: dict[str, float],
    market_regime: dict | None = None,
    min_tickers: int = 5,
    high_confidence_tickers: set[str] | None = None,
) -> list[str]:
    """
    Portföy için hisse seçimi — rejime göre getiri veya risk-ayarlı skor.
    high_confidence_tickers verilirse önce onlardan seç.
    """
    valid = list(returns_dict.keys())
    if len(valid) <= min_tickers:
        return valid

    if high_confidence_tickers:
        preferred = [t for t in valid if t in high_confidence_tickers]
        if len(preferred) >= min_tickers:
            valid = preferred

    regime = (market_regime or {}).get("regime", "sideways")

    def score(ticker: str) -> float:
        ret = returns_dict[ticker]
        risk = max(risk_dict.get(ticker, 0.05), 1e-6)
        if regime == "bear":
            return ret / risk
        if regime == "bull":
            return ret
        return ret / np.sqrt(risk)

    ranked = sorted(valid, key=score, reverse=True)
    positive = [t for t in ranked if returns_dict[t] > 0]

    if len(positive) >= min_tickers:
        return positive
    if len(positive) >= 2:
        return ranked[:min_tickers]
    return ranked[:min_tickers]


def filter_forecasts_by_confidence(
    forecasts_dict: dict,
    min_abs_return: float = MIN_SIGNAL_ABS_RETURN,
) -> dict:
    """Düşük sinyal gücü olan tahminleri portföy dışı bırakmak için işaretle."""
    annotate_forecast_confidence(forecasts_dict, min_abs_return)
    return forecasts_dict


# --- Faz 2: Black-Litterman + rejim bazlı portföy optimizasyonu ---

BL_RISK_AVERSION = 2.5
BL_TAU = 0.10
WEEKLY_HORIZON_DAYS = 5
MC_NUM_PORTFOLIOS = 25000


def build_horizon_cov_matrix(
    returns_df: pd.DataFrame,
    tickers: list[str],
    horizon_days: int = WEEKLY_HORIZON_DAYS,
) -> pd.DataFrame:
    """Günlük getirilerden ufuk kovaryans matrisi."""
    cols = [t for t in tickers if t in returns_df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return returns_df[cols].cov() * horizon_days


def apply_black_litterman_weekly(
    raw_views: dict[str, float],
    returns_df: pd.DataFrame,
    tickers: list[str] | None = None,
    horizon_days: int = WEEKLY_HORIZON_DAYS,
) -> dict[str, float]:
    """
    Haftalık ARIMA görüşlerini Black-Litterman ile dengele.
    Hata durumunda ham görüşleri döndürür.
    """
    if not raw_views:
        return {}

    tickers = tickers or list(raw_views.keys())
    tickers = [t for t in tickers if t in raw_views]
    if len(tickers) < 3:
        return dict(raw_views)

    try:
        from forecasting.black_litterman import calculate_black_litterman
        from forecasting.prices import get_market_caps

        views = pd.Series({t: raw_views[t] for t in tickers})
        cov = build_horizon_cov_matrix(returns_df, tickers, horizon_days)
        if cov.empty or cov.shape[0] < 3:
            return dict(raw_views)

        mcaps = get_market_caps(tickers)
        mcaps = mcaps.reindex(tickers).fillna(1e9)
        bl = calculate_black_litterman(
            mcaps, cov, views,
            risk_aversion=BL_RISK_AVERSION,
            tau=BL_TAU,
        )
        # Haftalık ufka uygun clip
        bl_weekly = bl.clip(-0.25, 0.25)
        result = dict(raw_views)
        result.update(bl_weekly.to_dict())
        return result
    except Exception:
        return dict(raw_views)


def get_portfolio_strategy_for_regime(market_regime: dict | None) -> str:
    """Backtest ile aynı rejim → strateji eşlemesi."""
    regime = (market_regime or {}).get("regime", "sideways")
    if regime == "bear":
        return "max_sharpe"
    if regime == "bull":
        return "max_return"
    return "balanced"


def optimize_portfolio_monte_carlo(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    strategy: str = "max_sharpe",
    num_portfolios: int = MC_NUM_PORTFOLIOS,
) -> np.ndarray | None:
    """
    Rejime göre Monte Carlo portföy ağırlıkları.
    strategy: max_sharpe | max_return | balanced
    """
    tickers = list(expected_returns.index)
    n = len(tickers)
    if n < 2:
        return np.array([1.0 / n] * n) if n == 1 else None

    er = expected_returns.values.astype(float)
    cov = cov_matrix.loc[tickers, tickers].values.astype(float)
    np.fill_diagonal(cov, np.maximum(np.diag(cov), 1e-8))

    best_sharpe = -np.inf
    best_return = -np.inf
    best_balanced = -np.inf
    w_sharpe = w_return = w_balanced = None

    for _ in range(num_portfolios):
        w = np.random.random(n)
        w /= w.sum()
        ret = float(np.dot(w, er))
        vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
        if vol <= 0:
            continue
        sharpe = ret / vol
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            w_sharpe = w.copy()
        if ret > best_return and vol < 0.50:
            best_return = ret
            w_return = w.copy()
        balanced = ret * 0.4 + sharpe * 0.6
        if balanced > best_balanced and vol < 0.50:
            best_balanced = balanced
            w_balanced = w.copy()

    if strategy == "max_return" and w_return is not None:
        return w_return
    if strategy == "balanced" and w_balanced is not None:
        return w_balanced
    if w_sharpe is not None:
        return w_sharpe
    return np.array([1.0 / n] * n)

