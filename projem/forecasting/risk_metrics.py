"""
Portföy risk metrikleri — GARCH + kovaryans matrisinden türetilir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.weekly_pipeline import calculate_up_probability

Z_95 = 1.645
Z_99 = 2.33


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        n = len(w)
        return np.ones(n) / n if n else w
    return w / total


def calculate_portfolio_var(
    weights: np.ndarray | list[float],
    expected_returns: np.ndarray | list[float] | pd.Series,
    cov_matrix: np.ndarray | pd.DataFrame,
    confidence: float = 0.95,
) -> dict[str, float]:
    """
    Parametrik portföy VaR (kayıp pozitif olarak döner).

    VaR = -(mu_p - z * sigma_p)
    """
    w = _normalize_weights(np.asarray(weights, dtype=float))
    if isinstance(expected_returns, pd.Series):
        er = expected_returns.values.astype(float)
    else:
        er = np.asarray(expected_returns, dtype=float)

    if isinstance(cov_matrix, pd.DataFrame):
        cov = cov_matrix.values.astype(float)
    else:
        cov = np.asarray(cov_matrix, dtype=float)

    if len(w) == 0 or er.shape[0] != len(w) or cov.shape[0] != len(w):
        return {
            "var_95": 0.0,
            "expected_return": 0.0,
            "portfolio_vol": 0.0,
            "probability_up": 0.5,
            "confidence": confidence,
        }

    mu_p = float(np.dot(w, er))
    sigma_p = float(np.sqrt(max(float(np.dot(w.T, np.dot(cov, w))), 0.0)))

    if confidence >= 0.99:
        z = Z_99
    elif confidence >= 0.95:
        z = Z_95
    else:
        z = Z_95 * (confidence / 0.95)

    var_loss = float(max(-(mu_p - z * sigma_p), 0.0))
    prob_up = calculate_up_probability(mu_p, sigma_p)

    return {
        "var_95": var_loss,
        "expected_return": mu_p,
        "portfolio_vol": sigma_p,
        "probability_up": prob_up,
        "confidence": confidence,
    }


def calculate_dynamic_stop_loss(
    weekly_vol: float,
    sigma_multiplier: float = 2.0,
    max_stop: float = 0.12,
) -> float:
    """GARCH haftalık volatilitesinden dinamik stop-loss önerisi."""
    vol = float(weekly_vol)
    if not np.isfinite(vol) or vol <= 0:
        return max_stop
    stop = sigma_multiplier * vol
    return float(min(max(stop, 0.02), max_stop))
