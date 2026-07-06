"""Black-Litterman portföy getiri birleştirme."""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_black_litterman(
    market_caps,
    cov_matrix,
    arima_views,
    risk_aversion=2.5,
    tau=0.05,
):
    """
    Black-Litterman: piyasa dengesi + ARIMA görüşleri.

    Returns:
        pd.Series: BL beklenen getirileri; hata durumunda arima_views
    """
    try:
        if len(market_caps) == 0:
            raise ValueError("Market caps boş")

        common_assets = market_caps.index.intersection(arima_views.index)
        if len(common_assets) == 0:
            raise ValueError("Market caps ve ARIMA views eşleşmiyor")

        market_caps = market_caps[common_assets]
        arima_views = arima_views[common_assets]

        if isinstance(cov_matrix, pd.DataFrame):
            cov_matrix = cov_matrix.loc[common_assets, common_assets].values

        total_mcap = market_caps.sum()
        market_weights = (market_caps / total_mcap).values
        pi = risk_aversion * np.dot(cov_matrix, market_weights)

        Q = arima_views.values.reshape(-1, 1)
        P = np.identity(len(market_caps))
        omega = np.dot(np.dot(P, tau * cov_matrix), P.T)

        if np.any(np.diag(omega) <= 0):
            omega += np.eye(len(omega)) * 1e-8

        tau_sigma = tau * cov_matrix
        try:
            tau_sigma_inv = np.linalg.inv(tau_sigma)
            omega_inv = np.linalg.inv(omega)
        except np.linalg.LinAlgError:
            tau_sigma_inv = np.linalg.pinv(tau_sigma)
            omega_inv = np.linalg.pinv(omega)

        left_term = np.linalg.inv(tau_sigma_inv + np.dot(np.dot(P.T, omega_inv), P))
        right_term = np.dot(tau_sigma_inv, pi) + np.dot(np.dot(P.T, omega_inv), Q.flatten())
        bl_returns = np.clip(np.dot(left_term, right_term), -1.0, 1.0)

        return pd.Series(bl_returns, index=common_assets)

    except Exception as e:
        logger.warning("Black-Litterman hatasi: %s — ARIMA kullanilacak", e)
        return arima_views
