"""Portföy optimizasyon ve BIST30 veri yükleme."""
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from scipy.optimize import minimize

from config import DB_PATH, BIST30_TICKERS, RISK_FREE_RATE
from db.schema import ensure_beklenen_getiriler_horizon_schema
from forecasting.batch_pipeline import HORIZON_MONTHS, get_horizon_forecast_days
from services.helpers import normalize_horizon_months


def adjust_returns_for_horizon(expected_returns, months):
    months = max(months, 1)
    period_fraction = months / 12.0
    if isinstance(expected_returns, pd.Series):
        safe_base = (1 + expected_returns).clip(lower=1e-6)
        return safe_base ** period_fraction - 1
    if isinstance(expected_returns, np.ndarray):
        safe_base = np.clip(1 + expected_returns, a_min=1e-6, a_max=None)
        return safe_base ** period_fraction - 1
    safe_base = max(1 + expected_returns, 1e-6)
    return safe_base ** period_fraction - 1


def adjust_covariance_for_horizon(cov_matrix, months):
    months = max(months, 1)
    return cov_matrix * (months / 12.0)


def annualize_period_return(period_return, months):
    months = max(months, 1)
    base = max(1 + period_return, 1e-4)
    return base ** (12.0 / months) - 1


def annualize_period_volatility(period_vol, months):
    months = max(months, 1)
    return period_vol * np.sqrt(12.0 / months)


def _has_horizon_schema(conn) -> bool:
    c = conn.cursor()
    c.execute("PRAGMA table_info(beklenen_getiriler)")
    cols = {row[1] for row in c.fetchall()}
    return "horizon_ay" in cols and "tahmini_donem_getiri" in cols


def _horizon_days_to_months(horizon_days: int) -> int:
    mapping = {5: 1, 22: 1, 66: 3, 126: 6, 252: 12}
    return normalize_horizon_months(mapping.get(int(horizon_days), max(1, int(horizon_days) // 21)))


def _read_ml_returns_from_db(months=12):
    """Vadeye ozel donem getirilerini oku (final ensemble oncelikli)."""
    months = normalize_horizon_months(months)
    conn = sqlite3.connect(DB_PATH)
    ensure_beklenen_getiriler_horizon_schema(conn)
    c = conn.cursor()
    placeholders = ",".join("?" for _ in BIST30_TICKERS)

    ml_returns_dict = {}
    ml_risk_dict = {}
    use_legacy_scaling = False

    if _has_horizon_schema(conn):
        c.execute(
            f"""
            SELECT ticker, tahmini_donem_getiri, tahmini_yillik_getiri, tahmini_yillik_risk
            FROM beklenen_getiriler
            WHERE ticker IN ({placeholders}) AND horizon_ay = ?
            """,
            (*BIST30_TICKERS, months),
        )
        rows = c.fetchall()

        if len(rows) < max(5, len(BIST30_TICKERS) // 3):
            c.execute(
                f"""
                SELECT ticker, tahmini_donem_getiri, tahmini_yillik_getiri, tahmini_yillik_risk
                FROM beklenen_getiriler
                WHERE ticker IN ({placeholders}) AND horizon_ay = 1
                """,
                BIST30_TICKERS,
            )
            rows = c.fetchall()
            use_legacy_scaling = True
    else:
        c.execute(
            f"""
            SELECT ticker, tahmini_yillik_getiri, tahmini_yillik_risk
            FROM beklenen_getiriler
            WHERE ticker IN ({placeholders})
            """,
            BIST30_TICKERS,
        )
        legacy_rows = c.fetchall()
        rows = [(t, None, annual, risk) for t, annual, risk in legacy_rows]
        use_legacy_scaling = True

    conn.close()

    for row in rows:
        ticker = row[0]
        period_return = row[1]
        annual_return = row[2] if len(row) > 2 else None
        risk = row[3] if len(row) > 3 else row[2] if len(row) == 3 else None

        if period_return is not None and not np.isnan(period_return):
            value = float(period_return)
        elif annual_return is not None and not np.isnan(annual_return):
            value = (
                adjust_returns_for_horizon(float(annual_return), months)
                if use_legacy_scaling
                else float(annual_return)
            )
        else:
            value = 0.0

        ml_returns_dict[ticker] = value
        ml_risk_dict[ticker] = risk if risk is not None and not np.isnan(risk) else 0.0001

    for t in BIST30_TICKERS:
        if t not in ml_returns_dict:
            ml_returns_dict[t] = 0.0
        if ml_risk_dict.get(t) is None or np.isnan(ml_risk_dict.get(t, 0)) or ml_risk_dict.get(t, 0) <= 0:
            ml_risk_dict[t] = 0.0001

    ml_expected_returns = pd.Series(
        [ml_returns_dict.get(t, 0) for t in BIST30_TICKERS], index=BIST30_TICKERS,
    )
    garch_volatilities = pd.Series(
        [ml_risk_dict.get(t, 0.0001) for t in BIST30_TICKERS], index=BIST30_TICKERS,
    )
    return ml_expected_returns.fillna(0), garch_volatilities


def _build_covariance_matrices(garch_volatilities, start_date="2018-01-01"):
    end_date = pd.to_datetime("today").strftime("%Y-%m-%d")
    prices = yf.download(BIST30_TICKERS, start=start_date, end=end_date, progress=False)["Close"]
    prices = prices.ffill().dropna(axis=0, how="any")
    daily_returns = prices.pct_change().dropna()
    recent_returns = daily_returns.tail(3 * 252)
    cov_matrix = recent_returns.cov() * 252
    historical_expected_returns = recent_returns.mean() * 252
    correlation_matrix = recent_returns.corr()
    d = np.diag(garch_volatilities)
    ml_cov_matrix = pd.DataFrame(
        np.dot(d, np.dot(correlation_matrix, d)),
        index=BIST30_TICKERS,
        columns=BIST30_TICKERS,
    ).fillna(0)
    return historical_expected_returns, cov_matrix, ml_cov_matrix, prices


def get_bist30_data_inputs(start_date="2018-01-01"):
    try:
        ml_period, garch_volatilities = _read_ml_returns_from_db(months=12)
        ml_expected_returns = pd.Series(
            {t: annualize_period_return(v, 12) for t, v in ml_period.items()},
            index=BIST30_TICKERS,
        )
    except Exception as e:
        raise Exception(f"Veritabanı okuma hatası: {e}") from e
    try:
        historical, cov, ml_cov, _ = _build_covariance_matrices(garch_volatilities, start_date)
    except Exception as e:
        raise Exception(f"Tarihsel risk/getiri verisi hatası: {e}") from e
    return ml_expected_returns, historical, cov, ml_cov


def get_bist30_ml_with_horizon(months=12):
    try:
        months = normalize_horizon_months(months)
        ml_expected_returns, garch_volatilities = _read_ml_returns_from_db(months=months)
        historical, cov_matrix, _, prices = _build_covariance_matrices(garch_volatilities)
        historical = adjust_returns_for_horizon(historical, months)
        cov_matrix = adjust_covariance_for_horizon(cov_matrix, months)
        period_fraction = months / 12.0
        garch_scaled = garch_volatilities * np.sqrt(period_fraction)
        recent_returns = prices.pct_change().dropna().tail(3 * 252)
        correlation_matrix = recent_returns.corr()
        d = np.diag(garch_scaled)
        ml_cov_matrix = pd.DataFrame(
            np.dot(d, np.dot(correlation_matrix.values, d)),
            index=BIST30_TICKERS,
            columns=BIST30_TICKERS,
        ).fillna(0)
        return ml_expected_returns, historical, cov_matrix, ml_cov_matrix, prices
    except Exception as e:
        raise Exception(f"ML tahmin hatası (veritabanı okuma): {e}") from e


def get_dynamic_bounds(n):
    if n <= 1:
        return 1.0, 1.0
    if n == 2:
        return 0.80, 0.20
    if n == 3:
        return 0.55, 0.10
    if n == 4:
        return 0.45, 0.05
    if n == 5:
        return 0.35, 0.05
    if n <= 10:
        return 0.25, 0.02
    if n <= 20:
        return 0.20, 0.01
    return 0.15, 0.01


def optimize_portfolio_analytical(
    expected_returns, cov_matrix, tickers, risk_free_rate=RISK_FREE_RATE, max_weight=None, min_weight=None
):
    n = len(tickers)
    er = expected_returns.values if isinstance(expected_returns, pd.Series) else np.array(expected_returns)
    cov = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else np.array(cov_matrix)

    if max_weight is None or min_weight is None:
        dynamic_max, dynamic_min = get_dynamic_bounds(n)
        max_weight = max_weight if max_weight is not None else dynamic_max
        min_weight = min_weight if min_weight is not None else dynamic_min

    safe_max_weight = max(max_weight, 1.0 / n + 0.01)
    safe_min_weight = max(0.0, min(min_weight, 1.0 / n - 0.01))

    def neg_sharpe(w):
        ret = np.dot(w, er)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        if vol <= 1e-10:
            return 1e10
        return -(ret - risk_free_rate) / vol

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((safe_min_weight, safe_max_weight) for _ in range(n))
    x0 = np.array([1.0 / n] * n)

    try:
        result = minimize(
            neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if result.success:
            w = result.x
            ret = float(np.dot(w, er))
            vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
            data = list(w) + [ret, vol, sharpe]
            index = list(tickers) + ["Getiri", "Risk", "SharpeOrani"]
            return pd.Series(data, index=index)
    except Exception as e:
        print(f"[!] Analitik optimizasyon hatasi: {e}")
    return None


def _select_portfolio_from_simulation(sim_results_df: pd.DataFrame, strategy: str = "max_sharpe") -> pd.Series:
    """Rejime gore Monte Carlo sonucundan portfoy sec."""
    if strategy == "min_vol":
        return sim_results_df.loc[sim_results_df["Risk"].idxmin()]

    candidates = sim_results_df[sim_results_df["Risk"] < 0.50]
    if candidates.empty:
        candidates = sim_results_df

    if strategy == "max_return":
        return candidates.loc[candidates["Getiri"].idxmax()]
    if strategy == "balanced":
        scores = candidates["Getiri"] * 0.4 + candidates["SharpeOrani"] * 0.6
        return candidates.loc[scores.idxmax()]
    return sim_results_df.loc[sim_results_df["SharpeOrani"].idxmax()]


def run_monte_carlo_simulation(
    expected_returns, cov_matrix, tickers, graph_title, max_weight=None, min_weight=None, months=12,
    strategy="max_sharpe",
):
    num_assets = len(tickers)
    num_portfolios = 20000

    if max_weight is None or min_weight is None:
        dynamic_max, dynamic_min = get_dynamic_bounds(num_assets)
        max_weight = max_weight if max_weight is not None else dynamic_max
        min_weight = min_weight if min_weight is not None else dynamic_min

    rf_horizon = (1.0 + RISK_FREE_RATE) ** (months / 12.0) - 1.0
    safe_max_weight = max(max_weight, 1.0 / num_assets + 0.01)
    safe_min_weight = max(0.0, min(min_weight, 1.0 / num_assets - 0.01))

    results_list = []
    weights_list = []

    for _ in range(num_portfolios):
        random_weights = np.random.dirichlet(np.ones(num_assets) * 2)
        for _ in range(30):
            random_weights = np.clip(random_weights, safe_min_weight, safe_max_weight)
            deficit = 1.0 - np.sum(random_weights)
            if np.abs(deficit) < 1e-10:
                break
            non_clipped = (random_weights > safe_min_weight) & (random_weights < safe_max_weight)
            num_non_clipped = np.sum(non_clipped)
            if num_non_clipped > 0:
                random_weights[non_clipped] += deficit / num_non_clipped
            else:
                random_weights = np.clip(random_weights + deficit / num_assets, safe_min_weight, safe_max_weight)

        port_return = np.dot(random_weights, expected_returns)
        port_volatility = np.sqrt(np.dot(random_weights.T, np.dot(cov_matrix, random_weights)))
        if np.isnan(port_return):
            port_return = 0
        if np.isnan(port_volatility) or port_volatility <= 0:
            port_volatility = 0.0001
        sharpe_ratio = (port_return - rf_horizon) / port_volatility
        results_list.append([port_return, port_volatility, sharpe_ratio])
        weights_list.append(random_weights)

    results_df = pd.DataFrame(results_list, columns=["Getiri", "Risk", "SharpeOrani"])
    weights_df = pd.DataFrame(weights_list, columns=tickers)
    sim_results_df = pd.concat([results_df, weights_df], axis=1)

    max_sharpe_port = _select_portfolio_from_simulation(sim_results_df, "max_sharpe")
    min_vol_port = _select_portfolio_from_simulation(sim_results_df, "min_vol")
    regime_port = _select_portfolio_from_simulation(sim_results_df, strategy)

    analytical_result = optimize_portfolio_analytical(
        expected_returns, cov_matrix, tickers, risk_free_rate=rf_horizon,
        max_weight=max_weight, min_weight=min_weight,
    )
    if analytical_result is not None:
        if strategy == "max_sharpe" and analytical_result["SharpeOrani"] > max_sharpe_port["SharpeOrani"]:
            max_sharpe_port = analytical_result
        elif strategy != "max_sharpe" and analytical_result["SharpeOrani"] > regime_port["SharpeOrani"]:
            regime_port = analytical_result

    primary_port = max_sharpe_port if strategy == "max_sharpe" else regime_port

    scatter_trace = go.Scatter(
        x=sim_results_df["Risk"], y=sim_results_df["Getiri"], mode="markers",
        marker=dict(color=sim_results_df["SharpeOrani"], colorscale="Viridis", showscale=True, size=5, opacity=0.4),
        hoverinfo="none",
    )
    max_sharpe_trace = go.Scatter(
        x=[primary_port["Risk"]], y=[primary_port["Getiri"]],
        mode="markers", marker=dict(color="yellow", size=15, symbol="star", line=dict(width=1, color="black")),
        name="Secilen Portfoy" if strategy != "max_sharpe" else "Maksimum Sharpe Oranı",
    )
    min_vol_trace = go.Scatter(
        x=[min_vol_port["Risk"]], y=[min_vol_port["Getiri"]],
        mode="markers", marker=dict(color="red", size=15, symbol="star", line=dict(width=1, color="black")),
        name="Minimum Risk",
    )
    layout = go.Layout(
        title=graph_title + " (Eğitim Amaçlı)",
        xaxis=dict(title="Risk (Yıllık Volatilite)"),
        yaxis=dict(title="Beklenen Yıllık Getiri"),
        showlegend=True,
    )
    fig = go.Figure(data=[scatter_trace, max_sharpe_trace, min_vol_trace], layout=layout)
    return primary_port, min_vol_port, fig.to_json()
