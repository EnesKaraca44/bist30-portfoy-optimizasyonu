"""Veritabanı erişim katmanı."""
from contextlib import contextmanager

import pandas as pd

from config import BIST30_TICKERS, DB_PATH
from db.schema import get_connection, table_exists


@contextmanager
def db_connection(timeout=30.0):
    conn = get_connection(timeout=timeout)
    try:
        yield conn
    finally:
        conn.close()


def get_expected_returns(tickers=None, months=12):
    from services.helpers import normalize_horizon_months

    months = normalize_horizon_months(months)
    tickers = tickers or BIST30_TICKERS
    placeholders = ",".join("?" for _ in tickers)
    query = (
        f"SELECT ticker, tahmini_donem_getiri, tahmini_yillik_getiri, tahmini_yillik_risk, tahmini_bl_getiri "
        f"FROM beklenen_getiriler WHERE ticker IN ({placeholders}) AND horizon_ay = ?"
    )
    with db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=[*tickers, months])
    return df


def save_weekly_forecast(ticker, tahmin_tarihi, horizon, tahmini_getiri, model_tipi):
    with db_connection() as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT OR REPLACE INTO tahminler
            (ticker, tahmin_tarihi, tahmin_horizon, tahmini_getiri, model_tipi)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticker, tahmin_tarihi, horizon, tahmini_getiri, model_tipi),
        )
        conn.commit()
