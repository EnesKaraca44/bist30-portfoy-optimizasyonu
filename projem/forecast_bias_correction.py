"""
Hisse bazlı sistematik tahmin hatası düzeltmesi (meta-model).

Geçmiş tahmin-gerçekleşme çiftlerinden ortalama signed error öğrenilir
ve yeni tahmine uygulanır.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from config import DB_PATH as DB_NAME

DEFAULT_LOOKBACK = 20
MIN_SAMPLES = 8
DEFAULT_SHRINKAGE = 0.5
MAX_BIAS = 0.03


def get_ticker_bias(
    ticker: str,
    lookback: int = DEFAULT_LOOKBACK,
    conn: sqlite3.Connection | None = None,
) -> float | None:
    """
    Ortalama signed hata: tahmini - gerçekleşen.
    Pozitif = model sistematik olarak iyimser tahmin ediyor.
    """
    should_close = False
    if conn is None:
        conn = sqlite3.connect(DB_NAME, timeout=30.0)
        should_close = True

    try:
        df = pd.read_sql_query(
            """
            SELECT tahmini_getiri, gerceklesen_getiri
            FROM tahminler
            WHERE ticker = ?
              AND gerceklesen_getiri IS NOT NULL
              AND tahmini_getiri IS NOT NULL
            ORDER BY tahmin_tarihi DESC
            LIMIT ?
            """,
            conn,
            params=(ticker, lookback),
        )
    finally:
        if should_close:
            conn.close()

    if len(df) < MIN_SAMPLES:
        return None

    errors = df["tahmini_getiri"] - df["gerceklesen_getiri"]
    bias = float(errors.mean())
    if not np.isfinite(bias):
        return None
    return float(np.clip(bias, -MAX_BIAS, MAX_BIAS))


def apply_bias_correction(
    ticker: str,
    raw_forecast: float,
    lookback: int = DEFAULT_LOOKBACK,
    shrinkage: float = DEFAULT_SHRINKAGE,
    conn: sqlite3.Connection | None = None,
) -> tuple[float, float | None]:
    """
    Düzeltilmiş tahmin döndürür.

    Returns:
        (corrected_forecast, applied_correction)
        applied_correction: ham tahminden çıkarılan miktar (shrinkage uygulanmış bias)
    """
    raw = float(raw_forecast)
    bias = get_ticker_bias(ticker, lookback=lookback, conn=conn)
    if bias is None:
        return raw, None

    correction = float(shrinkage * bias)
    corrected = raw - correction
    return corrected, correction
