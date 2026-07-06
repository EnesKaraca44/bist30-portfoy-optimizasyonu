#!/usr/bin/env python
"""Ham vs bias-düzeltilmiş tahmin backtest raporu."""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_PATH
from forecast_bias_correction import apply_bias_correction


def run_bias_backtest(lookback: int = 20, min_history: int = 8) -> dict:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT ticker, tahmin_tarihi, tahmini_getiri, gerceklesen_getiri, ham_tahmini_getiri
        FROM tahminler
        WHERE gerceklesen_getiri IS NOT NULL
          AND tahmini_getiri IS NOT NULL
        ORDER BY tahmin_tarihi ASC
        """,
        conn,
    )
    conn.close()

    if len(df) < min_history:
        return {"success": False, "message": "Backtest için yeterli tahmin yok."}

    corrected_preds = []
    raw_preds = []
    actuals = []

    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].sort_values("tahmin_tarihi")
        for i in range(min_history, len(ticker_df)):
            row = ticker_df.iloc[i]
            history = ticker_df.iloc[:i]
            if len(history) < min_history:
                continue

            raw = row.get("ham_tahmini_getiri")
            if raw is None or pd.isna(raw):
                raw = row["tahmini_getiri"]
            raw = float(raw)
            actual = float(row["gerceklesen_getiri"])

            errors = history["tahmini_getiri"] - history["gerceklesen_getiri"]
            bias = float(np.clip(errors.mean(), -0.03, 0.03))
            corrected = raw - 0.5 * bias

            raw_preds.append(raw)
            corrected_preds.append(corrected)
            actuals.append(actual)

    if not actuals:
        return {"success": False, "message": "Simülasyon üretilemedi."}

    raw_mae = float(np.mean(np.abs(np.array(raw_preds) - np.array(actuals))))
    corr_mae = float(np.mean(np.abs(np.array(corrected_preds) - np.array(actuals))))
    improvement_pct = ((raw_mae - corr_mae) / raw_mae * 100) if raw_mae > 0 else 0.0

    report = {
        "success": True,
        "samples": len(actuals),
        "raw_mae": round(raw_mae * 100, 3),
        "corrected_mae": round(corr_mae * 100, 3),
        "improvement_pct": round(improvement_pct, 2),
        "mae_improved": corr_mae < raw_mae,
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bias_backtest_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    report = run_bias_backtest()
    if not report.get("success"):
        print(report.get("message", "Hata"))
        sys.exit(0)
    print(f"Örnek: {report['samples']}")
    print(f"Ham MAE: %{report['raw_mae']}")
    print(f"Düzeltilmiş MAE: %{report['corrected_mae']}")
    print(f"İyileşme: %{report['improvement_pct']}")
