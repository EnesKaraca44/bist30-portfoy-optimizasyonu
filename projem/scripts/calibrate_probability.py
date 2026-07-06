#!/usr/bin/env python
"""Yükseliş olasılığı kalibrasyon raporu — Brier skoru ve bant isabeti."""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_PATH


def run_calibration_report(min_samples: int = 5) -> dict:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT ticker, tahmin_tarihi, tahmini_getiri, gerceklesen_getiri, yukselis_olasiligi
        FROM tahminler
        WHERE gerceklesen_getiri IS NOT NULL
          AND yukselis_olasiligi IS NOT NULL
        ORDER BY tahmin_tarihi DESC
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return {"success": False, "message": "Kalibrasyon için yeterli veri yok."}

    df["actual_up"] = (df["gerceklesen_getiri"] > 0).astype(int)
    df["prob"] = df["yukselis_olasiligi"].clip(0.05, 0.95)

    brier = float(np.mean((df["prob"] - df["actual_up"]) ** 2))
    baseline_brier = float(np.mean((0.5 - df["actual_up"]) ** 2))

    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    calibration = []
    for lo, hi in bins:
        band = df[(df["prob"] >= lo) & (df["prob"] < hi)]
        if len(band) < min_samples:
            continue
        calibration.append({
            "band": f"{int(lo*100)}-{int(min(hi, 1)*100)}%",
            "count": int(len(band)),
            "predicted_avg_prob": round(float(band["prob"].mean()) * 100, 1),
            "actual_up_rate": round(float(band["actual_up"].mean()) * 100, 1),
        })

    report = {
        "success": True,
        "total_samples": int(len(df)),
        "brier_score": round(brier, 4),
        "baseline_brier": round(baseline_brier, 4),
        "better_than_baseline": brier < baseline_brier,
        "calibration_bands": calibration,
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "probability_calibration.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    report = run_calibration_report()
    if not report.get("success"):
        print(report.get("message", "Hata"))
        sys.exit(1)
    print(f"Örnek sayısı: {report['total_samples']}")
    print(f"Brier skoru: {report['brier_score']} (baseline: {report['baseline_brier']})")
    print(f"Baseline'dan iyi: {report['better_than_baseline']}")
    for band in report.get("calibration_bands", []):
        print(f"  {band['band']}: n={band['count']}, tahmin=%{band['predicted_avg_prob']}, gerçek=%{band['actual_up_rate']}")
