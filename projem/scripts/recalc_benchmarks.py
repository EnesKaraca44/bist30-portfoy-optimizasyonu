import scripts._bootstrap  # noqa: F401
"""
Mevcut portföy performans kayıtlarındaki benchmark (BIST30) ve alpha değerlerini
yeniden hesaplar. Böylece eski timezone/tarih hatasından kalan -6 gibi yanlış
benchmark'lar, doğru tarih eşleştirmesiyle (ör. -1.58) güncellenir.

Kullanım: python recalc_benchmarks.py
"""
import sqlite3
import json
from portfolio_performance_tracker import (
    DB_NAME,
    add_business_days,
    get_benchmark_return,
    setup_portfolio_performance_table,
)

def main():
    setup_portfolio_performance_table()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT id, tahmin_tarihi, tahmin_horizon, portfoy_gerceklesen_getiri
        FROM portfoy_performans
        WHERE portfoy_gerceklesen_getiri IS NOT NULL
        ORDER BY tahmin_tarihi DESC
    """)
    rows = c.fetchall()
    updated = 0
    for row in rows:
        row_id, date_str, horizon, actual = row
        from datetime import datetime
        forecast_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        target_date = add_business_days(forecast_date, int(horizon))
        benchmark = get_benchmark_return(forecast_date, target_date)
        if benchmark is None:
            print(f"  [SKIP] {date_str[:10]} -> benchmark hesaplanamadi")
            continue
        alpha = actual - benchmark
        c.execute(
            "UPDATE portfoy_performans SET benchmark_getiri = ?, alpha = ? WHERE id = ?",
            (benchmark, alpha, row_id)
        )
        updated += 1
        print(f"  [OK] {date_str[:10]} benchmark: {benchmark*100:+.2f}%  alpha: {alpha*100:+.2f}%")
    conn.commit()
    conn.close()
    print(f"\nToplam {updated} kayit guncellendi. Arayuzu yenileyin.")

if __name__ == "__main__":
    main()
