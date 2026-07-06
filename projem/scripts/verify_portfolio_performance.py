import scripts._bootstrap  # noqa: F401
"""
Belirli bir tahmin tarihi için haftalik portfoy performansinin nasil hesaplandigini
hisse hisse gosterir. Boylece -4.95% gibi degerin nereden geldigini dogrulayabilirsiniz.

Kullanim: python verify_portfolio_performance.py
         python verify_portfolio_performance.py 2026-02-28
"""
import sys
import json
import sqlite3
from portfolio_performance_tracker import (
    get_portfolio_actual_breakdown,
    add_business_days,
    DB_NAME,
)

def main():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Tarih arguman veya son tamamlanan kayit
    if len(sys.argv) > 1:
        date_arg = sys.argv[1].strip()
        c.execute("""
            SELECT tahmin_tarihi, tahmin_horizon, portfoy_gerceklesen_getiri, portfoy_agirliklari
            FROM portfoy_performans
            WHERE DATE(tahmin_tarihi) = ?
            ORDER BY tahmin_tarihi DESC LIMIT 1
        """, (date_arg,))
    else:
        c.execute("""
            SELECT tahmin_tarihi, tahmin_horizon, portfoy_gerceklesen_getiri, portfoy_agirliklari
            FROM portfoy_performans
            WHERE portfoy_agirliklari IS NOT NULL
            ORDER BY tahmin_tarihi DESC LIMIT 1
        """)
    row = c.fetchone()
    conn.close()
    if not row:
        print("Kayit bulunamadi.")
        return
    date_str, horizon, db_actual, weights_json = row
    try:
        weights = json.loads(weights_json)
    except Exception:
        print("Agirliklar okunamadi.")
        return
    import pandas as pd
    forecast_date = pd.to_datetime(date_str)
    total_ret, target_date, breakdown = get_portfolio_actual_breakdown(
        forecast_date, int(horizon), weights
    )
    if total_ret is None:
        print("Hesaplama yapilamadi (veri eksik veya hata).")
        return
    print("=" * 80)
    print("HAFTALIK PORTFOY PERFORMANS DOGRULAMA")
    print("=" * 80)
    print(f"Tahmin tarihi (baslangic): {forecast_date.date()}")
    print(f"Hedef tarih (bitis)      : {target_date}  (tahmin + {horizon} is gunu)")
    print(f"Horizon (is gunu)        : {horizon}")
    print()
    print(f"Kodun hesapladigi toplam getiri: {total_ret*100:+.2f}%")
    if db_actual is not None:
        print(f"Veritabanindaki kayit          : {db_actual*100:+.2f}%")
        print()
    print("Hisse bazli detay (baslangic/bitis fiyat ve agirlikli katki):")
    print("-" * 80)
    for b in breakdown:
        print(f"  {b['ticker']:14} | {str(b['start_date'])[:10]} -> {str(b['end_date'])[:10]} | "
              f"Fiyat: {b['start_price']:.2f} -> {b['end_price']:.2f} | "
              f"Getiri: {b['ret_pct']:+.2f}% | Agirlik: %{b['weight']:.2f} | Katki: {b['contribution_pct']:+.4f}%")
    print("-" * 80)
    print(f"Toplam (agirlikli) getiri: {total_ret*100:+.2f}%")
    print()
    print("Not: Bitis tarihi 'tahmin tarihi + horizon is gunu' ile hesaplanir.")
    print("     Fiyatlar yfinance'dan; borsa acik olmayan gunlerde en yakin gun kullanilir.")

if __name__ == "__main__":
    main()
