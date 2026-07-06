import scripts._bootstrap  # noqa: F401
"""
Portföy başarı karnesinden hatalı/istenmeyen tarihleri siler.
- 2026-02-14: Eski BIST30 listesi (VAKBN vb.) ve hatalı benchmark ile oluşturuldu; karneden çıkarılır.
"""
import sqlite3

from config import DB_PATH as DB_NAME

# Karneden çıkarılacak tarihler (DATE(tahmin_tarihi) bu tarihlere eşit olan kayıtlar silinir)
DATES_TO_REMOVE = [
    '2026-03-07',  # Eski tahmin (yeni tahmine göre güncellendi)
    '2026-03-08',  # Eski tahmin (yeni tahmine göre güncellendi)
]

try:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    for date_str in DATES_TO_REMOVE:
        c.execute(
            "DELETE FROM portfoy_performans WHERE DATE(tahmin_tarihi) = ?",
            (date_str,)
        )
        deleted = c.rowcount
        conn.commit()
        print(f"Silinen kayıt sayısı ({date_str}): {deleted}")

    # Kalan son kayıtlar
    c.execute(
        "SELECT tahmin_tarihi, tahmin_horizon FROM portfoy_performans ORDER BY tahmin_tarihi DESC LIMIT 5"
    )
    rows = c.fetchall()
    print("\nKalan Son Kayıtlar:")
    for row in rows:
        print(f"  {row[0]}  horizon={row[1]}")

    conn.close()
except Exception as e:
    print(f"Hata: {e}")
