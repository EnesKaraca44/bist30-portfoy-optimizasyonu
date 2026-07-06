import scripts._bootstrap  # noqa: F401
import sqlite3
conn = sqlite3.connect('mpt_veritabani.db')
c = conn.cursor()
c.execute("SELECT id, tahmin_tarihi FROM portfoy_performans WHERE DATE(tahmin_tarihi) = '2026-02-21'")
rows = c.fetchall()
print("Bulunan kayıtlar:", rows)
if len(rows) > 1:
    c.execute("DELETE FROM portfoy_performans WHERE id = ?", (rows[0][0],))
    conn.commit()
    print(f"ID {rows[0][0]} silindi.")
conn.close()
