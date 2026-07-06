import scripts._bootstrap  # noqa: F401

import sqlite3
import pandas as pd
from datetime import datetime

conn = sqlite3.connect('mpt_veritabani.db')
c = conn.cursor()

# Get the latest forecast date
c.execute("SELECT MAX(tahmin_tarihi) FROM tahminler")
last_date = c.fetchone()[0]

print(f"En son tahmin tarihi: {last_date}")

if last_date:
    query = """
    SELECT ticker, tahmini_getiri, model_tipi
    FROM tahminler 
    WHERE tahmin_tarihi = ?
    ORDER BY tahmini_getiri DESC
    LIMIT 5
    """
    c.execute(query, (last_date,))
    rows = c.fetchall()
    
    print("\n--- EN YÜKSEK GETİRİ TAHMİNLERİ ---")
    for row in rows:
        ticker, ret, model = row
        print(f"{ticker}: %{ret*100:.2f} ({model})")

conn.close()
