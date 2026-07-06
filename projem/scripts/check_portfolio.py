import scripts._bootstrap  # noqa: F401

import sqlite3
import json
import pandas as pd

try:
    conn = sqlite3.connect('mpt_veritabani.db')
    c = conn.cursor()
    
    # Check if table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfoy_performans'")
    if not c.fetchone():
        print("Tablo bulunamadı: portfoy_performans")
    else:
        # Get latest forecast
        c.execute("""
            SELECT tahmin_tarihi, portfoy_tahmini_getiri, portfoy_agirliklari 
            FROM portfoy_performans 
            ORDER BY tahmin_tarihi DESC 
            LIMIT 1
        """)
        row = c.fetchone()
        
        if row:
            date, ret, weights_json = row
            print(f"Tarih: {date}")
            print(f"Beklenen Getiri: %{ret*100:.2f}")
            
            try:
                weights = json.loads(weights_json)
                # Sort by weight
                sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                
                print("\n--- MODEL PORTFÖY DAĞILIMI (TOP 5) ---")
                for ticker, weight in sorted_weights[:5]:
                    if weight > 0.01:
                        print(f"{ticker}: %{weight*100:.1f}")
            except:
                print("Ağırlıklar okunamadı.")
        else:
            print("Kayıtlı portföy tahmini yok.")
            
    conn.close()
except Exception as e:
    print(f"Hata: {e}")
