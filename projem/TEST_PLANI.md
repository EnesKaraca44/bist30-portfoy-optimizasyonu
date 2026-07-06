# Model Performansı Test Planı

## 📅 Haftalık Tahmin & Performans Takip Akışı

### 1️⃣ **CUMA KAPANIŞI (veya Cumartesi) - TAHMİN OLUŞTURMA**

```bash
python weekly_forecast.py
```

**Ne yapar:**
- Tüm BIST30 hisseleri için gelecek hafta (Pazartesi-Cuma) tahminleri oluşturur
- ARIMA + LSTM Ensemble (LSTM aktifse)
- Tahminleri `tahminler` tablosuna kaydeder
- Horizon: Bugünden Cuma'ya kadar işlem günü sayısı

**Kontrol:**
- Veritabanında `tahminler` tablosuna kayıtlar eklendi mi?

---

### 2️⃣ **ERTESİ HAFTA CUMA KAPANIŞ SONRASI - PERFORMANS GÜNCELLEME**

#### Seçenek A: Web Servisi ile (app.py çalışıyorsa)

**Endpoint:** `POST /update_performance`

**Tek hisse için:**
```bash
curl -X POST http://localhost:5000/update_performance \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AKBNK.IS", "horizon_days": 5, "lookback_days": 30}'
```

**Tüm hisseler için:**
```bash
curl -X POST http://localhost:5000/update_performance \
  -H "Content-Type: application/json" \
  -d '{"horizon_days": 5, "lookback_days": 30}'
```

#### Seçenek B: Python Script ile (direkt)

```python
from performance_tracker import backfill_actuals_for_ticker

# Tek hisse
backfill_actuals_for_ticker("AKBNK.IS", horizon_days=5, lookback_days=30)

# Tüm hisseler
import sqlite3
conn = sqlite3.connect('mpt_veritabani.db')
c = conn.cursor()
c.execute("SELECT DISTINCT ticker FROM tahminler")
tickers = [row[0] for row in c.fetchall()]
conn.close()

for ticker in tickers:
    count = backfill_actuals_for_ticker(ticker, horizon_days=5, lookback_days=30)
    print(f"{ticker}: {count} tahmin güncellendi")
```

**Ne yapar:**
- Horizon dolmuş tahminler için gerçekleşen getirileri hesaplar
- `tahminler` tablosunda `gerceklesen_getiri` ve `tahmin_hatasi` sütunlarını doldurur
- Sadece horizon dolmuş kayıtları günceller (5 işlem günü sonrası)

---

### 3️⃣ **PERFORMANS METRİKLERİNİ GÖRÜNTÜLEME**

#### Seçenek A: Web Servisi ile

**Tek hisse için:**
```bash
curl "http://localhost:5000/get_performance?ticker=AKBNK.IS"
```

**Tüm hisseler için:**
```bash
curl "http://localhost:5000/get_performance"
```

**Tarih aralığı ile:**
```bash
curl "http://localhost:5000/get_performance?ticker=AKBNK.IS&start_date=2024-12-01&end_date=2024-12-31"
```

#### Seçenek B: Python Script ile

```python
from performance_tracker import get_performance_metrics

# Tek hisse
metrics = get_performance_metrics("AKBNK.IS")
print(f"MAE: {metrics['mae']}")
print(f"RMSE: {metrics['rmse']}")
print(f"Direction Accuracy: {metrics['accuracy']}")
print(f"Valid Forecasts: {metrics['valid_forecasts']}/{metrics['total_forecasts']}")
```

---

### 4️⃣ **HORIZON KONTROLÜ**

Horizon doldu mu kontrol et:

```python
from performance_tracker import check_horizon_status

status = check_horizon_status("AKBNK.IS", "2024-12-13", horizon_days=5)
print(status['message'])  # "Horizon doldu, doğrulama yapılabilir." veya "X işlem günü kaldı."
```

---

## 🧪 TEST SENARYOSU (Bugün Cumartesi - Gelecek Hafta Test)

### Adım 1: Tahmin Oluştur (Bugün - Cumartesi)
```bash
python weekly_forecast.py
```
- Gelecek hafta (Pazartesi-Cuma) için tahminler oluşturulur
- Horizon: ~5 işlem günü (Pazartesi'den Cuma'ya)

### Adım 2: Veritabanını Kontrol Et
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('mpt_veritabani.db')
df = pd.read_sql_query("SELECT * FROM tahminler ORDER BY tahmin_tarihi DESC LIMIT 30", conn)
print(df[['ticker', 'tahmin_tarihi', 'tahmin_horizon', 'tahmini_getiri', 'gerceklesen_getiri']])
conn.close()
```

### Adım 3: Gelecek Hafta Sonu (Cuma Sonrası) - Performans Güncelle

**ÖNEMLİ:** Horizon dolmadan (5 işlem günü geçmeden) güncelleme yapılamaz!

Gelecek hafta Cuma sonrası:
```python
from performance_tracker import backfill_actuals_for_ticker

# Tüm hisseler için gerçekleşen getirileri hesapla
import sqlite3
conn = sqlite3.connect('mpt_veritabani.db')
c = conn.cursor()
c.execute("SELECT DISTINCT ticker FROM tahminler WHERE gerceklesen_getiri IS NULL")
tickers = [row[0] for row in c.fetchall()]
conn.close()

for ticker in tickers:
    count = backfill_actuals_for_ticker(ticker, horizon_days=5, lookback_days=30)
    if count > 0:
        print(f"✅ {ticker}: {count} tahmin güncellendi")
```

### Adım 4: Metrikleri Görüntüle
```python
from performance_tracker import get_performance_metrics

metrics = get_performance_metrics("AKBNK.IS")
print(f"""
Performans Metrikleri (AKBNK.IS):
- MAE (Mean Absolute Error): {metrics['mae']:.4f}
- RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}
- Direction Accuracy: {metrics['accuracy']:.2%}
- Geçerli Tahmin: {metrics['valid_forecasts']}/{metrics['total_forecasts']}
""")
```

---

## 🔍 HIZLI TEST (Geçmiş Tarih ile)

Eğer hemen test etmek isterseniz, geçmiş bir tahmini manuel güncelleyebilirsiniz:

```python
from datetime import datetime, timedelta
import sqlite3

# Geçmiş bir tahmin oluştur (horizon dolsun diye)
conn = sqlite3.connect('mpt_veritabani.db')
c = conn.cursor()

# 10 gün öncesinden bir tahmin ekle
past_date = (datetime.now() - timedelta(days=10)).isoformat()
c.execute('''
    INSERT OR REPLACE INTO tahminler 
    (ticker, tahmin_tarihi, tahmin_horizon, tahmini_getiri, model_tipi)
    VALUES (?, ?, ?, ?, ?)
''', ('AKBNK.IS', past_date, 5, 0.02, 'ARIMA'))

conn.commit()
conn.close()

# Şimdi backfill yap (horizon dolmuş olmalı)
from performance_tracker import backfill_actuals_for_ticker
count = backfill_actuals_for_ticker('AKBNK.IS', horizon_days=5, lookback_days=30)
print(f"Güncellenen tahmin sayısı: {count}")
```

---

## ✅ BAŞARILI TEST KRİTERLERİ

1. ✅ Tahminler oluşturuldu (`tahminler` tablosunda kayıtlar var)
2. ✅ Horizon dolduktan sonra backfill çalıştı
3. ✅ `gerceklesen_getiri` ve `tahmin_hatasi` sütunları dolduruldu
4. ✅ Performans metrikleri hesaplandı (MAE, RMSE, Accuracy)
5. ✅ Web servisi endpoint'leri çalışıyor

---

## 🚨 YAYGIN HATALAR

1. **"Horizon henüz dolmadı"** → 5 işlem günü geçmeli
2. **"No forecasts to update"** → Geçerli tahmin kaydı yok
3. **"Database locked"** → Başka bir process veritabanını kullanıyor (WAL mode kullanıyoruz, sorun olmamalı)
