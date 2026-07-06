# 📅 Haftalık Tahmin ve Performans Takip Rehberi

## 🎯 Yeni Strateji: Portföy Odaklı Performans Takibi

Artık **bireysel hisse performansı** yerine **portföy seviyesinde performans** takip ediyoruz. Bu projenin asıl amacına (portföy optimizasyonu) daha uygundur.

---

## 📆 CUMA (Bugün) - Haftalık Tahmin Yapma

### Adım 1: Haftalık Tahmin Oluştur

```bash
python weekly_forecast.py
```

**Bu komut ne yapar?**
- ✅ Tüm BIST30 hisseleri için gelecek hafta (Pazartesi-Cuma) tahminleri yapar
- ✅ Bireysel hisse tahminlerini `tahminler` tablosuna kaydeder
- ✅ **YENİ:** Portföy tahminini `portfoy_performans` tablosuna kaydeder
- ✅ Optimal portföy ağırlıklarını kaydeder

**Beklenen çıktı:**
```
[*] Haftalik Tahmin Isi Baslatiliyor...
[*] Haftalik Tahmin: 2025-12-19 -> 2025-12-26 (5 gun)
[+] Bu hafta: Bugunden Cuma'ya kadar
[*] Haftalik Tahminler Olusturuluyor...
  [+] AEFES.IS: Getiri=1.23%, Risk=2.45%
  [+] AKBNK.IS: Getiri=0.98%, Risk=2.12%
  ...
[+] 30 hisse icin tahminler kaydedildi!
[+] Portfoy tahmini kaydedildi: %1.45
```

### Adım 2: Web'de Kontrol Et

1. `app.py`'yi çalıştırın:
   ```bash
   python app.py
   ```

2. Tarayıcıda açın: `http://127.0.0.1:5000`

3. "Portföy Performans Takibi" butonuna tıklayın

4. Bu hafta için portföy tahminini görün:
   - Portföy tahmini getiri
   - Portföy ağırlıkları (hangi hisselere ne kadar yatırım)
   - Horizon: 5 gün (Pazartesi-Cuma)

---

## 📆 GELECEK HAFTA CUMA (Test Zamanı)

### Adım 1: Gerçekleşen Değerleri Doldur (Backfill)

**Seçenek A: Web Arayüzü ile**
1. `http://127.0.0.1:5000/portfolio_performance` sayfasına gidin
2. Sayfa otomatik olarak backfill yapar (API çağrısı sırasında)
3. Veya manuel olarak sayfayı yenileyin

**Seçenek B: Python ile**
```python
from portfolio_performance_tracker import backfill_portfolio_performance

# Tüm eksik portföy performans verilerini doldur
updated = backfill_portfolio_performance(lookback_days=30)
print(f"{updated} portföy tahmini güncellendi!")
```

**Seçenek C: Terminal ile**
```bash
python -c "from portfolio_performance_tracker import backfill_portfolio_performance; print(f'Güncellenen: {backfill_portfolio_performance(30)}')"
```

### Adım 2: Performansı Kontrol Et

Web arayüzünde (`/portfolio_performance`) göreceksiniz:

1. **Portföy Tahmini vs Gerçekleşen**
   - Model: "Portföy %X getiri sağlar" dedi
   - Gerçek: Portföy %Y getiri sağladı
   - Fark: %Z

2. **Yön Doğruluğu**
   - Model "yükselir" dedi, gerçekten yükseldi mi?
   - ✓ DOĞRU veya ✗ YANLIŞ

3. **Alpha (Benchmark'a Göre)**
   - Portföy BIST30 endeksini yendi mi?
   - Pozitif alpha = Başarılı! 🎉

4. **Portföy Ağırlıkları**
   - Hangi hisselere ne kadar yatırım yapıldı?
   - Bu ağırlıklar optimal miydi?

---

## 📊 Haftalık Rutin (Her Cuma)

### Cuma Öğleden Sonra (Tahmin)
```bash
python weekly_forecast.py
```

### Ertesi Cuma (Test)
1. Web'de `/portfolio_performance` sayfasını aç
2. Otomatik backfill yapılır
3. Sonuçları incele

---

## 🎯 Başarı Metrikleri

### İyi Performans Göstergeleri:
- ✅ **Yön Doğruluğu > %50**: Model yönü doğru tahmin ediyor
- ✅ **Ortalama Hata < %5**: Tahminler gerçeğe yakın
- ✅ **Pozitif Alpha**: Portföy benchmark'ı (BIST30) yendi
- ✅ **Tutarlılık**: Her hafta benzer sonuçlar

### İyileştirme Gerektiren Durumlar:
- ❌ Yön doğruluğu < %40
- ❌ Ortalama hata > %10
- ❌ Negatif alpha (benchmark'ın gerisinde)
- ❌ Çok değişken sonuçlar

---

## 🔍 Detaylı Analiz

### Bireysel Hisse Performansı (Diagnostic)
Eğer portföy performansı kötüyse, bireysel hisse tahminlerini kontrol edin:
- `/forecast_summary` sayfası
- Hangi hisseler yanlış tahmin edildi?
- Model hangi hisselerde başarısız?

### Portföy Optimizasyonu
- Portföy ağırlıkları optimal mi?
- Hangi hisselere daha fazla ağırlık verilmeli?
- Risk-return dengesi doğru mu?

---

## 💡 İpuçları

1. **Her Cuma düzenli tahmin yapın** - Veri birikimi önemli
2. **Her Cuma sonrası backfill yapın** - Gerçekleşen değerleri doldurun
3. **Sonuçları not edin** - Hangi haftalar başarılı, hangileri değil?
4. **Modeli iyileştirin** - Kötü performans gösteren hisseleri analiz edin

---

## 🚀 Hızlı Başlangıç

**Bugün (Cuma):**
```bash
# 1. Tahmin yap
python weekly_forecast.py

# 2. Web'i başlat
python app.py

# 3. Tarayıcıda aç
# http://127.0.0.1:5000/portfolio_performance
```

**Gelecek Cuma:**
```bash
# 1. Web'i aç
# http://127.0.0.1:5000/portfolio_performance
# (Otomatik backfill yapılır)

# 2. Sonuçları incele
# - Portföy tahmini vs gerçekleşen
# - Alpha (benchmark'a göre)
# - Yön doğruluğu
```

---

## ❓ Sorun Giderme

**"Henüz portföy performans verisi yok" hatası:**
- `weekly_forecast.py` çalıştırıldı mı?
- `portfoy_performans` tablosu oluşturuldu mu?

**"Gerçekleşen değer NULL" görünüyor:**
- Horizon doldu mu? (5 işlem günü geçti mi?)
- Backfill çalıştırıldı mı?
- Fiyat verileri çekilebildi mi?

**"Portföy tahmini kaydedilemedi" uyarısı:**
- `beklenen_getiriler` tablosunda veri var mı?
- Önce `batch_job.py` çalıştırıldı mı?

