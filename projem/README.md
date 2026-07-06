# 🤖 BIST 30 Portföy Optimizasyonu - ML Destekli Yatırım Sistemi

## 📋 Proje Özeti

Bu proje, Modern Portföy Teorisi (MPT) ile yapay zeka/derin öğrenme modellerini birleştirerek BIST 30 hisseleri için optimal portföy önerileri sunan bir web uygulamasıdır.

### 🎯 Ana Amaç

Geleneksel portföy optimizasyon yöntemlerini makine öğrenmesi modelleriyle geliştirerek, daha yüksek risk-ayarlı getiri (Sharpe Ratio) ve piyasayı aşan performans (Alpha) elde etmek.

---

## 🏆 Başarı Metrikleri (Backtest Sonuçları)

**Test Dönemi:** Aralık 2024 - Aralık 2025 (1 Yıl)

| Metrik | ML Model | BIST 30 Benchmark | Fark |
|--------|----------|-------------------|------|
| **Toplam Getiri** | %29.36 | %15.43 | **+13.93%** ✅ |
| **Ortalama Aylık Getiri** | %2.32 | %1.30 | +1.02% |
| **Sharpe Ratio** | 0.407 | 0.287 | +0.120 |
| **Kazanma Oranı** | %66.7 | - | - |
| **Max Drawdown** | -7.99% | - | - |
| **Volatilite** | 5.71% | 4.54% | +1.17% |

**SONUÇ:** ML modeli, BIST 30 endeksini **%13.93** oranında geride bırakmıştır! 🚀

---

## 🛠️ Teknolojiler

### Backend
- **Python 3.10+**
- **Flask** - Web framework
- **SQLite** - Veritabanı

### Makine Öğrenmesi & İstatistik
- **pmdarima** - ARIMA modeli
- **TensorFlow/Keras** - LSTM deep learning
- **arch** - GARCH volatilite modeli
- **scikit-learn** - Feature engineering

### Veri İşleme
- **pandas** - Veri manipülasyonu
- **numpy** - Sayısal hesaplamalar
- **yfinance** - Finansal veri çekme

### Görselleştirme
- **Plotly** - İnteraktif grafikler
- **HTML/CSS/JavaScript** - Frontend

---

## 📦 Kurulum

### 1. Gereksinimler

```bash
pip install flask pandas numpy yfinance pmdarima arch tensorflow scikit-learn plotly
```

### 2. Veritabanı ve Model Hazırlığı

```bash
# 1. ML modellerini eğit ve tahminleri oluştur
python batch_job.py

# 2. Haftalık tahminleri oluştur (isteğe bağlı)
python weekly_forecast.py
```

### 3. Web Uygulamasını Başlat

```bash
python app.py
```

Uygulama `http://127.0.0.1:5000` adresinde çalışacaktır.

---

## 📂 Proje Yapısı

```
projem/
├── app.py                          # Flask web uygulaması (Ana dosya)
├── backtest.py                     # Backtest ve performans analizi
├── batch_job.py                    # ML model eğitimi ve yıllık tahminler
├── weekly_forecast.py              # Haftalık tahmin oluşturma
├── portfolio_performance_tracker.py # Portföy seviyesi performans takibi
├── performance_tracker.py          # Bireysel hisse performans takibi
├── lstm_forecast.py                # LSTM deep learning modeli
├── feature_engineering.py          # 51 teknik gösterge hesaplama
├── model_selector.py               # Model seçimi ve ensemble
├── technical_indicators.py         # Teknik analiz göstergeleri
├── templates/                      # HTML şablonları
│   ├── index.html                  # Ana sayfa
│   ├── compare.html                # MPT vs ML karşılaştırma
│   ├── portfolio_performance.html  # Portföy performans takibi
│   └── forecast_summary.html       # Bireysel hisse performansı
├── lstm_models/                    # Eğitilmiş LSTM modelleri
├── mpt_veritabani.db               # SQLite veritabanı
└── README.md                       # Bu dosya
```

---

## 🎨 Özellikler

### 1. Ana Sayfa (`/`)
- BIST 30 portföy optimizasyonu
- ML destekli vs Klasik MPT karşılaştırması
- İnteraktif etkin sınır grafiği
- Portföy dağılımı görselleştirmesi
- Gelecek simülasyonu (Monte Carlo)

### 2. Karşılaştırma Sayfası (`/compare`)
- Klasik MPT ile ML sisteminin detaylı karşılaştırması
- Getiri, risk, Sharpe oranı karşılaştırması
- Metodoloji farkları

### 3. Portföy Performans Takibi (`/portfolio_performance`)
- Portföy seviyesinde performans metrikleri
- Gerçekleşen vs tahmin edilen getiri
- Alpha (benchmark'a göre fazladan getiri)
- Yön doğruluğu

### 4. Bireysel Hisse Performansı (`/forecast_summary`)
- Her hisse için tahmin doğruluğu
- Yön doğruluğu metrikleri
- Detaylı tahmin-geçekleşme karşılaştırması

---

## 🔬 Model Metodolojisi

### 1. Getiri Tahmini (3 Model Ensemble)

**ARIMA (Auto-Regressive Integrated Moving Average)**
- Zaman serisi tahmini
- Otomatik hiperparametre seçimi (pmdarima)
- Exogenous features desteği

**LSTM (Long Short-Term Memory)**
- Deep learning ile non-linear pattern öğrenme
- 51 teknik gösterge ile zenginleştirilmiş özellikler
- Seq2Seq mimarisi

**Black-Litterman Modeli**
- Tarihsel veriler ile ML tahminlerini birleştirme
- Daha dengeli ve güvenilir getiri tahminleri

### 2. Risk Tahmini (GARCH)

- GARCH(1,1) ile volatilite tahmini
- Dinamik risk ölçümü
- Kovaryans matrisi oluşturma

### 3. Portföy Optimizasyonu

- Monte Carlo simülasyonu (20,000 portföy)
- Maksimum Sharpe oranı stratejisi
- Minimum risk stratejisi
- Black-Litterman ile dengelenmiş getiri tahminleri

---

## 📊 Veri Kaynakları

- **Yahoo Finance (yfinance)**: BIST 30 hisse fiyat verileri
- **BIST 30 Endeksi (XU030.IS)**: Benchmark olarak kullanılıyor

---

## 🚀 Kullanım Senaryoları

### Senaryo 1: Yıllık Portföy Önerisi
```bash
# 1. ML modellerini güncelle
python batch_job.py

# 2. Web'i başlat
python app.py

# 3. Ana sayfada "BIST 30 ML Optimizasyonu" butonuna tıkla
```

### Senaryo 2: Haftalık Performans Takibi
```bash
# Her Cuma:
python weekly_forecast.py

# Ertesi Cuma:
# Web'de /portfolio_performance sayfasını aç
# Otomatik backfill yapılır
```

---

## 📈 Performans Sonuçları

### Backtest Özeti
- **Dönem:** Aralık 2024 - Aralık 2025 (1 yıl)
- **Başlangıç Sermayesi:** 100 TL
- **ML Model Sonuç:** 129.36 TL (%29.36 getiri)
- **BIST 30 Sonuç:** 115.43 TL (%15.43 getiri)
- **Alpha:** +13.93% (ML modeli %13.93 daha başarılı!)

### Gerçek Zamanlı Performans
- Portföy seviyesinde yön doğruluğu: %45.6 (son 3 tahmin)
- Geçen hafta başarı oranı: %60+ (en son tahmin)
- Ortalama hata: %4.1

---

## 🔧 Geliştirme Notları

### Model Parametreleri
- **LSTM:** 50 epoch, batch_size=32, learning_rate=0.001
- **ARIMA:** Otomatik seçim (pmdarima)
- **GARCH:** (1,1) parametreli
- **Monte Carlo:** 20,000 portföy simülasyonu

### Veritabanı Şeması
- `beklenen_getiriler`: Yıllık getiri/risk tahminleri
- `tahminler`: Haftalık tahminler ve gerçekleşen değerler
- `portfoy_performans`: Portföy seviyesi performans metrikleri

---

## 📝 Lisans

Bu proje bir bitirme tezi projesidir.

---

## 👤 Yazar

Bilgisayar Mühendisliği Bitirme Projesi
Tarih: Ocak 2025

---

## 🙏 Teşekkürler

- Yahoo Finance API
- Python finansal kütüphaneleri topluluğu
- Flask ve Plotly geliştiricileri

