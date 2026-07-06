# PortföyOptim — BIST 30 ML Destekli Portföy Optimizasyonu

Modern Portföy Teorisi (MPT), Black-Litterman dengeleme, ARIMA zaman serisi modelleri ve LSTM derin öğrenmeyi bir araya getirerek BIST 30 hisseleri için getiri/risk tahmini ve optimal portföy önerisi sunan Flask tabanlı web uygulaması.

> **Not:** Uygulama kaynak kodu `projem/` klasöründedir. Veritabanı (`mpt_veritabani.db`) ve eğitilmiş LSTM modelleri (`lstm_models/`) depo kökünde tutulur.

---

## İçindekiler

- [Ne Yapar?](#ne-yapar)
- [Mimari](#mimari)
- [ML Pipeline](#ml-pipeline)
- [Web Arayüzü ve API](#web-arayüzü-ve-api)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Proje Yapısı](#proje-yapısı)
- [Veritabanı](#veritabanı)
- [Konfigürasyon](#konfigürasyon)
- [Testler](#testler)
- [Backtest Sonuçları](#backtest-sonuçları)
- [Sorumluluk Reddi](#sorumluluk-reddi)

---

## Ne Yapar?

PortföyOptim, tek bir platformda şu işlevleri birleştirir:

| Alan | Açıklama |
|------|----------|
| **Uzun vade tahmin** | Her BIST 30 hissesi için 1, 3, 6 ve 12 aylık dönem getirisi (ARIMA + Black-Litterman + LSTM ensemble) |
| **Haftalık tahmin** | 5 iş günlük ufuk; yükseliş olasılığı, dinamik stop-loss, bias düzeltmesi |
| **Portföy optimizasyonu** | Monte Carlo simülasyonu ile risk profiline göre (düşük/orta/yüksek) optimal ağırlıklar |
| **Performans takibi** | Bireysel hisse ve portföy seviyesinde tahmin doğruluğu; benchmark karşılaştırması |
| **Makro rejim** | USD/TRY, BIST100, Brent petrol ve TCMB politika faizi ile stres skoru |
| **Risk radarı** | BIST100 çöküş olasılığı (RSI, SMA, volatilite sinyalleri) |
| **Halka arz analizi** | Geçmiş IPO verilerine dayalı tavan süresi tahmini ve backtest |
| **Kullanıcı yarışması** | Haftalık liderlik tablosu — kullanıcı portföyü vs AI portföyü |

Arayüzde uygulama adı **PortföyOptim** olarak görünür.

---

## Mimari

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Flask Web Uygulaması                        │
│  app.py  →  routes/views.py  →  portfolio/optimizer.py            │
│           auth.py (Flask-Login)    services/charts.py (Plotly)      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ SQLite okuma/yazma
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    mpt_veritabani.db (repo kökü)                    │
│  beklenen_getiriler │ tahminler │ portfoy_performans │ users │ ...  │
└───────────────────────────────▲─────────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                               │
┌───────┴──────────┐                         ┌──────────┴─────────┐
│   batch_job.py   │                         │ weekly_forecast.py │
│  1/3/6/12 ay     │                         │   5 iş günü        │
└───────┬──────────┘                         └──────────┬─────────┘
        │                                                 │
        ▼                                                 ▼
┌───────────────────────────────────────────────────────────────────┐
│                     forecasting/ paketi                           │
│  arima.py │ black_litterman.py │ garch.py │ batch_pipeline.py     │
│  weekly_pipeline.py │ risk_metrics.py                            │
└───────────────────────────────┬───────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ lstm_forecast │     │ feature_engineer│     │ macro_features.py  │
│ + lstm_models/│     │ 51 teknik gösterge│     │ evds_client.py     │
└───────────────┘     └─────────────────┘     └──────────────────┘
        ▲
        │ yfinance (BIST30, XU100, USDTRY, Brent)
        └─────────────────────────────────────────
```

### Veri kaynakları

- **Yahoo Finance (`yfinance`):** BIST 30 fiyatları, XU100 endeksi, USD/TRY, Brent petrol
- **TCMB EVDS:** Politika faizi serisi (`TP.APIFON4`) — ücretsiz API anahtarı gerekir
- **Yerel JSON:** `projem/data/ipo_history.json`, `upcoming_ipos.json`

---

## ML Pipeline

### Uzun vade — `batch_job.py`

Her BIST 30 hissesi için dört vade ayrı ayrı hesaplanır; sonuçlar `beklenen_getiriler` tablosuna yazılır.

| Vade | İş günü | Kullanım |
|------|---------|----------|
| 1 ay | 22 | Kısa vadeli portföy, freemium kullanıcılar |
| 3 ay | 66 | Orta vade |
| 6 ay | 126 | Orta-uzun vade |
| 12 ay | 252 | Yıllık strateji |

**Akış (her hisse × her vade):**

```
yfinance fiyat verisi (2018 → bugün)
    │
    ├─► Piyasa rejimi tespiti (bull / bear / sideways)
    ├─► Makro rejim overlay (kur şoku, endeks zayıflığı)
    ├─► GARCH(1,1) → yıllık volatilite (tahmini_yillik_risk)
    │
    └─► ARIMA (pmdarima auto-ARIMA + 51 teknik gösterge + makro exog)
            │
            ├─► finalize_arima_period_forecast()
            │     (tarihsel benchmark'a shrink + volatilite bazlı yumuşak clip)
            │
            ├─► Black-Litterman (piyasa cap ağırlıkları + ARIMA view'ları)
            │
            ├─► [USE_LSTM=True] LSTM dönem tahmini
            │     └─► ensemble_period_forecast() — dinamik ARIMA/BL vs LSTM ağırlığı
            │         (yön çelişkisinde LSTM atlanır, aşırı değerler sınırlanır)
            │
            └─► Rejim düzeltmesi → DB'ye kayıt
```

Model etiketi: `ARIMA+BL` veya `ARIMA+BL+LSTM`.

### Haftalık — `weekly_forecast.py`

- **Ufuk:** 5 iş günü (Pazartesi–Cuma)
- **Ek çıktılar:** `prob_up` (yükseliş olasılığı), `stop_loss`, `bias_correction`, portföy VaR95
- **Kayıt:** `tahminler` tablosu; duplicate gün kontrolü yapılır
- **Otomasyon:** `projem/run_weekly_forecast.bat` (Windows Task Scheduler — Cuma 19:30)

### Feature engineering

`feature_engineering.py` 51 teknik gösterge üretir: RSI, MACD, Bollinger bantları, hacim osilatörleri, trend göstergeleri vb. Bu özellikler hem ARIMA exogenous değişkenleri hem de LSTM girdileri olarak kullanılır.

### LSTM modelleri

- TensorFlow/Keras tabanlı 3 katmanlı multi-feature LSTM
- Modeller `lstm_models/{TICKER}_model.keras` olarak önbelleğe alınır (30 hisse × 3 dosya: model, scaler, meta)
- Multi-feature modda güvenilir pencere 5 gün; uzun ufuklarda geometrik extrapolasyon uygulanır
- TensorFlow kurulumu başarısızsa pipeline ARIMA+BL ile devam eder

### Makro katmanı

`macro_features.py` şu sinyalleri üretir:

- `usdtry_mom_5d`, `usdtry_vol_10d` — kur momentumu ve volatilitesi
- `xu100_mom_5d`, `xu100_trend` — endeks momentumu
- `brent_mom_10d` — petrol fiyatı
- TCMB politika faizi (`evds_client.py` üzerinden)

Makro stres skoru (0–1) rejim tespitini güçlendirir. Veri indirilemezse pipeline makrosuz çalışmaya devam eder.

---

## Web Arayüzü ve API

### Sayfalar

| Route | Açıklama |
|-------|----------|
| `/` | Giriş yoksa landing, varsa ana optimizasyon paneli |
| `/compare` | Klasik MPT vs AI/ML karşılaştırması |
| `/forecast_summary` | Bireysel hisse tahmin performansı |
| `/portfolio_performance` | Portföy seviyesi performans (alpha, VaR) |
| `/leaderboard` | Haftalık kullanıcı vs AI sıralaması |
| `/create_portfolio` | 5 hisselik yarışma portföyü oluşturma |
| `/abonelik` | Fiyatlandırma ve premium özellikler |

### Temel API uçları

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/api/bist30_live` | GET | Canlı BIST30 fiyat özeti |
| `/optimize` | POST | Seçili hisseler için klasik MPT |
| `/optimize_bist30_ml` | GET | ML destekli BIST30 Monte Carlo optimizasyonu |
| `/optimize_bist30_ml_with_horizon` | GET | Vade (`months`), `top_n`, `risk_profile` parametreleri |
| `/get_single_stock_forecast` | GET | Tek hisse ML tahmini (`ticker`, `months`) |
| `/api/compare_methods` | POST | MPT vs AI/ML JSON karşılaştırması |
| `/api/forecast_summary` | GET | Tahmin doğruluk özeti (otomatik backfill) |
| `/api/portfolio_performance` | GET | Portföy performans metrikleri |
| `/api/crash_radar` | GET | BIST100 çöküş olasılığı |
| `/api/macro_dashboard` | GET | Makro stres skoru ve kur/endeks metrikleri |
| `/api/predict_ipo` | POST | Halka arz tavan süresi tahmini |
| `/api/ipo_backtest` | GET | IPO modeli leave-one-out backtest |
| `/api/portfolios` | GET/POST | Kayıtlı portföyler |

### Freemium modeli

| Özellik | Ücretsiz | Premium / Admin |
|---------|----------|-----------------|
| ML portföy optimizasyonu | 1 kez demo | Sınırsız |
| Vade seçimi | Sadece 1 ay | 1 / 3 / 6 / 12 ay |
| Tek hisse tahmini | İstediği 2 hisse | Tüm BIST 30 |
| Portföy performans arşivi | Sınırlı | Tam erişim |

Premium aktivasyon (manuel): `python scripts/promote_premium.py kullanici@email.com`

---

## Kurulum

### Gereksinimler

- Python 3.10+
- Windows / Linux / macOS
- İnternet bağlantısı (yfinance, opsiyonel EVDS)

### Adımlar

```bash
# 1. Depoyu klonlayın
git clone <repo-url>
cd projem24032026

# 2. Bağımlılıkları kurun
cd projem
pip install -r requirements.txt

# 3. Ortam değişkenleri (opsiyonel ama önerilir)
# projem/.env dosyası oluşturun:
#   EVDS_API_KEY=your_key_here
#   SECRET_KEY=production-secret-key
# EVDS anahtarı: https://evds2.tcmb.gov.tr/

# 4. Uzun vade ML tahminlerini üretin (web arayüzü için zorunlu)
python batch_job.py

# 5. Haftalık tahminleri üretin (performans sayfaları için önerilir)
python weekly_forecast.py

# 6. Web sunucusunu başlatın
python app.py
```

Uygulama: **http://127.0.0.1:5000**

> `batch_job.py` tüm vadeler ve LSTM ile çalıştırıldığında 30–60+ dakika sürebilir.

### Bağımlılıklar

```
flask, flask-login, flask-bcrypt, python-dotenv, requests
pandas, numpy, scipy, yfinance
pmdarima, arch, tensorflow, scikit-learn
plotly, matplotlib, seaborn
```

---

## Çalıştırma

### Periyodik işler

| İş | Komut | Önerilen sıklık |
|----|-------|-----------------|
| Uzun vade ML batch | `cd projem && python batch_job.py` | Haftalık veya aylık |
| Haftalık tahmin | `cd projem && python weekly_forecast.py` | Her Cuma |
| Windows otomasyon | `projem/run_weekly_forecast.bat` | Task Scheduler |

### Yardımcı scriptler (`projem/scripts/`)

| Script | Amaç |
|--------|------|
| `promote_premium.py` | Kullanıcıyı premium yap |
| `promote_admin.py` | Admin yetkisi ver |
| `test_evds_connection.py` | EVDS API bağlantı testi |
| `run_ipo_backtest.py` | IPO model backtest |
| `run_macro_backtest_report.py` | Makro feature backtest raporu |
| `model_performance_report.py` | Model performans özeti |

Operasyon rehberi: [`projem/haftalik_tahmin_rehberi.md`](projem/haftalik_tahmin_rehberi.md)

---

## Proje Yapısı

```
projem24032026/
├── README.md                    # Bu dosya
├── mpt_veritabani.db            # Ana SQLite veritabanı
├── lstm_models/                 # Önceden eğitilmiş LSTM modelleri (30 hisse)
│   └── {TICKER}_model.keras | _scaler.pkl | _meta.pkl
│
└── projem/                      # Uygulama kaynak kodu
    ├── app.py                   # Flask giriş noktası
    ├── batch_job.py             # Uzun vade ML batch (1/3/6/12 ay)
    ├── weekly_forecast.py       # Haftalık 5 günlük tahmin
    ├── config.py                # Merkezi ayarlar
    ├── auth.py                  # Kullanıcı kimlik doğrulama
    ├── lstm_forecast.py         # LSTM eğitim ve tahmin
    ├── feature_engineering.py   # 51 teknik gösterge
    ├── macro_features.py        # Makro rejim ve stres skoru
    ├── evds_client.py           # TCMB EVDS API istemcisi
    ├── ipo_predictor.py         # Halka arz tavan süresi tahmini
    ├── crash_predictor.py       # BIST100 çöküş radarı
    ├── performance_tracker.py     # Bireysel hisse tahmin backfill
    ├── portfolio_performance_tracker.py
    ├── portfolio_vs_ai_scorer.py
    ├── model_selector.py        # Ensemble ağırlık seçimi
    ├── backtest.py              # Geçmiş performans analizi
    │
    ├── routes/
    │   └── views.py             # Tüm HTTP route'ları
    ├── portfolio/
    │   └── optimizer.py         # Monte Carlo optimizasyon
    ├── forecasting/
    │   ├── arima.py             # ARIMA + bileşik dönem getirisi
    │   ├── black_litterman.py   # Black-Litterman dengeleme
    │   ├── garch.py             # GARCH volatilite
    │   ├── batch_pipeline.py    # Vade utils, ensemble, ARIMA stabilize
    │   └── weekly_pipeline.py   # Haftalık pipeline
    ├── db/
    │   ├── schema.py            # Tablo şemaları
    │   ├── repository.py        # Veri erişim katmanı
    │   └── migrate_legacy.py    # Eski DB migrasyonu
    ├── services/
    │   ├── charts.py            # Plotly grafik üretimi
    │   └── helpers.py
    ├── templates/               # HTML şablonları (10 sayfa)
    ├── data/                    # IPO JSON verileri, backtest raporları
    ├── scripts/                 # Operasyon ve analiz scriptleri
    └── test_*.py                # unittest test paketi
```

### BIST 30 hisse listesi

30 hisse (`config.py`): AEFES, AKBNK, ASELS, ASTOR, BIMAS, DSTKF, EKGYO, ENKAI, EREGL, FROTO, GARAN, GUBRF, ISCTR, KCHOL, TRALT, KRDMD, MGROS, PETKM, PGSUS, SAHOL, SASA, SISE, TAVHL, TCELL, THYAO, TOASO, TTKOM, TUPRS, ULKER, YKBNK (`.IS` suffix ile).

---

## Veritabanı

Ana veritabanı: `mpt_veritabani.db` (repo kökü)

| Tablo | Amaç |
|-------|------|
| `beklenen_getiriler` | Uzun vade ML tahminleri — PK: `(ticker, horizon_ay)` |
| `tahminler` | Haftalık bireysel tahminler + gerçekleşen getiri |
| `portfoy_performans` | Haftalık portföy tahmini, benchmark, alpha, VaR95 |
| `users` | Auth, premium/demo bayrakları, `free_stock_views` |
| `user_portfolios` | Kayıtlı portföy ağırlıkları (JSON) |
| `leaderboard` | Haftalık kullanıcı vs AI getirisi |

`beklenen_getiriler` sütunları: `tahmini_donem_getiri`, `tahmini_yillik_getiri`, `tahmini_bl_getiri`, `tahmini_yillik_risk`, `model_tipi`, `guncelleme_tarihi`.

---

## Konfigürasyon

`projem/config.py` ve `projem/.env`:

| Ayar | Varsayılan | Açıklama |
|------|------------|----------|
| `USE_LSTM` | `True` | LSTM ensemble aktif |
| `USE_MACRO_FEATURES` | `True` | Makro exog + rejim overlay |
| `USE_EVDS_INTEREST` | `True` | TCMB politika faizi |
| `ANNUAL_ARIMA_CLIP_MAX` | `0.80` | Yıllık tahmin üst/alt sınırı (±%80) |
| `ENSEMBLE_ARIMA_W` / `ENSEMBLE_LSTM_W` | `0.5` / `0.5` | Sabit ensemble ağırlıkları (model_selector dinamik ayarlar) |
| `RISK_FREE_RATE` | `0.40` | Yıllık risksiz faiz (Türkiye bağlamı) |
| `EVDS_API_KEY` | `.env` | TCMB EVDS API anahtarı |
| `SECRET_KEY` | `.env` | Flask session güvenliği |
| `FLASK_DEBUG` | `false` | Debug modu |

---

## Testler

Tüm testler `unittest` tabanlıdır; `projem/` klasöründen çalıştırın:

```bash
cd projem

# Tüm test paketi
python -m unittest discover -v

# Önemli test modülleri
python test_integration.py          # Route ve entegrasyon (≥25 endpoint)
python test_accuracy_improvements.py # ML pipeline doğruluk testleri
python test_macro_features.py       # Makro feature hesaplamaları
python test_ipo_predictor.py        # IPO tahmin motoru
python test_evds_client.py          # EVDS API (key yoksa skip)
```

**Önkoşullar:** `mpt_veritabani.db` ve `lstm_models/` dizini mevcut olmalıdır. Bazı testler canlı yfinance ağına bağlıdır.

---

## Backtest Sonuçları

**Test dönemi:** Aralık 2024 – Aralık 2025 (1 yıl)

| Metrik | ML Model | BIST 30 Benchmark | Fark |
|--------|----------|-------------------|------|
| Toplam Getiri | %29.36 | %15.43 | **+%13.93** |
| Ortalama Aylık Getiri | %2.32 | %1.30 | +1.02% |
| Sharpe Ratio | 0.407 | 0.287 | +0.120 |
| Kazanma Oranı | %66.7 | — | — |
| Max Drawdown | -7.99% | — | — |

Detaylı backtest: `projem/backtest.py` ve `projem/scripts/model_performance_report.py`

---

## Sorumluluk Reddi

Bu proje bir **bitirme tezi** çalışmasıdır. Sunulan tahminler, portföy önerileri ve analizler yatırım tavsiyesi niteliği taşımaz. Geçmiş performans gelecekteki sonuçların garantisi değildir. Yatırım kararlarınızı kendi araştırmanız ve risk toleransınız doğrultusunda alınız.

---

## Ek Dokümantasyon

- [`projem/README.md`](projem/README.md) — Modül bazlı detaylı açıklamalar
- [`projem/haftalik_tahmin_rehberi.md`](projem/haftalik_tahmin_rehberi.md) — Haftalık tahmin operasyon rehberi

## Lisans

Bitirme tezi projesi — akademik kullanım.
