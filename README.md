# BIST 30 Portföy Optimizasyonu - ML Destekli Yatırım Sistemi

Uygulama kaynak kodu `projem/` klasöründedir. Aşağıdaki komutları bu klasörde çalıştırın.

```bash
cd projem
```

## Proje Özeti

Modern Portföy Teorisi (MPT) ile yapay zeka/derin öğrenme modellerini birleştirerek BIST 30 hisseleri için optimal portföy önerileri sunan Flask tabanlı web uygulaması.

## Başarı Metrikleri (Backtest)

| Metrik | ML Model | BIST 30 | Fark |
|--------|----------|---------|------|
| Toplam Getiri | %29.36 | %15.43 | +%13.93 |
| Sharpe Ratio | 0.407 | 0.287 | +0.120 |

## Kurulum

```bash
cd projem
pip install -r requirements.txt
python batch_job.py
python app.py
```

Uygulama: `http://127.0.0.1:5000`

## Özellikler

- 1 / 3 / 6 / 12 aylık vadeye özel ARIMA + Black-Litterman + LSTM ensemble
- Haftalık tahmin pipeline'ı
- Monte Carlo portföy optimizasyonu
- IPO ve makro rejim katmanları

## Proje Yapısı

```
projem/          # Ana uygulama (Flask, ML, veritabanı)
lstm_models/     # Eğitilmiş LSTM modelleri
mpt_veritabani.db
```

Detaylı dokümantasyon: [projem/README.md](projem/README.md)

## Lisans

Bitirme tezi projesi.
