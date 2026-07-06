# 🎓 BİTİRME PROJESİ SUNUM HAZIRLIK DOKÜMANI

## 📋 İÇİNDEKİLER

1. [Proje Özeti ve Mimarisi](#1-proje-özeti-ve-mimarisi)
2. [Sunum Konuşma Metni](#2-sunum-konuşma-metni)
3. [Kod Detayları ve Algoritma Açıklamaları](#3-kod-detayları-ve-algoritma-açıklamaları)

---

# 1. PROJE ÖZETİ VE MİMARİSİ

## 🎯 Proje Amacı

**BIST 30 Portföy Optimizasyonu - ML Destekli Yatırım Sistemi**

Bu proje, Modern Portföy Teorisi (MPT) ile yapay zeka/derin öğrenme modellerini birleştirerek BIST 30 hisseleri için optimal portföy önerileri sunan bir web uygulamasıdır.

### Ana Hedefler:
- Geleneksel portföy optimizasyon yöntemlerini makine öğrenmesi modelleriyle geliştirmek
- Daha yüksek risk-ayarlı getiri (Sharpe Ratio) elde etmek
- Piyasayı aşan performans (Alpha) sağlamak

---

## 🏗️ Sistem Mimarisi

### 1. **Veri Katmanı**
- **Kaynak:** Yahoo Finance (yfinance)
- **Veritabanı:** SQLite (`mpt_veritabani.db`)
- **Tablolar:**
  - `beklenen_getiriler`: Yıllık getiri/risk tahminleri
  - `tahminler`: Haftalık tahminler ve gerçekleşen değerler
  - `portfoy_performans`: Portföy seviyesi performans metrikleri

### 2. **Model Katmanı**

#### A. Getiri Tahmini (3 Model Ensemble)
1. **ARIMA (Auto-Regressive Integrated Moving Average)**
   - Zaman serisi tahmini
   - Otomatik hiperparametre seçimi (pmdarima)
   - Exogenous features desteği (51 teknik gösterge)

2. **LSTM (Long Short-Term Memory)**
   - Deep learning ile non-linear pattern öğrenme
   - 51 teknik gösterge ile zenginleştirilmiş özellikler
   - Seq2Seq mimarisi
   - Model cache sistemi (her tahminde yeniden eğitilmez)

3. **Black-Litterman Modeli**
   - Tarihsel veriler ile ML tahminlerini birleştirme
   - Piyasa dengesi (market equilibrium) + ML görüşleri
   - Daha dengeli ve güvenilir getiri tahminleri

#### B. Risk Tahmini
- **GARCH(1,1)**: Dinamik volatilite tahmini
- Kovaryans matrisi oluşturma (korelasyon + volatilite)

### 3. **Optimizasyon Katmanı**
- **Monte Carlo Simülasyonu:** 20,000-100,000 portföy
- **Stratejiler:**
  - Maksimum Sharpe oranı
  - Minimum risk
  - Dengeli (balanced)

### 4. **Web Arayüzü (Flask)**
- Ana sayfa: Portföy optimizasyonu
- Karşılaştırma: MPT vs ML
- Performans takibi: Portföy ve bireysel hisse

---

## 📊 Başarı Metrikleri

**Backtest Sonuçları (1 Yıl):**
- **Toplam Getiri:** %29.36 (ML) vs %15.43 (Benchmark) = **+13.93% Alpha**
- **Sharpe Ratio:** 0.407 (ML) vs 0.287 (Benchmark) = **+0.120**
- **Kazanma Oranı:** %66.7
- **Max Drawdown:** -7.99%
- **Volatilite:** 5.71%

---

# 2. SUNUM KONUŞMA METNİ

## 🎤 SUNUM BAŞLANGICI (2-3 dakika)

**"Merhaba, ben [İSİM]. Bugün sizlere bitirme projem olan 'BIST 30 Portföy Optimizasyonu - ML Destekli Yatırım Sistemi' projesini sunacağım."**

### Giriş

**"Modern Portföy Teorisi, 1952'den beri portföy optimizasyonunda kullanılan klasik bir yöntemdir. Ancak bu yöntem, sadece geçmiş verilere dayanır ve gelecek tahminleri yapmaz. Benim projem, bu klasik yöntemi yapay zeka ve derin öğrenme modelleriyle birleştirerek, daha yüksek performans elde etmeyi hedefliyor."**

### Problem Tanımı

**"Geleneksel MPT, geçmiş getiri ve risk verilerine dayanarak portföy oluşturur. Ancak finansal piyasalar dinamiktir ve geçmiş veriler geleceği her zaman yansıtmayabilir. Bu nedenle, gelecek tahminleri yapan ML modelleri kullanarak, daha akıllı portföy önerileri sunabiliriz."**

### Çözüm Yaklaşımı

**"Projemde üç ana model kullandım:**
1. **ARIMA** - Zaman serisi tahmini için
2. **LSTM** - Derin öğrenme ile non-linear pattern öğrenme
3. **Black-Litterman** - Tarihsel veriler ile ML tahminlerini birleştirme

**Bu modelleri ensemble yaparak, her hisse için gelecek getiri tahminleri yapıyorum. Sonra bu tahminleri kullanarak, Monte Carlo simülasyonu ile optimal portföy ağırlıklarını buluyorum."**

---

## 📈 METODOLOJİ BÖLÜMÜ (5-7 dakika)

### 1. Veri Toplama ve İşleme

**"BIST 30 hisselerinin fiyat verilerini Yahoo Finance'ten çekiyorum. 2018'den bugüne kadar olan verileri kullanıyorum. Bu verilerden günlük getirileri hesaplıyorum ve 51 teknik gösterge ekliyorum."**

**"Teknik göstergeler şunları içeriyor:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Momentum skorları
- Volatilite özellikleri
- Trend gücü

**Bu özellikler, modellerin daha iyi tahmin yapmasını sağlıyor."**

### 2. Getiri Tahmini Modelleri

#### ARIMA Modeli

**"ARIMA, zaman serisi tahmini için klasik bir yöntemdir. Auto-ARIMA kullanarak, her hisse için en iyi (p,d,q) parametrelerini otomatik seçiyorum. Ayrıca, 51 teknik göstergeyi exogenous variables olarak ekleyerek, SARIMAX modeli oluşturuyorum."**

**"ARIMA modeli, gelecek 22 günlük getirileri tahmin ediyor ve bunu yıllıklandırıyorum."**

#### LSTM Modeli

**"LSTM, derin öğrenme kullanarak non-linear pattern'leri öğrenir. Modelim 3 katmanlı LSTM içeriyor:**
- 128 nöron (ilk katman)
- 64 nöron (ikinci katman)
- 32 nöron (üçüncü katman)

**Her katmanda dropout (%30) kullanarak overfitting'i önlüyorum. Model, 60 günlük geçmiş veriyi kullanarak gelecek tahminleri yapıyor."**

**"Önemli bir optimizasyon: Model her tahminde sıfırdan eğitilmiyor. Eğitilmiş modeller kaydediliyor ve 30 gün geçerli kalıyor. Bu, hesaplama süresini önemli ölçüde azaltıyor."**

#### Black-Litterman Modeli

**"Black-Litterman, piyasa dengesi ile ML tahminlerini birleştirir. Piyasa dengesi, hisselerin piyasa değerlerine göre hesaplanır. ML tahminleri (ARIMA) ise 'görüşler' (views) olarak eklenir."**

**"Formül şu şekilde:**
```
μ_BL = (τΣ)^(-1) + P^T Ω^(-1) P)^(-1) × [(τΣ)^(-1) π + P^T Ω^(-1) Q]
```

**Burada:**
- `π`: Piyasa dengesi getirileri
- `Q`: ML tahminleri (ARIMA)
- `τ`: Görüş güveni (0.10)
- `Ω`: Görüş belirsizliği

**Black-Litterman, aşırı uçuk tahminleri dengeler ve daha güvenilir sonuçlar verir."**

### 3. Ensemble Yaklaşımı

**"ARIMA ve LSTM tahminlerini birleştiriyorum. Varsayılan olarak %50-50 ağırlık kullanıyorum, ancak model selector modülü ile her hisse için dinamik ağırlıklar hesaplanabilir."**

**"Ensemble formülü:**
```
Final_Tahmin = 0.5 × ARIMA_Tahmin + 0.5 × LSTM_Tahmin
```

**Backtest sonuçlarına göre, ensemble yaklaşımı tek başına ARIMA'dan %9.66 daha fazla alpha sağlıyor."**

### 4. Risk Tahmini (GARCH)

**"GARCH(1,1) modeli ile dinamik volatilite tahmini yapıyorum. GARCH, bugünkü riskin dünkü riske ve dünkü şoklara bağlı olduğunu varsayar."**

**"GARCH formülü:**
```
σ²_t = ω + α × ε²_(t-1) + β × σ²_(t-1)
```

**Bu volatilite tahminleri, kovaryans matrisini oluşturmak için kullanılır."**

### 5. Portföy Optimizasyonu

**"Monte Carlo simülasyonu ile 20,000-100,000 rastgele portföy oluşturuyorum. Her portföy için:**
- Beklenen getiri
- Risk (volatilite)
- Sharpe oranı

**hesaplanıyor. En yüksek Sharpe oranına sahip portföy seçiliyor."**

**"Sharpe oranı:**
```
Sharpe = (Beklenen Getiri - Risk-free Rate) / Risk
```

**Risk-free rate olarak 0 kullanıyorum (basitleştirme için)."**

---

## 💻 WEB UYGULAMASI BÖLÜMÜ (3-4 dakika)

### Flask Uygulaması

**"Web uygulaması Flask framework'ü ile geliştirildi. Ana sayfada BIST 30 portföy optimizasyonu yapılabiliyor."**

### Ana Özellikler

1. **Ana Sayfa (`/`)**
   - BIST 30 ML optimizasyonu
   - İnteraktif etkin sınır grafiği
   - Portföy dağılımı görselleştirmesi
   - Gelecek simülasyonu (Monte Carlo)

2. **Karşılaştırma Sayfası (`/compare`)**
   - Klasik MPT vs ML sistemi
   - Getiri, risk, Sharpe oranı karşılaştırması

3. **Portföy Performans Takibi (`/portfolio_performance`)**
   - Portföy seviyesinde performans metrikleri
   - Gerçekleşen vs tahmin edilen getiri
   - Alpha (benchmark'a göre fazladan getiri)

4. **Bireysel Hisse Performansı (`/forecast_summary`)**
   - Her hisse için tahmin doğruluğu
   - Yön doğruluğu metrikleri

### API Endpoint'leri

**"Backend'de RESTful API endpoint'leri var:**
- `/optimize_bist30_ml`: ML destekli optimizasyon
- `/api/portfolio_performance`: Portföy performans verileri
- `/api/forecast_summary`: Bireysel hisse performansı

**Frontend, bu API'leri kullanarak verileri çekiyor ve Plotly ile görselleştiriyor."**

---

## 📊 SONUÇLAR VE PERFORMANS (3-4 dakika)

### Backtest Sonuçları

**"Sistemimi 1 yıllık gerçek tarihsel verilerle test ettim (Aralık 2024 - Aralık 2025). Sonuçlar şu şekilde:"**

| Metrik | ML Model | BIST 30 Benchmark | Fark |
|--------|----------|-------------------|------|
| Toplam Getiri | %29.36 | %15.43 | **+13.93%** ✅ |
| Sharpe Ratio | 0.407 | 0.287 | +0.120 |
| Kazanma Oranı | %66.7 | - | - |
| Max Drawdown | -7.99% | - | - |

**"ML modeli, BIST 30 endeksini %13.93 oranında geride bırakmıştır. Bu, sistemin başarılı olduğunu gösteriyor."**

### Gerçek Zamanlı Performans

**"Ayrıca, sistemin canlı çalıştığını göstermek için son 2 haftadır gerçek zamanlı performans takibi yapıyorum. Backend'in doğru çalıştığını ve web arayüzünün gerçek verilerle entegre olduğunu bu şekilde doğruladım."**

**"Portföy seviyesinde yön doğruluğu: %45.6 (son 3 tahmin). Geçen hafta başarı oranı: %60+ (en son tahmin)."**

---

## 🔬 TEKNİK DETAYLAR (İsteğe Bağlı - Sorular Gelirse)

### Model Parametreleri

- **LSTM:** 15 epoch, batch_size=32, learning_rate=0.001 (Adam optimizer)
- **ARIMA:** Otomatik seçim (pmdarima), max_p=3, max_q=3
- **GARCH:** (1,1) parametreli
- **Monte Carlo:** 20,000-100,000 portföy simülasyonu
- **Black-Litterman:** risk_aversion=2.5, tau=0.10

### Veritabanı Şeması

- `beklenen_getiriler`: Yıllık getiri/risk tahminleri
- `tahminler`: Haftalık tahminler ve gerçekleşen değerler
- `portfoy_performans`: Portföy seviyesi performans metrikleri

---

## 🎯 SONUÇ VE GELECEK ÇALIŞMALAR (1-2 dakika)

### Sonuç

**"Projem, Modern Portföy Teorisi ile yapay zeka modellerini başarıyla birleştirdi. Backtest sonuçları, sistemin piyasayı %13.93 oranında geride bıraktığını gösteriyor. Web uygulaması, kullanıcı dostu bir arayüzle bu sistemi sunuyor."**

### Gelecek Çalışmalar

1. **Daha fazla veri:** Hacim verileri, makroekonomik göstergeler
2. **Daha gelişmiş modeller:** Transformer, Attention mekanizmaları
3. **Gerçek zamanlı trading:** Otomatik alım-satım sistemi
4. **Risk yönetimi:** Stop-loss, position sizing

### Teşekkür

**"Sorularınızı bekliyorum. Teşekkürler."**

---

# 3. KOD DETAYLARI VE ALGORİTMA AÇIKLAMALARI

## 📁 DOSYA YAPISI VE SORUMLULUKLAR

### 1. `app.py` - Flask Web Uygulaması

**Sorumluluk:** Web arayüzü ve API endpoint'leri

#### Önemli Fonksiyonlar:

##### `calculate_mpt(tickers, start_date='2018-01-01')`
```python
def calculate_mpt(tickers, start_date='2018-01-01'):
    """
    Klasik MPT hesaplama (kullanıcı girdisiyle)
    - Tarihsel verilerden ortalama getiri ve kovaryans hesaplar
    - Monte Carlo simülasyonu ile optimal portföy bulur
    """
```

**Algoritma:**
1. Yahoo Finance'ten fiyat verilerini çek
2. Günlük getirileri hesapla
3. Ortalama getiri ve kovaryans matrisini hesapla (yıllıklandır)
4. Monte Carlo simülasyonu (20,000 portföy)
5. En yüksek Sharpe oranı ve minimum risk portföylerini bul

##### `get_bist30_data_inputs(start_date='2018-01-01')`
```python
def get_bist30_data_inputs(start_date='2018-01-01'):
    """
    BIST 30 için ML tahminlerini veritabanından okur
    - Black-Litterman getirisi varsa onu kullan
    - Yoksa ARIMA getirisini kullan
    - GARCH riskleri ile kovaryans matrisi oluştur
    """
```

**Algoritma:**
1. Veritabanından ML tahminlerini oku
2. Black-Litterman getirisi varsa onu kullan, yoksa ARIMA'yı kullan
3. Tarihsel korelasyon matrisini çek
4. GARCH volatilite değerleri ile kovaryans matrisi oluştur:
   ```
   Σ_ML = D × Corr × D
   ```
   Burada D, volatilite değerlerinin köşegen matrisidir.

##### `run_black_litterman(prior_returns, view_returns, cov_matrix, tau=0.05, omega_scale=0.25)`
```python
def run_black_litterman(prior_returns, view_returns, cov_matrix, tau=0.05, omega_scale=0.25):
    """
    Black-Litterman modeli ile tarihsel ve ML tahminlerini birleştirir
    
    Formül:
    μ_BL = (τΣ)^(-1) + P^T Ω^(-1) P)^(-1) × [(τΣ)^(-1) π + P^T Ω^(-1) Q]
    
    Burada:
    - π: Prior returns (tarihsel ortalama)
    - Q: Views (ML tahminleri)
    - τ: Görüş güveni
    - Ω: Görüş belirsizliği
    """
```

**Algoritma:**
1. Piyasa dengesi getirilerini hesapla (π)
2. ML tahminlerini views olarak al (Q)
3. Görüş belirsizliği matrisini oluştur (Ω)
4. Black-Litterman formülünü uygula
5. Dengelenmiş getirileri döndür

##### `run_monte_carlo_simulation(expected_returns, cov_matrix, tickers, graph_title)`
```python
def run_monte_carlo_simulation(expected_returns, cov_matrix, tickers, graph_title):
    """
    Monte Carlo simülasyonu ile optimal portföy bulur
    
    Her portföy için:
    - Rastgele ağırlıklar oluştur (toplamı 1)
    - Beklenen getiri = w^T × μ
    - Risk = sqrt(w^T × Σ × w)
    - Sharpe = Getiri / Risk
    """
```

**Algoritma:**
1. 20,000 rastgele portföy oluştur
2. Her portföy için:
   - Ağırlıklar: `w = random_weights / sum(random_weights)`
   - Getiri: `μ_p = w^T × μ`
   - Risk: `σ_p = sqrt(w^T × Σ × w)`
   - Sharpe: `S = μ_p / σ_p`
3. En yüksek Sharpe ve minimum risk portföylerini bul
4. Plotly ile görselleştir

---

### 2. `batch_job.py` - ML Model Eğitimi ve Tahmin Üretimi

**Sorumluluk:** Yıllık getiri/risk tahminlerini oluşturur

#### Önemli Fonksiyonlar:

##### `get_arima_forecast(daily_returns, prices=None, use_features=True)`
```python
def get_arima_forecast(daily_returns, prices=None, use_features=True):
    """
    ARIMA ile gelecek getiri tahmini
    
    Algoritma:
    1. Feature engineering ile 51 teknik gösterge hesapla
    2. Auto-ARIMA ile en iyi (p,d,q) parametrelerini bul
    3. Exogenous features ile SARIMAX modeli eğit
    4. Gelecek 22 günlük tahmin yap
    5. Ortalama günlük getiriyi yıllıklandır (×252)
    """
```

**Detaylı Algoritma:**
1. **Feature Hazırlama:**
   - Fiyat serisinden 51 teknik gösterge çıkar
   - RSI, MACD, Bollinger Bands, Momentum, Volatilite, Trend
   - Exogenous features DataFrame'i oluştur

2. **Model Eğitimi:**
   ```python
   model = pm.auto_arima(
       daily_returns,
       X=exog_features,  # Exogenous features
       stepwise=True,
       max_p=3, max_q=3,
       seasonal=False
   )
   ```

3. **Tahmin:**
   ```python
   forecast = model.predict(n_periods=22, X=future_exog)
   annualized_return = forecast.mean() * 252
   ```

##### `get_garch_volatility(daily_returns)`
```python
def get_garch_volatility(daily_returns):
    """
    GARCH(1,1) ile volatilite tahmini
    
    Model:
    σ²_t = ω + α × ε²_(t-1) + β × σ²_(t-1)
    
    Algoritma:
    1. Getirileri ölçekle (×100)
    2. GARCH(1,1) modeli eğit
    3. Gelecek 22 günlük volatilite tahmini yap
    4. Ortalama volatiliteyi yıllıklandır (×sqrt(252))
    """
```

**Detaylı Algoritma:**
1. **Ölçekleme:**
   ```python
   returns_scaled = daily_returns * 100
   ```
   (GARCH küçük sayılarla zorlanır)

2. **Model Eğitimi:**
   ```python
   model = arch_model(returns_scaled, vol='Garch', p=1, q=1)
   model_fit = model.fit(disp='off')
   ```

3. **Tahmin:**
   ```python
   forecast = model_fit.forecast(horizon=22)
   next_month_volatility = np.sqrt(forecast.variance.values[-1, :])
   annualized_volatility = next_month_volatility.mean() / 100 * np.sqrt(252)
   ```

##### `calculate_black_litterman(market_caps, cov_matrix, arima_views, risk_aversion=2.5, tau=0.05)`
```python
def calculate_black_litterman(market_caps, cov_matrix, arima_views, risk_aversion=2.5, tau=0.05):
    """
    Black-Litterman modeli ile piyasa dengesi + ML görüşleri
    
    Algoritma:
    1. Piyasa ağırlıklarını hesapla (market_caps / total_mcap)
    2. Piyasa dengesi getirilerini hesapla: π = λ × Σ × w_market
    3. Görüş belirsizliği matrisini oluştur: Ω = τ × P × Σ × P^T
    4. Black-Litterman formülünü uygula
    5. Dengelenmiş getirileri döndür
    """
```

**Detaylı Algoritma:**
1. **Piyasa Ağırlıkları:**
   ```python
   total_mcap = market_caps.sum()
   market_weights = (market_caps / total_mcap).values
   ```

2. **Piyasa Dengesi Getirileri (Implied Equilibrium Returns):**
   ```python
   pi = risk_aversion * np.dot(cov_matrix, market_weights)
   ```
   Bu, CAPM'in tersidir: Eğer piyasa dengedeyse, getiriler risk ile orantılı olmalıdır.

3. **Görüşler (Views):**
   ```python
   Q = arima_views.values.reshape(-1, 1)  # ML tahminleri
   P = np.identity(len(market_caps))  # Her hisse için 1 görüş
   ```

4. **Görüş Belirsizliği:**
   ```python
   omega = np.dot(np.dot(P, tau * cov_matrix), P.T)
   ```

5. **Black-Litterman Formülü:**
   ```python
   tau_sigma = tau * cov_matrix
   tau_sigma_inv = np.linalg.inv(tau_sigma)
   omega_inv = np.linalg.inv(omega)
   
   left_term = np.linalg.inv(
       tau_sigma_inv + np.dot(np.dot(P.T, omega_inv), P)
   )
   
   right_term = (
       np.dot(tau_sigma_inv, pi) + 
       np.dot(np.dot(P.T, omega_inv), Q.flatten())
   )
   
   bl_returns = np.dot(left_term, right_term)
   ```

##### `run_bist30_ml_batch()`
```python
def run_bist30_ml_batch():
    """
    Ana batch işlemi: Tüm BIST 30 hisseleri için tahmin üretir
    
    Sıralama (BACKTEST İLE AYNI):
    1. ARIMA tahminleri (her hisse için)
    2. Black-Litterman dengelenmesi (tüm hisseler için)
    3. LSTM ensemble (her hisse için, BL sonrası)
    4. Veritabanına kaydet
    """
```

**Detaylı Algoritma:**
1. **Veri Hazırlama:**
   - Tarihsel verileri çek (2018-01-01'den bugüne)
   - Kovaryans matrisini hesapla
   - Piyasa değerlerini çek

2. **ARIMA Tahminleri:**
   ```python
   for ticker in valid_tickers:
       arima_mu = get_arima_forecast(series, prices=prices, use_features=True)
       arima_forecasts[ticker] = arima_mu
   ```

3. **Black-Litterman:**
   ```python
   bl_returns = calculate_black_litterman(
       market_caps, cov_matrix, arima_series,
       risk_aversion=2.5, tau=0.10
   )
   ```

4. **LSTM Ensemble:**
   ```python
   for ticker in valid_tickers:
       lstm_mu = forecast_lstm(series, forecast_days=22, ticker=ticker, ...)
       final_mu = ensemble_forecast(bl_value, lstm_mu, arima_weight=0.5, lstm_weight=0.5)
   ```

5. **Kayıt:**
   - Veritabanına kaydet (`beklenen_getiriler` tablosu)

---

### 3. `lstm_forecast.py` - LSTM Deep Learning Modeli

**Sorumluluk:** LSTM ile getiri tahmini

#### Önemli Fonksiyonlar:

##### `create_lstm_model(input_shape)`
```python
def create_lstm_model(input_shape):
    """
    LSTM model mimarisi oluşturur
    
    Multi-feature için:
    - 3 katmanlı LSTM (128, 64, 32 nöron)
    - Dropout (%30)
    - Dense katmanlar (16, 1)
    
    Single feature için:
    - 2 katmanlı LSTM (64, 32 nöron)
    - Dropout (%20)
    """
```

**Model Mimarisi (Multi-feature):**
```
Input: (batch_size, 60, 5)  # 60 gün, 5 feature
  ↓
LSTM(128, return_sequences=True)
  ↓ Dropout(0.3)
LSTM(64, return_sequences=True)
  ↓ Dropout(0.3)
LSTM(32, return_sequences=False)
  ↓ Dropout(0.2)
Dense(16, activation='relu')
  ↓
Dense(1)  # Output: gelecek getiri
```

##### `prepare_lstm_data_multifeature(prices, returns_series, lookback=60, scaler=None)`
```python
def prepare_lstm_data_multifeature(prices, returns_series, lookback=60, scaler=None):
    """
    LSTM için multi-feature veri hazırlığı
    
    Algoritma:
    1. Feature engineering ile 5 feature çıkar (RSI, MACD, BB, Momentum, Volatility)
    2. MinMaxScaler ile normalize et (0-1 arası)
    3. Sliding window ile (X, y) çiftleri oluştur
       - X: Son 60 günün feature'ları
       - y: 61. günün getirisi
    4. Shape: (samples, 60, 5)
    """
```

**Detaylı Algoritma:**
1. **Feature Hazırlama:**
   ```python
   feature_df = prepare_lstm_features(prices, returns_series, volume=None, n_features=5)
   # Returns, RSI, MACD, BB, Momentum, Volatility
   ```

2. **Normalizasyon:**
   ```python
   scaler = MinMaxScaler(feature_range=(0, 1))
   scaled_data = scaler.fit_transform(feature_df.values)
   ```

3. **Sliding Window:**
   ```python
   X, y = [], []
   for i in range(lookback, len(scaled_data)):
       X.append(scaled_data[i-lookback:i, :])  # Son 60 gün
       y.append(scaled_data[i, 0])  # 61. günün getirisi (ilk kolon)
   ```

##### `forecast_lstm(returns_series, forecast_days=22, ticker=None, prices=None, force_retrain=False, use_features=True)`
```python
def forecast_lstm(returns_series, forecast_days=22, ticker=None, prices=None, force_retrain=False, use_features=True):
    """
    LSTM ile tahmin yap (OPTİMİZE: Model cache sistemi)
    
    Algoritma:
    1. Model cache'den kontrol et (30 gün geçerli)
    2. Yoksa veya eskiyse eğit
    3. Son 60 günlük veriyi kullanarak inference yap
    4. Gelecek 22 günlük tahmin yap (iteratif)
    5. Yıllıklandır (×252)
    """
```

**Detaylı Algoritma:**
1. **Model Yükleme:**
   ```python
   if not force_retrain and is_model_valid(ticker):
       model, scaler = load_lstm_model(ticker)
   ```

2. **Model Eğitimi (Gerekirse):**
   ```python
   if model is None:
       model, scaler = train_lstm_model(ticker, returns_series, prices=prices, ...)
   ```

3. **Inference:**
   ```python
   last_sequence = X[-1].reshape(1, 60, n_features)
   predictions = []
   
   for _ in range(forecast_days):
       pred = model.predict(last_sequence, verbose=0)[0, 0]
       predictions.append(pred)
       
       # Sekansı kaydır (rolling window)
       last_sequence = np.roll(last_sequence, -1, axis=1)
       last_sequence[0, -1, 0] = pred  # Sadece returns'ü güncelle
   ```

4. **Inverse Transform:**
   ```python
   predictions_unscaled = scaler.inverse_transform(predictions_reshaped).flatten()
   daily_forecast = predictions_unscaled.mean()
   annualized = daily_forecast * 252
   ```

##### `ensemble_forecast(arima_pred, lstm_pred, arima_weight=0.6, lstm_weight=0.4)`
```python
def ensemble_forecast(arima_pred, lstm_pred, arima_weight=0.6, lstm_weight=0.4):
    """
    ARIMA ve LSTM tahminlerini birleştir
    
    Formül:
    Final = arima_weight × ARIMA + lstm_weight × LSTM
    """
```

---

### 4. `feature_engineering.py` - Feature Engineering

**Sorumluluk:** 51 teknik gösterge hesaplama

#### Önemli Fonksiyonlar:

##### `extract_all_features(prices, returns, volume=None, lookback=252)`
```python
def extract_all_features(prices, returns, volume=None, lookback=252):
    """
    Tüm feature'ları hesapla (51 teknik gösterge)
    
    Kategoriler:
    1. Technical Indicators: RSI, MACD, Bollinger Bands
    2. Volatility Features: 5, 10, 20, 30 günlük volatilite
    3. Trend Features: SMA, EMA, trend gücü
    4. Momentum Features: 5, 10, 20, 60 günlük momentum
    5. Volume Features: Hacim ortalaması, hacim trendi
    6. Statistical Features: Mean, std, skew, kurtosis
    """
```

**Detaylı Feature Listesi:**
- **RSI:** Relative Strength Index (0-100)
- **MACD:** MACD line, signal, histogram
- **Bollinger Bands:** Upper, middle, lower, position
- **Volatility:** 5d, 10d, 20d, 30d rolling volatility
- **Volatility Ratio:** Recent / overall volatility
- **SMA/EMA:** 10, 20, 50, 100 günlük
- **Price to SMA/EMA:** Fiyat / ortalama oranı
- **Momentum:** 5d, 10d, 20d, 60d price momentum
- **ROC:** Rate of Change
- **Trend Strength:** Short SMA / Long SMA
- **Volume:** SMA, ratio, trend
- **Statistics:** Mean, std, skew, kurtosis

##### `prepare_exogenous_features_for_arima(prices, returns, volume=None, n_features=5)`
```python
def prepare_exogenous_features_for_arima(prices, returns, volume=None, n_features=5):
    """
    ARIMA için exogenous features hazırla
    
    Öncelikli feature'lar:
    1. rsi_norm: RSI normalize (-1 ile 1 arası)
    2. macd_hist_norm: MACD histogram normalize
    3. bb_position: Bollinger Bands pozisyonu (0-1)
    4. momentum_20d: 20 günlük momentum
    5. trend_strength: Trend gücü
    """
```

##### `prepare_lstm_features(prices, returns, volume=None, n_features=5)`
```python
def prepare_lstm_features(prices, returns, volume=None, n_features=5):
    """
    LSTM için zaman serisi feature DataFrame'i oluştur
    
    Her gün için feature'lar hesaplanır (sliding window)
    Returns + 5 feature = 6 kolon
    """
```

---

### 5. `backtest.py` - Backtest Motoru

**Sorumluluk:** 1 yıllık gerçek verilerle sistem testi

#### Önemli Fonksiyonlar:

##### `run_quick_backtest()`
```python
def run_quick_backtest():
    """
    1 yıllık backtest (Aralık 2024 - Aralık 2025)
    
    Algoritma:
    1. Aylık dönemleri belirle
    2. Her ay için:
       a. Geçmiş verilerle ARIMA tahminleri yap
       b. Black-Litterman dengelenmesi
       c. LSTM ensemble
       d. Portföy optimizasyonu (Monte Carlo)
       e. Gerçekleşen getiriyi hesapla
       f. Portföy değerini güncelle
    3. Benchmark (BIST30 endeksi) ile karşılaştır
    4. Metrikleri hesapla ve görselleştir
    """
```

**Detaylı Algoritma:**
1. **Veri Hazırlama:**
   ```python
   monthly_dates = full_returns.loc[start_date:].resample('ME').last().index
   ```

2. **Her Ay İçin:**
   ```python
   for i in range(len(monthly_dates) - 1):
       current_date = monthly_dates[i]
       next_date = monthly_dates[i+1]
       
       # Geçmiş verilerle tahmin yap
       past_data = full_returns.loc[:current_date]
       
       # ARIMA tahminleri
       for ticker in BIST30:
           arima_mu = get_arima_forecast(past_data[ticker], ...)
       
       # Black-Litterman
       bl_returns = calculate_black_litterman(...)
       
       # LSTM Ensemble
       final_mu = ensemble_forecast(bl_value, lstm_value, ...)
       
       # Portföy optimizasyonu
       optimal_weights = optimize_portfolio(final_mu, cov_matrix, ...)
       
       # Gerçekleşen getiri
       period_return = (next_month_data.iloc[-1] / next_month_data.iloc[0]) - 1
       my_return = sum(weights[ticker] * period_return[ticker] for ticker in ...)
       
       portfolio_value *= (1 + my_return)
   ```

3. **Metrikler:**
   - Toplam getiri
   - Sharpe ratio
   - Kazanma oranı
   - Max drawdown
   - Volatilite

---

### 6. `weekly_forecast.py` - Haftalık Tahmin Üretimi

**Sorumluluk:** Haftalık tahminler oluşturur ve veritabanına kaydeder

#### Önemli Fonksiyonlar:

##### `forecast_weekly_returns(use_lstm=False, force_refresh=False)`
```python
def forecast_weekly_returns(use_lstm=False, force_refresh=False):
    """
    Tüm BIST30 hisseleri için haftalık tahmin oluştur
    
    Algoritma:
    1. Son 2 yıllık veriyi çek
    2. Her hisse için:
       a. ARIMA ile 5 günlük tahmin yap
       b. (Opsiyonel) LSTM ensemble
       c. GARCH ile risk tahmini
    3. Sonuçları döndür
    """
```

##### `save_weekly_forecasts(forecasts)`
```python
def save_weekly_forecasts(forecasts):
    """
    Haftalık tahminleri veritabanına kaydet
    
    Duplicate kontrolü:
    - Bugün için bu hisse için tahmin varsa güncelle
    - Yoksa yeni kayıt ekle
    """
```

---

### 7. `performance_tracker.py` - Performans Takibi

**Sorumluluk:** Bireysel hisse tahminlerinin gerçekleşen değerlerini hesaplar

#### Önemli Fonksiyonlar:

##### `add_business_days(start_date, num_days)`
```python
def add_business_days(start_date, num_days):
    """
    İş günü hesaplama (hafta sonlarını atla)
    
    Algoritma:
    - Başlangıç tarihinden itibaren günleri say
    - Hafta sonu (5=Cumartesi, 6=Pazar) ise atla
    - N iş günü sonrasını döndür
    """
```

**KRİTİK:** Bu fonksiyon, tahmin horizon'unun doğru hesaplanması için çok önemlidir. Örneğin, 5 iş günü = Pazartesi'den Cuma'ya kadar, hafta sonu dahil değil.

##### `backfill_actuals_for_ticker(ticker, horizon_days=5, lookback_days=30)`
```python
def backfill_actuals_for_ticker(ticker, horizon_days=5, lookback_days=30):
    """
    Eksik gerçekleşen getirileri otomatik doldur
    
    Algoritma:
    1. Horizon dolmuş ama gerçekleşen getiri boş olan tahminleri bul
    2. Her tahmin için:
       a. Tahmin tarihi + horizon (iş günü) = hedef tarih
       b. Tahmin tarihindeki fiyatı bul
       c. Hedef tarihteki fiyatı bul
       d. Gerçekleşen getiri = (hedef_fiyat / tahmin_fiyat) - 1
    3. Veritabanını güncelle
    """
```

**Detaylı Algoritma:**
1. **Tahminleri Bul:**
   ```python
   query = """
       SELECT id, tahmin_tarihi, tahmini_getiri, tahmin_horizon
       FROM tahminler 
       WHERE ticker = ? 
       AND gerceklesen_getiri IS NULL
   """
   ```

2. **İş Günü Hesaplama:**
   ```python
   forecast_date = pd.to_datetime(row['tahmin_tarihi']).date()
   target_date = add_business_days(forecast_date, horizon)
   ```

3. **Fiyat Bulma:**
   ```python
   # Başlangıç fiyatı
   if forecast_ts in prices.index:
       start_price = float(prices.loc[forecast_ts])
   else:
       past_prices = prices[prices.index <= forecast_ts]
       start_price = float(past_prices.iloc[-1])
   
   # Bitiş fiyatı
   if target_ts in prices.index:
       end_price = float(prices.loc[target_ts])
   else:
       future_prices = prices[prices.index >= target_ts]
       end_price = float(future_prices.iloc[0])
   ```

4. **Getiri Hesaplama:**
   ```python
   actual_return = (end_price / start_price) - 1
   error = abs(actual_return - predicted_return)
   ```

---

### 8. `portfolio_performance_tracker.py` - Portföy Performans Takibi

**Sorumluluk:** Portföy seviyesinde performans takibi

#### Önemli Fonksiyonlar:

##### `calculate_portfolio_forecast(forecast_date, horizon_days=5)`
```python
def calculate_portfolio_forecast(forecast_date, horizon_days=5):
    """
    Belirli bir tarih için portföy tahmini hesapla
    
    Algoritma:
    1. Veritabanından ML tahminlerini oku
    2. Kovaryans matrisi oluştur (korelasyon + GARCH volatilite)
    3. Monte Carlo optimizasyonu (50,000 portföy)
    4. En yüksek Sharpe portföyünü bul
    5. Horizon'a göre ölçekle (yıllık -> haftalık)
    """
```

##### `calculate_portfolio_actual(forecast_date, horizon_days, weights_dict)`
```python
def calculate_portfolio_actual(forecast_date, horizon_days, weights_dict):
    """
    Portföyün gerçekleşen getirisini hesapla
    
    Algoritma:
    1. Tahmin tarihi + horizon (iş günü) = hedef tarih
    2. Her hisse için:
       a. Tahmin tarihindeki fiyatı bul
       b. Hedef tarihteki fiyatı bul
       c. Getiri = (hedef_fiyat / tahmin_fiyat) - 1
    3. Portföy getirisi = sum(weight[ticker] × getiri[ticker])
    """
```

**Formül:**
```
Portföy Getirisi = Σ (w_i × r_i)

Burada:
- w_i: Hisse i'nin portföy ağırlığı
- r_i: Hisse i'nin gerçekleşen getirisi
```

---

## 🔑 ÖNEMLİ ALGORİTMA DETAYLARI

### 1. Black-Litterman Modeli Matematiksel Detay

**Piyasa Dengesi Getirileri (Implied Equilibrium Returns):**
```
π = λ × Σ × w_market

Burada:
- π: Piyasa dengesi getirileri (n×1 vektör)
- λ: Risk iştahı (risk_aversion = 2.5)
- Σ: Kovaryans matrisi (n×n)
- w_market: Piyasa ağırlıkları (n×1 vektör)
```

**Black-Litterman Formülü:**
```
μ_BL = [(τΣ)^(-1) + P^T Ω^(-1) P]^(-1) × [(τΣ)^(-1) π + P^T Ω^(-1) Q]

Burada:
- μ_BL: Black-Litterman beklenen getirileri
- τ: Görüş güveni (0.10)
- P: Görüş matrisi (n×n, identity)
- Ω: Görüş belirsizliği matrisi (n×n)
- Q: ML tahminleri (views)
```

**Görüş Belirsizliği:**
```
Ω = τ × P × Σ × P^T
```

### 2. Monte Carlo Portföy Optimizasyonu

**Amaç Fonksiyonu:**
```
Maximize: Sharpe Ratio = (w^T × μ) / sqrt(w^T × Σ × w)

Subject to:
- Σ w_i = 1  (Ağırlıklar toplamı 1)
- w_i ≥ 0    (Kısa satış yok)
```

**Algoritma:**
1. 20,000 rastgele portföy oluştur
2. Her portföy için:
   ```
   w = random_weights / sum(random_weights)  # Normalize
   μ_p = w^T × μ                              # Portföy getirisi
   σ_p = sqrt(w^T × Σ × w)                    # Portföy riski
   S = μ_p / σ_p                               # Sharpe oranı
   ```
3. En yüksek Sharpe portföyünü seç

### 3. GARCH(1,1) Modeli

**Model:**
```
σ²_t = ω + α × ε²_(t-1) + β × σ²_(t-1)

Burada:
- σ²_t: t zamanındaki varyans (risk)
- ε²_(t-1): t-1 zamanındaki şok (squared error)
- α, β: Model parametreleri (α + β < 1)
- ω: Sabit terim
```

**Tahmin:**
```
σ²_(t+1) = ω + α × ε²_t + β × σ²_t
```

### 4. LSTM Model Mimarisi

**Input Shape:**
- Single feature: `(batch_size, 60, 1)`
- Multi-feature: `(batch_size, 60, 5)`

**Model:**
```
Input(60, n_features)
  ↓
LSTM(128, return_sequences=True)
  ↓ Dropout(0.3)
LSTM(64, return_sequences=True)
  ↓ Dropout(0.3)
LSTM(32, return_sequences=False)
  ↓ Dropout(0.2)
Dense(16, activation='relu')
  ↓
Dense(1)  # Output: gelecek getiri
```

**Loss Function:**
```
MSE = mean((y_true - y_pred)²)
```

**Optimizer:**
```
Adam (learning_rate=0.001)
```

---

## 🎯 SUNUMDA KOD GÖSTERİMİ İÇİN ÖNERİLER

### 1. Black-Litterman Fonksiyonu Gösterimi

**Gösterilecek Kod:**
```python
def calculate_black_litterman(market_caps, cov_matrix, arima_views, risk_aversion=2.5, tau=0.05):
    # 1. Piyasa ağırlıkları
    total_mcap = market_caps.sum()
    market_weights = (market_caps / total_mcap).values
    
    # 2. Piyasa dengesi getirileri
    pi = risk_aversion * np.dot(cov_matrix, market_weights)
    
    # 3. Görüşler
    Q = arima_views.values.reshape(-1, 1)
    P = np.identity(len(market_caps))
    
    # 4. Görüş belirsizliği
    omega = np.dot(np.dot(P, tau * cov_matrix), P.T)
    
    # 5. Black-Litterman formülü
    tau_sigma = tau * cov_matrix
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(omega)
    
    left_term = np.linalg.inv(
        tau_sigma_inv + np.dot(np.dot(P.T, omega_inv), P)
    )
    
    right_term = (
        np.dot(tau_sigma_inv, pi) + 
        np.dot(np.dot(P.T, omega_inv), Q.flatten())
    )
    
    bl_returns = np.dot(left_term, right_term)
    return pd.Series(bl_returns, index=market_caps.index)
```

**Açıklama:**
- "Bu fonksiyon, piyasa dengesi ile ML tahminlerini birleştirir"
- "Piyasa ağırlıkları, hisselerin piyasa değerlerine göre hesaplanır"
- "Black-Litterman formülü, bu iki bilgiyi matematiksel olarak birleştirir"

### 2. Monte Carlo Optimizasyonu Gösterimi

**Gösterilecek Kod:**
```python
def run_monte_carlo_simulation(expected_returns, cov_matrix, tickers, graph_title):
    num_portfolios = 20000
    
    for _ in range(num_portfolios):
        # Rastgele ağırlıklar
        random_weights = np.random.random(num_assets)
        random_weights = random_weights / np.sum(random_weights)
        
        # Portföy getirisi
        port_return = np.dot(random_weights, expected_returns)
        
        # Portföy riski
        port_volatility = np.sqrt(
            np.dot(random_weights.T, np.dot(cov_matrix, random_weights))
        )
        
        # Sharpe oranı
        sharpe_ratio = port_return / port_volatility
        
        # En iyisini kaydet
        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_weights = random_weights
```

**Açıklama:**
- "20,000 rastgele portföy oluşturuyoruz"
- "Her portföy için getiri, risk ve Sharpe oranı hesaplanıyor"
- "En yüksek Sharpe oranına sahip portföy seçiliyor"

### 3. LSTM Model Mimarisi Gösterimi

**Gösterilecek Kod:**
```python
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

**Açıklama:**
- "3 katmanlı LSTM kullanıyoruz"
- "Her katmanda dropout kullanarak overfitting'i önlüyoruz"
- "Model, 60 günlük geçmiş veriyi kullanarak gelecek tahminleri yapıyor"

---

## 📝 SONUÇ

Bu doküman, bitirme projesi sunumu için gerekli tüm bilgileri içermektedir:

1. **Proje Özeti ve Mimarisi:** Sistemin genel yapısı
2. **Sunum Konuşma Metni:** Hocalara anlatılacak metin
3. **Kod Detayları:** Önemli algoritmaların açıklamaları

**Sunum sırasında:**
- Konuşma metnini takip edin
- Kod gösterimi yaparken yukarıdaki önerileri kullanın
- Sorular gelirse teknik detaylara girin
- Backtest sonuçlarını vurgulayın

**BAŞARILAR! 🎓✨**




