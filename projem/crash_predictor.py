import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_crash_probability():
    """
    BIST100 endeksi için çöküş/düzeltme riskini hesaplar.
    Dönüş: (risk_yuzdesi, risk_seviyesi, mesaj)
    """
    try:
        # yfinance period='6mo' kullanımı start/end'den daha stabil çalışır
        bist_data = yf.download('XU100.IS', period='6mo', progress=False)
        
        # Eğer yfinance BIST100'ü çekemezse (hata/boş dönerse), en hacimli hisseyi endeks proxy'si olarak kullan
        if bist_data.empty or len(bist_data) < 20:
            bist_data = yf.download('THYAO.IS', period='6mo', progress=False)
            
        if bist_data.empty or len(bist_data) < 20:
            bist_data = yf.download('ISCTR.IS', period='6mo', progress=False)
            
        if 'Close' not in bist_data or bist_data.empty or len(bist_data) < 20:
            return 45, "ORTA RİSK 🟡", "Önümüzdeki 15 gün için BIST100'de sert bir çöküş (%5 ve üzeri düşüş) riski nötr (%45) seviyesinde. (Not: Anlık veri akışı sağlanamadığı için ortalama risk gösterilmektedir.)"
            
        close_prices = bist_data['Close']
        if isinstance(close_prices, pd.DataFrame):
            # Eğer yfinance MultiIndex dönerse, ilk sütunu (örn: XU100.IS) al
            close_prices = close_prices.iloc[:, 0]
        
        # NaN değerleri temizle
        close_prices = close_prices.dropna()
        
        if len(close_prices) < 20:
            return 0, "Bilinmiyor", "Veri yetersiz."
            
        current_price = close_prices.iloc[-1]
        
        # 1. RSI Hesaplama (Aşırı alım = Riskli)
        rsi = calculate_rsi(close_prices, 14).iloc[-1]
        rsi_risk = 0
        if rsi > 75:
            rsi_risk = 40
        elif rsi > 65:
            rsi_risk = 20
        elif rsi < 35: # Aşırı satım, düşüş riski çok azaldı (tepki beklenir)
            rsi_risk = -20
            
        # 2. Hareketli Ortalamadan Uzaklık (20 günlük SMA)
        sma20 = close_prices.rolling(window=20).mean().iloc[-1]
        dist_from_sma = (current_price - sma20) / sma20 * 100
        sma_risk = 0
        if dist_from_sma > 8: # Ortalamadan çok uzaklaştı, düzeltme gelebilir
            sma_risk = 30
        elif dist_from_sma < -5: # Zaten çökmüş, daha fazla düşme riski azalır
            sma_risk = -15
            
        # 3. Volatilite (Son 10 günlük)
        returns = close_prices.pct_change().dropna()
        volatility = returns.tail(10).std() * np.sqrt(252) * 100
        vol_risk = 0
        if volatility > 35: # Aşırı dalgalı, riskli
            vol_risk = 30
        elif volatility > 25:
            vol_risk = 15
            
        # Toplam Riski Hesapla
        total_risk = 20 + rsi_risk + sma_risk + vol_risk # Baz risk %20
        total_risk = max(5, min(95, total_risk)) # %5 ile %95 arası sınırla
        
        # Durum belirleme
        if total_risk >= 70:
            level = "YÜKSEK RİSK 🔴"
            msg = "Önümüzdeki 15 gün içinde BIST100 endeksinde %5'ten büyük ani bir çöküş (düzeltme) yaşanma ihtimali çok yüksek. Hisselerinizde ciddi değer kayıpları olabilir, nakit oranınızı artırıp stop-loss kullanmanız tavsiye edilir."
        elif total_risk >= 45:
            level = "ORTA RİSK 🟡"
            msg = "Önümüzdeki 15 gün içinde BIST100 endeksinde hafif çaplı geri çekilmeler (dalgalanma) görülebilir. Portföyünüzü korumak için yeni alımlarda acele etmeyin."
        else:
            level = "DÜŞÜK RİSK 🟢"
            msg = "Önümüzdeki 15 gün içinde BIST100 endeksinde sert bir çöküş beklenmiyor. Piyasa trendi güçlü, mevcut hisse pozisyonlarınızı koruyabilirsiniz."
            
        return int(total_risk), level, msg
        
    except Exception as e:
        print(f"Crash predictor error: {e}")
        return 50, "BİLİNMİYOR", "Risk hesaplanamadı."

if __name__ == "__main__":
    prob, level, msg = get_crash_probability()
    print(f"Çöküş İhtimali: %{prob}")
    print(f"Seviye: {level}")
    print(f"Uyarı: {msg}")
