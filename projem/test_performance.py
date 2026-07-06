"""
Model Performansı Test Scripti
Haftalık tahmin ve performans takip sistemini test eder.
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from performance_tracker import (
    setup_performance_tables,
    backfill_actuals_for_ticker,
    get_performance_metrics,
    check_horizon_status
)

from config import DB_PATH as DB_NAME

def test_step_1_check_forecasts():
    """Adım 1: Tahminleri kontrol et"""
    print("\n" + "="*60)
    print("ADIM 1: TAHMİNLERİ KONTROL ET")
    print("="*60)
    
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("""
        SELECT ticker, tahmin_tarihi, tahmin_horizon, tahmini_getiri, 
               gerceklesen_getiri, model_tipi
        FROM tahminler 
        ORDER BY tahmin_tarihi DESC 
        LIMIT 30
    """, conn)
    conn.close()
    
    if df.empty:
        print("[!] Henüz tahmin kaydı yok. Önce weekly_forecast.py çalıştırın!")
        return False
    
    print(f"\n[+] Toplam {len(df)} tahmin kaydı bulundu.")
    print(f"\nSon 5 tahmin:")
    print(df.head().to_string(index=False))
    
    # Gerçekleşen getiri olmayan tahminler
    pending = df[df['gerceklesen_getiri'].isna()]
    print(f"\n[!] {len(pending)} tahmin henüz doğrulanmamış (horizon dolmamış olabilir)")
    
    return True

def test_step_2_check_horizon():
    """Adım 2: Horizon durumunu kontrol et"""
    print("\n" + "="*60)
    print("ADIM 2: HORIZON DURUMU")
    print("="*60)
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT ticker, tahmin_tarihi, tahmin_horizon 
        FROM tahminler 
        WHERE gerceklesen_getiri IS NULL
        ORDER BY tahmin_tarihi DESC
        LIMIT 5
    """)
    pending = c.fetchall()
    conn.close()
    
    if not pending:
        print("[+] Tüm tahminler doğrulanmış veya henüz tahmin yok.")
        return
    
    for ticker, forecast_date, horizon in pending:
        status = check_horizon_status(ticker, forecast_date, horizon or 5)
        print(f"\n{ticker} ({forecast_date}):")
        print(f"  - {status['message']}")
        print(f"  - Ready: {status['ready']}")

def test_step_3_backfill(ticker=None):
    """Adım 3: Gerçekleşen getirileri güncelle (backfill)"""
    print("\n" + "="*60)
    print("ADIM 3: PERFORMANS GÜNCELLEME (BACKFILL)")
    print("="*60)
    
    if ticker:
        print(f"\n[+] {ticker} için backfill yapılıyor...")
        count = backfill_actuals_for_ticker(ticker, horizon_days=5, lookback_days=30)
        print(f"[+] {count} tahmin güncellendi.")
    else:
        print("\n[+] Tüm hisseler için backfill yapılıyor...")
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT DISTINCT ticker FROM tahminler WHERE gerceklesen_getiri IS NULL")
        tickers = [row[0] for row in c.fetchall()]
        conn.close()
        
        total = 0
        for t in tickers:
            count = backfill_actuals_for_ticker(t, horizon_days=5, lookback_days=30)
            if count > 0:
                print(f"  [+] {t}: {count} tahmin güncellendi")
                total += count
        
        print(f"\n[+] Toplam {total} tahmin güncellendi ({len(tickers)} hisse)")

def test_step_4_view_metrics(ticker=None):
    """Adım 4: Performans metriklerini görüntüle"""
    print("\n" + "="*60)
    print("ADIM 4: PERFORMANS METRİKLERİ")
    print("="*60)
    
    if ticker:
        print(f"\n[+] {ticker} için metrikler:")
        metrics = get_performance_metrics(ticker)
        print_metrics(metrics)
    else:
        print("\n[+] Tüm hisseler için metrikler:")
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT DISTINCT ticker FROM tahminler")
        tickers = [row[0] for row in c.fetchall()]
        conn.close()
        
        for t in tickers[:10]:  # İlk 10 hisse
            metrics = get_performance_metrics(t)
            if metrics['valid_forecasts'] > 0:
                print(f"\n{t}:")
                print_metrics(metrics, indent="  ")

def print_metrics(metrics, indent=""):
    """Metrikleri yazdır"""
    if metrics['valid_forecasts'] == 0:
        print(f"{indent}[!] Henüz geçerli tahmin yok (horizon dolmamış)")
        return
    
    print(f"{indent}MAE (Mean Absolute Error): {metrics['mae']:.4f}")
    print(f"{indent}RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
    print(f"{indent}Direction Accuracy: {metrics['accuracy']:.2%}")
    print(f"{indent}Geçerli Tahmin: {metrics['valid_forecasts']}/{metrics['total_forecasts']}")

def main():
    """Ana test akışı"""
    print("\n" + "="*60)
    print("MODEL PERFORMANSI TEST SCRIPTI")
    print("="*60)
    
    # Tabloları oluştur
    setup_performance_tables()
    
    # Adım 1: Tahminleri kontrol et
    has_forecasts = test_step_1_check_forecasts()
    
    if not has_forecasts:
        print("\n[!] Önce weekly_forecast.py çalıştırarak tahminler oluşturun!")
        return
    
    # Adım 2: Horizon durumu
    test_step_2_check_horizon()
    
    # Adım 3: Backfill (kullanıcı isterse)
    print("\n[?] Backfill yapmak istiyor musunuz? (y/n): ", end="")
    # Otomatik: Tüm hisseler için yap
    test_step_3_backfill()
    
    # Adım 4: Metrikleri görüntüle
    print("\n[?] Hangi hisse için metrik görmek istersiniz? (boş = tümü): ", end="")
    # Örnek: AKBNK.IS veya boş
    test_step_4_view_metrics()

def test_weekly_performance(start_date='2025-12-01', end_date='2025-12-05'):
    """
    Belirli bir hafta için model vs piyasa performansını karşılaştır
    
    Args:
        start_date: Başlangıç tarihi (YYYY-MM-DD)
        end_date: Bitiş tarihi (YYYY-MM-DD)
    """
    import yfinance as yf
    
    print("\n" + "="*80)
    print(f"HAFTALIK PERFORMANS RAPORU: {start_date} - {end_date}")
    print("="*80)
    
    # 1. Tahminleri kontrol et
    conn = sqlite3.connect(DB_NAME)
    query = """
        SELECT tahmin_tarihi, ticker, tahmini_getiri, gerceklesen_getiri, 
               model_tipi, tahmin_horizon
        FROM tahminler 
        WHERE DATE(tahmin_tarihi) BETWEEN ? AND ?
        ORDER BY tahmin_tarihi, ticker
    """
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    conn.close()
    
    if df.empty:
        print(f"\n[!] {start_date} - {end_date} icin tahmin bulunamadi!")
        print("[!] Bu tarihte weekly_forecast.py calistirilmamis olabilir.")
        return
    
    print(f"\n[*] Toplam {len(df)} tahmin bulundu.")
    
    # 2. Gerçekleşen getirileri kontrol et
    filled_count = df['gerceklesen_getiri'].notna().sum()
    missing_count = df['gerceklesen_getiri'].isna().sum()
    
    print(f"[*] Gerceklesen getiri doldurulmus: {filled_count}")
    print(f"[*] Gerceklesen getiri eksik: {missing_count}")
    
    # 3. Eksik olanları doldur (backfill)
    if missing_count > 0:
        print("\n[*] Eksik gerceklesen getirileri dolduruluyor...\n")
        for ticker in df['ticker'].unique():
            updated = backfill_actuals_for_ticker(ticker, horizon_days=5, lookback_days=30)
            if updated > 0:
                print(f"  [+] {ticker}: {updated} tahmin guncellendi")
    
    # 4. Güncellenmiş verileri tekrar çek
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    conn.close()
    
    # 5. Geçerli tahminleri filtrele
    valid_df = df[df['gerceklesen_getiri'].notna() & df['tahmini_getiri'].notna()].copy()
    
    if valid_df.empty:
        print(f"\n[!] Gerceklesen getiri bulunamadi. Horizon henuz dolmamis olabilir.")
        horizon_end = pd.to_datetime(end_date) + timedelta(days=5)
        print(f"[!] {start_date}-{end_date} tahminleri icin en az {horizon_end.strftime('%Y-%m-%d')} tarihine kadar bekleyin.")
        return
    
    print(f"\n[+] {len(valid_df)} tahmin icin performans hesaplanabilir.\n")
    
    # 6. Model performansı hesapla
    valid_df['tahmin_hatasi'] = abs(valid_df['gerceklesen_getiri'] - valid_df['tahmini_getiri'])
    valid_df['yon_dogru'] = (
        ((valid_df['tahmini_getiri'] > 0) & (valid_df['gerceklesen_getiri'] > 0)) |
        ((valid_df['tahmini_getiri'] < 0) & (valid_df['gerceklesen_getiri'] < 0))
    )
    
    model_avg_predicted = valid_df['tahmini_getiri'].mean()
    model_avg_actual = valid_df['gerceklesen_getiri'].mean()
    
    # 7. BIST30 benchmark (haftalık getiri)
    try:
        # Hafta başı ve sonu için biraz önceki tarihleri de dahil et
        bench_start = (pd.to_datetime(start_date) - timedelta(days=3)).strftime('%Y-%m-%d')
        bench_end = (pd.to_datetime(end_date) + timedelta(days=3)).strftime('%Y-%m-%d')
        
        xu030 = yf.download('XU030.IS', start=bench_start, end=bench_end, progress=False)['Close']
        
        # Start_date ve end_date'e en yakın değerleri bul
        xu030_df = xu030.to_frame()
        xu030_df.index = pd.to_datetime(xu030_df.index).date
        
        # Başlangıç değeri
        start_dt = pd.to_datetime(start_date).date()
        if start_dt in xu030_df.index:
            start_val = xu030_df.loc[start_dt].values[0]
        else:
            # En yakın önceki tarihi bul
            earlier = xu030_df[xu030_df.index <= start_dt]
            start_val = earlier.iloc[-1].values[0] if len(earlier) > 0 else None
        
        # Bitiş değeri
        end_dt = pd.to_datetime(end_date).date()
        if end_dt in xu030_df.index:
            end_val = xu030_df.loc[end_dt].values[0]
        else:
            # En yakın sonraki tarihi bul
            later = xu030_df[xu030_df.index >= end_dt]
            end_val = later.iloc[0].values[0] if len(later) > 0 else None
        
        if start_val and end_val:
            benchmark_return = (end_val / start_val) - 1
        else:
            benchmark_return = None
    except Exception as e:
        print(f"[!] BIST30 verisi alinamadi: {e}")
        benchmark_return = None
    
    # 8. Sonuçları yazdır
    print("=" * 80)
    print("SONUCLAR")
    print("=" * 80)
    
    print(f"\n MODEL PERFORMANSI:")
    print(f"  Tahmini Ortalama Getiri: {model_avg_predicted*100:+.2f}%")
    print(f"  Gerceklesen Ortalama Getiri: {model_avg_actual*100:+.2f}%")
    print(f"  Ortalama Hata (MAE): {valid_df['tahmin_hatasi'].mean()*100:.2f}%")
    print(f"  Yon Dogrulugu: {valid_df['yon_dogru'].mean()*100:.1f}%")
    
    if benchmark_return is not None:
        print(f"\n BENCHMARK (BIST30):")
        print(f"  BIST30 Haftalik Getiri: {benchmark_return*100:+.2f}%")
        
        alpha = (model_avg_actual - benchmark_return) * 100
        print(f"\n ALPHA (Model - Piyasa):")
        if alpha > 0:
            print(f"  +{alpha:.2f}% - Model piyasayi GECTI! ")
        elif alpha < 0:
            print(f"  {alpha:.2f}% - Model piyasanin altinda kaldi ")
        else:
            print(f"  {alpha:.2f}% - Model piyasa ile ayni ")
    
    # En iyi ve en kötü tahminler
    print(f"\n EN IYI 5 TAHMIN:")
    best = valid_df.nlargest(5, 'gerceklesen_getiri')[['ticker', 'tahmini_getiri', 'gerceklesen_getiri']]
    for idx, row in best.iterrows():
        print(f"  {row['ticker']:12} Tahmin: {row['tahmini_getiri']*100:+6.2f}% | Gercek: {row['gerceklesen_getiri']*100:+6.2f}%")
    
    print(f"\n EN KOTU 5 TAHMIN:")
    worst = valid_df.nsmallest(5, 'gerceklesen_getiri')[['ticker', 'tahmini_getiri', 'gerceklesen_getiri']]
    for idx, row in worst.iterrows():
        print(f"  {row['ticker']:12} Tahmin: {row['tahmini_getiri']*100:+6.2f}% | Gercek: {row['gerceklesen_getiri']*100:+6.2f}%")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    # 9-10 Aralık arası performans testi (horizon dolmuş!)
    test_weekly_performance('2025-12-09', '2025-12-10')
