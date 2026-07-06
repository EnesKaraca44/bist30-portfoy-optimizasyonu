"""
BIST 30 ML Portföy Backtesti (Black-Litterman + LSTM Ensemble)
"""
import yfinance as yf
import pandas as pd
import numpy as np
import pmdarima as pm
from arch import arch_model
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns

warnings.filterwarnings("ignore")

# Grafik stili
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# LSTM modülünü yükle (opsiyonel)
try:
    from lstm_forecast import forecast_lstm, ensemble_forecast
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False
    print("[!] LSTM modulu yuklenemedi, sadece ARIMA kullanilacak")

# Technical Indicators modülünü yükle
try:
    from technical_indicators import detect_market_regime, calculate_momentum_score
    TECHNICAL_AVAILABLE = True
except:
    TECHNICAL_AVAILABLE = False
    print("[!] Technical Indicators modulu yuklenemedi")

# Feature Engineering modülünü yükle
try:
    from feature_engineering import prepare_exogenous_features_for_arima, prepare_rolling_exogenous_for_arima
    FEATURES_AVAILABLE = True
except:
    FEATURES_AVAILABLE = False
    print("[!] Feature engineering modulu yuklenemedi, sadece basit ARIMA kullanilacak")

# === AYARLAR ===
USE_BLACK_LITTERMAN = True  # Black-Litterman AÇIK (batch_job.py gibi)
USE_LSTM = True             # LSTM AÇIK (Deep Learning + Ensemble)
                            # NOT: Multi-feature LSTM + ARIMA = +%9.66 alpha!
ENSEMBLE_ARIMA_W = 0.5      # ARIMA ağırlığı
ENSEMBLE_LSTM_W = 0.5       # LSTM ağırlığı
CLIP_MAX = 0.80             # Aşırı uç tahminler için sınır (daha agresif)
NUM_PORTFOLIOS = 100000     # Monte Carlo simülasyon sayısı (daha tutarlı sonuçlar)
STRATEGY = 'max_sharpe'     # Portföy stratejisi (en iyi Sharpe oranı)
MOMENTUM_WEIGHT = 0.6       # Momentum ağırlığı (base: %60, technical indicators ile dinamik ayarlanır)

from config import BIST30
from forecasting import (
    get_arima_forecast,
    get_garch_volatility as get_garch_risk,
    calculate_black_litterman,
    get_market_caps,
)

start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
PERIOD_NAME = "Son 365 Gün"

# === PORTFÖY OPTİMİZASYONU ===
def optimize_portfolio(expected_returns, cov_matrix, strategy='balanced'):
    """
    Monte Carlo simülasyonu ile portföy optimizasyonu
    """
    num_assets = len(expected_returns)
    
    best_sharpe = -np.inf
    best_return = -np.inf
    best_balanced_score = -np.inf
    
    best_weights_sharpe = None
    best_weights_return = None
    best_weights_balanced = None
    
    er_np = expected_returns.values if isinstance(expected_returns, pd.Series) else expected_returns
    cov_np = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        
        ret = np.dot(w, er_np)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_np, w)))
        
        if vol > 0:
            sharpe = ret / vol
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights_sharpe = w
            
            # Risk limiti: %50'ye kadar (daha dengeli)
            if ret > best_return and vol < 0.50:
                best_return = ret
                best_weights_return = w
            
            # Sharpe'a daha fazla ağırlık (daha tutarlı sonuçlar için)
            balanced_score = ret * 0.4 + sharpe * 0.6  # Sharpe'a daha fazla ağırlık: %40 getiri + %60 Sharpe
            if balanced_score > best_balanced_score and vol < 0.50:
                best_balanced_score = balanced_score
                best_weights_balanced = w
    
    # Fallback stratejisi: En iyi Sharpe kullan (en güvenilir)
    # Strateji seçimi: max_sharpe öncelikli (en güvenilir)
    if strategy == 'max_sharpe' and best_weights_sharpe is not None:
        return best_weights_sharpe
    elif strategy == 'max_return' and best_weights_return is not None:
        return best_weights_return
    elif strategy == 'balanced' and best_weights_balanced is not None:
        return best_weights_balanced
    elif best_weights_sharpe is not None:
        return best_weights_sharpe
    
    # Son çare: Eşit ağırlıklı
    return np.array([1.0/num_assets] * num_assets)

# === BACKTEST MOTORU ===
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
    print("\n" + "="*70)
    print(f"HIZLI BACKTEST: {PERIOD_NAME}".center(70))
    print(f"     ({start_date} -> {end_date})".center(70))
    print("="*70)
    print(f"[*] ARIMA: max_p=3, max_q=3 (batch_job.py ile aynı)")
    print(f"[*] LSTM: {USE_LSTM} (Artık optimize: Model kaydedilir/yüklenir, her tahminde eğitilmez)")
    print(f"[*] Technical Indicators: {TECHNICAL_AVAILABLE} (Market Regime Detection)")
    print(f"[*] Black-Litterman: {USE_BLACK_LITTERMAN} (tau=0.10)")
    print(f"[*] Monte Carlo: {NUM_PORTFOLIOS:,} portföy")
    print(f"[*] Clip Max: +/-{CLIP_MAX*100:.0f}%")
    print(f"[*] Rastgele Seed: Her calistirmada farkli sonuc")
    print("="*70)
    
    # Seed YOK - her çalıştırmada farklı Monte Carlo sonuçları
    
    # Veri çek
    print("[*] Veriler indiriliyor...")
    full_data = yf.download(BIST30, start="2023-01-01", end=end_date, progress=False)['Close']
    full_returns = full_data.pct_change().dropna()
    
    # BIST30 Endeksi'ni benchmark olarak indir
    print("[*] BIST30 Endeksi (XU030.IS) benchmark olarak indiriliyor...")
    try:
        bist30_index = yf.download('XU030.IS', start="2023-01-01", end=end_date, progress=False)['Close']
    except:
        print("[!] XU030.IS indirilemedi, esit agirlikli portfoy kullanilacak")
        bist30_index = None
    
    # Piyasa Değerlerini Çek (BL için)
    if USE_BLACK_LITTERMAN:
        market_caps = get_market_caps()
        market_caps = market_caps[BIST30]  # Sadece geçerli olanlar
    
    # Aylık dönemleri belirle
    try:
        monthly_dates = full_returns.loc[start_date:].resample('ME').last().index
    except:
        monthly_dates = full_returns.loc[start_date:].resample('M').last().index
    
    portfolio_value = 100.0
    benchmark_value = 100.0
    
    # Görselleştirme için veri toplama
    tracking_data = {
        'dates': [],
        'portfolio_values': [100.0],
        'benchmark_values': [100.0],
        'monthly_model_returns': [],
        'monthly_bench_returns': [],
        'monthly_labels': [],
        'regimes': []
    }
    
    print(f"[*] Simulasyon Basliyor ({len(monthly_dates)-1} Ay)...")
    
    # Backtest döngüsü
    for i in range(len(monthly_dates) - 1):
        current_date = monthly_dates[i]
        next_date = monthly_dates[i+1]
        
        past_data = full_returns.loc[:current_date]
        
        # ARIMA tahminleri
        arima_preds = {}
        lstm_preds = {}
        vol_preds = {}
        valid_tickers = []
        
        for ticker in BIST30:
            try:
                series = past_data[ticker].dropna()
                prices = full_data[ticker].loc[:current_date].dropna()  # Fiyat serisi (feature'lar için)
                if len(series) < 100:
                    continue
                
                # ARIMA tahmini (Feature Engineering ile)
                arima_mu = get_arima_forecast(series, prices=prices, use_features=True)
                arima_preds[ticker] = arima_mu
                
                # LSTM tahmini (Multi-feature ile)
                if USE_LSTM and LSTM_AVAILABLE:
                    lstm_mu = forecast_lstm(series, forecast_days=22, ticker=ticker, 
                                          prices=prices, use_features=True, force_retrain=False)
                    lstm_preds[ticker] = lstm_mu
                
                # Risk tahmini
                vol = get_garch_risk(series)
                vol_preds[ticker] = vol
                
                valid_tickers.append(ticker)
            except:
                continue
        
        if len(valid_tickers) < 5:
            continue
        
        # Black-Litterman uygulaması
        arima_series = pd.Series(arima_preds)
        
        if USE_BLACK_LITTERMAN:
            # Geçmiş verinin kovaryans matrisi (COVID-19 etkisini temizlemek için son 3 yılı kullan)
            recent_past = past_data[valid_tickers].tail(3 * 252)
            cov_matrix_hist = recent_past.cov() * 252
            
            # Black-Litterman dengelemesi (batch_job.py parametreleri)
            bl_returns_series = calculate_black_litterman(
                market_caps[valid_tickers], 
                cov_matrix_hist, 
                arima_series,
                risk_aversion=2.5,
                tau=0.10  # Daha yüksek tau = ARIMA görüşlerine daha fazla güven
            )
            mu_series = bl_returns_series
        else:
            mu_series = arima_series
        
        # LSTM Ensemble
        if USE_LSTM and LSTM_AVAILABLE and len(lstm_preds) > 0:
            lstm_series = pd.Series(lstm_preds)
            # Ensemble: Her hisse için ağırlıklı ortalama
            final_mu = {}
            for ticker in valid_tickers:
                if ticker in mu_series.index and ticker in lstm_series.index:
                    final_mu[ticker] = ensemble_forecast(
                        mu_series[ticker],
                        lstm_series[ticker],
                        arima_weight=ENSEMBLE_ARIMA_W,
                        lstm_weight=ENSEMBLE_LSTM_W
                    )
                else:
                    final_mu[ticker] = mu_series.get(ticker, 0)
            mu_series = pd.Series(final_mu)
        
        # Aşırı uçuk tahminleri filtrele
        mu_series = mu_series.clip(-CLIP_MAX, CLIP_MAX)
        
        vol_series = pd.Series(vol_preds)
        
        # Kovaryans (COVID-19 etkisini temizlemek için son 3 yılı kullan)
        recent_past = past_data[valid_tickers].tail(3 * 252)
        corr_matrix = recent_past.corr()
        D = np.diag(vol_series)
        cov_matrix_np = np.dot(D, np.dot(corr_matrix.values, D))
        
        # Technical Indicators ile Market Regime Detection
        strategy_mode = STRATEGY
        regime_label = ''
        
        if TECHNICAL_AVAILABLE:
            # BIST30 ortalama momentum ile piyasa rejimini belirle
            avg_returns = past_data[valid_tickers].mean(axis=1)
            avg_prices = full_data[valid_tickers].mean(axis=1).loc[:current_date]
            
            if len(avg_prices) >= 60:
                regime_info = detect_market_regime(avg_prices, avg_returns, lookback=60)
                market_regime = regime_info['regime']
                
                # Stratejiyi rejime göre ayarla
                if market_regime == 'bear':
                    # BEAR: Defansif - Sharpe'a öncelik
                    strategy_mode = 'max_sharpe'
                    regime_label = '[DEF]'
                elif market_regime == 'bull':
                    # BULL: Agresif - Getiriye öncelik
                    strategy_mode = 'max_return'
                    regime_label = '[AGG]'
                else:  # sideways
                    # SIDEWAYS: Dengeli
                    strategy_mode = 'balanced'
                    regime_label = '[BAL]'
        
        # Optimizasyon (rejime göre ayarlanmış)
        optimal_weights = optimize_portfolio(mu_series, cov_matrix_np, strategy=strategy_mode)
        
        # Gerçekleşen getiri
        next_month_data = full_data.loc[current_date:next_date]
        if len(next_month_data) < 2:
            continue
        
        period_return = (next_month_data.iloc[-1] / next_month_data.iloc[0]) - 1
        
        # Model getirisi
        my_weights_dict = dict(zip(valid_tickers, optimal_weights))
        my_return = sum(
            my_weights_dict[ticker] * period_return[ticker] 
            for ticker in valid_tickers if ticker in period_return.index
        )
        
        portfolio_value *= (1 + my_return)
        
        # Benchmark (BIST30 Endeksi veya eşit ağırlıklı)
        if bist30_index is not None:
            try:
                index_period_data = bist30_index.loc[current_date:next_date]
                if len(index_period_data) >= 2:
                    bench_return = float((index_period_data.iloc[-1] / index_period_data.iloc[0]) - 1)
                else:
                    # Endeks verisi yoksa eşit ağırlıklı kullan
                    equal_weights = np.array([1/len(BIST30)] * len(BIST30))
                    bench_return = float(np.dot(equal_weights, period_return[BIST30].fillna(0).values))
            except Exception as e:
                equal_weights = np.array([1/len(BIST30)] * len(BIST30))
                bench_return = float(np.dot(equal_weights, period_return[BIST30].fillna(0).values))
        else:
            equal_weights = np.array([1/len(BIST30)] * len(BIST30))
            bench_return = float(np.dot(equal_weights, period_return[BIST30].fillna(0).values))
        
        benchmark_value *= (1 + bench_return)
        
        # Veri toplama (görselleştirme için)
        tracking_data['dates'].append(next_date)
        tracking_data['portfolio_values'].append(portfolio_value)
        tracking_data['benchmark_values'].append(benchmark_value)
        tracking_data['monthly_model_returns'].append(my_return * 100)
        tracking_data['monthly_bench_returns'].append(bench_return * 100)
        tracking_data['monthly_labels'].append(current_date.strftime('%Y-%m'))
        tracking_data['regimes'].append(regime_label)
        
        # Rejim etiketi ile birlikte göster
        print(f"[{current_date.date()}] -> Model: %{my_return*100:>6.2f} | Piyasa: %{bench_return*100:>6.2f} {regime_label}")

    # === SONUÇ RAPORU ===
    total_model = ((portfolio_value / 100) - 1) * 100
    total_bench = ((benchmark_value / 100) - 1) * 100
    diff = total_model - total_bench
    
    print("\n" + "="*70)
    print("SONUC RAPORU".center(70))
    print("="*70)
    print(f"Başlangıç Bakiyesi : 100.00 TL")
    print(f"Model Bitiş        : {portfolio_value:.2f} TL (Toplam: %{total_model:.2f})")
    print(f"Piyasa Bitiş       : {benchmark_value:.2f} TL (Toplam: %{total_bench:.2f})")
    print("-" * 70)
    
    if diff > 0:
        print(f"[+] BASARILI! Model piyasayi %{diff:.2f} farkla yendi.")
    else:
        print(f"[!] Model piyasanin %{abs(diff):.2f} gerisinde kaldi.")
    print("="*70)
    
    # === GÖRSELLEŞTİRME ===
    if len(tracking_data['dates']) > 0:
        create_visualization(tracking_data, total_model, total_bench, diff)

def create_visualization(data, total_model, total_bench, alpha):
    """
    Backtest sonuçlarını görselleştirir
    """
    print("\n[*] Grafikler olusturuluyor...")
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('BIST 30 ML PORTFÖY BACKTEST SONUÇLARI', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # --- Grafik 1: Portföy Değeri Karşılaştırması (En Önemli!) ---
    ax1 = fig.add_subplot(gs[0, :])
    
    dates = data['dates']
    portfolio_vals = data['portfolio_values'][1:]  # İlk 100'ü atla
    benchmark_vals = data['benchmark_values'][1:]
    
    ax1.plot(dates, portfolio_vals, 'b-', linewidth=2.5, label='ML Model Portföyü', marker='o', markersize=5)
    ax1.plot(dates, benchmark_vals, 'r--', linewidth=2.5, label='BIST30 Benchmark', marker='s', markersize=5)
    ax1.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Başlangıç (100 TL)')
    
    # Alpha vurgulama
    if alpha > 0:
        ax1.fill_between(dates, benchmark_vals, portfolio_vals, 
                         where=(np.array(portfolio_vals) >= np.array(benchmark_vals)),
                         alpha=0.3, color='green', label=f'Alpha (+{alpha:.2f}%)')
    
    ax1.set_title('📈 Portföy Değeri Gelişimi (100 TL Başlangıç)', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Tarih', fontsize=11)
    ax1.set_ylabel('Portföy Değeri (TL)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # --- Grafik 2: Aylık Getiri Karşılaştırması ---
    ax2 = fig.add_subplot(gs[1, 0])
    
    x_pos = np.arange(len(data['monthly_labels']))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, data['monthly_model_returns'], width, 
                    label='ML Model', color='#2E86AB', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, data['monthly_bench_returns'], width,
                    label='Benchmark', color='#A23B72', alpha=0.8)
    
    # Sıfır çizgisi
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax2.set_title('📊 Aylık Getiri Karşılaştırması', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Ay', fontsize=10)
    ax2.set_ylabel('Getiri (%)', fontsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(data['monthly_labels'], rotation=45, ha='right', fontsize=8)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # --- Grafik 3: Kümülatif Getiri ---
    ax3 = fig.add_subplot(gs[1, 1])
    
    cumulative_model = np.cumsum(data['monthly_model_returns'])
    cumulative_bench = np.cumsum(data['monthly_bench_returns'])
    
    ax3.plot(data['monthly_labels'], cumulative_model, 'b-', linewidth=2, 
             marker='o', markersize=4, label='ML Model')
    ax3.plot(data['monthly_labels'], cumulative_bench, 'r--', linewidth=2,
             marker='s', markersize=4, label='Benchmark')
    ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    
    ax3.set_title('📈 Kümülatif Getiri', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Ay', fontsize=10)
    ax3.set_ylabel('Kümülatif Getiri (%)', fontsize=10)
    ax3.set_xticklabels(data['monthly_labels'], rotation=45, ha='right', fontsize=8)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # --- Grafik 4: Özet Metrikler ---
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Metrik hesaplama
    model_returns = np.array(data['monthly_model_returns'])
    bench_returns = np.array(data['monthly_bench_returns'])
    
    model_vol = np.std(model_returns)
    bench_vol = np.std(bench_returns)
    
    model_sharpe = (np.mean(model_returns) / model_vol) if model_vol > 0 else 0
    bench_sharpe = (np.mean(bench_returns) / bench_vol) if bench_vol > 0 else 0
    
    # Kazanma oranı
    win_rate = np.sum(model_returns > bench_returns) / len(model_returns) * 100
    
    # Max Drawdown hesaplama
    cumulative_model_vals = np.array(data['portfolio_values'])
    running_max = np.maximum.accumulate(cumulative_model_vals)
    drawdown = (cumulative_model_vals - running_max) / running_max * 100
    max_drawdown = np.min(drawdown)
    
    # Tablo oluştur
    metrics = [
        ['Metrik', 'ML Model', 'Benchmark', 'Fark'],
        ['Toplam Getiri', f'{total_model:.2f}%', f'{total_bench:.2f}%', f'{alpha:+.2f}%'],
        ['Ortalama Aylık Getiri', f'{np.mean(model_returns):.2f}%', f'{np.mean(bench_returns):.2f}%', 
         f'{np.mean(model_returns) - np.mean(bench_returns):+.2f}%'],
        ['Volatilite (Std)', f'{model_vol:.2f}%', f'{bench_vol:.2f}%', 
         f'{model_vol - bench_vol:+.2f}%'],
        ['Sharpe Ratio (Aylık)', f'{model_sharpe:.3f}', f'{bench_sharpe:.3f}', 
         f'{model_sharpe - bench_sharpe:+.3f}'],
        ['Kazanma Oranı', f'{win_rate:.1f}%', '-', '-'],
        ['Max Drawdown', f'{max_drawdown:.2f}%', '-', '-']
    ]
    
    # Tablo çiz
    table = ax4.table(cellText=metrics, cellLoc='center', loc='center',
                     bbox=[0.1, 0.0, 0.8, 1.0])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Header styling
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white')
    
    # Row colors
    for i in range(1, len(metrics)):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('white')
            
            # Fark sütunu renklendirme
            if j == 3 and i > 0:
                text = metrics[i][j]
                if '+' in text and i != 5:  # Volatilite hariç
                    cell.set_facecolor('#C8E6C9')  # Açık yeşil
                elif '-' in text and i not in [5, 6]:
                    cell.set_facecolor('#FFCDD2')  # Açık kırmızı
    
    ax4.set_title('📊 Performans Metrikleri Özeti', fontsize=12, fontweight='bold', 
                 pad=20, loc='left', x=0.1)
    
    # Kaydet ve göster
    plt.tight_layout()
    
    # Benzersiz dosya adı için timestamp oluştur
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # PDF olarak kaydet
    try:
        output_file = f'backtest_sonuc_{timestamp}.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[✓] Grafikler PDF olarak kaydedildi: {output_file}")
    except Exception as e:
        print(f"[!] PDF kaydedilemedi: {e}")
    
    # PNG olarak kaydet
    try:
        output_file_png = f'backtest_sonuc_{timestamp}.png'
        plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
        print(f"[✓] Grafikler PNG olarak kaydedildi: {output_file_png}")
    except Exception as e:
        print(f"[!] PNG kaydedilemedi: {e}")
    
    plt.show()
    print("[✓] Gorselleştirme tamamlandi!")
    print(f"\n[*] Tum grafikler 'backtest_sonuc_{timestamp}.*' olarak kaydedildi.")

if __name__ == "__main__":
    run_quick_backtest()
