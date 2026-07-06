"""Portföy grafik ve simülasyon yardımcıları."""
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from portfolio.optimizer import run_monte_carlo_simulation

# --- YENİ: GELECEK SİMÜLASYON FONKSİYONU ---
def simulate_future_wealth(mu, sigma, initial_capital=100000, days=252, num_sims=1000):
    """
    Seçilen portföyün getiri (mu) ve riskine (sigma) göre 
    1 yıllık (252 gün) olası gelecek senaryolarını simüle eder.
    """
    dt = 1/252
    
    simulations = np.zeros((days, num_sims))
    simulations[0] = initial_capital
    
    # Hata önleme: mu veya sigma NaN ise durdur
    if np.isnan(mu) or np.isnan(sigma) or sigma <= 0:
        sigma = 0.0001
        mu = mu if not np.isnan(mu) else 0.05

    for t in range(1, days):
        Z = np.random.normal(0, 1, num_sims)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        simulations[t] = simulations[t-1] * np.exp(drift + diffusion)
        
    upper_bound = np.percentile(simulations, 95, axis=1)
    median_line = np.percentile(simulations, 50, axis=1)
    lower_bound = np.percentile(simulations, 5, axis=1)
    
    days_axis = list(range(days))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days_axis, y=upper_bound, mode='lines', line=dict(width=0)))
    fig.add_trace(go.Scatter(x=days_axis, y=lower_bound, mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(0,184,148,0.2)', name='Olası Aralık'))
    fig.add_trace(go.Scatter(x=days_axis, y=median_line, mode='lines',
                             line=dict(color='#6c5ce7', width=3), name='Beklenen Ortalama'))
    
    fig.update_layout(
        title="1 Yıllık Gelecek Simülasyonu (Eğitim Amaçlı - Yatırım Tavsiyesi Değildir)",
        xaxis_title="İşlem Günü",
        yaxis_title="Portföy Değeri (TL)",
        hovermode="x"
    )
    
    return fig.to_json(), {
        'upper': f"{upper_bound[-1]:,.2f}",
        'median': f"{median_line[-1]:,.2f}",
        'lower': f"{lower_bound[-1]:,.2f}"
    }

# --- 2. FONKSİYON: Klasik MPT (Kullanıcı Girdisiyle) ---

"""
    Klasik MPT hesaplama (kullanıcı girdisiyle)
    - Tarihsel verilerden ortalama getiri ve kovaryans hesaplar
    - Monte Carlo simülasyonu ile optimal portföy bulur
    """
def calculate_mpt(tickers, start_date='2018-01-01'):
    try:
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        
        if len(tickers) == 1:
            prices = prices.to_frame(name=tickers[0])

        prices = prices.dropna(axis=1, how='all') 
        prices = prices.ffill()
        prices = prices.dropna(axis=0, how='any')
        
        valid_tickers = list(prices.columns)
        if len(valid_tickers) == 0:
            return None, None, "Hata: Seçilen tarihler için hiçbir hisse verisi yok."
    except Exception as e:
        return None, None, f"Veri çekme hatası: {e}"
    
    daily_returns = prices.pct_change().dropna()

    # Kovaryans matrisi hesaplanırken COVID-19 etkisini temizlemek için sadece son 3 yılı kullan
    recent_returns = daily_returns.tail(3 * 252)
    expected_returns = daily_returns.mean() * 252
    cov_matrix = recent_returns.cov() * 252
    
    results_tuple = run_monte_carlo_simulation(expected_returns, cov_matrix, valid_tickers, "Klasik MPT")
    max_sharpe_port, min_vol_port, graph_json = results_tuple

    return max_sharpe_port.to_dict(), min_vol_port.to_dict(), graph_json


def generate_backtest_chart(weights_series, history_data_df):
    """
    Verilen portföy ağırlıkları ile son 1 yıllık geçmiş performansı BIST100 ile kıyaslar.
    """
    try:
        # Son 1 yıllık (252 işlem günü) veriyi al
        recent_data = history_data_df.tail(252)
        if len(recent_data) < 2:
            return None
        
        # Sadece ağırlığı olan hisseleri filtrele
        valid_tickers = [t for t in weights_series.index if t in recent_data.columns and weights_series[t] > 0.01]
        if not valid_tickers:
            return None
            
        # Günlük getiriler
        daily_returns = recent_data[valid_tickers].pct_change().dropna()
        weights = weights_series[valid_tickers].values
        weights = weights / np.sum(weights) # Normalize
        
        # Portföy günlük getirisi
        portfolio_returns = daily_returns.dot(weights)
        
        # Kümülatif getiriler (100 TL üzerinden)
        portfolio_cum = (1 + portfolio_returns).cumprod() * 100
        
        # Benchmark (BIST30 Eşit Ağırlıklı Proxy - BIST100'ü çekmek yavaşlatabilir)
        # BIST30 eşit ağırlıklı proxy olarak kullanalım:
        benchmark_returns = recent_data[recent_data.columns].pct_change().dropna().mean(axis=1)
        benchmark_cum = (1 + benchmark_returns).cumprod() * 100
        
        dates = portfolio_cum.index.strftime('%Y-%m-%d').tolist()
        
        trace_portfolio = go.Scatter(
            x=dates, y=portfolio_cum.values.tolist(),
            mode='lines', name='Yapay Zeka Portföyü',
            line=dict(color='#0ea5e9', width=3)
        )
        
        trace_benchmark = go.Scatter(
            x=dates, y=benchmark_cum.values.tolist(),
            mode='lines', name='BIST30 (Eşit Ağırlıklı)',
            line=dict(color='#ef4444', width=2, dash='dash')
        )
        
        layout = go.Layout(
            title="Geçmiş 1 Yıllık Performans Kıyaslaması (Backtest)",
            xaxis=dict(title='Tarih', showgrid=True, gridcolor='#f1f5f9'),
            yaxis=dict(title='Kümülatif Değer (100 TL Başlangıç)', showgrid=True, gridcolor='#f1f5f9'),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig = go.Figure(data=[trace_portfolio, trace_benchmark], layout=layout)
        return fig.to_json()
        
    except Exception as e:
        print(f"[!] Backtest chart error: {e}")
        return None


