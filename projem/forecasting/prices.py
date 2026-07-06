"""BIST fiyat ve piyasa değeri indirme."""
import time

import pandas as pd
import yfinance as yf

from config import BIST30_TICKERS

DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_DELAY = 12


def download_bist_prices(tickers=None, start_date=None, end_date=None, retries=DOWNLOAD_RETRIES):
    """
    BIST hisselerini tek tek, yeniden denemeli indirir.

    Returns:
        pd.DataFrame: Sütunlar ticker, indeks tarih
    """
    tickers = tickers or BIST30_TICKERS
    start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else start_date
    end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else end_date

    all_data = {}
    for ticker in tickers:
        success = False
        for attempt in range(1, retries + 1):
            try:
                df = yf.download(
                    ticker,
                    start=start_str,
                    end=end_str,
                    progress=False,
                    auto_adjust=False,
                    timeout=15,
                )
                if not df.empty and "Close" in df.columns:
                    series = df["Close"]
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]
                    all_data[ticker] = series
                    success = True
                    break
            except Exception:
                pass
            if not success and attempt < retries:
                time.sleep(DOWNLOAD_RETRY_DELAY if attempt > 1 else 2)

        if success:
            print(f"  [+] {ticker} verisi indirildi.")
        else:
            print(f"  [!] {ticker} verisi indirilemedi. Atlaniyor.")

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def get_market_caps(tickers=None):
    """Hisselerin piyasa değerlerini çeker."""
    tickers = tickers or BIST30_TICKERS
    print("[*] Piyasa Degerleri (Market Cap) indiriliyor...")
    mcaps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mcaps[t] = info.get("marketCap", 1e9)
        except Exception:
            mcaps[t] = 1e9
    return pd.Series(mcaps)
