"""Haftalık kullanıcı vs AI liderlik tablosu skorlaması."""
import json
import sqlite3
import warnings
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from config import DB_PATH as DB_NAME, BIST30_TICKERS


def compute_weighted_return(weights, actual_returns):
    """
    Ağırlıklı portföy getirisi hesapla.

    Args:
        weights: {ticker: weight} — ağırlıklar 0-1 arası (toplam ~1)
        actual_returns: pd.Series — ticker -> dönem getirisi (ondalık)

    Returns:
        float: Portföy getirisi
    """
    total = 0.0
    for ticker, weight in (weights or {}).items():
        key = str(ticker)
        if key in actual_returns.index:
            total += float(actual_returns[key]) * float(weight)
    return float(total)


def fetch_weekly_actual_returns(end_date=None, lookback_days=7):
    """
    BIST30 için haftalık gerçekleşen getirileri indirir.

    Returns:
        pd.Series veya None
    """
    end_date = end_date or datetime.today()
    start_date = end_date - timedelta(days=lookback_days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    try:
        data = yf.download(BIST30_TICKERS, start=start_str, end=end_str, progress=False)
        if data.empty:
            return None
        prices = data["Close"] if "Close" in data.columns else data
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        prices = prices.ffill().bfill()
        if prices.empty or len(prices) < 2:
            return None
        return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    except Exception:
        return None


def get_ai_benchmark_return(conn, actual_returns):
    """beklenen_getiriler tablosundan 12 aylik vade top-5 esit agirlikli AI getirisi."""
    c = conn.cursor()
    c.execute(
        """
        SELECT ticker FROM beklenen_getiriler
        WHERE horizon_ay = 12
        ORDER BY tahmini_yillik_getiri DESC
        LIMIT 5
        """
    )
    ai_picks = [row[0] for row in c.fetchall()]
    if not ai_picks:
        c.execute(
            "SELECT ticker FROM beklenen_getiriler ORDER BY tahmini_yillik_getiri DESC LIMIT 5"
        )
        ai_picks = [row[0] for row in c.fetchall()]
    if not ai_picks:
        return 0.0, []

    weight = 1.0 / len(ai_picks)
    ai_return = sum(
        float(actual_returns[ticker]) * weight
        for ticker in ai_picks
        if ticker in actual_returns.index
    )
    return float(ai_return), ai_picks


def run_leaderboard_scoring(end_date=None, conn=None, actual_returns=None):
    """
    Kullanıcı portföylerini AI benchmark ile karşılaştırır, leaderboard'a yazar.

    Returns:
        dict: success, users_scored, ai_return, week_start, week_end, message
    """
    print("[*] Kullanici vs AI Liderlik Tablosu Hesaplaniyor...")

    end_date = end_date or datetime.today()
    start_date = end_date - timedelta(days=7)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if actual_returns is None:
        print(f"[*] Fiyatlar cekiliyor ({start_str} - {end_str})...")
        actual_returns = fetch_weekly_actual_returns(end_date=end_date)

    if actual_returns is None or len(actual_returns) == 0:
        msg = "Borsa fiyat verisi alinamadi; liderlik tablosu atlandi."
        print(f"[-] {msg}")
        return {"success": False, "users_scored": 0, "ai_return": None, "message": msg}

    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(DB_NAME)

    try:
        ai_return, ai_picks = get_ai_benchmark_return(conn, actual_returns)
        print(f"[+] AI (Top-5 esit agirlik) Getirisi: %{ai_return * 100:.2f}")

        c = conn.cursor()
        c.execute("SELECT id, user_id, weights_json FROM user_portfolios")
        user_portfolios = c.fetchall()

        if not user_portfolios:
            msg = "Kayitli kullanici portfoyu yok; liderlik tablosu atlandi."
            print(f"[-] {msg}")
            return {
                "success": True,
                "users_scored": 0,
                "ai_return": ai_return,
                "week_start": start_str,
                "week_end": end_str,
                "message": msg,
            }

        c.execute("DELETE FROM leaderboard WHERE week_end_date = ?", (end_str,))

        scored = 0
        for portfolio_id, user_id, weights_json in user_portfolios:
            try:
                weights = json.loads(weights_json)
                user_return = compute_weighted_return(weights, actual_returns)
                is_winner = 1 if user_return > ai_return else 0
                c.execute(
                    """
                    INSERT INTO leaderboard
                    (user_id, week_start_date, week_end_date, user_return, ai_return, is_winner)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, start_str, end_str, user_return, ai_return, is_winner),
                )
                scored += 1
                status = "KAZANDI" if is_winner else "KAYBETTI"
                print(f"   [+] User {user_id}: %{user_return * 100:.2f} -> {status}")
            except Exception as e:
                print(f"   [-] User {user_id} hata: {e}")

        conn.commit()
        print(f"[+] Liderlik tablosu guncellendi ({scored} kullanici).")
        return {
            "success": True,
            "users_scored": scored,
            "ai_return": ai_return,
            "ai_picks": ai_picks,
            "week_start": start_str,
            "week_end": end_str,
            "message": f"{scored} kullanici skorlandi.",
        }
    finally:
        if own_conn:
            conn.close()


# Geriye dönük uyumluluk
calculate_weekly_scorer = run_leaderboard_scoring


if __name__ == "__main__":
    run_leaderboard_scoring()
