"""
Geçmiş halka arz kayıtlarında tavan gün sayısını fiyat verisiyle doğrular.

Yöntem: İlk işlem gününden itibaren ardışık iş günlerinde
Close ≈ High (günlük tavan) sayılır; seri kırılınca durulur.
"""
import json
import os
import sys
import time
from datetime import timedelta

import pandas as pd
import yfinance as yf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ipo_predictor import load_ipo_history

DATA_PATH = os.path.join(_ROOT, "data", "ipo_history.json")
TAVAN_GETIRI_ESIK = 0.095  # BIST %10 tavan bandına yakın kapanış
TAVAN_TOLERANS = 0.002  # Close, High'a %0.2 içinde


def _is_ceiling_day(close: float, prev_close: float, high: float) -> bool:
    if prev_close <= 0 or close <= 0:
        return False
    if (close / prev_close - 1.0) >= TAVAN_GETIRI_ESIK:
        return True
    if high > 0 and abs(high - close) / high <= TAVAN_TOLERANS:
        return True
    return False


def _count_ceiling_days(df: pd.DataFrame) -> int | None:
    """İlk işlem gününden başlayarak ardışık tavan günlerini say."""
    if df is None or df.empty:
        return None
    df = df.sort_index()
    count = 0
    prev_close = None
    for _, row in df.iterrows():
        close = float(row["Close"])
        high = float(row["High"])
        if close <= 0:
            break
        if prev_close is None:
            # İlk işlem günü: genelde tavan açılış/kapanış
            if high > 0 and abs(high - close) / high <= TAVAN_TOLERANS:
                count += 1
            prev_close = close
            continue
        if _is_ceiling_day(close, prev_close, high):
            count += 1
            prev_close = close
        else:
            break
    return count


def verify_ticker(ticker: str, ipo_date: str) -> dict:
    sym = ticker if ticker.endswith(".IS") else f"{ticker}.IS"
    start = pd.Timestamp(ipo_date) - timedelta(days=30)
    end = pd.Timestamp(ipo_date) + timedelta(days=120)

    df = pd.DataFrame()
    for attempt in range(3):
        try:
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                break
        except Exception:
            pass
        time.sleep(1.5)

    if df is None or df.empty:
        return {"ticker": ticker, "error": "fiyat verisi yok"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close"])
    computed = _count_ceiling_days(df)
    first_day = str(df.index[0].date()) if not df.empty else None
    return {
        "ticker": ticker,
        "ilk_islem": first_day,
        "hesaplanan_gun": computed,
        "error": None,
    }


def run_verify(update_json: bool = False) -> dict:
    history = load_ipo_history()
    rows = []
    updates = []

    for rec in history:
        start_date = rec.get("first_trade_date") or rec.get("ipo_date")
        v = verify_ticker(rec["ticker"], start_date)
        kayitli = int(rec["actual_ceiling_days"])
        hesap = v.get("hesaplanan_gun")
        if v.get("error"):
            status = "HATA"
        elif hesap is None:
            status = "?"
        elif hesap == kayitli:
            status = "OK"
        elif abs(hesap - kayitli) == 1:
            status = "YAKIN"
        else:
            status = "FARK"

        row = {
            "borsa_kodu": rec["ticker"],
            "kayitli": kayitli,
            "hesaplanan": hesap,
            "durum": status,
            "ilk_islem": v.get("ilk_islem"),
            "hata": v.get("error"),
        }
        rows.append(row)

        if update_json and status == "FARK" and hesap is not None and hesap >= 1:
            updates.append((rec["ticker"], kayitli, hesap))

        time.sleep(0.3)

    summary = {
        "toplam": len(rows),
        "ok": sum(1 for r in rows if r["durum"] == "OK"),
        "yakin": sum(1 for r in rows if r["durum"] == "YAKIN"),
        "fark": sum(1 for r in rows if r["durum"] == "FARK"),
        "hata": sum(1 for r in rows if r["durum"] == "HATA"),
        "detaylar": rows,
        "guncelleme_onerileri": [
            {"borsa_kodu": t, "eski": e, "yeni": n} for t, e, n in updates
        ],
    }
    return summary


def apply_updates(summary: dict) -> int:
    if not summary.get("guncelleme_onerileri"):
        return 0
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    kod_map = {u["borsa_kodu"]: u["yeni"] for u in summary["guncelleme_onerileri"]}
    changed = 0
    for rec in data.get("kayitlar", []):
        kod = rec.get("borsa_kodu")
        if kod in kod_map:
            rec["gercek_tavan_gunu"] = kod_map[kod]
            rec["tavan_dogrulama"] = "yfinance_auto"
            changed += 1
    if changed:
        data["son_guncelleme"] = pd.Timestamp.today().strftime("%Y-%m-%d")
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return changed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--apply", action="store_true", help="FARK kayıtlarını otomatik güncelle")
    args = parser.parse_args()

    summary = run_verify()
    if args.apply:
        n = apply_updates(summary)
        summary["uygulanan_guncelleme"] = n

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"\n=== Tavan Gün Doğrulama ({summary['toplam']} kayıt) ===")
    print(f"OK: {summary['ok']}  YAKIN(±1): {summary['yakin']}  FARK: {summary['fark']}  HATA: {summary['hata']}\n")
    for r in summary["detaylar"]:
        if r["durum"] == "OK":
            continue
        extra = f" ilk_islem={r['ilk_islem']}" if r.get("ilk_islem") else ""
        err = f" ({r['hata']})" if r.get("hata") else ""
        print(
            f"  {r['borsa_kodu']:8} {r['durum']:5} "
            f"kayitli={r['kayitli']} hesap={r['hesaplanan']}{extra}{err}"
        )
    if summary.get("guncelleme_onerileri"):
        print("\nOnerilen guncellemeler (--apply ile uygulanabilir):")
        for u in summary["guncelleme_onerileri"]:
            print(f"  {u['borsa_kodu']}: {u['eski']} -> {u['yeni']}")


if __name__ == "__main__":
    main()
