"""
Makro backtest ve tahmin karsilastirma raporu.

- Iki tahmin tarihini karsilastirir (makro oncesi/sonrasi)
- Hisse gruplarina gore (ihracatci, banka, vb.) ozet uretir
- Gerceklesen tahminler icin yon dogrulugu analizi
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime

TICKER_CATEGORIES = {
    "ihracatci": {
        "FROTO.IS", "TOASO.IS", "EREGL.IS", "KRDMD.IS", "SISE.IS", "SASA.IS",
        "PETKM.IS", "GUBRF.IS", "ENKAI.IS", "ASELS.IS", "ASTOR.IS",
    },
    "banka": {"AKBNK.IS", "GARAN.IS", "ISCTR.IS", "YKBNK.IS", "DSTKF.IS", "EKGYO.IS"},
    "havacilik_petrol": {"THYAO.IS", "PGSUS.IS", "TAVHL.IS", "TUPRS.IS"},
    "defansif": {"BIMAS.IS", "MGROS.IS", "ULKER.IS", "TCELL.IS", "TTKOM.IS", "AEFES.IS"},
}


def get_ticker_category(ticker: str) -> str:
    for cat, tickers in TICKER_CATEGORIES.items():
        if ticker in tickers:
            return cat
    return "diger"


def load_forecasts_by_date(conn, date_str: str) -> dict[str, float]:
    c = conn.cursor()
    c.execute(
        """
        SELECT ticker, tahmini_getiri
        FROM tahminler
        WHERE DATE(tahmin_tarihi) = ?
        """,
        (date_str,),
    )
    return {row[0]: float(row[1]) for row in c.fetchall()}


def get_forecast_dates(conn, limit: int = 10) -> list[str]:
    c = conn.cursor()
    c.execute(
        """
        SELECT DISTINCT DATE(tahmin_tarihi) AS d
        FROM tahminler
        ORDER BY d DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [row[0] for row in c.fetchall()]


def compare_forecast_snapshots(old: dict[str, float], new: dict[str, float]) -> dict:
    """Iki tahmin anini karsilastirir."""
    rows = []
    by_category: dict[str, list[float]] = defaultdict(list)

    for ticker in sorted(set(old) & set(new)):
        o, n = old[ticker], new[ticker]
        diff = n - o
        flipped = (o >= 0) != (n >= 0)
        cat = get_ticker_category(ticker)
        rows.append({
            "ticker": ticker,
            "category": cat,
            "old_pct": round(o * 100, 2),
            "new_pct": round(n * 100, 2),
            "diff_pp": round(diff * 100, 2),
            "direction_flipped": flipped,
        })
        by_category[cat].append(abs(diff))

    abs_diffs = [abs(r["diff_pp"]) for r in rows]
    flips = [r for r in rows if r["direction_flipped"]]

    category_summary = {}
    for cat, diffs in by_category.items():
        category_summary[cat] = {
            "count": len(diffs),
            "avg_abs_diff_pp": round(sum(diffs) * 100 / len(diffs), 2) if diffs else 0.0,
        }

    return {
        "compared_tickers": len(rows),
        "avg_abs_diff_pp": round(sum(abs_diffs) / len(abs_diffs), 2) if abs_diffs else 0.0,
        "max_diff": max(rows, key=lambda r: abs(r["diff_pp"])) if rows else None,
        "direction_flips": len(flips),
        "flip_tickers": [r["ticker"] for r in flips],
        "by_category": category_summary,
        "rows": rows,
    }


def analyze_direction_accuracy(conn, limit: int = 100) -> dict:
    """Gerceklesen tahminlerde yon dogrulugu."""
    c = conn.cursor()
    c.execute(
        """
        SELECT ticker, tahmini_getiri, gerceklesen_getiri
        FROM tahminler
        WHERE gerceklesen_getiri IS NOT NULL
        ORDER BY tahmin_tarihi DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = c.fetchall()
    if not rows:
        return {"total": 0, "correct": 0, "accuracy_pct": None, "by_category": {}}

    correct = 0
    by_cat: dict[str, list[bool]] = defaultdict(list)

    for ticker, pred, actual in rows:
        pred_f, act_f = float(pred), float(actual)
        hit = (pred_f >= 0) == (act_f >= 0)
        correct += int(hit)
        by_cat[get_ticker_category(ticker)].append(hit)

    cat_acc = {
        cat: round(sum(hits) / len(hits) * 100, 1)
        for cat, hits in by_cat.items()
        if hits
    }

    return {
        "total": len(rows),
        "correct": correct,
        "accuracy_pct": round(correct / len(rows) * 100, 1),
        "by_category": cat_acc,
    }


def build_macro_backtest_report(conn, date_old: str | None = None, date_new: str | None = None) -> dict:
    """
    Tam rapor: son iki tahmin tarihini karsilastir + yon dogrulugu.
    """
    dates = get_forecast_dates(conn, limit=5)
    if len(dates) < 2 and (date_old is None or date_new is None):
        return {
            "success": False,
            "message": "Karsilastirma icin en az 2 farkli tahmin tarihi gerekli.",
            "available_dates": dates,
        }

    date_new = date_new or dates[0]
    date_old = date_old or dates[1]

    old = load_forecasts_by_date(conn, date_old)
    new = load_forecasts_by_date(conn, date_new)
    comparison = compare_forecast_snapshots(old, new)
    accuracy = analyze_direction_accuracy(conn)

    return {
        "success": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date_old": date_old,
        "date_new": date_new,
        "comparison": comparison,
        "direction_accuracy": accuracy,
    }


def format_report_text(report: dict) -> str:
    """Raporu konsol ciktisi icin metne cevir."""
    if not report.get("success"):
        return f"[-] {report.get('message', 'Rapor olusturulamadi')}"

    lines = [
        "=" * 72,
        "MAKRO BACKTEST / TAHMIN KARSILASTIRMA RAPORU",
        "=" * 72,
        f"Tarih eski : {report['date_old']}",
        f"Tarih yeni : {report['date_new']}",
        "",
    ]

    comp = report["comparison"]
    lines.append(f"Karsilastirilan hisse: {comp['compared_tickers']}")
    lines.append(f"Ortalama mutlak fark: {comp['avg_abs_diff_pp']} puan")
    lines.append(f"Yon degisen: {comp['direction_flips']} hisse")
    if comp.get("max_diff"):
        m = comp["max_diff"]
        lines.append(f"En buyuk fark: {m['ticker']} ({m['diff_pp']:+.2f} puan)")

    lines.append("")
    lines.append("--- Kategori bazli ortalama fark ---")
    for cat, info in sorted(comp.get("by_category", {}).items()):
        lines.append(f"  {cat:18s}: {info['avg_abs_diff_pp']:.2f} puan ({info['count']} hisse)")

    acc = report.get("direction_accuracy", {})
    if acc.get("total", 0) > 0:
        lines.append("")
        lines.append("--- Gerceklesen tahminler yon dogrulugu ---")
        lines.append(f"  Genel: %{acc['accuracy_pct']} ({acc['correct']}/{acc['total']})")
        for cat, pct in sorted(acc.get("by_category", {}).items()):
            lines.append(f"  {cat:18s}: %{pct}")

    lines.append("")
    lines.append("--- Hisse detay (ilk 10) ---")
    for row in comp.get("rows", [])[:10]:
        flip = " [YON DEGISTI]" if row["direction_flipped"] else ""
        lines.append(
            f"  {row['ticker']:12s} {row['old_pct']:+6.2f}% -> {row['new_pct']:+6.2f}% "
            f"({row['diff_pp']:+5.2f}p){flip}"
        )

    return "\n".join(lines)
