"""Makro oncesi vs sonrasi tahmin karsilastirmasi."""
import sqlite3

import _bootstrap  # noqa: F401
from config import DB_PATH

OLD_DATE = "2026-07-03"
NEW_DATE = "2026-07-04"


def get_forecasts(conn, date_str):
    c = conn.cursor()
    c.execute(
        """
        SELECT ticker, tahmini_getiri, model_tipi
        FROM tahminler
        WHERE DATE(tahmin_tarihi) = ?
        ORDER BY ticker
        """,
        (date_str,),
    )
    return {r[0]: {"ret": float(r[1]), "model": r[2]} for r in c.fetchall()}


def main():
    conn = sqlite3.connect(DB_PATH)
    old = get_forecasts(conn, OLD_DATE)
    new = get_forecasts(conn, NEW_DATE)

    print("=" * 72)
    print(f"KARSILASTIRMA: Makro ONCESI ({OLD_DATE}) vs Makro SONRASI ({NEW_DATE})")
    print("=" * 72)
    print(f"{'Hisse':12s} {'Eski':>8s} {'Yeni':>8s} {'Fark':>8s}  Yon")
    print("-" * 72)

    diffs = []
    for t in sorted(old.keys()):
        if t not in new:
            continue
        o, n = old[t]["ret"], new[t]["ret"]
        d = n - o
        diffs.append((t, o, n, d))
        old_dir = "+" if o >= 0 else "-"
        new_dir = "+" if n >= 0 else "-"
        flip = " *** YON DEGISTI" if (o >= 0) != (n >= 0) else ""
        print(f"{t:12s} {o*100:+7.2f}% {n*100:+7.2f}% {d*100:+7.2f}p  {old_dir}->{new_dir}{flip}")

    print("-" * 72)
    abs_diffs = [abs(d[3]) for d in diffs]
    print(f"Hisse sayisi: {len(diffs)}")
    print(f"Ortalama mutlak fark: {sum(abs_diffs)/len(abs_diffs)*100:.2f} puan")
    biggest = max(diffs, key=lambda x: abs(x[3]))
    print(f"En buyuk fark: {biggest[0]} ({abs(biggest[3])*100:.2f} puan)")
    flips = [d for d in diffs if (d[1] >= 0) != (d[2] >= 0)]
    print(f"Yon degisen hisse: {len(flips)}")
    for t, o, n, d in sorted(flips, key=lambda x: abs(x[3]), reverse=True):
        print(f"  {t}: {o*100:+.2f}% -> {n*100:+.2f}%")

    c = conn.cursor()
    c.execute(
        """
        SELECT DATE(tahmin_tarihi), tahmin_edilen_getiri
        FROM portfoy_performans
        ORDER BY tahmin_tarihi DESC
        LIMIT 5
        """
    )
    ports = c.fetchall()
    print()
    print("=== PORTFOY TAHMINLERI (son kayitlar) ===")
    for date, ret in ports:
        label = "YENI" if date == NEW_DATE else "ESKI" if date == OLD_DATE else date
        print(f"  {label}: %{float(ret)*100:.2f}")

    conn.close()


if __name__ == "__main__":
    main()
