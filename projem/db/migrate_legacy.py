"""Eski projem/mpt_veritabani.db verilerini ana DB'ye tasir."""
import os
import sqlite3

from config import BASE_DIR, DB_PATH

LEGACY_DB = os.path.join(BASE_DIR, "mpt_veritabani.db")

TABLES_INSERT_OR_IGNORE = {
    "portfoy_performans": (
        "tahmin_tarihi, tahmin_horizon, portfoy_tahmini_getiri, portfoy_gerceklesen_getiri, "
        "portfoy_hatasi, portfoy_agirliklari, benchmark_getiri, alpha",
        "tahmin_tarihi, tahmin_horizon",
    ),
    "tahminler": (
        "ticker, tahmin_tarihi, tahmin_horizon, tahmini_getiri, gerceklesen_getiri, "
        "tahmin_hatasi, model_tipi",
        "ticker, tahmin_tarihi, tahmin_horizon",
    ),
    "duygu_analizi": (
        "ticker, duygu_skoru, algis_durumu, son_guncelleme",
        "ticker",
    ),
}


def _table_exists(conn, name):
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return c.fetchone() is not None


def _copy_table(src_conn, dst_conn, table, columns, conflict_cols):
    sc = src_conn.cursor()
    dc = dst_conn.cursor()
    if not _table_exists(src_conn, table):
        return 0
    if not _table_exists(dst_conn, table):
        sc.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
        ddl = sc.fetchone()
        if ddl and ddl[0]:
            dc.execute(ddl[0])
    col_list = [c.strip() for c in columns.split(",")]
    placeholders = ", ".join("?" for _ in col_list)
    sc.execute(f"SELECT {columns} FROM {table}")
    rows = sc.fetchall()
    inserted = 0
    for row in rows:
        try:
            dc.execute(
                f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})",
                row,
            )
            if dc.rowcount:
                inserted += 1
        except sqlite3.OperationalError:
            # Kolon uyumsuzlugu — ortak kolonlari dene
            pass
    return inserted


def migrate_legacy_data():
    """Legacy DB'deki eksik kayitlari ana DB'ye kopyala."""
    if not os.path.isfile(LEGACY_DB):
        return
    if os.path.normpath(LEGACY_DB) == os.path.normpath(DB_PATH):
        return

    try:
        src = sqlite3.connect(LEGACY_DB)
        dst = sqlite3.connect(DB_PATH, timeout=30.0)
        dst.execute("PRAGMA journal_mode=WAL")
        total = 0
        for table, (cols, _) in TABLES_INSERT_OR_IGNORE.items():
            n = _copy_table(src, dst, table, cols, _)
            if n:
                print(f"[+] Legacy aktarim: {table} +{n} kayit")
                total += n
        dst.commit()
        src.close()
        dst.close()
        if total:
            print(f"[+] Toplam {total} legacy kayit ana DB'ye eklendi")
    except Exception as e:
        print(f"[!] Legacy veri aktarimi atlandi: {e}")
