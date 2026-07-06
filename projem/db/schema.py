"""Veritabanı şema tanımları."""
import sqlite3

from config import DB_PATH


def get_connection(timeout=30.0):
    conn = sqlite3.connect(DB_PATH, timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def table_exists(conn, table_name):
    c = conn.cursor()
    c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return c.fetchone() is not None


REQUIRED_TABLES = [
    "beklenen_getiriler",
    "tahminler",
    "users",
    "user_portfolios",
]


def ensure_beklenen_getiriler_horizon_schema(conn=None):
    """Vade bazli (ticker, horizon_ay) tablo yapisini garanti et."""
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    try:
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='beklenen_getiriler'")
        exists = c.fetchone() is not None
        if exists:
            c.execute("PRAGMA table_info(beklenen_getiriler)")
            col_names = {row[1] for row in c.fetchall()}
            if "horizon_ay" in col_names and "tahmini_donem_getiri" in col_names:
                return
            c.execute("DROP TABLE beklenen_getiriler")
        c.execute(
            """
            CREATE TABLE beklenen_getiriler (
                ticker TEXT NOT NULL,
                horizon_ay INTEGER NOT NULL,
                tahmini_donem_getiri REAL,
                tahmini_yillik_getiri REAL,
                tahmini_yillik_risk REAL,
                tahmini_bl_getiri REAL,
                son_guncelleme TEXT,
                PRIMARY KEY (ticker, horizon_ay)
            )
            """
        )
        conn.commit()
    finally:
        if close_after:
            conn.close()
