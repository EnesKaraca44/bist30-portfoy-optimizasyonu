"""Makro backtest raporu CLI."""
import json
import sqlite3
import sys

import _bootstrap  # noqa: F401
from config import DB_PATH
from macro_backtest import build_macro_backtest_report, format_report_text


def main():
    date_old = sys.argv[1] if len(sys.argv) > 1 else None
    date_new = sys.argv[2] if len(sys.argv) > 2 else None

    conn = sqlite3.connect(DB_PATH)
    report = build_macro_backtest_report(conn, date_old=date_old, date_new=date_new)
    conn.close()

    print(format_report_text(report))

    out_path = _bootstrap.ROOT / "data" / "macro_backtest_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[+] JSON rapor: {out_path}")

    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
