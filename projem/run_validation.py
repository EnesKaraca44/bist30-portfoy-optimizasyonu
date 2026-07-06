#!/usr/bin/env python
"""11 madde sonrası tam doğrulama — unittest + smoke + entegrasyon."""
import importlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  [OK] {name}")
    except Exception as e:
        FAILURES.append(f"{name}: {e}")
        print(f"  [FAIL] {name}: {e}")


def smoke_imports():
    core = (
        "config",
        "forecasting",
        "forecasting.arima",
        "forecasting.garch",
        "forecasting.weekly_pipeline",
        "forecasting.risk_metrics",
        "forecast_bias_correction",
        "forecasting.batch_pipeline",
        "portfolio.optimizer",
        "db.repository",
        "db.schema",
        "utils.dates",
        "services.charts",
        "services.helpers",
        "routes.views",
        "weekly_forecast",
        "performance_tracker",
    )
    for mod in core:
        importlib.import_module(mod)
    # TensorFlow bağımlı modüller opsiyonel
    for mod in ("lstm_forecast", "batch_job"):
        try:
            importlib.import_module(mod)
        except ImportError as e:
            print(f"  [WARN] {mod} atlandi (opsiyonel): {e}")


def smoke_config():
    from config import DB_PATH, LSTM_MODELS_DIR, BIST30_TICKERS

    assert len(BIST30_TICKERS) == 30
    assert "TRALT.IS" in BIST30_TICKERS
    assert "KOZAL.IS" not in BIST30_TICKERS
    assert os.path.exists(DB_PATH), f"DB yok: {DB_PATH}"
    assert os.path.isdir(LSTM_MODELS_DIR), f"LSTM klasörü yok: {LSTM_MODELS_DIR}"


def smoke_db():
    from config import DB_PATH
    from db.schema import get_connection, table_exists, REQUIRED_TABLES
    import weekly_forecast

    conn = get_connection()
    try:
        for table in REQUIRED_TABLES:
            assert table_exists(conn, table), f"Tablo eksik: {table}"
    finally:
        conn.close()

    assert weekly_forecast.DB_NAME == DB_PATH


def smoke_portfolio():
    import numpy as np
    import pandas as pd
    from portfolio.optimizer import run_monte_carlo_simulation

    tickers = ["A.IS", "B.IS", "C.IS"]
    er = pd.Series([0.1, 0.12, 0.08], index=tickers)
    cov = pd.DataFrame(np.eye(3) * 0.04, index=tickers, columns=tickers)
    max_p, min_p, _ = run_monte_carlo_simulation(er, cov, tickers, "Test")
    assert abs(max_p[tickers].sum() - 1.0) < 0.01


def smoke_flask():
    from app import app, DB_NAME
    from config import DB_PATH

    assert DB_NAME == DB_PATH
    client = app.test_client()
    r = client.get("/login")
    assert r.status_code == 200


def run_unittests():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "unittest",
         "test_improvements", "test_integration", "test_accuracy_improvements",
         "test_up_probability", "test_forecast_bias", "test_portfolio_var",
         "test_ipo_data", "test_ipo_predictor", "test_ipo_api", "test_ipo_backtest", "-v"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True,
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print(result.stderr[-1000:])
    if result.returncode != 0:
        FAILURES.append(f"unittest exit code {result.returncode}")


def main():
    print("=== Faz 6: Final Doğrulama ===\n")
    print("1. Import smoke")
    check("modül importları", smoke_imports)
    print("\n2. Config")
    check("config yolları ve ticker listesi", smoke_config)
    print("\n3. Veritabanı")
    check("DB tutarlılık ve tablolar", smoke_db)
    print("\n4. Portföy optimizasyon")
    check("Monte Carlo smoke", smoke_portfolio)
    print("\n5. Flask")
    check("Flask test client", smoke_flask)
    print("\n6. Unit testler")
    run_unittests()

    print("\n" + "=" * 40)
    if FAILURES:
        print(f"BAŞARISIZ ({len(FAILURES)} hata):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("TÜM DOĞRULAMALAR GEÇTİ")
    sys.exit(0)


if __name__ == "__main__":
    main()
