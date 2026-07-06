"""Merkezi konfigürasyon — tüm modüller buradan import eder."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DB_PATH = os.path.join(PROJECT_ROOT, "mpt_veritabani.db")
DB_NAME = DB_PATH  # geriye dönük uyumluluk

LSTM_MODELS_DIR = os.path.join(PROJECT_ROOT, "lstm_models")

BIST30_TICKERS = [
    "AEFES.IS", "AKBNK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "DSTKF.IS",
    "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS",
    "ISCTR.IS", "KCHOL.IS", "TRALT.IS", "KRDMD.IS", "MGROS.IS", "PETKM.IS",
    "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TAVHL.IS", "TCELL.IS",
    "THYAO.IS", "TOASO.IS", "TTKOM.IS", "TUPRS.IS", "ULKER.IS", "YKBNK.IS",
]
BIST30 = BIST30_TICKERS  # kısa alias

RISK_FREE_RATE = 0.40
ENSEMBLE_ARIMA_W = 0.5
ENSEMBLE_LSTM_W = 0.5
USE_LSTM = True

# Yillik ARIMA/LSTM tahmin clip (backtest ile ayni)
ANNUAL_ARIMA_CLIP_MAX = 0.80

# Makro özellikler (USDTRY, XU100, Brent) — ARIMA exog + rejim tespiti
USE_MACRO_FEATURES = True
MACRO_SYMBOLS = {
    "usdtry": "TRY=X",    # Dolar/TL kuru
    "xu100": "XU100.IS",  # BIST100 endeksi
    "brent": "BZ=F",      # Brent petrol
}
# ARIMA exog'a eklenecek makro feature sayısı (teknik feature'lara ek)
MACRO_N_FEATURES = 4

# TCMB EVDS — politika faizi (ucretsiz API key: https://evds2.tcmb.gov.tr/)
USE_EVDS_INTEREST = True
EVDS_API_KEY = os.environ.get("EVDS_API_KEY", "")
EVDS_POLICY_RATE_SERIES = "TP.APIFON4"  # TCMB agirlikli ortalama fonlama (politika faizi)

SECRET_KEY = os.environ.get("SECRET_KEY", "bist30-portfolio-secret-key-2024-dev-only")
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
