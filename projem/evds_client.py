"""
TCMB EVDS API istemcisi — politika faizi serisi.

API anahtari: https://evds2.tcmb.gov.tr/ uzerinden ucretsiz alinir.
.env dosyasina EVDS_API_KEY=... ekleyin.

2024-04 sonrasi: key HTTP header'da gonderilir (URL'de degil).
"""
import warnings
from datetime import datetime, timedelta
from urllib.parse import urlencode

import pandas as pd

warnings.filterwarnings("ignore")

EVDS_ENDPOINTS = [
    "https://evds2.tcmb.gov.tr/service/evds/",
    "https://evds3.tcmb.gov.tr/igmevdsms-dis/",
]
_last_evds_error = None


def get_last_evds_error():
    """Son EVDS hata mesaji (debug icin)."""
    return _last_evds_error


def _format_evds_date(dt):
    if hasattr(dt, "strftime"):
        return dt.strftime("%d-%m-%Y")
    return dt


def _parse_evds_date(value):
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y-%m"):
        try:
            ts = pd.Timestamp(datetime.strptime(str(value).strip(), fmt))
            return ts
        except ValueError:
            continue
    return pd.NaT


def _http_get(url, api_key, timeout=25):
    """EVDS HTTP istegi — once requests, yoksa urllib."""
    headers = {"key": api_key}

    try:
        import requests

        resp = requests.get(url, headers=headers, timeout=timeout)
        return resp.status_code, resp.text, resp.content
    except ImportError:
        import urllib.request

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, body.decode("utf-8", errors="replace"), body


def fetch_evds_series(series_code, start_date, end_date, api_key=None, timeout=25):
    """
    EVDS'ten tek seri indirir.

    Returns:
        pd.Series (index=tarih, values=float) veya None
    """
    global _last_evds_error

    if not api_key:
        _last_evds_error = "API key tanimli degil"
        return None

    _last_evds_error = None

    params = {
        "series": series_code,
        "startDate": _format_evds_date(start_date),
        "endDate": _format_evds_date(end_date),
        "type": "json",
    }
    query = urlencode(params)

    try:
        last_text = ""
        for base in EVDS_ENDPOINTS:
            url = f"{base}{query}"
            status, text, _raw = _http_get(url, api_key, timeout=timeout)
            last_text = (text or "").strip()

            if status != 200:
                _last_evds_error = f"HTTP {status}: {last_text[:120].strip()}"
                continue

            if not last_text:
                _last_evds_error = "EVDS bos yanit dondurdu"
                continue

            if not last_text.startswith("{"):
                _last_evds_error = (
                    "EVDS HTML sayfasi dondu — API key gecersiz/eksik olabilir "
                    f"(HTTP {status})"
                )
                continue

            import json

            payload = json.loads(last_text)
            items = payload.get("items") or []
            if not items:
                _last_evds_error = "EVDS items listesi bos"
                continue

            col_key = series_code.replace(".", "_")
            dates, values = [], []

            for row in items:
                dt = _parse_evds_date(row.get("Tarih"))
                raw_val = row.get(col_key)
                if raw_val in (None, "", "ND"):
                    for k, v in row.items():
                        if k not in ("Tarih", "UNIXTIME") and v not in (None, "", "ND"):
                            raw_val = v
                            break
                if pd.isna(dt) or raw_val in (None, "", "ND"):
                    continue
                try:
                    val = float(str(raw_val).replace(",", "."))
                except ValueError:
                    continue
                dates.append(dt)
                values.append(val)

            if not dates:
                _last_evds_error = f"Parse edilebilir veri yok (kolon: {col_key})"
                continue

            series = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()
            series.index = series.index.tz_localize(None)
            _last_evds_error = None
            return series

        return None

    except Exception as exc:
        _last_evds_error = str(exc)[:200]
        return None


def get_policy_rate_series(start_date=None, end_date=None, api_key=None, series_code=None):
    """Politika faizi serisini EVDS'ten ceker."""
    try:
        from config import EVDS_API_KEY, EVDS_POLICY_RATE_SERIES, USE_EVDS_INTEREST
    except ImportError:
        USE_EVDS_INTEREST = False
        EVDS_API_KEY = ""
        EVDS_POLICY_RATE_SERIES = "TP.APIFON4"

    if not USE_EVDS_INTEREST:
        return None

    api_key = api_key or EVDS_API_KEY
    end_date = end_date or pd.Timestamp.now()
    start_date = start_date or (end_date - timedelta(days=365))

    primary = series_code or EVDS_POLICY_RATE_SERIES
    result = fetch_evds_series(primary, start_date, end_date, api_key=api_key)
    if result is not None:
        return result

    # Yedek seri kodu
    if primary != "TP.MB.AO.A12":
        return fetch_evds_series("TP.MB.AO.A12", start_date, end_date, api_key=api_key)

    return None


def compute_policy_rate_features(policy_series):
    """Politika faizi serisinden makro feature'lar."""
    if policy_series is None or policy_series.empty:
        return None

    s = policy_series.ffill().dropna()
    if len(s) < 5:
        return None

    feats = pd.DataFrame(index=s.index)
    feats["policy_rate_level"] = s / 100.0
    feats["policy_rate_chg_20d"] = s.diff(20) / 100.0
    feats["policy_rate_mom_5d"] = s.pct_change(5)
    return feats.replace([float("inf"), float("-inf")], pd.NA).dropna(how="all")
