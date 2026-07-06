"""EVDS baglanti testi — kullanici yerelde calistirir."""
import _bootstrap  # noqa: F401
import config
from evds_client import get_last_evds_error, get_policy_rate_series


def main():
    print("[*] EVDS baglanti testi")
    print(f"    Key tanimli: {'Evet' if config.EVDS_API_KEY else 'Hayir'}")
    print(f"    Seri: {config.EVDS_POLICY_RATE_SERIES}")

    series = get_policy_rate_series()
    if series is not None and not series.empty:
        print(f"[+] BASARILI — son faiz: %{series.iloc[-1]:.2f} ({series.index[-1].date()})")
        return 0

    err = get_last_evds_error() or "Bilinmeyen hata"
    print(f"[-] BASARISIZ — {err}")
    print()
    if "HTML" in err or "JSON" in err:
        print("Bu hata genelde API key yanlis/eksik oldugunda olur.")
        print("TCMB panelinden YENI key alip .env dosyasina yapistirin.")
    print()
    print("Kontrol listesi:")
    print("  1) https://evds2.tcmb.gov.tr/ → Profil → API Key")
    print("  2) Key'i basindan sonuna kadar kopyalayin (bosluk olmasin)")
    print("  3) projem/.env → EVDS_API_KEY=... satirini guncelleyin")
    print("  4) Tekrar: python scripts/test_evds_connection.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
