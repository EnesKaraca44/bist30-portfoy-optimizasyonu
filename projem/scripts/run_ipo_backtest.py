"""Geçmiş halka arz verisi doğrulama + leave-one-out backtest CLI."""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ipo_predictor import optimize_backtest_weights, run_ipo_backtest, validate_ipo_history


def _print_validation(v: dict) -> None:
    print(f"\n=== Veri Doğrulama ({v['kayit_sayisi']} kayıt) ===")
    if v["temiz"]:
        print("Tüm kayıtlar temiz görünüyor.")
        return
    print(f"Uyarı: {v['sorun_sayisi']} sorun bulundu")
    for item in v["sorunlar"]:
        print(f"  - {item['ticker']}: {item['problem']}")


def _print_backtest(result: dict, show_details: bool) -> None:
    print("\n=== Backtest (leave-one-out) ===")
    print(f"Test edilen kayıt: {result['test_edilen']}/{result['kayit_sayisi']}")
    print(f"Ortalama mutlak hata: {result['ortalama_hata_gun']:.2f} gün")
    print(f"Aralık isabeti: %{result['aralik_isabet_yuzde']}")
    print(f"±1 gün içinde: %{result['bir_gun_icinde_yuzde']}")
    print(f"hist_weight: {result['hist_weight']}")

    if not show_details:
        return

    print("\nDetay:")
    for row in result["detaylar"]:
        if "hata" in row:
            print(f"  {row['borsa_kodu']}: HATA — {row['hata']}")
            continue
        hit = "OK" if row["aralik_tuttu"] else "X"
        print(
            f"  {row['borsa_kodu']:8} gerçek={row['gercek_gun']} "
            f"tahmin={row['tahmin_ort']} ({row['tahmin_aralik']}) "
            f"hata={row['hata_gun']} {hit}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="IPO geçmiş veri doğrulama + backtest")
    parser.add_argument("--detay", action="store_true", help="Hisse bazlı sonuçları yazdır")
    parser.add_argument("--optimize", action="store_true", help="hist_weight grid araması")
    parser.add_argument("--json", action="store_true", help="JSON çıktı")
    parser.add_argument("--weight", type=float, default=0.85, help="Geçmiş/kural harman ağırlığı")
    args = parser.parse_args()

    validation = validate_ipo_history()
    if args.optimize:
        result = optimize_backtest_weights()
    else:
        result = run_ipo_backtest(hist_weight=args.weight)

    if args.json:
        print(json.dumps({"dogrulama": validation, "backtest": result}, ensure_ascii=False, indent=2))
        return

    _print_validation(validation)
    _print_backtest(result, show_details=args.detay)


if __name__ == "__main__":
    main()
