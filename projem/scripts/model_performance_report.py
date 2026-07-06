"""Hisse bazlı model performans raporu — haftalık tahmin sonrası teşhis."""
import scripts._bootstrap  # noqa: F401

from model_selector import generate_model_performance_report


def main():
    print("[*] Model performans raporu olusturuluyor...")
    df = generate_model_performance_report(lookback_forecasts=20)
    if df.empty:
        print("[-] Yeterli tahmin gecmisi yok.")
        return
    print(df.to_string(index=False))
    weak = df[df["arima_direction_acc"].fillna(0) < 0.45]
    if len(weak):
        print(f"\n[!] Dusuk yon dogrulugu: {', '.join(weak['ticker'].tolist())}")


if __name__ == "__main__":
    main()
