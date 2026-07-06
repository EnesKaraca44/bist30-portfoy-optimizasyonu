"""
Geçmiş halka arz verisi — elle doğrulama aracı.

Kullanım:
  python scripts/ipo_manual_verification.py export   # şablonu oluştur/güncelle
  python scripts/ipo_manual_verification.py status     # kaç kayıt doğrulandı
  python scripts/ipo_manual_verification.py apply      # doldurulan veriyi ipo_history.json'a yaz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DATA_DIR = os.path.join(_ROOT, "data")
HISTORY_PATH = os.path.join(DATA_DIR, "ipo_history.json")
MANUAL_PATH = os.path.join(DATA_DIR, "ipo_manual_dogrulama.json")

REHBER = {
    "nasil_doldurulur": [
        "Investing.com, Matriks veya KAP grafiğinden hisseyi aç.",
        "borsada_islem_tarihi: Gong / ilk işlem günü (YYYY-MM-DD).",
        "gercek_tavan_gunu: İlk işlem gününden itibaren ardışık tavan kapanış sayısı.",
        "Tavan kırıldığı gün saymayı bırak (o gün dahil değil).",
        "Emin olduğunda dogrulandi: true yap.",
        "python scripts/ipo_manual_verification.py apply komutu ile ana dosyaya aktar.",
    ],
    "ornek": {
        "borsa_kodu": "LILAK",
        "borsada_islem_tarihi": "2024-02-22",
        "gercek_tavan_gunu": 6,
        "dogrulandi": True,
        "kaynak_url": "https://tr.investing.com/equities/lila-kagit",
        "notlar": "Grafikten sayıldı",
    },
}


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_template() -> None:
    history = _load_json(HISTORY_PATH)
    existing = {}
    if os.path.isfile(MANUAL_PATH):
        old = _load_json(MANUAL_PATH)
        existing = {r["borsa_kodu"]: r for r in old.get("kayitlar", [])}

    kayitlar = []
    for rec in history.get("kayitlar", []):
        kod = rec["borsa_kodu"]
        prev = existing.get(kod, {})
        verified = rec.get("tavan_dogrulama") in ("manuel", "haber")
        kayitlar.append({
            "borsa_kodu": kod,
            "sirket_adi": rec.get("sirket_adi"),
            "sektor": rec.get("sektor"),
            "_mevcut_tavan_gun": rec.get("gercek_tavan_gunu"),
            "_mevcut_islem_tarihi": rec.get("borsada_islem_tarihi"),
            "borsada_islem_tarihi": prev.get("borsada_islem_tarihi") or rec.get("borsada_islem_tarihi") or "",
            "gercek_tavan_gunu": prev.get("gercek_tavan_gunu") if "gercek_tavan_gunu" in prev else (
                rec.get("gercek_tavan_gunu") if verified else None
            ),
            "dogrulandi": prev.get("dogrulandi", verified),
            "kaynak_url": prev.get("kaynak_url") or rec.get("kaynak_url") or "",
            "notlar": prev.get("notlar") or rec.get("notlar") or "",
        })

    out = {
        "aciklama": "Elle doğrulama dosyası — dogrulandi: true olan kayıtlar apply ile ipo_history.json'a yazılır.",
        "rehber": REHBER,
        "son_sablon": date.today().isoformat(),
        "kayitlar": kayitlar,
    }
    _save_json(MANUAL_PATH, out)
    done = sum(1 for k in kayitlar if k.get("dogrulandi"))
    print(f"Şablon yazıldı: {MANUAL_PATH}")
    print(f"  Toplam: {len(kayitlar)}  Doğrulanmış: {done}  Bekleyen: {len(kayitlar) - done}")


def show_status() -> None:
    if not os.path.isfile(MANUAL_PATH):
        print(f"Dosya yok. Önce: python scripts/ipo_manual_verification.py export")
        return
    data = _load_json(MANUAL_PATH)
    kayitlar = data.get("kayitlar", [])
    done = [k for k in kayitlar if k.get("dogrulandi")]
    pending = [k for k in kayitlar if not k.get("dogrulandi")]
    print(f"\n=== Elle Doğrulama Durumu ===")
    print(f"Doğrulanmış: {len(done)}/{len(kayitlar)}")
    if done:
        print("\nTamamlanan:")
        for k in done:
            print(f"  {k['borsa_kodu']:8} tavan={k.get('gercek_tavan_gunu')}  islem={k.get('borsada_islem_tarihi') or '-'}")
    if pending:
        print(f"\nBekleyen ({len(pending)}):")
        for k in pending:
            print(f"  {k['borsa_kodu']:8} (mevcut tahmin: {k.get('_mevcut_tavan_gun')} gün)")


def apply_manual() -> None:
    if not os.path.isfile(MANUAL_PATH):
        print(f"Önce export çalıştır: python scripts/ipo_manual_verification.py export")
        sys.exit(1)

    manual = _load_json(MANUAL_PATH)
    history = _load_json(HISTORY_PATH)
    by_kod = {r["borsa_kodu"]: r for r in manual.get("kayitlar", [])}

    updated = 0
    skipped = []
    for rec in history.get("kayitlar", []):
        kod = rec["borsa_kodu"]
        m = by_kod.get(kod)
        if not m or not m.get("dogrulandi"):
            continue
        days = m.get("gercek_tavan_gunu")
        if days is None or not isinstance(days, (int, float)) or days < 1 or days > 15:
            skipped.append(f"{kod}: gercek_tavan_gunu geçersiz ({days})")
            continue
        rec["gercek_tavan_gunu"] = int(days)
        islem = (m.get("borsada_islem_tarihi") or "").strip()
        if islem:
            rec["borsada_islem_tarihi"] = islem
        rec["tavan_dogrulama"] = "manuel"
        if m.get("kaynak_url"):
            rec["kaynak_url"] = m["kaynak_url"]
        if m.get("notlar"):
            rec["notlar"] = m["notlar"]
        updated += 1

    history["son_guncelleme"] = date.today().isoformat()
    _save_json(HISTORY_PATH, history)

    print(f"\n=== Uygulama Sonucu ===")
    print(f"Güncellenen kayıt: {updated}")
    if skipped:
        print("Atlanan:")
        for s in skipped:
            print(f"  - {s}")

    from ipo_predictor import run_ipo_backtest, validate_ipo_history
    v = validate_ipo_history()
    bt = run_ipo_backtest()
    print(f"\nDoğrulama: {v['kayit_sayisi'] - v.get('dogrulanmamis_sayisi', 0)}/{v['kayit_sayisi']} manuel/haber doğrulandı")
    print(f"Backtest: ort.hata={bt['ortalama_hata_gun']} gün, aralık=%{bt['aralik_isabet_yuzde']}")


def main():
    parser = argparse.ArgumentParser(description="IPO geçmiş veri elle doğrulama")
    parser.add_argument("command", choices=["export", "status", "apply"])
    args = parser.parse_args()
    if args.command == "export":
        export_template()
    elif args.command == "status":
        show_status()
    else:
        apply_manual()


if __name__ == "__main__":
    main()
