"""Halka arz tavan süresi tahmin motoru — geçmiş benzerlik + kural harmanı."""
from __future__ import annotations

import json
import math
import os
from typing import Any

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

VALID_SECTORS = (
    "Perakende", "Enerji", "Finans", "Teknoloji", "Sanayi",
    "Gıda", "İnşaat", "Sağlık", "Diğer",
)

DISCLAIMER = (
    "Geçmiş BIST halka arz verilerine dayalı senaryo analizidir. "
    "Yatırım tavsiyesi değildir; gerçekleşen tavan süresi farklı olabilir."
)

STATUS_LABELS = {
    "planlanan": "Planlanan",
    "onaylandi": "Onaylandı",
    "talep_topluyor": "Talep Topluyor",
    "islem_bekliyor": "İşlem Bekliyor",
}


def _load_json(filename: str) -> dict:
    path = os.path.join(_DATA_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pick(record: dict, *keys, default=None):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _normalize_history(raw: dict) -> dict:
    """JSON (Türkçe veya eski İngilizce anahtar) → dahili format."""
    return {
        "ticker": _pick(raw, "borsa_kodu", "ticker"),
        "name": _pick(raw, "sirket_adi", "name"),
        "ipo_date": _pick(raw, "halka_arz_tarihi", "ipo_date"),
        "size_mln": _pick(raw, "buyukluk_milyon_tl", "size_mln"),
        "inst_ratio": _pick(raw, "kurumsal_oran_yuzde", "inst_ratio"),
        "discount": _pick(raw, "iskonto_yuzde", "discount"),
        "sector": _pick(raw, "sektor", "sector"),
        "actual_ceiling_days": _pick(raw, "gercek_tavan_gunu", "actual_ceiling_days"),
        "first_trade_date": _pick(raw, "borsada_islem_tarihi", "first_trade_date"),
        "ceiling_verified": _pick(raw, "tavan_dogrulama", "ceiling_verified"),
        "first_day_return_pct": _pick(raw, "ilk_gun_getiri_yuzde", "first_day_return_pct"),
        "source_url": _pick(raw, "kaynak_url", "source_url"),
    }


def _normalize_upcoming(raw: dict) -> dict:
    """JSON (Türkçe veya eski İngilizce anahtar) → dahili format."""
    veri = _pick(raw, "veri_durumu", "data_completeness", default="eksik")
    if veri == "partial":
        veri = "kismi"
    elif veri == "full":
        veri = "tam"
    return {
        "id": _pick(raw, "borsa_kodu", "id"),
        "name": _pick(raw, "sirket_adi", "name"),
        "status": _pick(raw, "durum", "status", default="planlanan"),
        "expected_trade_date": _pick(raw, "borsada_islem_tarihi", "expected_trade_date"),
        "size_mln": _pick(raw, "buyukluk_milyon_tl", "size_mln"),
        "inst_ratio": _pick(raw, "kurumsal_oran_yuzde", "inst_ratio"),
        "discount": _pick(raw, "iskonto_yuzde", "discount"),
        "sector": _pick(raw, "sektor", "sector"),
        "kap_url": _pick(raw, "kap_linki", "kap_url"),
        "data_completeness": veri,
        "notes": _pick(raw, "notlar", "notes"),
        "missing_fields": raw.get("eksik_alanlar") or raw.get("missing_fields") or [],
        "extra": {
            k: raw[k] for k in (
                "talep_toplama_baslangic", "talep_toplama_bitis",
                "bireysel_oran_yuzde", "halka_arz_fiyati_tl",
                "halka_aciklik_orani_yuzde", "fiyat_istikrari_gun", "pazar",
                "buyukluk_ek_satış_dahil_milyon_tl",
            ) if k in raw and raw[k] is not None
        },
    }


def load_ipo_history() -> list[dict]:
    data = _load_json("ipo_history.json")
    kayitlar = data.get("kayitlar") or data.get("records", [])
    return [_normalize_history(r) for r in kayitlar]


def load_upcoming_ipos() -> list[dict]:
    data = _load_json("upcoming_ipos.json")
    liste = data.get("halka_arzlar") or data.get("ipos", [])
    return [_normalize_upcoming(r) for r in liste]


def get_upcoming_ipo_by_id(ipo_id: str) -> dict | None:
    for ipo in load_upcoming_ipos():
        if ipo.get("id") == ipo_id:
            return ipo
    return None


def get_active_ipos() -> list[dict]:
    """Geriye uyumluluk + zengin liste."""
    result = []
    for ipo in load_upcoming_ipos():
        result.append({
            "id": ipo["id"],
            "name": _format_upcoming_name(ipo),
            "status": ipo.get("status", "planlanan"),
            "status_label": STATUS_LABELS.get(ipo.get("status", ""), ipo.get("status", "")),
            "expected_trade_date": ipo.get("expected_trade_date"),
            "size_mln": ipo.get("size_mln"),
            "inst_ratio": ipo.get("inst_ratio"),
            "discount": ipo.get("discount"),
            "sector": ipo.get("sector"),
            "kap_url": ipo.get("kap_url"),
            "data_completeness": ipo.get("data_completeness", "minimal"),
        })
    return result


def _format_upcoming_name(ipo: dict) -> str:
    parts = [ipo.get("name", ipo.get("id", ""))]
    if ipo.get("expected_trade_date"):
        parts.append(f"({ipo['expected_trade_date']})")
    return " ".join(parts)


def _norm_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1.0)
    return min(abs(a - b) / denom, 1.0)


def compute_similarity(
    size_mln: float,
    inst_ratio: float,
    discount: float,
    sector: str | None,
    hist: dict,
) -> float:
    size_score = 1.0 - _norm_diff(size_mln, float(hist["size_mln"]))
    inst_score = 1.0 - min(abs(inst_ratio - float(hist["inst_ratio"])) / 100.0, 1.0)
    disc_score = 1.0 - min(abs(discount - float(hist["discount"])) / 100.0, 1.0)
    sector_bonus = 1.0 if sector and sector == hist.get("sector") else 0.0
    return 100.0 * (
        0.35 * size_score + 0.35 * inst_score + 0.20 * disc_score + 0.10 * sector_bonus
    )


def find_similar_ipos(
    size_mln: float,
    inst_ratio: float,
    discount: float,
    sector: str | None = None,
    top_n: int = 5,
    exclude_tickers: set[str] | None = None,
    history: list[dict] | None = None,
) -> list[dict]:
    scored = []
    pool = history if history is not None else load_ipo_history()
    skip = exclude_tickers or set()
    for hist in pool:
        if hist["ticker"] in skip:
            continue
        sim = compute_similarity(size_mln, inst_ratio, discount, sector, hist)
        scored.append({
            "ticker": hist["ticker"],
            "name": hist["name"],
            "days": int(hist["actual_ceiling_days"]),
            "similarity_pct": round(sim, 1),
            "ipo_date": hist.get("ipo_date"),
            "sector": hist.get("sector"),
        })
    scored.sort(key=lambda x: x["similarity_pct"], reverse=True)
    return scored[:top_n]


def _rule_based_days(size_mln: float, inst_ratio: float, discount: float) -> float:
    if size_mln < 1000:
        base = 8.0
    elif size_mln < 3000:
        base = 5.0
    elif size_mln < 5000:
        base = 3.0
    else:
        base = 1.0

    if inst_ratio > 80:
        base += 3.0
    elif inst_ratio > 60:
        base += 1.0
    elif inst_ratio < 40:
        base -= 1.0

    if discount > 25:
        base += 2.0
    elif discount < 15:
        base -= 1.0

    return max(1.0, min(14.0, base))


def _sector_avg_days(
    sector: str | None,
    exclude_tickers: set[str] | None = None,
    history: list[dict] | None = None,
) -> float | None:
    if not sector:
        return None
    skip = exclude_tickers or set()
    pool = history if history is not None else load_ipo_history()
    days = [
        float(h["actual_ceiling_days"])
        for h in pool
        if h.get("sector") == sector and h["ticker"] not in skip
    ]
    if not days:
        return None
    return sum(days) / len(days)


def format_buyukluk_tl(size_mln: float) -> str:
    """Milyon TL → kullanıcı dostu Türkçe metin."""
    if size_mln >= 1000:
        milyar = size_mln / 1000
        s = f"{milyar:.2f}".rstrip("0").rstrip(".")
        return s.replace(".", ",") + " Milyar TL"
    n = int(round(size_mln))
    return f"{n:,}".replace(",", ".") + " Milyon TL"


def build_factor_breakdown(
    size_mln: float,
    inst_ratio: float,
    discount: float,
    sector: str | None = None,
) -> list[dict]:
    factors = []

    buyukluk_metin = format_buyukluk_tl(size_mln)
    if size_mln < 1000:
        effect, detail = 2, f"{buyukluk_metin} — küçük tahta, tavan süresi uzayabilir"
    elif size_mln < 3000:
        effect, detail = 0, f"{buyukluk_metin} — orta tahta"
    elif size_mln < 5000:
        effect, detail = -1, f"{buyukluk_metin} — büyük tahta"
    else:
        effect, detail = -2, f"{buyukluk_metin} — çok büyük tahta, kısa tavan beklenir"
    factors.append({"label": "Halka arz büyüklüğü", "effect_days": effect, "detail": detail})

    if inst_ratio > 80:
        effect, detail = 2, f"%{inst_ratio:.0f} — yüksek kurumsal kilit"
    elif inst_ratio > 60:
        effect, detail = 1, f"%{inst_ratio:.0f} — orta-yüksek kurumsal"
    elif inst_ratio < 40:
        effect, detail = -1, f"%{inst_ratio:.0f} — düşük kurumsal, bireysel talep güçlü"
    else:
        effect, detail = 0, f"%{inst_ratio:.0f} — orta kurumsal oran"
    factors.append({"label": "Kurumsal yatırımcı oranı", "effect_days": effect, "detail": detail})

    if discount > 25:
        effect, detail = 2, f"%{discount:.0f} — yüksek iskonto, talep artırıcı"
    elif discount < 15:
        effect, detail = -1, f"%{discount:.0f} — düşük iskonto"
    else:
        effect, detail = 0, f"%{discount:.0f} — standart iskonto"
    factors.append({"label": "İskonto oranı", "effect_days": effect, "detail": detail})

    sec_avg = _sector_avg_days(sector)
    if sector and sec_avg is not None:
        factors.append({
            "label": f"Sektör ortalaması ({sector})",
            "effect_days": 0,
            "detail": f"Geçmiş {sector} halka arzları ortalama {sec_avg:.1f} gün tavan yaptı",
        })

    return factors


def _ceiling_strength(size_mln: float, inst_ratio: float, discount: float) -> tuple[int, str]:
    score = 50.0
    if size_mln < 1500:
        score += 20
    elif size_mln > 6000:
        score -= 25
    elif size_mln > 4000:
        score -= 12

    if inst_ratio < 25:
        score += 15
    elif inst_ratio > 65:
        score -= 15

    if discount > 22:
        score += 12
    elif discount < 14:
        score -= 8

    score = int(max(5, min(95, round(score))))
    if score >= 70:
        label = "Güçlü tavan potansiyeli"
    elif score >= 45:
        label = "Orta tavan potansiyeli"
    else:
        label = "Zayıf / kısa tavan beklenir"
    return score, label


def _confidence_level(
    size_mln: float | None,
    inst_ratio: float | None,
    discount: float | None,
    top_similarity: float,
) -> tuple[str, str]:
    filled = sum(x is not None for x in (size_mln, inst_ratio, discount))
    if filled == 3 and top_similarity >= 75:
        return "high", "Parametreler tam; en benzer geçmiş halka arz yüksek eşleşme gösteriyor."
    if filled >= 2 and top_similarity >= 50:
        return "medium", "Orta güven; benzer geçmiş kayıtlar kısmen eşleşiyor."
    return "low", "Düşük güven; eksik parametre veya zayıf benzerlik — aralık geniş tutuldu."


def _predict_core(
    size_mln: float,
    inst_ratio: float,
    discount: float,
    sector: str | None = None,
    exclude_tickers: set[str] | None = None,
    history: list[dict] | None = None,
    hist_weight: float = 0.85,
) -> dict[str, Any]:
    """Tahmin motoru — backtest için leave-one-out destekli."""
    pool = history if history is not None else load_ipo_history()
    similar = find_similar_ipos(
        size_mln, inst_ratio, discount, sector,
        top_n=5, exclude_tickers=exclude_tickers, history=pool,
    )
    if not similar:
        return {"error": "Benzer kayıt bulunamadı"}

    weights = [s["similarity_pct"] for s in similar]
    days = [s["days"] for s in similar]
    w_sum = sum(weights) or 1.0
    hist_avg = sum(d * w for d, w in zip(days, weights)) / w_sum

    if len(days) > 1:
        mean = sum(days) / len(days)
        hist_std = math.sqrt(sum((d - mean) ** 2 for d in days) / len(days))
    else:
        hist_std = 0.5
    hist_std = max(hist_std, 0.5)

    rule_days = _rule_based_days(size_mln, inst_ratio, discount)
    blended = hist_weight * hist_avg + (1.0 - hist_weight) * rule_days

    confidence, confidence_note = _confidence_level(
        size_mln, inst_ratio, discount, similar[0]["similarity_pct"],
    )
    width_extra = 1.0 if confidence == "low" else (0.5 if confidence == "medium" else 0.0)

    min_days = max(1, int(math.floor(blended - hist_std - width_extra)))
    max_days = min(15, int(math.ceil(blended + hist_std + width_extra)))
    if min_days > max_days:
        min_days = max(1, max_days - 1)

    expected_avg = int(round(blended))
    strength, strength_label = _ceiling_strength(size_mln, inst_ratio, discount)

    return {
        "success": True,
        "expected_days_str": f"{min_days} - {max_days} iş günü",
        "expected_days_avg": expected_avg,
        "min_days": min_days,
        "max_days": max_days,
        "ceiling_strength": strength,
        "ceiling_label": strength_label,
        "confidence": confidence,
        "confidence_note": confidence_note,
        "similar_ipos": similar[:3],
        "sector_avg_days": round(_sector_avg_days(sector, exclude_tickers, pool) or 0, 1)
        if sector else None,
    }


def validate_ipo_history() -> dict[str, Any]:
    """Geçmiş kayıt bütünlük kontrolü."""
    records = load_ipo_history()
    issues = []
    tickers = []
    for rec in records:
        t = rec.get("ticker")
        if not t:
            issues.append({"ticker": "?", "problem": "borsa_kodu eksik"})
            continue
        if t in tickers:
            issues.append({"ticker": t, "problem": "mükerrer kayıt"})
        tickers.append(t)
        for field in ("size_mln", "inst_ratio", "discount", "actual_ceiling_days"):
            val = rec.get(field)
            if val is None:
                issues.append({"ticker": t, "problem": f"{field} eksik"})
        days = rec.get("actual_ceiling_days")
        if days is not None and (days < 1 or days > 15):
            issues.append({"ticker": t, "problem": f"tavan günü şüpheli: {days}"})
        if rec.get("size_mln") is not None and rec["size_mln"] <= 0:
            issues.append({"ticker": t, "problem": "büyüklük geçersiz"})
        if not _is_ceiling_verified(rec):
            issues.append({"ticker": t, "problem": "tavan günü henüz doğrulanmadı", "tip": "uyari"})

    uyari = [i for i in issues if i.get("tip") == "uyari"]
    kritik = [i for i in issues if i.get("tip") != "uyari"]

    return {
        "kayit_sayisi": len(records),
        "sorun_sayisi": len(kritik),
        "dogrulanmamis_sayisi": len(uyari),
        "temiz": len(kritik) == 0,
        "sorunlar": kritik,
        "uyarilar": uyari,
    }


def _is_ceiling_verified(rec: dict) -> bool:
    v = rec.get("ceiling_verified")
    return bool(v) and v not in ("eksik", "hayir", False)


def get_ipo_user_tasks() -> dict[str, Any]:
    """Kullanıcının manuel tamamlayabileceği görev listesi."""
    upcoming = load_upcoming_ipos()
    history = load_ipo_history()

    yaklasan = []
    for ipo in upcoming:
        missing = list(ipo.get("missing_fields") or [])
        if missing:
            labels = {
                "borsada_islem_tarihi": "Borsada ilk işlem tarihi (KAP duyurusu)",
            }
            yaklasan.append({
                "borsa_kodu": ipo["id"],
                "sirket": ipo.get("name"),
                "eksik": [labels.get(m, m) for m in missing],
            })

    dogrulanmamis = [
        {
            "borsa_kodu": r["ticker"],
            "sirket": r.get("name"),
        }
        for r in history
        if not _is_ceiling_verified(r)
    ]

    return {
        "yaklasan_guncelleme": yaklasan,
        "gecmis_dogrulama": dogrulanmamis,
        "ozet": {
            "yaklasan_eksik": len(yaklasan),
            "gecmis_dogrulanmamis": len(dogrulanmamis),
        },
    }


def run_ipo_backtest(hist_weight: float = 0.85) -> dict[str, Any]:
    """
    Leave-one-out backtest: her geçmiş halka arz kendi tahmininde kullanılmaz.
    """
    history = load_ipo_history()
    rows = []
    abs_errors = []
    range_hits = 0
    within_one = 0

    for rec in history:
        actual = int(rec["actual_ceiling_days"])
        pred = _predict_core(
            float(rec["size_mln"]),
            float(rec["inst_ratio"]),
            float(rec["discount"]),
            rec.get("sector"),
            exclude_tickers={rec["ticker"]},
            history=history,
            hist_weight=hist_weight,
        )
        if pred.get("error"):
            rows.append({
                "borsa_kodu": rec["ticker"],
                "gercek_gun": actual,
                "hata": pred["error"],
            })
            continue

        avg = pred["expected_days_avg"]
        err = abs(avg - actual)
        abs_errors.append(err)
        in_range = pred["min_days"] <= actual <= pred["max_days"]
        if in_range:
            range_hits += 1
        if err <= 1:
            within_one += 1

        rows.append({
            "borsa_kodu": rec["ticker"],
            "sektor": rec.get("sector"),
            "gercek_gun": actual,
            "tahmin_ort": avg,
            "tahmin_aralik": f"{pred['min_days']}-{pred['max_days']}",
            "hata_gun": err,
            "aralik_tuttu": in_range,
            "en_benzer": pred["similar_ipos"][0]["ticker"] if pred["similar_ipos"] else None,
        })

    n = len(abs_errors)
    mae = sum(abs_errors) / n if n else 0.0
    return {
        "kayit_sayisi": len(history),
        "test_edilen": n,
        "ortalama_hata_gun": round(mae, 2),
        "aralik_isabet_yuzde": round(100.0 * range_hits / n, 1) if n else 0.0,
        "bir_gun_icinde_yuzde": round(100.0 * within_one / n, 1) if n else 0.0,
        "hist_weight": hist_weight,
        "dogrulama": validate_ipo_history(),
        "detaylar": rows,
    }


def optimize_backtest_weights() -> dict[str, Any]:
    """hist_weight için basit grid arama."""
    best = None
    for w in [x / 100 for x in range(50, 86, 5)]:
        result = run_ipo_backtest(hist_weight=w)
        mae = result["ortalama_hata_gun"]
        if best is None or mae < best["ortalama_hata_gun"]:
            best = result
    return best or run_ipo_backtest()


def get_ipo_model_dashboard() -> dict[str, Any]:
    """UI için backtest + veri kalitesi + Türkçe açıklamalar."""
    history = load_ipo_history()
    dogrulama = validate_ipo_history()
    backtest = run_ipo_backtest()
    tasks = get_ipo_user_tasks()

    verified = sum(1 for r in history if _is_ceiling_verified(r))
    total = len(history)
    mae = backtest["ortalama_hata_gun"]
    range_pct = backtest["aralik_isabet_yuzde"]
    within1 = backtest["bir_gun_icinde_yuzde"]

    if mae <= 1.5:
        kalite, kalite_renk = "iyi", "#10b981"
        yorum = "Geçmiş arzlarda tahminler genelde güvenilir; yine de aralık olarak değerlendirin."
    elif mae <= 2.5:
        kalite, kalite_renk = "orta", "#f59e0b"
        yorum = "Model yönü iyi yakalar; tek bir gün rakamına fazla anlam yüklemeyin."
    else:
        kalite, kalite_renk = "dikkat", "#ef4444"
        yorum = "Tahminler senaryo niteliğindedir; mutlaka geniş aralıkla okuyun."

    kalite_etiket = {"iyi": "İyi", "orta": "Orta", "dikkat": "Dikkatli oku"}[kalite]

    return {
        "ortalama_hata_gun": mae,
        "aralik_isabet_yuzde": range_pct,
        "bir_gun_icinde_yuzde": within1,
        "test_edilen": backtest["test_edilen"],
        "dogrulama": dogrulama,
        "dogrulanmis_kayit": verified,
        "toplam_kayit": total,
        "veri_tamam": verified >= total and total > 0,
        "kalite_seviyesi": kalite,
        "kalite_etiket": kalite_etiket,
        "kalite_renk": kalite_renk,
        "kalite_yorumu": yorum,
        "nasil_calisir": [
            "Benzer büyüklük, kurumsal tahsis ve iskontoya sahip geçmiş BIST halka arzları bulunur.",
            "Bu arzlardaki gerçek tavan günleri ile kural tabanlı tahmin harmanlanır.",
            "Sonuç tek gün değil; alt–üst sınır içeren bir iş günü aralığıdır.",
        ],
        "backtest_aciklama": (
            f"Son {total} halka arzın her biri ayrı test edildi — tahmin yapılırken "
            "o arzın kendi verisi kullanılmadı."
        ),
        "metrikler": [
            {
                "baslik": "Ortalama sapma",
                "deger": mae,
                "birim": "gün",
                "aciklama": "Tahmin ortalaması ile gerçek tavan günü arasındaki fark.",
            },
            {
                "baslik": "Aralık isabeti",
                "deger": range_pct,
                "birim": "%",
                "aciklama": "Gerçek tavan günü, verilen min–max aralığına girdi mi?",
            },
            {
                "baslik": "±1 gün isabet",
                "deger": within1,
                "birim": "%",
                "aciklama": "Tahmin, gerçek değerden en fazla 1 gün saptı.",
            },
        ],
        "yaklasan_eksikler": tasks.get("yaklasan_guncelleme", []),
    }


def predict_ceiling_days(
    size_mln: float | None = None,
    inst_ratio: float | None = None,
    discount: float | None = None,
    sector: str | None = None,
    ipo_id: str | None = None,
) -> dict[str, Any]:
    ipo_info = None
    if ipo_id:
        upcoming = get_upcoming_ipo_by_id(ipo_id)
        if not upcoming:
            return {"error": "Halka arz bulunamadı"}
        ipo_info = {
            "id": upcoming["id"],
            "name": upcoming.get("name"),
            "status": upcoming.get("status"),
            "status_label": STATUS_LABELS.get(upcoming.get("status", ""), ""),
            "expected_trade_date": upcoming.get("expected_trade_date"),
            "kap_url": upcoming.get("kap_url"),
        }
        size_mln = size_mln if size_mln is not None else upcoming.get("size_mln")
        inst_ratio = inst_ratio if inst_ratio is not None else upcoming.get("inst_ratio")
        discount = discount if discount is not None else upcoming.get("discount")
        sector = sector or upcoming.get("sector")
        if upcoming.get("missing_fields"):
            ipo_info["missing_fields"] = upcoming["missing_fields"]
        if upcoming.get("extra"):
            ipo_info["extra"] = upcoming["extra"]

    discount_estimated = False
    if discount is None and size_mln is not None and inst_ratio is not None:
        discount = 15.0
        discount_estimated = True

    missing = [k for k, v in {
        "size_mln": size_mln, "inst_ratio": inst_ratio,
    }.items() if v is None]
    if missing:
        return {"error": f"Gerekli parametreler eksik: {', '.join(missing)}"}
    if discount is None:
        return {"error": "Gerekli parametreler eksik: iskonto_yuzde (discount)"}

    size_mln = float(size_mln)
    inst_ratio = float(inst_ratio)
    discount = float(discount)

    core = _predict_core(size_mln, inst_ratio, discount, sector)
    if core.get("error"):
        return core

    confidence = core["confidence"]
    confidence_note = core["confidence_note"]
    if discount_estimated:
        confidence = "low" if confidence == "high" else confidence
        confidence_note += " İskonto izahnamede yok; tahmin için %15 varsayıldı."

    similar = core["similar_ipos"]
    comment = _build_comment(core["expected_days_avg"], similar, core["ceiling_label"])

    return {
        **core,
        "confidence": confidence,
        "confidence_note": confidence_note,
        "comment": comment,
        "factors": build_factor_breakdown(size_mln, inst_ratio, discount, sector),
        "ipo_info": ipo_info,
        "disclaimer": DISCLAIMER,
        "details": {
            "size_mln": size_mln,
            "buyukluk_metin": format_buyukluk_tl(size_mln),
            "inst_ratio": inst_ratio,
            "discount": discount,
            "sector": sector,
        },
    }


def _build_comment(expected_avg: int, similar: list[dict], strength_label: str) -> str:
    names = ", ".join(f"{s['ticker']} ({s['days']} gün)" for s in similar[:2])
    return (
        f"Benzer geçmiş halka arzlara ({names}) göre tahmini {expected_avg} iş günü tavan "
        f"bekleniyor. Değerlendirme: {strength_label}."
    )


def predict_ipo_performance(
    ipo_id: str | None = None,
    size_mln: float | None = None,
    inst_ratio: float | None = None,
    discount: float | None = None,
    sector: str | None = None,
) -> dict[str, Any]:
    """API uyumlu ana giriş noktası."""
    try:
        return predict_ceiling_days(
            size_mln=size_mln,
            inst_ratio=inst_ratio,
            discount=discount,
            sector=sector,
            ipo_id=ipo_id,
        )
    except Exception as e:
        return {"error": str(e)}


def get_ipo_history(sector: str | None = None, limit: int = 40) -> list[dict]:
    records = load_ipo_history()
    if sector:
        records = [r for r in records if r.get("sector") == sector]
    return records[:limit]
