"""Faz 0 — IPO veri dosyaları doğrulama (Türkçe alan adları)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

GEREKLI_GECMIS_ALANLAR = {
    "borsa_kodu", "sirket_adi", "halka_arz_tarihi", "buyukluk_milyon_tl",
    "kurumsal_oran_yuzde", "iskonto_yuzde", "sektor", "gercek_tavan_gunu", "kaynak_url",
}
GEREKLI_YAKLASAN_ALANLAR = {
    "borsa_kodu", "sirket_adi", "durum", "sektor", "veri_durumu", "kap_linki",
}
GECERLI_DURUMLAR = {"planlanan", "onaylandi", "talep_topluyor", "islem_bekliyor"}
GECERLI_SEKTORLER = {
    "Perakende", "Enerji", "Finans", "Teknoloji", "Sanayi",
    "Gıda", "İnşaat", "Sağlık", "Diğer",
}


class TestIpoDataFiles(unittest.TestCase):
    def _load(self, filename):
        path = os.path.join(DATA_DIR, filename)
        self.assertTrue(os.path.isfile(path), f"Eksik dosya: {filename}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_history_file_structure(self):
        data = self._load("ipo_history.json")
        kayitlar = data.get("kayitlar") or data.get("records", [])
        self.assertGreaterEqual(len(kayitlar), 25)
        for rec in kayitlar:
            missing = GEREKLI_GECMIS_ALANLAR - set(rec.keys())
            self.assertFalse(missing, f"{rec.get('borsa_kodu')}: eksik alan {missing}")
            self.assertIn(rec["sektor"], GECERLI_SEKTORLER)
            self.assertGreater(rec["gercek_tavan_gunu"], 0)
            self.assertLessEqual(rec["gercek_tavan_gunu"], 15)

    def test_upcoming_file_structure(self):
        data = self._load("upcoming_ipos.json")
        liste = data.get("halka_arzlar") or data.get("ipos", [])
        self.assertGreater(len(liste), 0)
        for ipo in liste:
            missing = GEREKLI_YAKLASAN_ALANLAR - set(ipo.keys())
            self.assertFalse(missing, f"{ipo.get('borsa_kodu')}: eksik alan {missing}")
            self.assertIn(ipo["durum"], GECERLI_DURUMLAR)

    def test_koton_in_history(self):
        data = self._load("ipo_history.json")
        kodlar = {r["borsa_kodu"] for r in data.get("kayitlar", [])}
        self.assertIn("KOTON", kodlar)

    def test_ssaat_in_upcoming(self):
        data = self._load("upcoming_ipos.json")
        kodlar = {r["borsa_kodu"] for r in data.get("halka_arzlar", [])}
        self.assertIn("SSAAT", kodlar)


if __name__ == "__main__":
    unittest.main(verbosity=2)
