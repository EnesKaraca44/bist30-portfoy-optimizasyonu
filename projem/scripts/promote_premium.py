import scripts._bootstrap  # noqa: F401
"""
Kullanıcıyı Premium yapmak için yardımcı script.
Ödeme entegrasyonu öncesi manuel aktivasyon için kullanılır.

Kullanım: python promote_premium.py <email>
"""
import sqlite3
import sys

from config import DB_PATH as DB_NAME

def promote_to_premium(email):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute("SELECT id, email FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    
    if not user:
        print(f"[-] HATA: '{email}' adresli kullanıcı bulunamadı.")
        conn.close()
        return
    
    c.execute("UPDATE users SET is_premium = 1 WHERE id = ?", (user[0],))
    conn.commit()
    conn.close()
    print(f"[+] BAŞARILI: {email} kullanıcısı Premium yapıldı.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python promote_premium.py <email>")
        print("Örnek: python promote_premium.py kullanici@email.com")
    else:
        promote_to_premium(sys.argv[1])
