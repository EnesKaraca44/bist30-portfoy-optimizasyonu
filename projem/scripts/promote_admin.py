import scripts._bootstrap  # noqa: F401
import sqlite3
import sys

from config import DB_PATH as DB_NAME

def promote_to_admin(email):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Kullanıcıyı bul
    c.execute("SELECT id, email FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    
    if not user:
        print(f"[-] HATA: '{email}' adresli kullanıcı bulunamadı.")
        conn.close()
        return
    
    # Admin yetkisi ver
    c.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user[0],))
    conn.commit()
    conn.close()
    print(f"[+] BAŞARILI: {email} kullanıcısı Admin yapıldı.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python promote_admin.py <email>")
    else:
        promote_to_admin(sys.argv[1])
