"""
Kullanıcı Kimlik Doğrulama Modülü
Flask-Login ile entegre oturum yönetimi
"""
import os
import sqlite3
from flask_login import UserMixin
from flask_bcrypt import Bcrypt

from config import DB_PATH as DB_NAME, BASE_DIR
bcrypt = Bcrypt()


class User(UserMixin):
    """
    Flask-Login için kullanıcı modeli.
    UserMixin: is_authenticated, is_active, is_anonymous, get_id() sağlar.
    """
    def __init__(self, id, email, password_hash, is_premium=False, is_admin=False, demo_used=False, created_at=None):
        self.id = id
        self.email = email
        self.password_hash = password_hash
        self.is_premium = is_premium
        self.is_admin = is_admin
        self.demo_used = demo_used
        self.created_at = created_at
    
    @staticmethod
    def get_by_id(user_id):
        """ID ile kullanıcı bul"""
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, email, password_hash, is_premium, is_admin, demo_used, created_at FROM users WHERE id = ?", (user_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            return User(id=row[0], email=row[1], password_hash=row[2], 
                       is_premium=bool(row[3]), is_admin=bool(row[4]), 
                       demo_used=bool(row[5]), created_at=row[6])
        return None
    
    @staticmethod
    def get_by_email(email):
        """Email ile kullanıcı bul"""
        normalized_email = (email or "").strip()
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            "SELECT id, email, password_hash, is_premium, is_admin, demo_used, created_at "
            "FROM users WHERE lower(email) = lower(?)",
            (normalized_email,)
        )
        row = c.fetchone()
        conn.close()
        
        if row:
            return User(id=row[0], email=row[1], password_hash=row[2], 
                       is_premium=bool(row[3]), is_admin=bool(row[4]), 
                       demo_used=bool(row[5]), created_at=row[6])
        return None
    
    @staticmethod
    def create(email, password):
        """Yeni kullanıcı oluştur"""
        normalized_email = (email or "").strip().lower()
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        try:
            c.execute("""
                INSERT INTO users (email, password_hash, is_premium, is_admin, demo_used)
                VALUES (?, ?, 0, 0, 0)
            """, (normalized_email, password_hash))
            conn.commit()
            user_id = c.lastrowid
            conn.close()
            return User.get_by_id(user_id)
        except sqlite3.IntegrityError:
            conn.close()
            return None  # Email zaten mevcut
    
    def check_password(self, password):
        """Şifre kontrolü"""
        return bcrypt.check_password_hash(self.password_hash, password)


def _migrate_users_from_legacy_db():
    """
    Eski projem/mpt_veritabani.db kullanicilarini ana DB'ye tasir.
    Refactoring oncesi web app bu dosyayi kullaniyordu.
    """
    legacy_db = os.path.join(BASE_DIR, "mpt_veritabani.db")
    if not os.path.isfile(legacy_db) or os.path.normpath(legacy_db) == os.path.normpath(DB_NAME):
        return

    try:
        src = sqlite3.connect(legacy_db)
        sc = src.cursor()
        sc.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not sc.fetchone():
            src.close()
            return
        sc.execute(
            "SELECT email, password_hash, is_premium, is_admin, demo_used, created_at FROM users"
        )
        rows = sc.fetchall()
        src.close()
        if not rows:
            return

        dst = sqlite3.connect(DB_NAME)
        dc = dst.cursor()
        migrated = 0
        for row in rows:
            email, pw_hash, premium, admin, demo, created = row
            dc.execute("SELECT id FROM users WHERE lower(email)=lower(?)", (email,))
            existing = dc.fetchone()
            if existing:
                dc.execute(
                    """UPDATE users SET password_hash=?, is_premium=?, is_admin=?, demo_used=?
                       WHERE lower(email)=lower(?)""",
                    (pw_hash, premium, admin, demo, email),
                )
            else:
                dc.execute(
                    """INSERT INTO users (email, password_hash, is_premium, is_admin, demo_used, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (email, pw_hash, premium, admin, demo, created),
                )
            migrated += 1
        dst.commit()
        dst.close()
        if migrated:
            print(f"[+] {migrated} kullanici eski DB'den aktarildi ({legacy_db})")
    except Exception as e:
        print(f"[!] Eski DB kullanici aktarimi atlandi: {e}")


def setup_users_table():
    """Kullanıcı tablosunu oluştur"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_premium INTEGER DEFAULT 0,
            is_admin INTEGER DEFAULT 0,
            demo_used INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Kolonlar yoksa ekle (Upgrade senaryosu)
    try:
        c.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
        print("[*] is_admin kolonu eklendi.")
    except sqlite3.OperationalError:
        pass # Zaten var

    try:
        c.execute("ALTER TABLE users ADD COLUMN demo_used INTEGER DEFAULT 0")
        print("[*] demo_used kolonu eklendi.")
    except sqlite3.OperationalError:
        pass  # Zaten var
    try:
        c.execute("ALTER TABLE users ADD COLUMN free_stock_views TEXT DEFAULT '[]'")
        print("[*] free_stock_views kolonu eklendi.")
    except sqlite3.OperationalError:
        pass  # Zaten var
    conn.commit()
    conn.close()
    _migrate_users_from_legacy_db()
    print("[+] Users tablosu hazır.")


def setup_user_portfolios_table():
    """Kullanıcı portföylerini saklamak için tablo"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            portfolio_name TEXT NOT NULL,
            weights_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()
    print("[+] User portfolios tablosu hazır.")


def setup_leaderboard_table():
    """Liderlik tablosunu oluştur"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            week_start_date TEXT NOT NULL,
            week_end_date TEXT NOT NULL,
            user_return REAL NOT NULL,
            ai_return REAL NOT NULL,
            is_winner INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()
    print("[+] Leaderboard tablosu hazır.")


if __name__ == "__main__":
    setup_users_table()
    setup_user_portfolios_table()
    setup_leaderboard_table()
    print("[+] Tüm tablolar oluşturuldu!")


# --- PORTFÖY YÖNETİM FONKSİYONLARI ---

def save_portfolio(user_id, portfolio_name, weights_dict):
    """Kullanıcı için yeni portföy kaydet"""
    import json
    weights_json = json.dumps(weights_dict)
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO user_portfolios (user_id, portfolio_name, weights_json)
        VALUES (?, ?, ?)
    """, (user_id, portfolio_name, weights_json))
    conn.commit()
    portfolio_id = c.lastrowid
    conn.close()
    return portfolio_id


def get_user_portfolios(user_id):
    """Kullanıcının tüm portföylerini getir"""
    import json
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT id, portfolio_name, weights_json, created_at 
        FROM user_portfolios 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    """, (user_id,))
    rows = c.fetchall()
    conn.close()
    
    portfolios = []
    for row in rows:
        portfolios.append({
            'id': row[0],
            'name': row[1],
            'weights': json.loads(row[2]),
            'created_at': row[3]
        })
    return portfolios


def delete_portfolio(user_id, portfolio_id):
    """Kullanıcının portföyünü sil (sadece kendi portföyünü silebilir)"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        DELETE FROM user_portfolios 
        WHERE id = ? AND user_id = ?
    """, (portfolio_id, user_id))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def get_portfolio_by_id(user_id, portfolio_id):
    """Belirli bir portföyü getir"""
    import json
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT id, portfolio_name, weights_json, created_at 
        FROM user_portfolios 
        WHERE id = ? AND user_id = ?
    """, (portfolio_id, user_id))
    row = c.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'name': row[1],
            'weights': json.loads(row[2]),
            'created_at': row[3]
        }
    return None
