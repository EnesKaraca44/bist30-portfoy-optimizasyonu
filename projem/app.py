import logging

from flask import Flask, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager

from config import DB_PATH, SECRET_KEY, FLASK_DEBUG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# --- 1. Flask Uygulaması, DB ve BIST 30 Listesi ---
app = Flask(__name__)
app.secret_key = SECRET_KEY


@app.errorhandler(500)
def internal_error(error):
    logger.exception("Sunucu hatasi: %s", error)
    if request.path.startswith("/api/"):
        return jsonify({"error": "Sunucu hatasi. Lütfen daha sonra tekrar deneyin."}), 500
    flash("Beklenmeyen bir hata oluştu.", "error")
    return redirect(url_for("home")), 500

# Flask-Login yapılandırması
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Bu sayfayı görmek için giriş yapmalısınız.'

# Auth modülünü import et ve bcrypt'i başlat
from auth import User, bcrypt, setup_users_table, setup_user_portfolios_table, setup_leaderboard_table
bcrypt.init_app(app)

# Kullanıcı tablolarını oluştur
setup_users_table()
setup_user_portfolios_table()
setup_leaderboard_table()

from performance_tracker import setup_performance_tables
from portfolio_performance_tracker import setup_portfolio_performance_table
from db.migrate_legacy import migrate_legacy_data

setup_performance_tables()
setup_portfolio_performance_table()
migrate_legacy_data()

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(int(user_id))

DB_NAME = DB_PATH

from routes import register_all

register_all(app)

# --- 6. Sunucuyu Çalıştırma ---
if __name__ == '__main__':
    app.run(debug=FLASK_DEBUG, use_reloader=False)
