"""Flask route kayıtları."""
import json
import traceback

import numpy as np
import pandas as pd
import sqlite3
import yfinance as yf
from flask import request, jsonify, render_template, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user

from auth import User
from config import DB_PATH, BIST30_TICKERS
from portfolio.optimizer import (
    get_bist30_data_inputs,
    get_bist30_ml_with_horizon,
    run_monte_carlo_simulation,
)
from services.helpers import sanitize_months, normalize_horizon_months
from services.charts import simulate_future_wealth, calculate_mpt, generate_backtest_chart

DB_NAME = DB_PATH


def register_routes(app):
    @app.route('/')
    def home():
        if current_user.is_authenticated:
            return render_template('index.html', bist30_list=BIST30_TICKERS, user=current_user)
        else:
            return render_template('landing.html')

    @app.route('/api/bist30_live', methods=['GET'])
    def bist30_live():
        """Üst barda gösterilecek canlı BIST30 özet verileri."""
        try:
            data = yf.download(BIST30_TICKERS, period='5d', interval='1d', progress=False)['Close']
            if data is None or data.empty:
                return jsonify({'success': False, 'message': 'Canlı veri alınamadı.'}), 503

            # Tek sembol durumunda Series gelebilir
            if isinstance(data, pd.Series):
                data = data.to_frame(name=BIST30_TICKERS[0])

            data = data.ffill().dropna(axis=1, how='all')
            if data.shape[0] < 2:
                return jsonify({'success': False, 'message': 'Yeterli fiyat geçmişi yok.'}), 503

            last = data.iloc[-1]
            prev = data.iloc[-2]

            ticker_rows = []
            for t in data.columns:
                last_price = float(last[t]) if pd.notna(last[t]) else None
                prev_price = float(prev[t]) if pd.notna(prev[t]) else None
                if not last_price or not prev_price or prev_price <= 0:
                    continue
                change_pct = ((last_price / prev_price) - 1) * 100.0
                ticker_rows.append({
                    'ticker': t,
                    'price': round(last_price, 2),
                    'change_pct': round(change_pct, 2)
                })

            if not ticker_rows:
                return jsonify({'success': False, 'message': 'Geçerli canlı veri bulunamadı.'}), 503

            # Tüm BIST30 listesini, orijinal sıra ile döndür
            ticker_map = {x['ticker']: x for x in ticker_rows}
            full_tickers = [ticker_map[t] for t in BIST30_TICKERS if t in ticker_map]

            # Basit endeks proxy: BIST30 hisselerinin eşit ağırlıklı ortalama günlük değişimi
            avg_change = float(np.mean([x['change_pct'] for x in ticker_rows])) if ticker_rows else 0.0
            return jsonify({
                'success': True,
                'as_of': pd.to_datetime(data.index[-1]).strftime('%Y-%m-%d'),
                'bist30_avg_change_pct': round(avg_change, 2),
                'tickers': full_tickers
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Canlı veri hatası: {e}'}), 500


    # --- AUTH ROUTES ---

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('home'))
    
        if request.method == 'POST':
            email = (request.form.get('email') or '').strip()
            password = request.form.get('password')
        
            user = User.get_by_email(email)
            if user and user.check_password(password):
                login_user(user)
                return redirect(url_for('home'))
            else:
                return render_template('login.html', error='E-posta veya şifre hatalı.')
    
        return render_template('login.html')


    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for('home'))
    
        if request.method == 'POST':
            email = (request.form.get('email') or '').strip().lower()
            password = request.form.get('password')
            password_confirm = request.form.get('password_confirm')
        
            if len(password) < 6:
                return render_template('register.html', error='Şifre en az 6 karakter olmalıdır.')
        
            if password != password_confirm:
                return render_template('register.html', error='Şifreler eşleşmiyor.')
        
            user = User.create(email, password)
            if user:
                login_user(user)
                return redirect(url_for('home'))
            else:
                return render_template('register.html', error='Bu e-posta adresi zaten kayıtlı.')
    
        return render_template('register.html')


    @app.route('/logout')
    def logout():
        logout_user()
        return redirect(url_for('home'))


    @app.route('/abonelik')
    def abonelik():
        """Premium abonelik / fiyatlandırma sayfası"""
        return render_template('abonelik.html')


    @app.route('/odeme')
    def odeme():
        """Ödeme sayfası (placeholder - Stripe/iyzico entegrasyonu eklenecek)"""
        plan = request.args.get('plan', 'monthly')
        return render_template('abonelik.html', plan=plan)


    # --- PORTFÖY API ROUTE'LARI ---

    @app.route('/api/portfolios', methods=['GET'])
    @login_required
    def get_portfolios():
        """Kullanıcının kayıtlı portföylerini getir"""
        from auth import get_user_portfolios
        portfolios = get_user_portfolios(current_user.id)
        return jsonify({'success': True, 'portfolios': portfolios})


    @app.route('/api/portfolios', methods=['POST'])
    @login_required
    def save_portfolio_api():
        """Yeni portföy kaydet"""
        from auth import save_portfolio
        data = request.json
    
        portfolio_name = data.get('name', 'Portföyüm')
        weights = data.get('weights', {})
    
        if not weights:
            return jsonify({'success': False, 'error': 'Portföy ağırlıkları boş olamaz'}), 400
    
        portfolio_id = save_portfolio(current_user.id, portfolio_name, weights)
        return jsonify({'success': True, 'portfolio_id': portfolio_id, 'message': 'Portföy kaydedildi!'})


    @app.route('/api/portfolios/<int:portfolio_id>', methods=['DELETE'])
    @login_required
    def delete_portfolio_api(portfolio_id):
        """Portföyü sil"""
        from auth import delete_portfolio
        deleted = delete_portfolio(current_user.id, portfolio_id)
    
        if deleted:
            return jsonify({'success': True, 'message': 'Portföy silindi'})
        else:
            return jsonify({'success': False, 'error': 'Portföy bulunamadı'}), 404


    @app.route('/api/portfolios/<int:portfolio_id>', methods=['GET'])
    @login_required
    def get_portfolio_detail(portfolio_id):
        """Belirli bir portföyün detaylarını getir"""
        from auth import get_portfolio_by_id
        portfolio = get_portfolio_by_id(current_user.id, portfolio_id)
    
        if portfolio:
            return jsonify({'success': True, 'portfolio': portfolio})
        else:
            return jsonify({'success': False, 'error': 'Portföy bulunamadı'}), 404

    @app.route('/create_portfolio', methods=['GET', 'POST'])
    @login_required
    def create_portfolio():
        """Kullanıcının yarışma için kendi 5 hisselik portföyünü oluşturma ekranı"""
        if request.method == 'POST':
            stock1 = request.form.get('stock1')
            stock2 = request.form.get('stock2')
            stock3 = request.form.get('stock3')
            stock4 = request.form.get('stock4')
            stock5 = request.form.get('stock5')
        
            weight1 = request.form.get('weight1', 20)
            weight2 = request.form.get('weight2', 20)
            weight3 = request.form.get('weight3', 20)
            weight4 = request.form.get('weight4', 20)
            weight5 = request.form.get('weight5', 20)
        
            tickers = [stock1, stock2, stock3, stock4, stock5]
        
            # Seçimlerin benzersizliğini ve boş olmadığını kontrol et
            if not all(tickers) or len(set(tickers)) != 5:
                return render_template('create_portfolio.html', 
                                       error='Lütfen BIST30 listesinden birbirinden farklı 5 hisse seçin.', 
                                       bist30=BIST30_TICKERS)
        
            try:
                raw_weights = [float(weight1), float(weight2), float(weight3), float(weight4), float(weight5)]
            except ValueError:
                return render_template('create_portfolio.html', 
                                       error='Geçersiz ağırlık değeri girdiniz.', 
                                       bist30=BIST30_TICKERS)
            
            total_weight = sum(raw_weights)
            if abs(total_weight - 100) > 1: # İzin verilen küçük sapma
                return render_template('create_portfolio.html', 
                                       error='Ağırlıkların toplamı %100 olmalıdır.', 
                                       bist30=BIST30_TICKERS)
        
            # Her hissenin ağırlığı (%100 -> 1.0 sistemine normalize ediliyor)
            weights = {tickers[i]: raw_weights[i] / total_weight for i in range(5)}
        
            # Check if saved portfolio for this week already exists and update it if so, otherwise just create new
            # save_portfolio handles updates and inserts behind scenes?
            from auth import save_portfolio
            save_portfolio(current_user.id, "Haftalık Yarışma Portföyü", weights)
        
            return redirect(url_for('leaderboard'))
        
        return render_template('create_portfolio.html', bist30=BIST30_TICKERS)


    @app.route('/leaderboard')
    def leaderboard():
        """Kullanıcı vs Yapay Zeka Sıralaması"""
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        # En son haftanın sonuçlarını al
        c.execute('''
            SELECT u.email, l.user_return, l.ai_return, l.is_winner, l.week_end_date
            FROM leaderboard l
            JOIN users u ON l.user_id = u.id
            ORDER BY l.week_end_date DESC, l.user_return DESC
            LIMIT 50
        ''')
        results = c.fetchall()
        conn.close()
    
        leaderboard_data = []
        ai_return = None
        week_end_date = None
    
        for row in results:
            email, u_ret, a_ret, is_win, w_date = row
            username = email.split('@')[0]  # Emailin ilk kısmını kullanıcı adı olarak göster
        
            if ai_return is None:
                ai_return = a_ret * 100 if a_ret else 0
                week_end_date = w_date
            
            leaderboard_data.append({
                'username': username,
                'return': u_ret * 100,
                'is_winner': is_win
            })
        
        if ai_return is None:
            ai_return = 0
        
        return render_template('leaderboard.html', 
                              leaderboard=leaderboard_data, 
                              ai_return=ai_return, 
                              week_end_date=week_end_date)


    @app.route('/optimize', methods=['POST'])
    def optimize_portfolio():
        try:
            data = request.json
            tickers = data.get('tickers') 
            if not tickers:
                return jsonify({'error': 'Tickers listesi gerekli.'}), 400
        
            max_sharpe, min_risk, graph_json = calculate_mpt(tickers)
        
            if max_sharpe is None:
                return jsonify({'error': min_risk}), 500

            return jsonify({
                'max_sharpe_portfolio': max_sharpe,
                'min_risk_portfolio': min_risk,
                'graph_json': graph_json
            })
        
        except Exception as e:
            return jsonify({'error': f'Sunucu hatası: {e}'}), 500

    @app.route('/optimize_bist30_ml', methods=['GET'])
    @login_required
    def optimize_bist30_portfolio_ml():
        try:
            # --- ERİŞİM KONTROLÜ ---
            if not current_user.is_admin and not current_user.is_premium:
                if current_user.demo_used:
                    return jsonify({
                        'error': 'DEMO_EXPIRED', 
                        'message': 'Demo kullanım hakkınız dolmuştur. Yeni analizler için Premium aboneliğe geçmelisiniz.'
                    }), 403
            
                # Demo kullanıldı olarak işaretle
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("UPDATE users SET demo_used = 1 WHERE id = ?", (current_user.id,))
                conn.commit()
                conn.close()
            # --- KRİTİK: DOĞRU SIRA ---
            ml_ret, cl_ret, cl_cov, ml_cov = get_bist30_data_inputs()
        
            ml_max_sharpe, ml_min_risk, ml_graph = run_monte_carlo_simulation(
                ml_ret, ml_cov, BIST30_TICKERS, "BIST 30 (ML Destekli)"
            )
            cl_max_sharpe, cl_min_risk, _ = run_monte_carlo_simulation(
                cl_ret, cl_cov, BIST30_TICKERS, "BIST 30 (Klasik)"
            )

            # --- NaN hatası burada çözüldü ---
            sim_graph, sim_stats = simulate_future_wealth(
                ml_max_sharpe['Getiri'], ml_max_sharpe['Risk']
            )
        
            return jsonify({
                'ml_max_sharpe': ml_max_sharpe.to_dict(),
                'ml_min_risk': ml_min_risk.to_dict(),
                'graph_json': ml_graph,
                'classic_max_sharpe': cl_max_sharpe.to_dict(),
                'classic_min_risk': cl_min_risk.to_dict(),
                'simulation_graph': sim_graph,
                'simulation_stats': sim_stats
            })
        
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Sunucu hatası: {e}'}), 500

    @app.route('/optimize_bist30_ml_with_horizon', methods=['GET'])
    @login_required
    def optimize_bist30_ml_with_horizon():
        """
        Seçilen tahmin ufku için BIST 30 ML optimizasyonu
        Query param: months (1, 3, 6, 12), top_n (3, 5, 10, 30)
        """
        try:
            months = normalize_horizon_months(request.args.get('months', 12), default=12)
            top_n_raw = request.args.get('top_n', '30')
            try:
                top_n = int(top_n_raw)
                if top_n not in (3, 5, 10, 30):
                    top_n = 30
            except (ValueError, TypeError):
                top_n = 30
        
            risk_profile = request.args.get('risk_profile', 'medium')
        
            # --- ERİŞİM KONTROLÜ ---
            if not current_user.is_admin and not current_user.is_premium:
                if months > 1:
                    return jsonify({
                        'error': 'PERMISSION_DENIED',
                        'message': '3, 6 ve 12 aylık vade tahminleri sadece Premium üyelere özeldir.'
                    }), 403
        
            # Seçilen ufuk için ML tahminleri al
            ml_ret, cl_ret, cl_cov, ml_cov, prices_df = get_bist30_ml_with_horizon(months)

            from forecasting.weekly_pipeline import (
                detect_market_regime_from_panel,
                get_portfolio_strategy_for_regime,
                apply_regime_adjustment,
            )
            from forecasting.batch_pipeline import build_regime_explanation_for_user
            returns_panel = prices_df.pct_change().dropna()
            market_regime = detect_market_regime_from_panel(prices_df, returns_panel)
            mc_strategy = get_portfolio_strategy_for_regime(market_regime)
            ml_ret = pd.Series(
                {t: apply_regime_adjustment(float(v), market_regime) for t, v in ml_ret.items()}
            )
        
            # Önce tam 30 hisse ile optimizasyon yap
            all_tickers = list(ml_ret.index)
            ml_max_full, ml_min_risk_full, ml_graph = run_monte_carlo_simulation(
                ml_ret.values, ml_cov.values, all_tickers, f"BIST 30 ML ({months} Ay)",
                months=months, strategy=mc_strategy,
            )
            cl_max_full, cl_min_risk_full, _ = run_monte_carlo_simulation(
                cl_ret.values, cl_cov.values, all_tickers, f"BIST 30 Klasik ({months} Ay)",
                months=months, strategy="max_sharpe",
            )
        
            # Hisse sayısına göre revizyon: 30 hisselik ağırlıklardan top N al, doğrudan o hisseler üzerinde yeniden optimize et
            meta = ['Getiri', 'Risk', 'SharpeOrani']
            weights_series = ml_max_full.drop(labels=meta, errors='ignore')
            if top_n < 30 and len(weights_series) > top_n:
                top_by_weight = weights_series.sort_values(ascending=False).head(top_n)
                top_tickers = list(top_by_weight.index)
            
                # Seçilen top N hisse için verileri dilimle
                ml_ret_sliced = ml_ret[top_tickers]
                ml_cov_sliced = ml_cov.loc[top_tickers, top_tickers]
                cl_ret_sliced = cl_ret[top_tickers]
                cl_cov_sliced = cl_cov.loc[top_tickers, top_tickers]
            
                # Dilimlenmiş hisseler üzerinde doğrudan optimizasyon yap (böylece dinamik kısıtlar uygulanır ve eşit ağırlık engellenir)
                ml_max_sliced, ml_min_sliced, ml_graph = run_monte_carlo_simulation(
                    ml_ret_sliced.values, ml_cov_sliced.values, top_tickers, f"BIST 30 ML ({months} Ay - {top_n} Hisse)",
                    months=months, strategy=mc_strategy,
                )
                cl_max_sliced, cl_min_sliced, _ = run_monte_carlo_simulation(
                    cl_ret_sliced.values, cl_cov_sliced.values, top_tickers, f"BIST 30 Klasik ({months} Ay - {top_n} Hisse)",
                    months=months
                )
            
                # Tüm 30 hisselik formatta boş ağırlık serileri oluştur
                ml_max_sharpe = pd.Series(0.0, index=all_tickers)
                ml_min_risk = pd.Series(0.0, index=all_tickers)
                cl_max_full = pd.Series(0.0, index=all_tickers)
                cl_min_risk = pd.Series(0.0, index=all_tickers)
            
                # Optimize edilen ağırlıkları ata
                for t in top_tickers:
                    ml_max_sharpe[t] = ml_max_sliced[t]
                    ml_min_risk[t] = ml_min_sliced[t]
                    cl_max_full[t] = cl_max_sliced[t]
                    cl_min_risk[t] = cl_min_sliced[t]
                
                # Performans metriklerini ata
                for m_key in meta:
                    ml_max_sharpe[m_key] = ml_max_sliced[m_key]
                    ml_min_risk[m_key] = ml_min_sliced[m_key]
                    cl_max_full[m_key] = cl_max_sliced[m_key]
                    cl_min_risk[m_key] = cl_min_sliced[m_key]
                
                valid_tickers = top_tickers
            else:
                ml_max_sharpe = ml_max_full
                ml_min_risk = ml_min_risk_full
                cl_max_full = cl_max_full
                cl_min_risk = cl_min_risk_full
                valid_tickers = all_tickers
        
            regime_label = market_regime.get("regime", "sideways")
            regime_info = build_regime_explanation_for_user(
                market_regime, mc_strategy, risk_profile=risk_profile,
            )

            # Seçilen risk profiline göre hedef portföyü belirle
            if risk_profile == 'low':
                selected_portfolio = ml_min_risk
            elif risk_profile == 'high':
                selected_portfolio = ml_max_sharpe
            else: # medium
                # Dengeli portföy: %50 max sharpe, %50 min risk
                selected_portfolio = (ml_max_sharpe.drop(labels=meta, errors='ignore') * 0.5) + (ml_min_risk.drop(labels=meta, errors='ignore') * 0.5)
                # Meta verileri ekle
                for m_key in meta:
                    selected_portfolio[m_key] = (ml_max_sharpe[m_key] * 0.5) + (ml_min_risk[m_key] * 0.5)

            explain_text = regime_info["full_explain_text"]
            
            # Backtest grafiğini oluştur
            backtest_graph = generate_backtest_chart(selected_portfolio.drop(labels=meta, errors='ignore'), prices_df)
        
            return jsonify({
                'ml_max_sharpe': ml_max_sharpe.to_dict(),
                'ml_min_risk': ml_min_risk.to_dict(),
                'selected_portfolio': selected_portfolio.to_dict(),
                'explain_text': explain_text,
                'backtest_graph_json': backtest_graph,
                'graph_json': ml_graph,
                'classic_max_sharpe': cl_max_full.to_dict(),
                'classic_min_risk': cl_min_risk.to_dict(),
                'horizon_months': months,
                'top_n': len(valid_tickers),
                'market_regime': regime_label,
                'optimization_strategy': mc_strategy,
                'regime_info': regime_info,
            })
        
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Sunucu hatası: {e}'}), 500
    

    @app.route('/get_single_stock_forecast', methods=['GET'])
    @login_required
    def get_single_stock_forecast():
        """
        Bireysel hisse için ML tahmini (Backtest'teki başarılı model mantığı ile)
        Veritabanındaki final_forecasts değerini kullanır (ARIMA+BL+LSTM ensemble)
        """
        ticker = request.args.get('ticker')
        months = normalize_horizon_months(request.args.get('months', 12), default=12)
        if not ticker:
            return jsonify({'error': 'Ticker gerekli.'}), 400
    
        # --- ERİŞİM KONTROLÜ: Ücretsiz hesaplar istediği 2 hisseyi ücretsiz görür ---
        if not current_user.is_admin and not current_user.is_premium:
            conn_acc = sqlite3.connect(DB_NAME)
            c_acc = conn_acc.cursor()
            c_acc.execute("SELECT free_stock_views FROM users WHERE id = ?", (current_user.id,))
            row = c_acc.fetchone()
            free_list = []
            if row and row[0]:
                try:
                    free_list = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or [])
                except (ValueError, TypeError):
                    free_list = []
            if not isinstance(free_list, list):
                free_list = []
            free_list = [t for t in free_list if t][:2]  # Max 2, temizle

            if ticker not in free_list and len(free_list) >= 2:
                conn_acc.close()
                return jsonify({
                    'error': 'PERMISSION_DENIED',
                    'message': f'{ticker} tahmini Premium üyeliğe özeldir. Ücretsiz hesaplar istediği 2 hisseyi görüntüleyebilir; 2 hisse kotanızı kullandınız.'
                }), 403

            # Yeni hisse görüntüleniyorsa listeye ekle
            if ticker not in free_list and len(free_list) < 2:
                free_list.append(ticker)
                c_acc.execute("UPDATE users SET free_stock_views = ? WHERE id = ?", (json.dumps(free_list), current_user.id))
                conn_acc.commit()
            conn_acc.close()

        try:
            from db.schema import ensure_beklenen_getiriler_horizon_schema

            conn = sqlite3.connect(DB_NAME)
            ensure_beklenen_getiriler_horizon_schema(conn)
            c = conn.cursor()
            c.execute(
                """
                SELECT tahmini_donem_getiri, tahmini_yillik_getiri, tahmini_yillik_risk, tahmini_bl_getiri
                FROM beklenen_getiriler
                WHERE ticker = ? AND horizon_ay = ?
                """,
                (ticker, months),
            )
            result = c.fetchone()
            if result is None:
                c.execute(
                    """
                    SELECT tahmini_donem_getiri, tahmini_yillik_getiri, tahmini_yillik_risk, tahmini_bl_getiri
                    FROM beklenen_getiriler
                    WHERE ticker = ?
                    ORDER BY horizon_ay DESC
                    LIMIT 1
                    """,
                    (ticker,),
                )
                result = c.fetchone()
            conn.close()
        
            if result:
                period_getiri = result[0] if result[0] is not None else None
                yillik_getiri = result[1] if result[1] is not None else 0
                risk = result[2] if result[2] is not None else 0
                bl_getiri = result[3] if len(result) > 3 and result[3] is not None else None

                if period_getiri is not None and not np.isnan(period_getiri):
                    final_forecast = float(period_getiri)
                else:
                    from portfolio.optimizer import adjust_returns_for_horizon
                    final_forecast = float(adjust_returns_for_horizon(yillik_getiri, months))
            
                # Sharpe oranı hesapla (kullanıcı için bilgilendirme)
                period_risk = risk * np.sqrt(months / 12.0) if risk else 0
                sharpe_ratio = final_forecast / period_risk if period_risk and period_risk > 0 else 0

                from forecasting.weekly_pipeline import calculate_up_probability

                probability_up = calculate_up_probability(final_forecast, period_risk if period_risk else 0.15)

                # Haftalık tahmin varsa onu da göster (daha güncel olasılık)
                weekly_prob = None
                try:
                    conn2 = sqlite3.connect(DB_NAME)
                    c2 = conn2.cursor()
                    c2.execute(
                        """
                        SELECT yukselis_olasiligi, tahmini_getiri
                        FROM tahminler
                        WHERE ticker = ?
                        ORDER BY tahmin_tarihi DESC
                        LIMIT 1
                        """,
                        (ticker,),
                    )
                    row2 = c2.fetchone()
                    conn2.close()
                    if row2 and row2[0] is not None:
                        weekly_prob = round(float(row2[0]) * 100, 1)
                except Exception:
                    pass
            
                return jsonify({
                    'ticker': ticker,
                    'forecast': final_forecast,
                    'risk_forecast': period_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'yillik_getiri': yillik_getiri,
                    'bl_forecast': bl_getiri,
                    'probability_up': round(probability_up * 100, 1),
                    'weekly_probability_up': weekly_prob,
                    'horizon_months': months,
                    'horizon': f'{months} Ay',
                    'description': (
                        f'{months} aylik beklenen getiri tahmini. Yükseliş olasılığı: %{probability_up*100:.0f}. '
                        f'Sharpe Oranı: {sharpe_ratio:.2f}'
                    ),
                })
            else:
                return jsonify({'error': f'{ticker} için ML verisi bulunamadı. Lütfen önce batch_job.py çalıştırın.'}), 404
            
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Veritabanı hatası: {e}'}), 500

    # --- MODEL PERFORMANS ENDPOINT'LERİ ---

    @app.route('/get_performance', methods=['GET'])
    def get_performance():
        """
        Model Performansı sayfası için metrikleri getirir.
    
        Query params:
            ticker: Hisse sembolü (opsiyonel, yoksa tüm hisseler)
            start_date: Başlangıç tarihi (opsiyonel)
            end_date: Bitiş tarihi (opsiyonel)
        """
        try:
            from performance_tracker import get_performance_metrics, setup_performance_tables
        
            # Tabloları oluştur (yoksa)
            setup_performance_tables()
        
            ticker = request.args.get('ticker')
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
        
            if ticker:
                # Tek hisse için metrikler
                metrics = get_performance_metrics(ticker, start_date, end_date)
                return jsonify({
                    'ticker': ticker,
                    'metrics': metrics
                })
            else:
                # Tüm hisseler için özet
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("SELECT DISTINCT ticker FROM tahminler ORDER BY ticker")
                tickers = [row[0] for row in c.fetchall()]
                conn.close()
            
                all_metrics = {}
                for t in tickers:
                    metrics = get_performance_metrics(t, start_date, end_date)
                    if metrics['valid_forecasts'] > 0:
                        all_metrics[t] = metrics
            
                return jsonify({
                    'all_tickers': all_metrics,
                    'total_tickers': len(all_metrics)
                })
            
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Performans metrikleri alınamadı: {e}'}), 500

    @app.route('/update_performance', methods=['POST'])
    def update_performance():
        """
        Model Performansı - "Güncelle" butonu.
        Horizon dolmuş tahminler için gerçekleşen getirileri hesaplar (backfill).
    
        Body:
            ticker: Hisse sembolü (opsiyonel, yoksa tüm hisseler)
            horizon_days: Tahmin ufku (varsayılan: 5 gün)
        """
        try:
            from performance_tracker import backfill_actuals_for_ticker, setup_performance_tables, check_horizon_status
        
            # Tabloları oluştur (yoksa)
            setup_performance_tables()
        
            data = request.json or {}
            ticker = data.get('ticker')
            horizon_days = data.get('horizon_days', 5)
            lookback_days = data.get('lookback_days', 30)
        
            if ticker:
                # Tek hisse için backfill
                updated_count = backfill_actuals_for_ticker(ticker, horizon_days, lookback_days)
                return jsonify({
                    'ticker': ticker,
                    'updated_forecasts': updated_count,
                    'message': f'{ticker} için {updated_count} tahmin güncellendi.'
                })
            else:
                # Tüm hisseler için backfill
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("SELECT DISTINCT ticker FROM tahminler ORDER BY ticker")
                tickers = [row[0] for row in c.fetchall()]
                conn.close()
            
                total_updated = 0
                results = {}
                for t in tickers:
                    count = backfill_actuals_for_ticker(t, horizon_days, lookback_days)
                    if count > 0:
                        results[t] = count
                        total_updated += count
            
                return jsonify({
                    'total_updated': total_updated,
                    'ticker_results': results,
                    'message': f'Toplam {total_updated} tahmin güncellendi ({len(results)} hisse).'
                })
            
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Performans güncellenemedi: {e}'}), 500

    @app.route('/check_horizon', methods=['GET'])
    def check_horizon():
        """
        Horizon durumunu kontrol et (tahminin doğrulanabilir olup olmadığını gösterir).
    
        Query params:
            ticker: Hisse sembolü
            forecast_date: Tahmin tarihi (ISO format)
            horizon_days: Tahmin ufku (varsayılan: 5 gün)
        """
        try:
            from performance_tracker import check_horizon_status
        
            ticker = request.args.get('ticker')
            forecast_date = request.args.get('forecast_date')
            horizon_days = int(request.args.get('horizon_days', 5))
        
            if not ticker or not forecast_date:
                return jsonify({'error': 'ticker ve forecast_date gerekli.'}), 400
        
            status = check_horizon_status(ticker, forecast_date, horizon_days)
            return jsonify(status)
        
        except Exception as e:
            return jsonify({'error': f'Horizon kontrolü yapılamadı: {e}'}), 500


    # --- KARŞILAŞTIRMA: MPT vs AI/ML ---
    @app.route('/compare')
    def compare_page():
        """Karşılaştırma sayfası"""
        return render_template('compare.html', bist30_list=BIST30_TICKERS)


    @app.route('/api/compare_methods', methods=['POST'])
    def compare_methods():
        """
        Klasik MPT ile AI/ML sistemini karşılaştır
    
        Returns:
            JSON: {
                'mpt': {...},      # Klasik MPT sonuçları
                'aiml': {...},     # AI/ML sistemi sonuçları
                'comparison': {...} # Karşılaştırma metrikleri
            }
        """
        try:
            data = request.json
            months = normalize_horizon_months(data.get('months', 12))
        
            # 1. Klasik MPT Optimizasyonu
            print("[*] Klasik MPT hesaplaniyor...")
            max_sharpe_mpt, min_risk_mpt, _ = calculate_mpt(BIST30_TICKERS)
        
            if max_sharpe_mpt is None:
                return jsonify({'error': 'Klasik MPT hesaplanamadı'}), 500
        
            # 2. AI/ML Sistemi Optimizasyonu
            print("[*] AI/ML sistemi hesaplaniyor...")
            ml_ret, cl_ret, cl_cov, ml_cov = get_bist30_ml_with_horizon(months=months)
        
            # AI/ML için Monte Carlo
            ml_max_sharpe, ml_min_risk, _ = run_monte_carlo_simulation(
                ml_ret, ml_cov, BIST30_TICKERS, 
                f"AI/ML Portfolio Optimization ({months} Months)",
                months=months
            )
        
            # 3. MPT sonuçlarını standart formata çevir
            mpt_formatted = {
                'return': max_sharpe_mpt.get('Getiri', 0),
                'risk': max_sharpe_mpt.get('Risk', 0),
                'sharpe': max_sharpe_mpt.get('SharpeOrani', 0),
                'weights': {k: v for k, v in max_sharpe_mpt.items() 
                           if k not in ['Getiri', 'Risk', 'SharpeOrani']}
            }
        
            # AI/ML sonuçlarını da standart formata çevir (Series ise)
            if isinstance(ml_max_sharpe, pd.Series):
                aiml_formatted = {
                    'return': ml_max_sharpe.get('Getiri', 0),
                    'risk': ml_max_sharpe.get('Risk', 0),
                    'sharpe': ml_max_sharpe.get('SharpeOrani', 0),
                    'weights': {k: v for k, v in ml_max_sharpe.items() 
                               if k not in ['Getiri', 'Risk', 'SharpeOrani']}
                }
            else:
                aiml_formatted = ml_max_sharpe
        
            # Min risk portföylerini de formatlayalım
            if isinstance(min_risk_mpt, pd.Series):
                mpt_min_risk_formatted = {
                    'return': min_risk_mpt.get('Getiri', 0),
                    'risk': min_risk_mpt.get('Risk', 0),
                    'sharpe': min_risk_mpt.get('SharpeOrani', 0),
                    'weights': {k: v for k, v in min_risk_mpt.items() 
                               if k not in ['Getiri', 'Risk', 'SharpeOrani']}
                }
            else:
                mpt_min_risk_formatted = min_risk_mpt
        
            if isinstance(ml_min_risk, pd.Series):
                aiml_min_risk_formatted = {
                    'return': ml_min_risk.get('Getiri', 0),
                    'risk': ml_min_risk.get('Risk', 0),
                    'sharpe': ml_min_risk.get('SharpeOrani', 0),
                    'weights': {k: v for k, v in ml_min_risk.items() 
                               if k not in ['Getiri', 'Risk', 'SharpeOrani']}
                }
            else:
                aiml_min_risk_formatted = ml_min_risk
        
            # 4. Karşılaştırma Metrikleri
            comparison = {
                'return_improvement': round((aiml_formatted['return'] - mpt_formatted['return']) * 100, 2),
                'risk_improvement': round((mpt_formatted['risk'] - aiml_formatted['risk']) * 100, 2),
                'sharpe_improvement': round((aiml_formatted['sharpe'] - mpt_formatted['sharpe']) * 100, 2),
                'mpt_method': 'Geçmiş Ortalama + Markowitz',
                'aiml_method': 'LSTM+ARIMA Ensemble + Black-Litterman',
                'mpt_features': 'Sadece getiri ve risk',
                'aiml_features': '51 teknik gösterge + AI tahminleri'
            }
        
            # 5. Sonuçları döndür
            return jsonify({
                'success': True,
                'mpt': {
                    'name': 'Klasik MPT (Markowitz)',
                    'max_sharpe': mpt_formatted,
                    'min_risk': mpt_min_risk_formatted,
                    'description': 'Geçmiş veri bazlı optimizasyon'
                },
                'aiml': {
                    'name': 'Gelişmiş AI/ML Sistemi',
                    'max_sharpe': aiml_formatted,
                    'min_risk': aiml_min_risk_formatted,
                    'description': 'Deep Learning + Black-Litterman optimizasyon'
                },
                'comparison': comparison,
                'horizon_months': months
            })
        
        except Exception as e:
            print(f"[!] Karşılaştırma hatası: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Karşılaştırma yapılamadı: {str(e)}'}), 500


    # --- TAHMİN PERFORMANS ÖZETİ ---
    @app.route('/forecast_summary')
    def forecast_summary_page():
        """Tahmin performans özeti sayfası (Bireysel hisse - diagnostic)"""
        return render_template('forecast_summary.html', bist30_list=BIST30_TICKERS)

    @app.route('/portfolio_performance')
    def portfolio_performance_page():
        """Portföy seviyesinde performans takibi sayfası"""
        return render_template('portfolio_performance.html', bist30_list=BIST30_TICKERS)

    @app.route('/api/forecast_summary', methods=['GET'])
    def get_forecast_summary():
        """
        Model tahminlerinin basit özeti - Herkesin anlayabileceği şekilde
        OTOMATİK BACKFILL YAPILIR!
        """
        try:
            # ÖNCE OTOMATİK BACKFILL YAP (Portföy sayfası gibi!)
            from performance_tracker import backfill_actuals_for_ticker, setup_performance_tables

            # Şema migrasyonu (yukselis_olasiligi kolonu eski DB'lerde olmayabilir)
            setup_performance_tables()

            conn_check = sqlite3.connect(DB_NAME)
            c = conn_check.cursor()
            c.execute("SELECT DISTINCT ticker FROM tahminler WHERE gerceklesen_getiri IS NULL")
            pending_tickers = [row[0] for row in c.fetchall()]
            conn_check.close()
        
            updated_count = 0
            if pending_tickers:
                print(f"[Forecast Summary] {len(pending_tickers)} hisse için backfill yapılıyor...")
                for ticker in pending_tickers:
                    try:
                        count = backfill_actuals_for_ticker(ticker, horizon_days=5, lookback_days=30)
                        if count > 0:
                            updated_count += count
                            print(f"[Forecast Summary] {ticker}: {count} tahmin güncellendi")
                    except Exception as e:
                        print(f"[Forecast Summary] {ticker} backfill hatası: {e}")
                        continue
            
                if updated_count > 0:
                    print(f"[Forecast Summary] Toplam {updated_count} bireysel hisse tahmini güncellendi")
        
            # --- EKLENEN ÖZELLİK: OTOMATİK PORTFÖY TAHMİNİ OLUŞTURMA (FIX #2) ---
            # Eğer bu hafta için hisse tahminleri varsa ama portföy tahmini YOKSA, otomatik oluştur.
            try:
                from datetime import datetime, timedelta
                import weekly_forecast
                from portfolio_performance_tracker import calculate_portfolio_forecast, save_portfolio_forecast
            
                # 1. Bu hafta için güncel tahmin var mı?
                conn_pro = sqlite3.connect(DB_NAME)
                c_pro = conn_pro.cursor()
                # Bu haftanın pazartesisi veya sonrası için tahmin
                today = datetime.now()
                start_of_week = (today - timedelta(days=today.weekday())).strftime('%Y-%m-%d')
            
                c_pro.execute("SELECT count(*) FROM tahminler WHERE tahmin_tarihi >= ?", (start_of_week,))
                forecast_count = c_pro.fetchone()[0]
            
                # 2. Bu hafta için portföy tahmini var mı?
                c_pro.execute("SELECT count(*) FROM portfoy_performans WHERE tahmin_tarihi >= ?", (start_of_week,))
                portfolio_count = c_pro.fetchone()[0]
                conn_pro.close()
            
                if forecast_count > 0 and portfolio_count == 0:
                    print(f"[Auto-Portfolio] Hisse tahminleri var ({forecast_count}) ama portföy yok. Oluşturuluyor...")
                
                    # Tahminleri çek
                    forecasts = weekly_forecast.forecast_weekly_returns()
                
                    if forecasts:
                        # Portföyü hesapla
                        portfolio_forecast = calculate_portfolio_forecast(
                            datetime.now(), 
                            horizon_days=5, 
                            forecasts_dict=forecasts
                        )
                    
                        if portfolio_forecast:
                            save_portfolio_forecast(
                                datetime.now(),
                                horizon_days=5,
                                predicted_return=portfolio_forecast['predicted_return'],
                                weights_dict=portfolio_forecast['weights'],
                                var_95=portfolio_forecast.get('var_95'),
                                portfolio_vol=portfolio_forecast.get('portfolio_vol'),
                                probability_up=portfolio_forecast.get('probability_up'),
                            )
                            print(f"[Auto-Portfolio] BAŞARILI: Portföy tahmini karneye eklendi. Getiri: %{portfolio_forecast['predicted_return']*100:.2f}")
            except Exception as e:
                print(f"[Auto-Portfolio] HATA: {e}")
                # Ana akışı bozma
                pass
        
            # ŞİMDİ VERİLERİ GETİR
            conn = sqlite3.connect(DB_NAME)
        
            # Tüm tahminleri ve gerçekleşen değerleri çek
            query = """
                SELECT ticker, tahmin_tarihi, tahmini_getiri, gerceklesen_getiri, tahmin_hatasi, yukselis_olasiligi
                FROM tahminler
                ORDER BY ticker, tahmin_tarihi DESC
            """
        
            df_all = pd.read_sql_query(query, conn)
            conn.close()
        
            df = df_all[df_all['gerceklesen_getiri'].notna()]
            df_pending = df_all[df_all['gerceklesen_getiri'].isna()]
        
            if df.empty:
                return jsonify({
                    'success': False,
                    'message': 'Henüz gerçekleşen değer yok. Lütfen önce haftalık tahmin yapın (weekly_forecast.py) ve 5 iş günü bekleyin.',
                    'summary': {},
                    'details': {},
                    'updated': updated_count
                })
        
            # Her hisse için özet
            summary = {}
            details = {}
        
            for ticker in df['ticker'].unique():
                ticker_data = df[df['ticker'] == ticker].copy()
            
                # DÜZELTME: Her hisse için EN SON 3 TAHMİNİ al
                ticker_data = ticker_data.sort_values('tahmin_tarihi', ascending=False).head(3).copy()
            
                # Yüzdelere çevir
                ticker_data['tahmin_yuzde'] = ticker_data['tahmini_getiri'] * 100
                ticker_data['gercek_yuzde'] = ticker_data['gerceklesen_getiri'] * 100
            
                # Yön doğruluğu
                ticker_data['tahmin_yukari'] = ticker_data['tahmin_yuzde'] > 0
                ticker_data['gercek_yukari'] = ticker_data['gercek_yuzde'] > 0
                ticker_data['yon_dogru'] = ticker_data['tahmin_yukari'] == ticker_data['gercek_yukari']
            
                dogru_sayisi = ticker_data['yon_dogru'].sum()
                toplam_sayisi = len(ticker_data)
                basari_orani = (dogru_sayisi / toplam_sayisi * 100) if toplam_sayisi > 0 else 0
                ortalama_hata = ticker_data['tahmin_hatasi'].abs().mean() * 100
            
                # En güncel tahminin yükseliş olasılığı (bekleyen varsa o, yoksa son gerçekleşen)
                ticker_all = df_all[df_all['ticker'] == ticker].sort_values('tahmin_tarihi', ascending=False)
                latest_prob = None
                if not ticker_all.empty:
                    prob_val = ticker_all.iloc[0].get('yukselis_olasiligi')
                    if prob_val is not None and pd.notna(prob_val):
                        latest_prob = round(float(prob_val) * 100, 1)

                summary[ticker] = {
                    'total_forecasts': int(toplam_sayisi),
                    'success_rate': round(basari_orani, 1),
                    'correct_predictions': int(dogru_sayisi),
                    'avg_error': round(ortalama_hata, 2),
                    'probability_up': latest_prob
                }
            
                # Detaylı tahminler (son 3 tahmin, en yeni önce)
                details[ticker] = []
            
                # Bekleyenleri ekle
                ticker_pending = df_pending[df_pending['ticker'] == ticker].sort_values('tahmin_tarihi', ascending=False)
                for _, row in ticker_pending.iterrows():
                    prob_val = row.get('yukselis_olasiligi')
                    details[ticker].append({
                        'date': pd.to_datetime(row['tahmin_tarihi']).strftime('%Y-%m-%d'),
                        'predicted': round(row['tahmini_getiri'] * 100, 2),
                        'actual': None,
                        'error': None,
                        'direction_correct': None,
                        'probability_up': round(float(prob_val) * 100, 1) if prob_val is not None and pd.notna(prob_val) else None,
                        'status': 'pending'
                    })
            
                for _, row in ticker_data.iterrows():
                    details[ticker].append({
                        'date': pd.to_datetime(row['tahmin_tarihi']).strftime('%Y-%m-%d'),
                        'predicted': round(row['tahmin_yuzde'], 2),
                        'actual': round(row['gercek_yuzde'], 2),
                        'error': round(abs(row['tahmin_yuzde'] - row['gercek_yuzde']), 2),
                        'direction_correct': bool(row['yon_dogru']),
                        'status': 'completed'
                    })
        
            # Genel özet - HER HİSSE İÇİN SON 3 TAHMİN
            # Her hisse için en son 3 tahmini al ve birleştir
            df_filtered_list = []
            for ticker in df['ticker'].unique():
                ticker_data = df[df['ticker'] == ticker].copy()
                # En son 3 tahmin
                ticker_recent = ticker_data.sort_values('tahmin_tarihi', ascending=False).head(3)
                df_filtered_list.append(ticker_recent)
        
            df_filtered = pd.concat(df_filtered_list, ignore_index=True) if df_filtered_list else pd.DataFrame()
        
            # Genel özet hesapla (son 3 tahmin üzerinden)
            df_filtered['tahmin_yuzde'] = df_filtered['tahmini_getiri'] * 100
            df_filtered['gercek_yuzde'] = df_filtered['gerceklesen_getiri'] * 100
            df_filtered['tahmin_yukari'] = df_filtered['tahmin_yuzde'] > 0
            df_filtered['gercek_yukari'] = df_filtered['gercek_yuzde'] > 0
            df_filtered['yon_dogru'] = df_filtered['tahmin_yukari'] == df_filtered['gercek_yukari']
        
            toplam_tahmin = len(df_filtered)
            toplam_dogru = df_filtered['yon_dogru'].sum()
            genel_basari = (toplam_dogru / toplam_tahmin * 100) if toplam_tahmin > 0 else 0
            genel_hata = df_filtered['tahmin_hatasi'].abs().mean() * 100
        
            # GEÇEN HAFTA (SON TAHMİN) PERFORMANSI
            # Her hisse için en son tahmini al
            df_last_week_list = []
            for ticker in df['ticker'].unique():
                ticker_data = df[df['ticker'] == ticker].copy()
                # En son tahmin
                ticker_last = ticker_data.sort_values('tahmin_tarihi', ascending=False).head(1)
                if len(ticker_last) > 0:
                    df_last_week_list.append(ticker_last)
        
            df_last_week = pd.concat(df_last_week_list, ignore_index=True) if df_last_week_list else pd.DataFrame()
        
            if len(df_last_week) > 0:
                df_last_week['tahmin_yuzde'] = df_last_week['tahmini_getiri'] * 100
                df_last_week['gercek_yuzde'] = df_last_week['gerceklesen_getiri'] * 100
                df_last_week['tahmin_yukari'] = df_last_week['tahmin_yuzde'] > 0
                df_last_week['gercek_yukari'] = df_last_week['gercek_yuzde'] > 0
                df_last_week['yon_dogru'] = df_last_week['tahmin_yukari'] == df_last_week['gercek_yukari']
            
                last_week_tahmin = len(df_last_week)
                last_week_dogru = df_last_week['yon_dogru'].sum()
                last_week_basari = (last_week_dogru / last_week_tahmin * 100) if last_week_tahmin > 0 else 0
                last_week_hata = df_last_week['tahmin_hatasi'].abs().mean() * 100
            else:
                last_week_tahmin = 0
                last_week_dogru = 0
                last_week_basari = 0
                last_week_hata = 0
        
            return jsonify({
                'success': True,
                'updated': updated_count,
                'general': {
                    'total_forecasts': int(toplam_tahmin),
                    'overall_success_rate': round(genel_basari, 1),
                    'correct_predictions': int(toplam_dogru),
                    'avg_error': round(genel_hata, 2),
                    # Geçen hafta (son tahmin) metrikleri
                    'last_week_total': int(last_week_tahmin),
                    'last_week_success_rate': round(last_week_basari, 1),
                    'last_week_correct': int(last_week_dogru),
                    'last_week_avg_error': round(last_week_hata, 2) if last_week_tahmin > 0 else 0
                },
                'summary': summary,
                'details': details
            })
        
        except Exception as e:
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/portfolio_performance', methods=['GET'])
    def get_portfolio_performance():
        """
        Portföy seviyesinde performans metrikleri.
        Giriş zorunlu. Ücretsiz kullanıcılar sadece en güncel 1 haftayı görür.
        """
        if not current_user.is_authenticated:
            return jsonify({'success': False, 'error': 'UNAUTHORIZED', 'message': 'Haftalık tahminleri görmek için giriş yapın.'}), 401
        try:
            from portfolio_performance_tracker import (
                setup_portfolio_performance_table,
                backfill_portfolio_performance
            )
        
            setup_portfolio_performance_table()
        
            # Backfill yap (eksik verileri doldur)
            updated = backfill_portfolio_performance(lookback_days=30)
        
            # Verileri çek (hem gerçekleşen hem de henüz gerçekleşmeyen tahminleri göster)
            conn = sqlite3.connect(DB_NAME)
            query = """
                SELECT tahmin_tarihi, tahmin_horizon, portfoy_tahmini_getiri, 
                       portfoy_gerceklesen_getiri, portfoy_hatasi, portfoy_agirliklari,
                       benchmark_getiri, alpha, var_95, portfoy_vol, yukselis_olasiligi
                FROM portfoy_performans
                ORDER BY tahmin_tarihi DESC
            """
        
            df = pd.read_sql_query(query, conn)
            conn.close()
        
            if df.empty:
                return jsonify({
                    'success': False,
                    'message': 'Henüz portföy performans verisi yok. Lütfen önce haftalık tahmin yapın.',
                    'updated': updated
                })
        
            # Gerçekleşen değeri olan ve olmayan kayıtları ayır
            df_completed = df[df['portfoy_gerceklesen_getiri'].notna()]
            df_pending = df[df['portfoy_gerceklesen_getiri'].isna()]
        
            # Genel istatistikler (sadece gerçekleşen değerleri olanlar için)
            if len(df_completed) > 0:
                df_completed['tahmin_yuzde'] = df_completed['portfoy_tahmini_getiri'] * 100
                df_completed['gercek_yuzde'] = df_completed['portfoy_gerceklesen_getiri'] * 100
                df_completed['piyasa_yonu_dogru'] = (
                    np.sign(df_completed['portfoy_gerceklesen_getiri']) == np.sign(df_completed['benchmark_getiri'])
                )
                df_completed['benchmark_gecildi'] = df_completed['alpha'] > 0
                # Yön doğruluğunu benchmark bazlı hesapla:
                # predicted_alpha = tahmin - benchmark, actual_alpha = gerçekleşen - benchmark
                # Eğer benchmark verisi yoksa eski mutlak yön kıyası fallback olarak kullanılır.
                df_completed['predicted_alpha'] = df_completed['portfoy_tahmini_getiri'] - df_completed['benchmark_getiri']
                df_completed['actual_alpha'] = df_completed['portfoy_gerceklesen_getiri'] - df_completed['benchmark_getiri']
                benchmark_available = df_completed['benchmark_getiri'].notna()
                df_completed['yon_dogru'] = np.where(
                    benchmark_available,
                    np.sign(df_completed['predicted_alpha']) == np.sign(df_completed['actual_alpha']),
                    (df_completed['portfoy_tahmini_getiri'] > 0) == (df_completed['portfoy_gerceklesen_getiri'] > 0)
                )
                # Kullanıcıya gösterilecek ana metrik: benchmark'ı geçme oranı (alpha > 0)
                beat_series = df_completed['benchmark_gecildi'].dropna()
                benchmark_beat_rate = float(beat_series.mean() * 100) if len(beat_series) > 0 else 0.0
                market_dir_series = df_completed.loc[df_completed['benchmark_getiri'].notna(), 'piyasa_yonu_dogru']
                market_direction_rate = float(market_dir_series.mean() * 100) if len(market_dir_series) > 0 else 0.0
                alpha_dir_series = df_completed.loc[df_completed['benchmark_getiri'].notna(), 'yon_dogru']
                alpha_direction_rate = float(alpha_dir_series.mean() * 100) if len(alpha_dir_series) > 0 else 0.0
            
                toplam_tahmin = len(df_completed)
                toplam_dogru = df_completed['yon_dogru'].sum()
                genel_basari = (toplam_dogru / toplam_tahmin * 100) if toplam_tahmin > 0 else 0
                genel_hata = df_completed['portfoy_hatasi'].abs().mean() * 100
            
                # Alpha (benchmark'a göre)
                avg_alpha = df_completed['alpha'].mean() * 100 if 'alpha' in df_completed.columns and df_completed['alpha'].notna().any() else None

                # Kümülatif Portföy - Benchmark Farkı (19.12.2025'ten itibaren)
                baslangic = pd.Timestamp('2025-12-19')
                df_from_start = df_completed[pd.to_datetime(df_completed['tahmin_tarihi']).dt.date >= baslangic.date()]
                if len(df_from_start) > 0 and 'alpha' in df_from_start.columns and df_from_start['alpha'].notna().any():
                    cumulative_diff = float((df_from_start['alpha'] * 100).sum())
                else:
                    cumulative_diff = None
            else:
                # Henüz gerçekleşen değer yok
                toplam_tahmin = 0
                toplam_dogru = 0
                genel_basari = 0
                benchmark_beat_rate = 0
                market_direction_rate = 0
                alpha_direction_rate = 0
                genel_hata = 0
                avg_alpha = None
                cumulative_diff = None
        
            # Detaylar (tüm kayıtlar - hem gerçekleşen hem bekleyen)
            details = []
        
            def _risk_fields(row):
                var_val = row.get('var_95') if hasattr(row, 'get') else None
                vol_val = row.get('portfoy_vol') if hasattr(row, 'get') else None
                prob_val = row.get('yukselis_olasiligi') if hasattr(row, 'get') else None
                return {
                    'var_95': round(float(var_val) * 100, 2) if var_val is not None and pd.notna(var_val) else None,
                    'portfolio_vol': round(float(vol_val) * 100, 2) if vol_val is not None and pd.notna(vol_val) else None,
                    'probability_up': round(float(prob_val) * 100, 1) if prob_val is not None and pd.notna(prob_val) else None,
                }

            # Önce gerçekleşen değerleri olanlar
            if len(df_completed) > 0:
                for _, row in df_completed.iterrows():
                    risk = _risk_fields(row)
                    details.append({
                        'date': pd.to_datetime(row['tahmin_tarihi']).strftime('%Y-%m-%d'),
                        'predicted': round(row['portfoy_tahmini_getiri'] * 100, 2),
                        'actual': round(row['portfoy_gerceklesen_getiri'] * 100, 2),
                        'error': round(row['portfoy_hatasi'] * 100, 2) if pd.notna(row['portfoy_hatasi']) else None,
                        'direction_correct': bool(row['yon_dogru']),
                        'market_direction_correct': bool(row['piyasa_yonu_dogru']) if pd.notna(row['benchmark_getiri']) else None,
                        'beat_benchmark': bool(row['benchmark_gecildi']) if pd.notna(row['alpha']) else None,
                        'benchmark': round(row['benchmark_getiri'] * 100, 2) if pd.notna(row['benchmark_getiri']) else None,
                        'alpha': round(row['alpha'] * 100, 2) if pd.notna(row['alpha']) else None,
                        'weights': json.loads(row['portfoy_agirliklari']) if row['portfoy_agirliklari'] else None,
                        'status': 'completed',
                        **risk,
                    })
        
            # Sonra bekleyen tahminler (henüz gerçekleşmemiş)
            for _, row in df_pending.iterrows():
                risk = _risk_fields(row)
                details.append({
                    'date': pd.to_datetime(row['tahmin_tarihi']).strftime('%Y-%m-%d'),
                    'predicted': round(row['portfoy_tahmini_getiri'] * 100, 2),
                    'actual': None,
                    'error': None,
                    'direction_correct': None,
                    'market_direction_correct': None,
                    'beat_benchmark': None,
                    'benchmark': None,
                    'alpha': None,
                    'weights': json.loads(row['portfoy_agirliklari']) if row['portfoy_agirliklari'] else None,
                    'status': 'pending',
                    'horizon': int(row['tahmin_horizon']),
                    **risk,
                })
        
            # Haftalık bülten için: GELECEK haftanın tahmini (pending) öncelikli
            # Pending = henüz gerçekleşmemiş = gelecek hafta tahmini; Completed = geçmiş hafta gerçekleşmiş
            pending_list = [d for d in details if d['status'] == 'pending']
            latest_for_bulletin = max(pending_list, key=lambda x: x['date']) if pending_list else (max(details, key=lambda x: x['date']) if details else None)
        
            is_premium_user = current_user.is_authenticated and (current_user.is_premium or current_user.is_admin)
        
            return jsonify({
                'success': True,
                'updated': updated,
                'is_premium_user': is_premium_user,
                'latest_portfolio': latest_for_bulletin,
                'general': {
                    'total_forecasts': int(toplam_tahmin),
                    'overall_success_rate': round(benchmark_beat_rate, 1),
                    'correct_predictions': int(toplam_dogru),
                    'benchmark_beats': int(beat_series.sum()) if len(df_completed) > 0 else 0,
                    'market_direction_rate': round(market_direction_rate, 1),
                    'alpha_direction_rate': round(alpha_direction_rate, 1),
                    'avg_error': round(genel_hata, 2),
                    'avg_alpha': round(avg_alpha, 2) if avg_alpha is not None else None,
                    'cumulative_diff': round(cumulative_diff, 2) if cumulative_diff is not None else None
                },
                'details': details
            })
        
        except Exception as e:
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500

    from crash_predictor import get_crash_probability
    from ipo_predictor import (
        get_active_ipos,
        get_ipo_history,
        get_ipo_user_tasks,
        load_upcoming_ipos,
        predict_ipo_performance,
        run_ipo_backtest,
        get_ipo_model_dashboard,
        validate_ipo_history,
    )
    import json
    import os
    from config import BASE_DIR

    def _ipo_data_updated_at(filename):
        path = os.path.join(BASE_DIR, "data", filename)
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("updated_at")
        except OSError:
            return None

    @app.route('/api/crash_radar', methods=['GET'])
    def api_crash_radar():
        """BIST100 çöküş ihtimalini döndürür."""
        prob, level, msg = get_crash_probability()
        return jsonify({
            'probability': prob,
            'level': level,
            'message': msg
        })

    @app.route('/api/macro_dashboard', methods=['GET'])
    def api_macro_dashboard():
        """Makro stres skoru ve kur/endeks metriklerini döndürür."""
        try:
            from macro_features import get_macro_dashboard
            return jsonify(get_macro_dashboard())
        except Exception as e:
            return jsonify({
                'enabled': False,
                'available': False,
                'message': f'Makro verisi alınamadı: {e}',
            }), 503

    @app.route('/api/upcoming_ipos', methods=['GET'])
    def api_upcoming_ipos():
        """Yaklaşan halka arz listesi (tam parametrelerle)."""
        return jsonify({
            'ipos': get_active_ipos(),
            'updated_at': _ipo_data_updated_at("upcoming_ipos.json"),
        })

    @app.route('/api/active_ipos', methods=['GET'])
    def api_active_ipos():
        """Geriye uyumluluk — yaklaşan halka arz listesi."""
        return jsonify(get_active_ipos())

    @app.route('/api/ipo_history', methods=['GET'])
    def api_ipo_history():
        """Geçmiş BIST halka arz referans verisi."""
        sector = request.args.get('sector')
        limit = request.args.get('limit', 40, type=int)
        return jsonify({
            'records': get_ipo_history(sector=sector or None, limit=min(limit, 100)),
            'updated_at': _ipo_data_updated_at("ipo_history.json"),
        })

    @app.route('/api/predict_ipo', methods=['POST'])
    def api_predict_ipo():
        """Halka arz tavan süresi tahmini."""
        data = request.json
        if not data:
            return jsonify({"error": "Veri gönderilmedi"}), 400

        ipo_id = data.get('ipo_id')
        size_mln = data.get('size_mln')
        inst_ratio = data.get('inst_ratio')
        discount = data.get('discount')
        sector = data.get('sector')

        result = predict_ipo_performance(
            ipo_id=ipo_id,
            size_mln=size_mln,
            inst_ratio=inst_ratio,
            discount=discount,
            sector=sector,
        )
        if "error" in result and not result.get("success"):
            return jsonify(result), 400

        return jsonify(result)

    @app.route('/api/ipo_backtest', methods=['GET'])
    def api_ipo_backtest():
        """Geçmiş veri doğrulama + leave-one-out backtest + UI açıklamaları."""
        return jsonify(get_ipo_model_dashboard())

    @app.route('/api/ipo_validate', methods=['GET'])
    def api_ipo_validate():
        """Geçmiş halka arz kayıt bütünlük kontrolü."""
        return jsonify(validate_ipo_history())

    @app.route('/api/ipo_user_tasks', methods=['GET'])
    def api_ipo_user_tasks():
        """Kullanıcının manuel tamamlayabileceği IPO görevleri."""
        return jsonify(get_ipo_user_tasks())

