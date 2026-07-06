@echo off
REM ==============================================
REM BIST 30 Haftalik Tahmin Otomasyonu
REM Her Cuma 19:30'da Windows Gorev Zamanlayici tarafindan calistirilir
REM ==============================================

cd /d "%~dp0"
echo [%date% %time%] Haftalik tahmin basladi..
echo Tahminler basliyor...
python weekly_forecast.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [%date% %time%] BASARILI: Haftalik tahminler tamamlandi.
    echo [%date% %time%] BASARILI: Haftalik tahminler tamamlandi. >> weekly_forecast_log.txt
) else (
    echo.
    echo [%date% %time%] HATA: Tahmin islemi sirasinda bir hata olustu. Hata Kodu: %ERRORLEVEL%
    echo [%date% %time%] HATA: Tahmin islemi basarisiz oldu. Kodu: %ERRORLEVEL% >> weekly_forecast_log.txt
)

echo. >> weekly_forecast_log.txt
