"""Tarih yardımcı fonksiyonları."""
from datetime import timedelta


def add_business_days(start_date, num_days):
    """Başlangıç tarihinden itibaren N iş günü ekle (hafta sonlarını atla)."""
    current_date = start_date
    days_added = 0

    while days_added < num_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            days_added += 1

    return current_date
