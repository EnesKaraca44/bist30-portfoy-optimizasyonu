SUPPORTED_HORIZON_MONTHS = (1, 3, 6, 12)


def sanitize_months(value, default=12):
    try:
        months = int(value)
        if months <= 0:
            return default
        return months
    except (TypeError, ValueError):
        return default


def normalize_horizon_months(value, default=12):
    """Desteklenen vadelere (1, 3, 6, 12) yuvarlar."""
    months = sanitize_months(value, default=default)
    if months in SUPPORTED_HORIZON_MONTHS:
        return months
    return min(SUPPORTED_HORIZON_MONTHS, key=lambda m: abs(m - months))


def str_to_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"false", "0", "no", "hayir", "hayır"}:
        return False
    if value in {"true", "1", "yes", "evet"}:
        return True
    return default
