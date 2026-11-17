"""IST timezone helpers, NSE trading calendar guards, and intraday session progress."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

# Official NSE equity weekday holidays (2026) — weekend holidays excluded except Muhurat special date
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Shri Ram Navami
    date(2026, 3, 31),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Id
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali-Balipratipada
    date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
    date(2026, 12, 25),  # Christmas
}

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
SESSION_TOTAL_MINUTES = 375  # 09:15 to 15:30 IST


def now_ist() -> datetime:
    return datetime.now(IST)


def is_nse_trading_day(d: date | None = None) -> bool:
    d = d or now_ist().date()
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if d in NSE_HOLIDAYS_2026:
        return False
    return True


def is_nse_market_open(dt: datetime | None = None, buffer_after_close_min: int = 5) -> bool:
    dt = dt or now_ist()
    if not is_nse_trading_day(dt.date()):
        return False
    close_dt = datetime.combine(dt.date(), MARKET_CLOSE, tzinfo=IST) + timedelta(minutes=buffer_after_close_min)
    open_dt = datetime.combine(dt.date(), MARKET_OPEN, tzinfo=IST)
    return open_dt <= dt <= close_dt


def is_nse_candle_incomplete_today(dt: datetime | None = None) -> bool:
    """
    Return True on an NSE trading day before the 15:30 IST closing bell,
    when any daily bar stamped with today's date from yfinance is partial/incomplete.
    At or after 15:30 IST (and on weekends/holidays), returns False.
    """
    dt = dt or now_ist()
    if not is_nse_trading_day(dt.date()):
        return False
    return dt.time() < MARKET_CLOSE


def session_elapsed_minutes(dt: datetime | None = None) -> float:
    """Return minutes elapsed since 09:15 IST, clamped to [0, 375]."""
    dt = dt or now_ist()
    open_dt = datetime.combine(dt.date(), MARKET_OPEN, tzinfo=IST)
    elapsed = (dt - open_dt).total_seconds() / 60.0
    return max(0.0, min(float(SESSION_TOTAL_MINUTES), elapsed))


def expected_intraday_volume_fraction(elapsed_minutes: float) -> float:
    """
    Approximate U-shaped NSE intraday cumulative volume profile so live RVOL
    doesn't under-report morning breakouts or explode in the first 5 minutes.
    - First 30 mins typically carry ~18% of daily volume.
    - Midday carries slower flow; final 45 mins accelerate.
    """
    if elapsed_minutes <= 5:
        return 0.08
    frac_linear = elapsed_minutes / SESSION_TOTAL_MINUTES
    # Blend linear with mild front-loaded U-shape curve
    if elapsed_minutes <= 45:
        return max(0.10, 0.08 + (elapsed_minutes / 45.0) * 0.16)
    if elapsed_minutes >= 330:
        return min(1.0, 0.82 + ((elapsed_minutes - 330.0) / 45.0) * 0.18)
    return min(0.85, 0.24 + ((elapsed_minutes - 45.0) / 285.0) * 0.58)
