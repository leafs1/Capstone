from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

# ---------- Time helpers ----------
@staticmethod
def utc_yesterday_today() -> Tuple[int, int]:
    """Calendar UTC yesterday 00:00:00 → today 23:59:59 (in UTC)."""
    now_utc = datetime.now(timezone.utc)
    today_utc = datetime(year=now_utc.year, month=now_utc.month, day=now_utc.day, tzinfo=timezone.utc)
    y0 = (today_utc - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    t1 = (today_utc + timedelta(days=0)).replace(hour=23, minute=59, second=59, microsecond=0)
    return int(y0.timestamp()), int(t1.timestamp())

def iso_to_ts(s: str) -> int:
    """Convert ISO timestamp to unix timestamp"""
    return int(pd.Timestamp(s, tz="UTC").timestamp())