import os
import time
from collections import deque
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
import duckdb
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

BASE_URL = "https://api.polygon.io"

class RateLimiter:
    def __init__(self, max_requests: int = 5, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._calls: deque[float] = deque()

    def wait(self) -> None:
        now = time.time()
        while self._calls and now - self._calls[0] >= self.window:
            self._calls.popleft()
        if len(self._calls) >= self.max_requests:
            sleep_for = self.window - (now - self._calls[0]) + 0.05
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._calls.append(time.time())

class PolygonClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        max_requests_per_minute: int = 5,
        session: Optional[requests.Session] = None,
        timeout: float = 30.0,
    ) -> None:
        key = api_key or os.getenv("POLYGON_API_KEY")
        if not key:
            raise ValueError("Set POLYGON_API_KEY env var or pass api_key")
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.limiter = RateLimiter(max_requests=max_requests_per_minute, window_seconds=60)

    def _request(self, method: str, url: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 5) -> Dict[str, Any]:
        params = dict(params or {})
        if "apiKey" not in params and "apiKey=" not in url:
            params["apiKey"] = self.api_key

        attempt = 0
        backoff = 2.0
        while True:
            self.limiter.wait()
            try:
                resp = self.session.request(method, url, params=params, timeout=self.timeout)
            except requests.RequestException:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue

            if resp.status_code == 200:
                return resp.json() if resp.content else {}

            if resp.status_code in (429, 500, 502, 503, 504):
                attempt += 1
                if attempt > max_retries:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After")
                try:
                    sleep_for = float(retry_after) if retry_after else backoff
                except ValueError:
                    sleep_for = backoff
                time.sleep(sleep_for)
                backoff = min(backoff * 2, 60.0)
                continue

            resp.raise_for_status()

    def fetch_minutes(self, ticker: str, start: str, end: str, adjusted: bool = True, limit: int = 50000, sort: str = "asc") -> pd.DataFrame:
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}"
        params: Dict[str, Any] = {
            "adjusted": str(adjusted).lower(),
            "sort": sort,
            "limit": int(limit),
            "apiKey": self.api_key,
        }

        results: List[Dict[str, Any]] = []
        while True:
            data = self._request("GET", url, params=params)
            results.extend(data.get("results", []))
            next_url = data.get("next_url")
            if not next_url:
                break
            url = next_url
            params = {"apiKey": self.api_key}

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df["ts_utc"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "trades"})
        cols = ["ts_utc", "open", "high", "low", "close", "volume", "vwap", "trades"]
        df["ticker"] = ticker
        return df[["ticker"] + cols].sort_values("ts_utc").reset_index(drop=True)

def ensure_security_prices_table(conn: duckdb.DuckDBPyConnection, table: str = "security_prices") -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            ticker TEXT,
            ts_utc TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            vwap DOUBLE,
            trades BIGINT,
            PRIMARY KEY (ticker, ts_utc)
        );
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ts ON {table}(ts_utc);")

def upsert_security_prices(conn: duckdb.DuckDBPyConnection, frame: pd.DataFrame, table: str = "security_prices") -> int:
    if frame is None or frame.empty:
        return 0

    # Normalize columns
    expected = ["ticker", "ts_utc", "open", "high", "low", "close", "volume", "vwap", "trades"]
    for c in expected:
        if c not in frame.columns:
            frame[c] = pd.NA
    df = frame[expected].copy()

    view_name = "_sec_src_view"
    tmp_name = "_sec_src"

    # Clean any leftovers
    try:
        conn.unregister(view_name)
    except Exception:
        pass
    conn.execute(f"DROP TABLE IF EXISTS {tmp_name};")

    # Register → materialize → MERGE in a transaction
    conn.register(view_name, df)
    try:
        conn.execute("BEGIN;")
        conn.execute(f"CREATE TEMP TABLE {tmp_name} AS SELECT * FROM {view_name};")
        conn.execute(f"""
            MERGE INTO {table} t
            USING {tmp_name} s
            ON t.ticker = s.ticker AND t.ts_utc = s.ts_utc
            WHEN MATCHED THEN UPDATE SET
                open = s.open, high = s.high, low = s.low, close = s.close,
                volume = s.volume, vwap = s.vwap, trades = s.trades
            WHEN NOT MATCHED THEN INSERT (ticker, ts_utc, open, high, low, close, volume, vwap, trades)
            VALUES (s.ticker, s.ts_utc, s.open, s.high, s.low, s.close, s.volume, s.vwap, s.trades);
        """)
        n = conn.execute(f"SELECT COUNT(*) FROM {tmp_name};").fetchone()[0]
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.execute(f"DROP TABLE IF EXISTS {tmp_name};")
        try:
            conn.unregister(view_name)
        except Exception:
            pass

    return int(n)

def _get_last_ts_utc(conn: duckdb.DuckDBPyConnection, table: str, ticker: str):
    row = conn.execute(f"SELECT max(ts_utc) FROM {table} WHERE ticker = ?", [ticker]).fetchone()
    return row[0]

def save_minutes_to_duckdb(
    tickers: List[str],
    start: Optional[str],
    end: str,
    db_path: str = "./data/markets.duckdb",
    table: str = "security_prices",
    max_requests_per_minute: int = 5,
    api_key: Optional[str] = None,
    incremental: bool = False,   # new
) -> int:
    """
    Fetch minute bars for a list of tickers within [start, end) and upsert into DuckDB.
    Returns total rows processed from API (not deduped count).
    """
    client = PolygonClient(api_key=api_key, max_requests_per_minute=max_requests_per_minute)
    conn = duckdb.connect(db_path)
    ensure_security_prices_table(conn, table)

    total = 0
    try:
        for tkr in tickers:
            eff_start = start
            if incremental:
                last = _get_last_ts_utc(conn, table, tkr)
                if last is not None:
                    eff_start = (pd.to_datetime(last, utc=True) + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            df = client.fetch_minutes(tkr, eff_start or start, end)
            if df is not None and not df.empty:
                total += len(df)
                upsert_security_prices(conn, df, table=table)
                print(f"[{tkr}] upserted {len(df)} rows into {table}")
            else:
                print(f"[{tkr}] no data returned")
    finally:
        conn.close()
    return total

if __name__ == "__main__":
    # Example: SPY and QQQ for first week of Jan 2025
    tickers = ["SPY"]
    rows = save_minutes_to_duckdb(tickers, start="2025-01-01", end="2025-01-02")
    print("Total fetched rows:", rows)