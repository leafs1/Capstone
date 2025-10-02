# load_market_prices.py
import sys
from datetime import timezone
from typing import Dict, List, Tuple, Optional

import duckdb
import pandas as pd
import yfinance as yf

DB_PATH = "markets.duckdb"
TABLE   = "market_prices"

ALIASES: Dict[str, List[str]] = {
    "spy":  ["SPY"],
    "qqq":  ["QQQ"],
    "vif":  ["VFV.TO"],  # change if you want a different fund
    "gold": ["GLD"],     # or ["GLD", "IAU"] to allow fallback
}

INPUT_LABELS: List[str] = ["Spy", "Qqq", "Vif", "Gold"]

DOWNLOAD_PLANS = [
    {"period": "1d",  "interval": "1m",  "prepost": True},
    {"period": "5d",  "interval": "5m",  "prepost": True},
    {"period": "1mo", "interval": "1d",  "prepost": False},
]

def normalize_labels(labels: List[str], alias_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name in labels:
        key = name.strip().lower()
        out[name.strip()] = alias_map.get(key, [name.strip().upper()])
    return out

def try_fetch_latest(ticker: str) -> Optional[Tuple[pd.Timestamp, float]]:
    for plan in DOWNLOAD_PLANS:
        df = yf.download(
            tickers=ticker,
            period=plan["period"],
            interval=plan["interval"],
            prepost=plan["prepost"],
            progress=False,
            group_by="column",
            multi_level_index=False,
            threads=False,
            timeout=20,
            auto_adjust=True,  # explicitly use adjusted 'Close' and silence FutureWarning
        )
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            df = df.dropna(subset=["Close"])
            if df.empty:
                continue
            last_ts = df.index[-1]
            ts_utc = (last_ts.tz_localize(timezone.utc) if last_ts.tzinfo is None
                      else last_ts.tz_convert(timezone.utc))
            last_px = float(df.iloc[-1]["Close"])
            return ts_utc.to_pydatetime(), last_px
    return None

def ensure_table(conn: duckdb.DuckDBPyConnection, table: str) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            ts_utc TIMESTAMP,
            ticker VARCHAR,
            price DOUBLE
        );
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ts_ticker ON {table}(ts_utc, ticker);")

def upsert_row(conn: duckdb.DuckDBPyConnection, table: str, ts_utc, ticker: str, price: float) -> None:
    conn.execute(f"""
        MERGE INTO {table} t
        USING (SELECT ?::TIMESTAMP AS ts_utc, ?::VARCHAR AS ticker, ?::DOUBLE AS price) s
        ON t.ts_utc = s.ts_utc AND t.ticker = s.ticker
        WHEN MATCHED THEN UPDATE SET price = s.price
        WHEN NOT MATCHED THEN INSERT (ts_utc, ticker, price) VALUES (s.ts_utc, s.ticker, s.price);
    """, [ts_utc, ticker, price])

def main() -> int:
    label_to_candidates = normalize_labels(INPUT_LABELS, ALIASES)
    conn = duckdb.connect(DB_PATH)
    ensure_table(conn, TABLE)

    failures: List[str] = []
    for label, candidates in label_to_candidates.items():
        got = False
        for candidate in candidates:
            try:
                res = try_fetch_latest(candidate)
                if res is None:
                    continue
                ts_utc, price = res
                upsert_row(conn, TABLE, ts_utc, candidate, price)
                print(f"Inserted: ts_utc={ts_utc.isoformat()} | ticker={candidate} | price={price}")
                got = True
                break
            except Exception:
                continue
        if not got:
            failures.append(f"{label} → {candidates}: no data")

    print("\nLatest rows:")
    preview = conn.execute(f"""
        SELECT * FROM {TABLE}
        ORDER BY ts_utc DESC, ticker
        LIMIT 20
    """).fetchdf()
    print(preview)

    if failures:
        print("\nWarnings:")
        for msg in failures:
            print(" -", msg)

    conn.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
