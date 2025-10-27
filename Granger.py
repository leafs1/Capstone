import os
import duckdb
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

DB_MKT = os.getenv("MKT_DB", "./data/markets.duckdb")          # equity DB file
DB_POLY = os.getenv("POLY_DB", "./data/research.duckdb")       # polymarket DB file
DB_OUT = os.getenv("OUT_DB", DB_MKT)                           # where to store results

# Resolve absolute paths and pick the correct alias for OUT_DB
_MKT_PATH  = os.path.abspath(DB_MKT)
_POLY_PATH = os.path.abspath(DB_POLY)
_OUT_PATH  = os.path.abspath(DB_OUT)

if _OUT_PATH == _MKT_PATH:
    OUT_ALIAS = "mkt"
elif _OUT_PATH == _POLY_PATH:
    OUT_ALIAS = "poly"
else:
    OUT_ALIAS = "out"

POLY_TABLE = "poly.main.prices"
EQUITY_TABLE = "mkt.main.security_bbo_1m"
RESULTS_TABLE = f"{OUT_ALIAS}.main.granger_results"

def _attach(conn: duckdb.DuckDBPyConnection, alias: str, path: str) -> None:
    p = os.path.abspath(path).replace("'", "''")
    conn.execute(f"ATTACH DATABASE '{p}' AS {alias}")

def get_conn() -> duckdb.DuckDBPyConnection:
    """
    In-memory router that ATTACHes:
      - mkt  -> equities DB
      - poly -> polymarket DB
      - out  -> results DB (only if a distinct file)
    """
    conn = duckdb.connect()  # in-memory
    _attach(conn, "mkt", DB_MKT)
    if _POLY_PATH != _MKT_PATH:
        _attach(conn, "poly", DB_POLY)
    else:
        # If POLY and MKT are same file, ensure alias exists
        conn.execute("CREATE SCHEMA IF NOT EXISTS poly")
    if OUT_ALIAS == "out":
        _attach(conn, "out", DB_OUT)
    return conn

def load_polymarket(conn, token_id: str, start: str, end: str) -> pd.Series:
    """
    Return a 1‑minute, forward‑filled Polymarket mid series (UTC tz-aware).
    """
    q = f"""
        SELECT
            to_timestamp(ts)::TIMESTAMP AS ts_utc,
            price
        FROM {POLY_TABLE}
        WHERE token_id = ? AND ts BETWEEN epoch(?::TIMESTAMP) AND epoch(?::TIMESTAMP)
        ORDER BY ts
    """
    df = conn.execute(q, [token_id, start, end]).df()
    if df.empty:
        return pd.Series(dtype="float64")
    s = (
        df.set_index(pd.to_datetime(df["ts_utc"], utc=True))
          .drop(columns=["ts_utc"])["price"]
          .resample("1min").last().ffill()
    )
    return s

def load_equity(conn, ticker: str, start: str, end: str) -> pd.Series:
    """
    Return a 1‑minute, forward‑filled equity mid series (UTC tz-aware).
    """
    q = f"""
        SELECT ts_utc, mid_px
        FROM {EQUITY_TABLE}
        WHERE ticker = ? AND ts_utc BETWEEN ? AND ?
        ORDER BY ts_utc
    """
    df = conn.execute(q, [ticker, start, end]).df()
    if df.empty:
        return pd.Series(dtype="float64")
    s = (
        df.set_index(pd.to_datetime(df["ts_utc"], utc=True))
          .drop(columns=["ts_utc"])["mid_px"]
          .resample("1min").last().ffill()
    )
    return s

def make_returns(s: pd.Series) -> pd.Series:
    return np.log(s).diff().dropna()

def granger_pair(token_id: str, ticker: str, start: str, end: str, maxlag: int = 30):
    # Use attached DBs
    conn = get_conn()
    try:
        poly = load_polymarket(conn, token_id, start, end)
        eq = load_equity(conn, ticker, start, end)
    finally:
        conn.close()

    df = pd.concat({"poly": poly, "eq": eq}, axis=1).dropna()
    if df.empty:
        print("No overlapping data.")
        return

    r = df.apply(make_returns).dropna()
    if len(r) < maxlag * 10:
        print(f"Too few observations for maxlag={maxlag}: got {len(r)} rows.")
        return

    print(f"Rows: {len(r)}, window: {r.index.min()} -> {r.index.max()}")
    print("Testing: equity -> polymarket")
    res1 = grangercausalitytests(r[["poly", "eq"]], maxlag=maxlag, verbose=False)
    pvals1 = {lag: res[0]["ssr_ftest"][1] for lag, res in res1.items()}
    print("min p-value (eq→poly):", min(pvals1.items(), key=lambda x: x[1]))

    print("Testing: polymarket -> equity")
    res2 = grangercausalitytests(r[["eq", "poly"]], maxlag=maxlag, verbose=False)
    pvals2 = {lag: res[0]["ssr_ftest"][1] for lag, res in res2.items()}
    print("min p-value (poly→eq):", min(pvals2.items(), key=lambda x: x[1]))

def list_tokens(conn, min_rows: int = 200) -> pd.DataFrame:
    q = f"""
        SELECT token_id, COUNT(*) AS n, MIN(ts) AS min_ts, MAX(ts) AS max_ts
        FROM {POLY_TABLE}
        GROUP BY token_id
        HAVING COUNT(*) >= ?
        ORDER BY n DESC
    """
    return conn.execute(q, [min_rows]).df()

def equity_window(conn, ticker: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    q = f"SELECT MIN(ts_utc) AS s, MAX(ts_utc) AS e FROM {EQUITY_TABLE} WHERE ticker = ?"
    df = conn.execute(q, [ticker]).df()
    if df.empty or pd.isna(df.loc[0, "s"]) or pd.isna(df.loc[0, "e"]):
        return None
    return pd.to_datetime(df.loc[0, "s"], utc=True), pd.to_datetime(df.loc[0, "e"], utc=True)

def token_window(conn, token_id: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    q = f"SELECT to_timestamp(MIN(ts)) AS s, to_timestamp(MAX(ts)) AS e FROM {POLY_TABLE} WHERE token_id = ?"
    df = conn.execute(q, [token_id]).df()
    if df.empty or pd.isna(df.loc[0, "s"]) or pd.isna(df.loc[0, "e"]):
        return None
    return pd.to_datetime(df.loc[0, "s"], utc=True), pd.to_datetime(df.loc[0, "e"], utc=True)

def overlap_window(conn, token_id: str, ticker: str) -> tuple[str, str] | None:
    tw = token_window(conn, token_id)
    ew = equity_window(conn, ticker)
    if not tw or not ew:
        return None
    start = max(tw[0], ew[0])
    end = min(tw[1], ew[1])
    if start >= end:
        return None
    return start.isoformat(), end.isoformat()

def granger_summary(conn, token_id: str, ticker: str, start: str, end: str, maxlag: int = 30):
    poly = load_polymarket(conn, token_id, start, end)
    eq = load_equity(conn, ticker, start, end)

    df = pd.concat({"poly": poly, "eq": eq}, axis=1).dropna()
    if df.empty:
        return None

    r = df.apply(make_returns).dropna()
    if len(r) < max(60, maxlag * 5):  # ensure enough data
        return None

    res1 = grangercausalitytests(r[["poly", "eq"]], maxlag=maxlag, verbose=False)
    pvals1 = {lag: res[0]["ssr_ftest"][1] for lag, res in res1.items()}
    lag_eq_to_poly, p_eq_to_poly = min(pvals1.items(), key=lambda x: x[1])

    res2 = grangercausalitytests(r[["eq", "poly"]], maxlag=maxlag, verbose=False)
    pvals2 = {lag: res[0]["ssr_ftest"][1] for lag, res in res2.items()}
    lag_poly_to_eq, p_poly_to_eq = min(pvals2.items(), key=lambda x: x[1])

    return {
        "token_id": token_id,
        "ticker": ticker,
        "start": r.index.min().isoformat(),
        "end": r.index.max().isoformat(),
        "n_obs": int(len(r)),
        "lag_eq_to_poly": int(lag_eq_to_poly),
        "p_eq_to_poly": float(p_eq_to_poly),
        "lag_poly_to_eq": int(lag_poly_to_eq),
        "p_poly_to_eq": float(p_poly_to_eq),
    }

def ensure_results_table(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
            token_id TEXT,
            ticker TEXT,
            start_ts TIMESTAMP,
            end_ts TIMESTAMP,
            n_obs INTEGER,
            lag_eq_to_poly INTEGER,
            p_eq_to_poly DOUBLE,
            lag_poly_to_eq INTEGER,
            p_poly_to_eq DOUBLE
        )
    """)

def run_all_tokens(ticker: str, maxlag: int = 30, min_rows: int = 200, limit: int | None = None) -> pd.DataFrame:
    conn = get_conn()
    ensure_results_table(conn)
    try:
        toks = list_tokens(conn, min_rows=min_rows)
        if limit:
            toks = toks.head(limit)
        results = []
        for token_id in toks["token_id"]:
            ow = overlap_window(conn, token_id, ticker)
            if not ow:
                continue
            start, end = ow
            summary = granger_summary(conn, token_id, ticker, start, end, maxlag=maxlag)
            if summary:
                results.append(summary)
                conn.execute(f"DELETE FROM {RESULTS_TABLE} WHERE token_id = ? AND ticker = ?", [token_id, ticker])
                conn.execute(f"""
                    INSERT INTO {RESULTS_TABLE}
                    (token_id, ticker, start_ts, end_ts, n_obs, lag_eq_to_poly, p_eq_to_poly, lag_poly_to_eq, p_poly_to_eq)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    summary["token_id"], summary["ticker"],
                    summary["start"], summary["end"], summary["n_obs"],
                    summary["lag_eq_to_poly"], summary["p_eq_to_poly"],
                    summary["lag_poly_to_eq"], summary["p_poly_to_eq"],
                ])
        return pd.DataFrame(results).sort_values("p_poly_to_eq").reset_index(drop=True)
    finally:
        conn.close()

if __name__ == "__main__":
    TICKER = os.getenv("GRANGER_TICKER", "SPY")
    MAXLAG = int(os.getenv("GRANGER_MAXLAG", "30"))
    MIN_ROWS = int(os.getenv("GRANGER_MIN_ROWS", "200"))
    LIMIT = int(os.getenv("GRANGER_LIMIT", "0")) or None

    df = run_all_tokens(TICKER, maxlag=MAXLAG, min_rows=MIN_ROWS, limit=LIMIT)
    if df.empty:
        print("No results.")
    else:
        print(df.head(20).to_string(index=False))