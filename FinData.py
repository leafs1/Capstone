import os
import glob
from typing import Iterable, List, Optional

import duckdb
import pandas as pd

# --- Optional: only imported when reading DBN ---
try:
    from databento import DBNStore
except Exception:
    DBNStore = None

# ---------------------------
# Normalization for BBO-1m
# ---------------------------
def _normalize_bbo_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Databento BBO-1m-like data to:
    ['ticker','ts_utc','bid_px','ask_px','bid_sz','ask_sz','bid_ct','ask_ct','mid_px','spread'].
    Uses ts_recv for timestamp.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "ticker","ts_utc","bid_px","ask_px","bid_sz","ask_sz","bid_ct","ask_ct","mid_px","spread"
        ])

    print(df)
    # Reset index to make ts_recv a column if it's the index
    if df.index.name == "ts_recv" or "ts_recv" in df.index.names:
        df = df.reset_index()

    # --- Harmonize column names from DBN/CSV ---
    df = df.rename(columns={
        "symbol": "ticker",
        "bid_px_00": "bid_px",
        "ask_px_00": "ask_px",
        "bid_sz_00": "bid_sz",
        "ask_sz_00": "ask_sz",
        "bid_ct_00": "bid_ct",
        "ask_ct_00": "ask_ct",
    }).copy()

    for col in ["ticker","bid_px","ask_px","bid_sz","ask_sz","bid_ct","ask_ct"]:
        if col not in df.columns:
            df[col] = pd.NA

    # --- Use ts_recv for timestamp ---
    if "ts_recv" not in df.columns:
        raise KeyError("No ts_recv column found.")

    if pd.api.types.is_integer_dtype(df["ts_recv"]):
        df["ts_utc"] = pd.to_datetime(df["ts_recv"], unit="ns", utc=True, errors="coerce")
    else:
        df["ts_utc"] = pd.to_datetime(df["ts_recv"], utc=True, errors="coerce")

    # --- Derived fields ---
    df["mid_px"] = (
        pd.to_numeric(df["bid_px"], errors="coerce") +
        pd.to_numeric(df["ask_px"], errors="coerce")
    ) / 2.0
    df["spread"] = (
        pd.to_numeric(df["ask_px"], errors="coerce") -
        pd.to_numeric(df["bid_px"], errors="coerce")
    )

    cols = ["ticker","ts_utc","bid_px","ask_px","bid_sz","ask_sz","bid_ct","ask_ct","mid_px","spread"]
    out = df[cols].sort_values(["ticker","ts_utc"]).reset_index(drop=True)
    return out


# ---------------------------
# DuckDB helpers
# ---------------------------
def ensure_bbo_table(conn: duckdb.DuckDBPyConnection, table: str = "security_bbo_1m") -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            ticker TEXT,
            ts_utc TIMESTAMP,
            bid_px DOUBLE,
            ask_px DOUBLE,
            bid_sz BIGINT,
            ask_sz BIGINT,
            bid_ct BIGINT,
            ask_ct BIGINT,
            mid_px DOUBLE,
            spread DOUBLE,
            PRIMARY KEY (ticker, ts_utc)
        );
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ts ON {table}(ts_utc);")


def upsert_bbo_1m(conn: duckdb.DuckDBPyConnection, frame: pd.DataFrame, table: str = "security_bbo_1m") -> int:
    if frame is None or frame.empty:
        return 0

    expected = ["ticker","ts_utc","bid_px","ask_px","bid_sz","ask_sz","bid_ct","ask_ct","mid_px","spread"]
    for c in expected:
        if c not in frame.columns:
            frame[c] = pd.NA
    df = frame[expected].copy()

    view_name = "_bbo_src_view"
    tmp_name = "_bbo_src"

    try:
        conn.unregister(view_name)
    except Exception:
        pass
    conn.execute(f"DROP TABLE IF EXISTS {tmp_name};")

    conn.register(view_name, df)
    try:
        conn.execute("BEGIN;")
        conn.execute(f"CREATE TEMP TABLE {tmp_name} AS SELECT * FROM {view_name};")
        conn.execute(f"""
            MERGE INTO {table} t
            USING {tmp_name} s
            ON t.ticker = s.ticker AND t.ts_utc = s.ts_utc
            WHEN MATCHED THEN UPDATE SET
                bid_px = s.bid_px, ask_px = s.ask_px,
                bid_sz = s.bid_sz, ask_sz = s.ask_sz,
                bid_ct = s.bid_ct, ask_ct = s.ask_ct,
                mid_px = s.mid_px, spread = s.spread
            WHEN NOT MATCHED THEN INSERT (
                ticker, ts_utc, bid_px, ask_px, bid_sz, ask_sz, bid_ct, ask_ct, mid_px, spread
            ) VALUES (
                s.ticker, s.ts_utc, s.bid_px, s.ask_px, s.bid_sz, s.ask_sz, s.bid_ct, s.ask_ct, s.mid_px, s.spread
            );
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


# ---------------------------
# Readers: DBN and CSV
# ---------------------------
def _read_dbn_file(path: str, *, chunk_rows: Optional[int] = None) -> Iterable[pd.DataFrame]:
    if DBNStore is None:
        raise RuntimeError("databento>=0.63 required to read DBN files (DBNStore).")
    store = DBNStore.from_file(path)
    if chunk_rows and chunk_rows > 0:
        offset = 0
        while True:
            df = store.to_df(count=chunk_rows, offset=offset)
            if df.empty:
                break
            yield df
            offset += len(df)
    else:
        yield store.to_df()


def _read_csv_file(path: str, **read_csv_kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **read_csv_kwargs)


def ingest_download_center_folder(
    downloads_dir: str,
    db_path: str = "./data/markets.duckdb",
    table: str = "security_bbo_1m",
    chunk_rows: Optional[int] = None,
) -> int:
    """
    Ingest all DBN/DBN.zst and CSV files from downloads_dir into DuckDB.
    Returns number of rows upserted.
    """
    dbn_files = glob.glob(os.path.join(downloads_dir, "**", "*.dbn"), recursive=True)
    dbn_files += glob.glob(os.path.join(downloads_dir, "**", "*.dbn.zst"), recursive=True)
    csv_files = glob.glob(os.path.join(downloads_dir, "**", "*.csv"), recursive=True)

    total_rows = 0
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = duckdb.connect(db_path)
    ensure_bbo_table(conn, table)

    try:
        for f in dbn_files:
            print(f"Reading DBN: {f}")
            for raw in _read_dbn_file(f, chunk_rows=chunk_rows):
                norm = _normalize_bbo_df(raw)
                norm = densify_minutes(norm)
                n = upsert_bbo_1m(conn, norm, table=table)
                total_rows += n
                print(f"  Upserted {n} rows from {f}")

        for f in csv_files:
            print(f"Reading CSV: {f}")
            raw = _read_csv_file(f)
            norm = _normalize_bbo_df(raw)
            n = upsert_bbo_1m(conn, norm, table=table)
            total_rows += n
            print(f"  Upserted {n} rows from {f}")
    finally:
        conn.close()

    return total_rows


def densify_minutes(
    df: pd.DataFrame,
    rth_only: bool = True,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Build a dense 1‑minute grid per ticker using forward-fill.
    Fills within each trading day (or RTH window) and does NOT fill across days.
    """
    if df.empty:
        return df
    df = df.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True).dt.floor("1min")
    value_cols = ["bid_px","ask_px","bid_sz","ask_sz","bid_ct","ask_ct","mid_px","spread"]

    out = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = g.drop_duplicates(subset=["ts_utc"]).set_index("ts_utc").sort_index()

        # tag local day
        g["ts_local"] = g.index.tz_convert(tz)
        g["session_day"] = g["ts_local"].dt.normalize()

        for day, gd in g.groupby("session_day", sort=False):
            if rth_only:
                start_l = day + pd.Timedelta(hours=9, minutes=30)
                end_l = day + pd.Timedelta(hours=16)
                idx_local = pd.date_range(start_l, end_l, freq="1min", tz=tz)
            else:
                # full-day span based on available data that day
                day_mask = (g["ts_local"].dt.normalize() == day)
                day_times = g.index[day_mask]
                if day_times.empty:
                    continue
                idx_local = pd.date_range(
                    day_times.min().tz_convert(tz),
                    day_times.max().tz_convert(tz),
                    freq="1min",
                    tz=tz,
                )

            idx_utc = idx_local.tz_convert("UTC")
            gd = gd.drop(columns=["ts_local","session_day"], errors="ignore")
            gd = gd.reindex(idx_utc)

            # unlimited forward-fill within this day/session
            gd[value_cols] = gd[value_cols].ffill()

            gd["ticker"] = tkr
            out.append(gd.reset_index().rename(columns={"index": "ts_utc"}))

    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    folder = os.path.expanduser("./Spy25:10:2024-2025")
    inserted = ingest_download_center_folder(
        downloads_dir=folder,
        db_path="./data/markets.duckdb",
        table="security_bbo_1m",
        chunk_rows=None,
    )
    print(f"Total ingested: {inserted} rows into DuckDB.")
