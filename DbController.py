# db_controller.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Dict, Any

import duckdb
import pandas as pd

'''
.tables

SHOW markets;
'''

@dataclass
class DuckDBConfig:
    path: str = "data/research.duckdb"
    read_only: bool = False


class DuckDBController:
    def __init__(self, config: DuckDBConfig = DuckDBConfig()):
        self.config = config
        Path(self.config.path).parent.mkdir(parents=True, exist_ok=True)
        # open/create DB file
        self.con = duckdb.connect(self.config.path, read_only=self.config.read_only)
        # sensible pragmas
        self.con.execute("PRAGMA threads = 4;")
        self.con.execute("PRAGMA memory_limit = '4GB';")
        self.init_schema()

    def close(self):
        if self.con:
            self.con.close()

    # ---------- Schema ----------
    def init_schema(self):
        # Prices with theme
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            token_id  TEXT,
            ts        BIGINT,      -- unix seconds (UTC)
            price     DOUBLE,
            theme     TEXT,
            PRIMARY KEY (token_id, ts)
        );
        """)
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_prices_ts ON prices (ts);")

        # Full markets schema (matches upsert_markets)
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            market_id     TEXT PRIMARY KEY,
            question      TEXT,
            theme         TEXT,
            slug          TEXT,
            active        BOOLEAN,
            closed        BOOLEAN,
            startDateIso  TEXT,
            endDateIso    TEXT,
            closedTime    TEXT,
            conditionId   TEXT,
            liquidityNum  DOUBLE,
            volumeNum     DOUBLE
        );
        """)

        # Full tokens schema (matches upsert_tokens)
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            token_id  TEXT PRIMARY KEY,
            market_id TEXT,
            outcome   TEXT
        );
        """)

        # Lightweight migration for older DBs that might miss columns
        self._ensure_columns("markets", {
            "theme": "TEXT", "slug": "TEXT", "active": "BOOLEAN", "closed": "BOOLEAN",
            "startDateIso": "TEXT", "endDateIso": "TEXT", "closedTime": "TEXT",
            "conditionId": "TEXT", "liquidityNum": "DOUBLE", "volumeNum": "DOUBLE"
        })
        self._ensure_columns("tokens", {"outcome": "TEXT"})

    def _ensure_columns(self, table: str, cols: Dict[str, str]) -> None:
        info = self.con.execute(f"PRAGMA table_info('{table}')").df()
        existing = set(info["name"].tolist())
        for col, typ in cols.items():
            if col not in existing:
                self.con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ};")

    # ---------- Utilities ----------
    @staticmethod
    def _normalize_prices_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize incoming DataFrame to ['token_id','ts','price','theme'].
        Accepts aliases for timestamp.
        """
        df = df.copy()
        if "ts" not in df.columns:
            for cand in ("timestamp", "time"):
                if cand in df.columns:
                    df.rename(columns={cand: "ts"}, inplace=True)
                    break
        needed = ["token_id", "ts", "price", "theme"]
        for c in needed:
            if c not in df.columns:
                df[c] = pd.NA if c != "price" else df.get(c)
        df = df[needed].dropna(subset=["token_id", "ts", "price"])
        df["token_id"] = df["token_id"].astype("string")
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("int64")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])
        return df[needed]

    @staticmethod
    def _normalize_markets_df(df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "market_id","question","theme","slug","active","closed",
            "startDateIso","endDateIso","closedTime","conditionId","liquidityNum","volumeNum"
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA

        df = df.copy()
        df["market_id"] = df["market_id"].astype("string")
        df["question"] = df["question"].astype("string")
        df["theme"] = df["theme"].astype("string")
        df["slug"] = df["slug"].astype("string")
        df["active"] = df["active"].astype("boolean")
        df["closed"] = df["closed"].astype("boolean")
        df["startDateIso"] = df["startDateIso"].astype("string")
        df["endDateIso"] = df["endDateIso"].astype("string")
        df["closedTime"] = df["closedTime"].astype("string")
        df["conditionId"] = df["conditionId"].astype("string")
        df["liquidityNum"] = pd.to_numeric(df["liquidityNum"], errors="coerce")
        df["volumeNum"] = pd.to_numeric(df["volumeNum"], errors="coerce")

        return df[cols]


    @staticmethod
    def _normalize_tokens_df(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["token_id","market_id","outcome"]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        df["token_id"] = df["token_id"].astype("string")
        df["market_id"] = df["market_id"].astype("string")
        df["outcome"] = df["outcome"].astype("string")
        return df[cols]

    # ---------- Upserts ----------
    def upsert_prices(self, df: pd.DataFrame):
        """
        Upsert into prices (token_id, ts) PK, updating price/theme.
        """
        if df is None or df.empty:
            return 0
        df = self._normalize_prices_df(df)
        self.con.register("prices_src", df)
        self.con.execute("CREATE TEMP TABLE _prices_src AS SELECT * FROM prices_src;")
        self.con.execute("""
            MERGE INTO prices AS t
            USING _prices_src AS s
            ON t.token_id = s.token_id AND t.ts = s.ts
            WHEN MATCHED THEN UPDATE SET
                price = COALESCE(s.price, t.price),
                theme = COALESCE(s.theme, t.theme)
            WHEN NOT MATCHED THEN INSERT (token_id, ts, price, theme)
            VALUES (s.token_id, s.ts, s.price, s.theme);
        """)
        cnt = self.con.execute("SELECT COUNT(*) FROM _prices_src;").fetchone()[0]
        self.con.execute("DROP TABLE _prices_src;")
        self.con.unregister("prices_src")
        return cnt

    def upsert_markets(self, df: pd.DataFrame):
        if df is None or df.empty:
            return 0
        df = self._normalize_markets_df(df)
        self.con.register("markets_src", df)
        self.con.execute("CREATE TEMP TABLE _markets_src AS SELECT * FROM markets_src;")
        self.con.execute("""
            MERGE INTO markets AS t
            USING _markets_src AS s
            ON t.market_id = s.market_id
            WHEN MATCHED THEN UPDATE SET
                question     = COALESCE(s.question, t.question),
                theme        = COALESCE(s.theme, t.theme),
                slug         = COALESCE(s.slug, t.slug),
                active       = COALESCE(s.active, t.active),
                closed       = COALESCE(s.closed, t.closed),
                startDateIso = COALESCE(s.startDateIso, t.startDateIso),
                endDateIso   = COALESCE(s.endDateIso, t.endDateIso),
                closedTime   = COALESCE(s.closedTime, t.closedTime),
                conditionId  = COALESCE(s.conditionId, t.conditionId),
                liquidityNum = COALESCE(s.liquidityNum, t.liquidityNum),
                volumeNum    = COALESCE(s.volumeNum, t.volumeNum)
            WHEN NOT MATCHED THEN INSERT (
                market_id, question, theme, slug, active, closed,
                startDateIso, endDateIso, closedTime, conditionId, liquidityNum, volumeNum
            ) VALUES (
                s.market_id, s.question, s.theme, s.slug, s.active, s.closed,
                s.startDateIso, s.endDateIso, s.closedTime, s.conditionId, s.liquidityNum, s.volumeNum
            );
        """)
        cnt = self.con.execute("SELECT COUNT(*) FROM _markets_src;").fetchone()[0]
        self.con.execute("DROP TABLE _markets_src;")
        self.con.unregister("markets_src")
        return cnt

    def upsert_tokens(self, df: pd.DataFrame):
        if df is None or df.empty:
            return 0
        df = self._normalize_tokens_df(df)
        self.con.register("tokens_src", df)
        self.con.execute("CREATE TEMP TABLE _tokens_src AS SELECT * FROM tokens_src;")
        self.con.execute("""
            MERGE INTO tokens AS t
            USING _tokens_src AS s
            ON t.token_id = s.token_id
            WHEN MATCHED THEN UPDATE SET
                market_id = COALESCE(s.market_id, t.market_id),
                outcome   = COALESCE(s.outcome, t.outcome)
            WHEN NOT MATCHED THEN INSERT (token_id, market_id, outcome)
            VALUES (s.token_id, s.market_id, s.outcome);
        """)
        cnt = self.con.execute("SELECT COUNT(*) FROM _tokens_src;").fetchone()[0]
        self.con.execute("DROP TABLE _tokens_src;")
        self.con.unregister("tokens_src")
        return cnt

    # ---------- Queries ----------
    def query_prices_df(
        self,
        token_ids: Optional[Iterable[str]] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        themes: Optional[Iterable[str]] = None,
        order_by: str = "token_id, ts"
    ) -> pd.DataFrame:
        sel_cols = "token_id, ts, price, theme"
        conds = []
        params: Dict[str, Any] = {}
        if token_ids:
            conds.append("token_id IN (SELECT * FROM UNNEST($token_ids))")
            params["token_ids"] = list(map(str, token_ids))
        if themes:
            conds.append("theme IN (SELECT * FROM UNNEST($themes))")
            params["themes"] = list(map(str, themes))
        if start_ts is not None:
            conds.append("ts >= $start_ts")
            params["start_ts"] = int(start_ts)
        if end_ts is not None:
            conds.append("ts <= $end_ts")
            params["end_ts"] = int(end_ts)
        where = f"WHERE {' AND '.join(conds)}" if conds else ""
        q = f"SELECT {sel_cols} FROM prices {where} ORDER BY {order_by};"
        return self.con.execute(q, params).df()

    def query_markets_df(self, theme: Optional[str] = None) -> pd.DataFrame:
        if theme:
            return self.con.execute(
                "SELECT * FROM markets WHERE theme = $theme ORDER BY market_id;",
                {"theme": theme},
            ).df()
        return self.con.execute("SELECT * FROM markets ORDER BY market_id;").df()

    def raw(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Run arbitrary SQL and get a DataFrame back."""
        return self.con.execute(sql, params or {}).df()

    # ---------- Maintenance ----------
    def optimize(self):
        # DuckDB has no ANALYZE pragma. VACUUM works to reclaim space / rewrite DB.
        # You can also run CHECKPOINT to persist changes.
        try:
            self.con.execute("VACUUM;")
        except Exception as e:
            print("VACUUM failed:", e)
        try:
            self.con.execute("CHECKPOINT;")
        except Exception as e:
            print("CHECKPOINT failed:", e)

