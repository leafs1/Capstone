# polymarket_data.py
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

import utils

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"


@dataclass
class MarketTokenInfo:
    market_id: str
    question: str
    clob_token_ids: List[str]
    active: bool
    closed: bool
    end_date_iso: Optional[str]


class PolymarketData:
    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.s = session or requests.Session()

    def _pull_markets_page(
        self,
        offset: int,
        limit: int,
        active: Optional[bool],
        closed: Optional[bool],
        start_date_min: Optional[str],
        start_date_max: Optional[str],
        end_date_min: Optional[str],
        end_date_max: Optional[str],
        order: str = "liquidity",
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        if start_date_min: params["start_date_min"] = start_date_min
        if start_date_max: params["start_date_max"] = start_date_max
        if end_date_min:   params["end_date_min"]   = end_date_min
        if end_date_max:   params["end_date_max"]   = end_date_max

        r = self.s.get(f"{GAMMA_BASE}/markets", params=params, timeout=30)
        r.raise_for_status()
        return r.json() or []

    @staticmethod
    def _theme_regexes(user_themes: Optional[Dict[str, List[str]]] = None) -> Dict[str, re.Pattern]:
        """Default macro themes + user overrides. Each value is a compiled OR regex."""
        defaults: Dict[str, List[str]] = {
            # Original macro event themes
            "fomc":        [r"\bFOMC\b", r"Federal Reserve", r"\bFed\b", r"rate decision", r"dot ?plot", r"FOMC statement"],
            "cpi":         [r"\bCPI\b", r"Consumer Price Index", r"\binflation\b", r"core CPI"],
            "pce":         [r"\bPCE\b", r"personal consumption expenditures", r"core PCE"],
            "payrolls":    [r"nonfarm payrolls", r"\bNFP\b", r"jobs report"],
            "unemployment":[r"\bunemployment\b", r"\bjobless rate\b", r"U-3"],
            "gdp":         [r"\bGDP\b", r"gross domestic product"],
            "rates":       [r"interest rate", r"rate hike", r"rate cut", r"basis points", r"\bbps\b"],
            "ecb":         [r"\bECB\b", r"European Central Bank"],
            "boe":         [r"\bBOE\b", r"Bank of England", r"Monetary Policy Committee"],
            "core_inflation":[r"core inflation", r"ex-?food (and|&) energy"],
            
            # NEW: Direct market prediction themes
            "spy":         [r"\bSPY\b", r"S&P 500", r"S&P500", r"\bSPX\b"],
            "market_direction": [r"stock market", r"\bstocks\b", r"equity market", r"market (up|down|crash|rally|correction)", r"bull market", r"bear market"],
            "volatility":  [r"\bVIX\b", r"volatility", r"market volatility"],
            
            # Corporate & earnings
            "tech_sector": [r"tech (stocks|sector)", r"FAANG", r"Magnificent 7", r"tech giants", r"technology stocks"],
            
            # Geopolitical
            "geopolitics": [r"trade war", r"tariffs"],
            "china":       [r"\bChina\b", r"US-China", r"Taiwan", r"Chinese"],
            
            # Energy & commodities
            "commodities": [r"\bgold\b", r"commodity", r"metals", r"silver"],
            
            # Government & policy
            "government":  [r"government shutdown", r"debt ceiling", r"fiscal", r"budget"],
            
            # Banking & finance
            "banking":     [r"\bbank\b", r"financial crisis", r"credit", r"banking sector"],
            "recession":   [r"recession", r"economic downturn", r"bear market", r"depression"],
        }
        if user_themes:
            for k, words in user_themes.items():
                defaults[k] = words  # override or add
        return {k: re.compile("|".join(words), re.IGNORECASE) for k, words in defaults.items()}

    def get_macro_event_markets(
        self,
        themes: Optional[Dict[str, List[str]]] = None,
        start_date: Optional[str] = None,  # e.g. "2024-01-01T00:00:00Z"
        end_date: Optional[str]   = None,  # e.g. "2025-12-31T23:59:59Z"
        active: Optional[bool] = None,     # True for live only; False for inactive; None for all
        closed: Optional[bool] = None,     # True for closed only; None for all
        max_pages: int = 20,
        page_size: int = 200,
    ) -> pd.DataFrame:
        """
        Sweep Gamma markets and return a DataFrame of macro-event markets
        matching the given themes (defaults included).
        """
        regexes = self._theme_regexes(themes)
        rows: List[Dict[str, Any]] = []

        # Optional date window routed to Gamma filters (ISO 8601)
        sd_min = start_date
        sd_max = None
        ed_min = None
        ed_max = end_date

        offset = 0
        for _ in range(max_pages):
            batch = self._pull_markets_page(
                offset=offset, limit=page_size,
                active=active, closed=closed,
                start_date_min=sd_min, start_date_max=sd_max,
                end_date_min=ed_min, end_date_max=ed_max,
            )
            if not batch:
                break
            offset += len(batch)

            for m in batch:
                text = " ".join([
                    str(m.get("question", "")),
                    str(m.get("description", "")),
                    " ".join(map(lambda t: t.get("label","") if isinstance(t, dict) else str(t), m.get("tags", []) or []))
                ])

                hit_theme: Optional[str] = None
                for name, rx in regexes.items():
                    if rx.search(text):
                        hit_theme = name
                        break
                if not hit_theme:
                    continue

                tokens = self._extract_clob_token_ids(m)
                tag_labels = []
                if isinstance(m.get("tags"), list):
                    for t in m["tags"]:
                        if isinstance(t, dict) and t.get("label"):
                            tag_labels.append(t["label"])

                rows.append({
                    "market_id": str(m.get("id", "")),
                    "theme": hit_theme,
                    "question": m.get("question"),
                    "slug": m.get("slug"),
                    "active": m.get("active"),
                    "closed": m.get("closed"),
                    "startDateIso": m.get("startDateIso") or m.get("startDate"),
                    "endDateIso": m.get("endDateIso") or m.get("endDate"),
                    "closedTime": m.get("closedTime"),
                    "conditionId": m.get("conditionId"),
                    "tokens": tokens,
                    "tag_labels": tag_labels,
                    "liquidityNum": m.get("liquidityNum"),
                    "volumeNum": m.get("volumeNum"),
                })

        if not rows:
            return pd.DataFrame(columns=[
                "market_id","theme","question","slug","active","closed",
                "startDateIso","endDateIso","closedTime","conditionId","tokens","tag_labels",
                "liquidityNum","volumeNum"
            ])

        df = pd.DataFrame(rows).drop_duplicates(subset=["market_id","theme"]).reset_index(drop=True)
        return df

    # ---------- Market discovery ----------
    def get_top_liquidity_live_market(self) -> Dict[str, Any]:
        """A handy picker: highest-liquidity open market."""
        r = self.s.get(
            f"{GAMMA_BASE}/markets",
            params={"active": "true", "closed": "false", "order": "liquidity", "ascending": "false", "limit": 1},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()[0]

    def _extract_clob_token_ids(self, m: Dict[str, Any]) -> List[str]:
        """Gamma sometimes returns clobTokenIds as JSON string. Normalize to list[str]."""
        ids = m.get("clobTokenIds")
        if isinstance(ids, list):
            return [str(x) for x in ids]
        if isinstance(ids, str):
            try:
                parsed = json.loads(ids)
                return [str(x) for x in parsed]
            except json.JSONDecodeError:
                # fallback: try comma split
                return [x.strip() for x in ids.split(",") if x.strip()]
        return []


    # ---------- Price history ----------
    def get_price_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        fidelity: Optional[int] = 1,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch price history from the CLOB API and return a pandas DataFrame with:
        ['timestamp', 'price']
        - Use either (start_ts/end_ts) or 'interval' (e.g., 'max', '1d'). If both are provided, interval wins.
        - fidelity (int) is minutes per bar; e.g., 1, 5, 15, 60. Omit for server default.
        """
        params: Dict[str, Union[str, int]] = {"market": token_id}
        if interval:
            params["interval"] = interval
        else:
            if start_ts is None or end_ts is None:
                raise ValueError("Provide start_ts and end_ts when 'interval' is not used.")
            params["startTs"] = int(start_ts)
            params["endTs"] = int(end_ts)
            if fidelity is not None:
                params["fidelity"] = int(fidelity)

        r = self.s.get(f"{CLOB_BASE}/prices-history", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        hist = data.get("history", []) or data.get("data", [])

        if not hist:
            # Return empty, well-typed frame for consistency
            return pd.DataFrame(columns=["timestamp", "price"])

        # Normalize rows
        # Common shapes seen: {"t": 1670951746, "p": 0.52} or OHLCV-style keys.
        records = []
        for row in hist:
            t = row.get("t") or row.get("timestamp") or row.get("time")
            iso = datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat()
            records.append(
                {
                    "timestamp": int(t),
                    "time_utc": iso,
                    "price": row.get("p"),
                }
            )

        df = pd.DataFrame.from_records(records).sort_values("timestamp").reset_index(drop=True)
        return df

    # ---------- Display ----------
    @staticmethod
    def pretty_print_history(df: pd.DataFrame, rows: int = 5) -> None:
        if df.empty:
            print("No history points returned.")
            return

        t0 = df["timestamp"].min()
        t1 = df["timestamp"].max()
        span = t1 - t0 if pd.notnull(t1) and pd.notnull(t0) else None

        print("── Price History Summary ──")
        print(f"Rows: {len(df)}")
        if pd.notnull(t0) and pd.notnull(t1):
            print(f"Range UTC: {datetime.utcfromtimestamp(int(t0)).isoformat()}Z → {datetime.utcfromtimestamp(int(t1)).isoformat()}Z")
            print(f"Span (s): {span}")
        cols = [c for c in ["price"] if c in df.columns]
        if cols:
            print("\nColumns present:", ", ".join(cols))
        print("\nHead:")
        print(df.head(rows).to_string(index=False))
        print("\nTail:")
        print(df.tail(rows).to_string(index=False))