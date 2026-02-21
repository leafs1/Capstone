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
        """
        Enhanced macro event themes with precise matching.
        Returns compiled regex patterns with IGNORECASE and UNICODE flags.
        
        IMPORTANT: Patterns are checked in order, so more specific patterns should come first.
        """
        FLAGS = re.IGNORECASE | re.UNICODE
        
        defaults: Dict[str, List[str]] = {
            # === Monetary Policy & Central Banks ===
            "fomc": [
                r"\bFOMC\b",
                r"\bFederal Reserve (?:meeting|decision|announcement|policy|statement)\b",
                r"\bFed (?:meeting|decision|announcement|policy|statement|minutes)\b",
                r"\bJerome Powell\b",
                r"\bFed Chair\b",
                r"\bFed funds rate\b",
                r"\bSOFR\b",
                r"\bdot plot\b",
                r"\bSummary of Economic Projections\b"
            ],
            "ecb": [
                r"\bECB\b",
                r"\bEuropean Central Bank\b",
                r"\bChristine Lagarde\b"
            ],
            "boe": [
                r"\bBank of England\b", 
                r"\bMPC meeting\b",
                r"\bAndrew Bailey\b"
            ],
            
            # === Inflation Indicators ===
            "cpi": [
                r"\bCPI\b(?:\s+(?:data|report|release|print|reading|inflation|number))?",
                r"\bConsumer Price Index\b",
                r"\b(?:headline|core)\s+(?:CPI|inflation)\b",
                r"\binflation\s+(?:report|data|reading|print)\b"
            ],
            "pce": [
                r"\bPCE\b(?:\s+(?:data|report|release|print|reading|inflation))?",
                r"\bPersonal Consumption Expenditures\b",
                r"\bcore PCE\b"
            ],
            "ppi": [
                r"\bPPI\b(?:\s+(?:data|report|release|print|reading))?",
                r"\bProducer Price Index\b"
            ],
            
            # === Labor Market ===
            "labor": [
                r"\bnonfarm payrolls?\b",
                r"\bNFP\b(?:\s+(?:data|report|release|print|reading))?",
                r"\bunemployment rate\b",
                r"\bjobless claims\b",
                r"\bJOLTS\b(?:\s+(?:data|report|release))?",
                r"\bjobs report\b",
                r"\blabor market\b"
            ],
            
            # === Growth & Output ===
            "gdp": [
                r"\bGDP\s+(?:growth|data|report|release|print|reading|contraction|expansion)\b",
                r"\bgross domestic product\b",
                r"\b(?:Q[1-4]|quarter|quarterly)\s+GDP\b",
                r"\bGDP\s+(?:in\s+)?Q[1-4]\s+\d{4}\b",  # "GDP in Q3 2025"
                r"\bNegative GDP\b",
                r"\bGDP contraction\b"
            ],
            "ism": [
                r"\bISM\s+(?:Manufacturing|Services|PMI)\b",
                r"\bPMI\s+(?:data|report|reading)\b"
            ],
            
            # === Consumer Spending ===
            "retail_sales": [
                r"\bretail sales\b(?:\s+(?:data|report|release))?",
                r"\bcontrol group sales\b"
            ],
            "durables": [
                r"\bdurable goods\b(?:\s+(?:orders|data|report))?",
                r"\bcore capital goods\b"
            ],
            
            # === Interest Rates & Fixed Income ===
            "rates": [
                r"\binterest rate\s+(?:hike|cut|decision|change)\b",
                r"\brate\s+(?:hike|cut)\b",
                r"\b(?:basis points?|bps)\b",
                r"\byield curve\b",
                r"\b2s10s\b",
                r"\bcurve inversion\b",
                r"\bterm premium\b",
                r"\bquantitative (?:tightening|easing)\b",
                r"\bQT\b",
                r"\bQE\b"
            ],
            "treasury": [
                r"\bT-?(?:bill|note|bond)s?\b",
                r"\b(?:10|30)-?year(?:\s+treasury)?\b",
                r"\bUST\b",
                r"\btreasury\s+(?:auction|yield)\b"
            ],
            
            # === Equity Markets ===
            "spy": [
                r"\bSPY\s+(?:price|close|level|above|below|hit|reach)\b",
                r"\bS\s*&\s*P\s*500\s+(?:index|close|level|above|below|hit|reach)\b",
                r"\bSPX\s+(?:close|level|above|below|hit|reach)\b",
                r"\bE-?mini\s+S\s*&\s*P\b"
            ],
            "volatility": [
                r"\bVIX\s+(?:above|below|close|level|spike)\b",
                r"\bvolatility\s+(?:index|spike|surge)\b",
                r"\bimplied\s+volatility\b"
            ],
            "market_direction": [
                r"\bstock\s+market\s+(?:crash|correction|rally|selloff|decline|rise)\b",
                r"\bequity\s+market\s+(?:crash|correction|rally|selloff)\b",
                r"\b(?:bull|bear)\s+market\b",
                r"\bcircuit\s+breaker\b",
                r"\bmarket\s+halt\b",
                r"\bDow\s+Jones\b",
                r"\bNasdaq\s+(?:100|Composite)\b"
            ],
            
            # === Sectors ===
            "tech_sector": [
                r"\btech\s+(?:stock|sector|selloff|rally)\b",
                r"\bFAANG\b",
                r"\bMagnificent\s+7\b",
                r"\btechnology\s+sector\b",
                r"\bsemiconductor\s+(?:stocks?|sector)\b"
            ],
            "banking": [
                r"\bbank(?:ing)?\s+(?:sector|crisis|failure|collapse|run)\b",
                r"\bregional\s+banks?\b",
                r"\bfinancial\s+(?:crisis|sector|stability)\b",
                r"\bSVB\b",
                r"\bFirst\s+Republic\b",
                r"\bCredit\s+Suisse\b"
            ],
            
            # === Commodities & Energy ===
            "energy": [
                r"\b(?:WTI|Brent)\s+(?:crude|oil)\b",
                r"\bcrude\s+oil\s+(?:price|barrel)\b",
                r"\boil\s+price\b",
                r"\bgasoline\s+price\b",
                r"\benergy\s+(?:price|sector)\b"
            ],
            "commodities": [
                r"\bgold\s+(?:price|oz|ounce|above|below)\b",
                r"\bsilver\s+(?:price|oz|ounce)\b",
                r"\bcopper\s+price\b",
                r"\bcommodit(?:y|ies)\s+(?:price|market|index)\b"
            ],
            
            # === Foreign Exchange ===
            "fx": [
                r"\bUSD(?:/|vs)(?:EUR|JPY|GBP|CHF|CAD|AUD)\b",
                r"\b(?:Euro|Yen|Pound|Dollar)\s+(?:parity|exchange)\b",
                r"\bdollar\s+index\b",
                r"\bDXY\b"
            ],
            
            # === Geopolitics & Trade ===
            "geopolitics": [
                r"\btrade\s+war\b",
                r"\btariff(?:s)?\s+(?:on|against|increase|decrease)\b",
                r"\bUS-China\s+(?:trade|relations|tensions)\b",
                r"\bTaiwan\s+(?:invasion|conflict|crisis)\b",
                r"\beconomic\s+sanctions\b"
            ],
            
            # === Government & Policy ===
            "government": [
                r"\bgovernment\s+shutdown\b",
                r"\bdebt\s+ceiling\b",
                r"\bfiscal\s+(?:policy|cliff|stimulus)\b",
                r"\bContinuing\s+Resolution\b",
                r"\bbudget\s+(?:deficit|surplus|deal)\b"
            ],
            
            # === Economic Conditions ===
            "recession": [
                r"\brecession\s+(?:in|by|before)\b",
                r"\beconomic\s+(?:downturn|contraction)\b",
                r"\b(?:hard|soft)\s+landing\b",
                r"\beconomic\s+depression\b"
            ],
        }
        
        if user_themes:
            for k, words in user_themes.items():
                defaults[k] = words  # override or add
        
        return {k: re.compile("|".join(words), FLAGS) for k, words in defaults.items()}

    def get_macro_event_markets(
        self,
        themes: Optional[Dict[str, List[str]]] = None,
        start_date: Optional[str] = None,  # e.g. "2024-01-01T00:00:00Z"
        end_date: Optional[str]   = None,  # e.g. "2025-12-31T23:59:59Z"
        active: Optional[bool] = None,     # True for live only; False for inactive; None for all
        closed: Optional[bool] = None,     # True for closed only; None for all
    ) -> pd.DataFrame:
        """
        Sweep ALL Gamma markets in the date range and return a DataFrame of macro-event markets
        matching the given themes (defaults included).
        
        Automatically paginate through all available markets with no artificial limits.
        """
        regexes = self._theme_regexes(themes)
        rows: List[Dict[str, Any]] = []

        # Optional date window routed to Gamma filters (ISO 8601)
        sd_min = start_date
        sd_max = None
        ed_min = None
        ed_max = end_date

        # API typically allows up to 100 markets per page
        page_size = 100
        offset = 0
        page_num = 0
        
        # Paginate until we get an empty batch
        while True:
            page_num += 1
            batch = self._pull_markets_page(
                offset=offset, limit=page_size,
                active=active, closed=closed,
                start_date_min=sd_min, start_date_max=sd_max,
                end_date_min=ed_min, end_date_max=ed_max,
            )
            
            # Stop if no more results
            if not batch or len(batch) == 0:
                print(f"Finished pagination at page {page_num-1}, total offset {offset}")
                break
            
            print(f"Page {page_num}: fetched {len(batch)} markets (offset {offset})")
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