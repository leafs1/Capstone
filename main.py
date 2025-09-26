from PolyMarket import PolymarketData
import pandas as pd
from DbController import DuckDBController, DuckDBConfig

def fetchPolymarketData():
    pm = PolymarketData()

    # Aggregate everything macro since Jan 1, 2023 (UTC)
    df = pm.get_macro_event_markets(
        start_date="2023-01-01T00:00:00Z",
        end_date="2025-12-31T23:59:59Z",
        active=None,   # both live and historical
        closed=None,   # both live and historical
        max_pages=30   # increase to sweep more
    )
    print(f"Found {len(df)} macro-related markets")
    print(df[["theme","question","startDateIso","endDateIso","closedTime"]].head(10).to_string(index=False))

    # Get all CPI-only markets (live + closed):
    cpi_df = pm.get_macro_event_markets(themes={"cpi": [r"\bCPI\b", r"Consumer Price Index"]},
                                        active=None, closed=None, max_pages=30)
    # Expand tokens to one row per token for direct joins to price history
    cpi_tokens = cpi_df.explode("tokens")[["theme","market_id","question","tokens"]].rename(columns={"tokens":"token_id"})
    print(cpi_tokens.head(10).to_string(index=False))


def testDuck():
    db = DuckDBController(DuckDBConfig(path="data/research.duckdb"))

    # 1) Upsert some price history (df_hist must have at least token_id, ts, price)
    df_hist = pd.DataFrame({
        "token_id": ["abc","abc","abc"],
        "ts": [1758565024, 1758565084, 1758565143],
        "price": [0.505, 0.500, 0.500],
        "theme": ["fomc","fomc","fomc"]
    })
    db.upsert_prices(df_hist)

    # 2) Upsert markets metadata
    df_mk = pd.DataFrame({
        "market_id": ["m1"],
        "question": ["Will the Fed hike at next meeting?"],
        "theme": ["fomc"],
        "active": [True],
        "closed": [False]
    })
    db.upsert_markets(df_mk)

    # 3) Query back a window
    out = db.query_prices_df(token_ids=["abc"], start_ts=1758565000, end_ts=1758565200)
    print(out)

    # 4) Housekeeping
    db.optimize()
    db.close()

# -------------------- Example usage --------------------
if __name__ == "__main__":
    testDuck()


