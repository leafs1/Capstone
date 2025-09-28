from PolyMarket import PolymarketData
import pandas as pd
from DbController import DuckDBController, DuckDBConfig
from utils import iso_to_ts
import time

def fetchPolymarketData():
    pm = PolymarketData()
    # Max time delta for API (1,000,000 seconds ≈ 11.6 days)
    MAX_TIME_DELTA = 1755006400 - 1754006400  # 1,000,000 seconds

    # Aggregate everything macro since Jan 1, 2023 (UTC)
    df = pm.get_macro_event_markets(
        start_date="2025-01-01T00:00:00Z",
        end_date="2025-12-31T23:59:59Z",
        active=None,   # both live and historical
        closed=None,   # both live and historical
        max_pages=30   # increase to sweep more
    )
    print(f"Found {len(df)} macro-related markets")
    print(df[["theme","question","startDateIso","endDateIso","closedTime"]].head(100).to_string(index=False))

    # Get price history for each market
    db = DuckDBController(DuckDBConfig(path="data/research.duckdb"))

    for idx, row in df.iterrows():
        market_id = row.get('market_id')
        theme = row.get('theme', 'unknown')
        question = row.get('question', 'No question')

        # Get market timing
        start_date = row.get('startDateIso')
        end_date = row.get('endDateIso') or row.get('closedTime')

        if not start_date or not end_date:
            print(f"Skipping market {market_id}: missing start/end dates")
            continue

        try:
            start_ts = iso_to_ts(start_date)
            end_ts = iso_to_ts(end_date)
        except Exception as e:
            print(f"Skipping market {market_id}: invalid dates - {e}")
            continue

        # Get token IDs from the market
        tokens = row.get('tokens', [])

        for token_id in tokens:
            if token_id:
                print(f"Fetching price history for token {token_id} ({question[:50]}...)")

                try:
                    # Batch the requests if time range is too large
                    total_price_df = pd.DataFrame()
                    current_start = start_ts

                    while current_start < end_ts:
                        current_end = min(current_start + MAX_TIME_DELTA, end_ts)

                        print(f"  Fetching batch: {current_start} to {current_end} (delta: {current_end - current_start})")

                        # Use your existing get_price_history method for this batch with retry logic
                        max_retries = 3
                        retry_count = 0
                        batch_price_df = pd.DataFrame()

                        while retry_count < max_retries:
                            try:
                                batch_price_df = pm.get_price_history(
                                    token_id=str(token_id),
                                    start_ts=current_start,
                                    end_ts=current_end,
                                    fidelity=10  # 10-minute intervals
                                )
                                break  # Success, exit retry loop
                            except Exception as e:
                                if "429" in str(e) or "rate limit" in str(e).lower():
                                    retry_count += 1
                                    print(f"    Rate limited. Waiting 10 seconds... (attempt {retry_count}/{max_retries})")
                                    time.sleep(10)
                                elif "503" in str(e) or "502" in str(e):
                                    retry_count += 1
                                    print(f"    Server error. Waiting 5 seconds... (attempt {retry_count}/{max_retries})")
                                    time.sleep(5)
                                else:
                                    # Other error, don't retry
                                    raise e

                        if not batch_price_df.empty:
                            total_price_df = pd.concat([total_price_df, batch_price_df], ignore_index=True)

                        # Move to next batch
                        current_start = current_end

                        # Small delay between batches to be nice to API
                        time.sleep(1)

                    if not total_price_df.empty:
                        # Remove duplicates and sort by timestamp
                        total_price_df = total_price_df.drop_duplicates().sort_values('timestamp')

                        # Prepare for database - rename timestamp to ts and add metadata
                        total_price_df = total_price_df.rename(columns={'timestamp': 'ts'})
                        total_price_df['token_id'] = str(token_id)
                        total_price_df['theme'] = theme

                        # Select only needed columns
                        total_price_df = total_price_df[['token_id', 'ts', 'price', 'theme']]

                        # Store in database
                        try:
                            db.upsert_prices(total_price_df)
                            print(f"Stored {len(total_price_df)} price points for token {token_id}")
                        except Exception as e:
                            print(f"Error storing prices for token {token_id}: {e}")
                    else:
                        print(f"No price data found for token {token_id}")

                except Exception as e:
                    print(f"Error fetching price history for token {token_id}: {e}")

    # Close database connection
    db.close()
    print("Finished fetching and storing price histories")

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
    fetchPolymarketData()


