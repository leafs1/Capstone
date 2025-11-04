import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from Granger import get_conn, load_polymarket, load_equity, merge_poly_with_equity, make_returns, overlap_window

# Test with first token
conn = get_conn()

# Get first token
q = "SELECT token_id FROM poly.main.prices GROUP BY token_id HAVING COUNT(*) >= 200 ORDER BY COUNT(*) DESC LIMIT 1"
token_id = conn.execute(q).fetchone()[0]

print(f"Testing token: {token_id[:30]}...")

# Get overlap window
ow = overlap_window(conn, token_id, "SPY")
if not ow:
    print("No overlap!")
    sys.exit(1)

start, end = ow
print(f"Overlap: {start} -> {end}")

# Load data
poly = load_polymarket(conn, token_id, start, end)
eq = load_equity(conn, "SPY", start, end)

print(f"\nPolymarket data points: {len(poly)}")
print(f"Equity data points: {len(eq)}")

# Merge
df = merge_poly_with_equity(poly, eq)
print(f"\nMerged data points: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Calculate returns
if not df.empty:
    returns = df.apply(make_returns).dropna()
    print(f"Return observations: {len(returns)}")
    print(f"\nFirst few rows:")
    print(returns.head())
    print(f"\nSufficient for maxlag=30? {len(returns) >= 90}")

conn.close()
