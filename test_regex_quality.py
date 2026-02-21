#!/usr/bin/env python3
"""Quick test to verify regex quality - no downloading."""

from PolyMarket import PolymarketData

pm = PolymarketData()

print("Testing enhanced regex patterns...\n")
print("="*80)

# Get markets
df = pm.get_macro_event_markets(
    start_date="2025-01-01T00:00:00Z",
    end_date="2025-11-03T23:59:59Z",
    active=None,
    closed=None,
    max_pages=30
)

print(f"Found {len(df)} macro-related markets\n")
print("Theme breakdown:")
print(df['theme'].value_counts().to_string())
print("\n" + "="*80)
print("Sample markets by theme:\n")
print(df[['theme', 'question']].head(30).to_string(index=False))
