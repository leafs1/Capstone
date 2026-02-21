#!/usr/bin/env python3
"""
Backfill checkpoint table from existing price data.
Run this once if you already have data downloaded before the checkpoint system was added.
"""

from DbController import DuckDBController, DuckDBConfig

def backfill():
    db = DuckDBController(DuckDBConfig(path="data/research.duckdb"))
    
    # Check existing data
    existing = db.con.execute("""
        SELECT COUNT(DISTINCT token_id) as tokens, COUNT(*) as prices 
        FROM prices
    """).fetchone()
    
    print("\n" + "="*70)
    print("BACKFILL CHECKPOINTS FROM EXISTING DATA")
    print("="*70)
    print(f"Found existing data:")
    print(f"  📊 Tokens: {existing[0]}")
    print(f"  📊 Price points: {existing[1]:,}")
    
    # Check current checkpoint status
    checkpoint_count = db.con.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    print(f"\nCurrent checkpoints: {checkpoint_count}")
    
    if checkpoint_count > 0:
        print("\n⚠️  Warning: Checkpoint table already has entries.")
        response = input("Continue with backfill? This will add any missing tokens. (y/n): ")
        if response.lower() != 'y':
            print("❌ Cancelled.")
            db.close()
            return
    
    print("\n🔄 Backfilling checkpoints...")
    count = db.backfill_checkpoints_from_existing_data()
    
    # Show updated stats
    stats = db.get_checkpoint_stats()
    print("\n" + "="*70)
    print("UPDATED CHECKPOINT STATUS:")
    print(f"  ✅ Completed tokens: {stats['completed']}")
    print(f"  ❌ Failed tokens: {stats['failed']}")
    print(f"  📊 Total prices: {stats['total_prices']:,}")
    print("="*70)
    print(f"\n✅ Done! Now when you run main.py, it will skip these {stats['completed']} tokens.")
    print("\n" + "="*70 + "\n")
    
    db.close()

if __name__ == "__main__":
    backfill()
