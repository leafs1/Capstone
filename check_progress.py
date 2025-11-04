#!/usr/bin/env python3
"""
Script to check the progress of data collection.
Shows checkpoint statistics and lists any failed tokens.
"""

from DbController import DuckDBController, DuckDBConfig
import sys

def check_progress():
    db = DuckDBController(DuckDBConfig(path="data/research.duckdb"))
    
    # Get overall statistics
    stats = db.get_checkpoint_stats()
    
    print("\n" + "="*70)
    print("DATA COLLECTION PROGRESS")
    print("="*70)
    print(f"✅ Completed tokens:     {stats['completed']}")
    print(f"❌ Failed tokens:        {stats['failed']}")
    if stats.get('incomplete', 0) > 0:
        print(f"⚠️  Incomplete tokens:    {stats['incomplete']}")
    print(f"📊 Total prices stored:  {stats['total_prices']:,}")
    total = stats['completed'] + stats['failed'] + stats.get('incomplete', 0)
    if total > 0:
        success_rate = (stats['completed'] / total) * 100
        print(f"📈 Success rate:         {success_rate:.1f}%")
    print("="*70)
    
    # Show recent completions
    print("\n📋 Recent Completions (last 10):")
    recent = db.raw("""
        SELECT token_id, market_id, processed_at, num_prices
        FROM checkpoints
        WHERE status = 'completed'
        ORDER BY processed_at DESC
        LIMIT 10;
    """)
    if not recent.empty:
        for _, row in recent.iterrows():
            print(f"  {row['token_id'][:16]}... - {row['num_prices']:,} prices at {row['processed_at']}")
    else:
        print("  (none yet)")
    
    # Show failed tokens if any
    failed = db.raw("""
        SELECT token_id, market_id, error_msg, processed_at
        FROM checkpoints
        WHERE status = 'failed'
        ORDER BY processed_at DESC;
    """)
    
    if not failed.empty:
        print(f"\n⚠️  Failed Tokens ({len(failed)}):")
        for _, row in failed.iterrows():
            error_preview = row['error_msg'][:60] + "..." if len(row['error_msg']) > 60 else row['error_msg']
            print(f"  {row['token_id'][:16]}... - {error_preview}")
    else:
        print("\n✅ No failed tokens!")
    
    # Show incomplete tokens if any
    incomplete = db.raw("""
        SELECT token_id, market_id, error_msg, num_prices, processed_at
        FROM checkpoints
        WHERE status = 'incomplete'
        ORDER BY processed_at DESC;
    """)
    
    if not incomplete.empty:
        print(f"\n⚠️  Incomplete Tokens ({len(incomplete)}) - will be re-downloaded:")
        for _, row in incomplete.iterrows():
            print(f"  {row['token_id'][:16]}... - {row['num_prices']} prices (incomplete range)")
    
    # Check if we can estimate total work
    try:
        total_tokens = db.raw("SELECT COUNT(DISTINCT token_id) FROM tokens;")
        if not total_tokens.empty:
            total_count = total_tokens.iloc[0, 0]
            remaining = total_count - stats['completed']
            print(f"\n🎯 Progress: {stats['completed']}/{total_count} tokens ({(stats['completed']/total_count*100):.1f}%)")
            print(f"   Remaining: {remaining} tokens")
    except:
        pass
    
    print("\n" + "="*70 + "\n")
    
    db.close()

def reset_failed():
    """Reset failed tokens to allow retry."""
    db = DuckDBController(DuckDBConfig(path="data/research.duckdb"))
    
    result = db.con.execute("SELECT COUNT(*) FROM checkpoints WHERE status = 'failed';").fetchone()
    count = result[0]
    
    if count == 0:
        print("No failed tokens to reset.")
        db.close()
        return
    
    print(f"Found {count} failed tokens.")
    response = input("Reset these to allow retry? (y/n): ")
    
    if response.lower() == 'y':
        db.con.execute("DELETE FROM checkpoints WHERE status = 'failed';")
        print(f"✅ Reset {count} failed tokens. They will be retried on next run.")
    else:
        print("❌ Cancelled.")
    
    db.close()

def reset_incomplete():
    """Reset incomplete tokens to allow retry."""
    db = DuckDBController(DuckDBConfig(path="data/research.duckdb"))
    
    result = db.con.execute("SELECT COUNT(*) FROM checkpoints WHERE status = 'incomplete';").fetchone()
    count = result[0]
    
    if count == 0:
        print("No incomplete tokens to reset.")
        db.close()
        return
    
    print(f"Found {count} incomplete tokens.")
    response = input("Reset these to allow retry? (y/n): ")
    
    if response.lower() == 'y':
        db.con.execute("DELETE FROM checkpoints WHERE status = 'incomplete';")
        print(f"✅ Reset {count} incomplete tokens. They will be retried on next run.")
    else:
        print("❌ Cancelled.")
    
    db.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--reset-failed":
        reset_failed()
    elif len(sys.argv) > 1 and sys.argv[1] == "--reset-incomplete":
        reset_incomplete()
    else:
        check_progress()
