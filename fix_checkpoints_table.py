#!/usr/bin/env python3
"""
Fix the checkpoints table to add missing columns.
Run this once to update your database schema.
"""

from DbController import DuckDBController, DuckDBConfig

def fix_checkpoints():
    db = DuckDBController(DuckDBConfig(path="data/research.duckdb"))
    
    print("Checking checkpoints table...")
    
    # Check if table exists
    try:
        result = db.con.execute("SELECT * FROM checkpoints LIMIT 0;").fetchall()
        print("✅ Checkpoints table exists")
    except:
        print("❌ Checkpoints table doesn't exist - it will be created automatically")
        db.close()
        return
    
    # Check current columns
    info = db.con.execute("PRAGMA table_info('checkpoints')").df()
    existing_cols = set(info['name'].tolist())
    print(f"\nExisting columns: {', '.join(sorted(existing_cols))}")
    
    # Add missing columns
    required_cols = {
        'processed_at': 'TIMESTAMP',
        'error_msg': 'TEXT',
        'num_prices': 'BIGINT',
        'status': 'TEXT',
        'market_id': 'TEXT'
    }
    
    added = []
    for col, typ in required_cols.items():
        if col not in existing_cols:
            print(f"Adding column: {col} {typ}")
            db.con.execute(f"ALTER TABLE checkpoints ADD COLUMN {col} {typ};")
            added.append(col)
    
    if added:
        print(f"\n✅ Added {len(added)} column(s): {', '.join(added)}")
    else:
        print("\n✅ All columns already exist!")
    
    # Show final schema
    info = db.con.execute("PRAGMA table_info('checkpoints')").df()
    print("\nFinal schema:")
    print(info[['name', 'type']].to_string(index=False))
    
    db.close()

if __name__ == "__main__":
    fix_checkpoints()
