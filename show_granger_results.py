#!/usr/bin/env python3
"""
Display Granger causality results with market questions and analysis windows.
Shows tokens that have significant Granger causality relationships with SPY.
"""
import os
import duckdb
import pandas as pd

DB_MKT = os.getenv("MKT_DB", "./data/markets.duckdb")
DB_POLY = os.getenv("POLY_DB", "./data/research.duckdb")

def get_conn():
    """Create connection with attached databases."""
    conn = duckdb.connect()
    mkt_path = os.path.abspath(DB_MKT).replace("'", "''")
    poly_path = os.path.abspath(DB_POLY).replace("'", "''")
    
    conn.execute(f"ATTACH DATABASE '{mkt_path}' AS mkt")
    if poly_path != mkt_path:
        conn.execute(f"ATTACH DATABASE '{poly_path}' AS poly")
    return conn

def show_results(min_sig_level: float = 0.05, direction: str = "both"):
    """
    Display Granger causality results.
    
    Args:
        min_sig_level: Minimum significance level (corrected p-value threshold)
        direction: Filter by direction - "both" (bidirectional), "poly_to_eq", "eq_to_poly", or "any"
    """
    conn = get_conn()
    
    try:
        # Build query based on direction filter
        if direction == "both":
            direction_filter = "AND g.sig_poly_to_eq = TRUE AND g.sig_eq_to_poly = TRUE"
            title = "BIDIRECTIONAL GRANGER CAUSALITY"
        elif direction == "poly_to_eq":
            direction_filter = "AND g.sig_poly_to_eq = TRUE"
            title = "POLYMARKET → EQUITY CAUSALITY"
        elif direction == "eq_to_poly":
            direction_filter = "AND g.sig_eq_to_poly = TRUE"
            title = "EQUITY → POLYMARKET CAUSALITY"
        else:  # "any"
            direction_filter = "AND (g.sig_poly_to_eq = TRUE OR g.sig_eq_to_poly = TRUE)"
            title = "SIGNIFICANT GRANGER CAUSALITY (ANY DIRECTION)"
        
        query = f"""
            SELECT 
                g.token_id,
                m.question,
                m.slug as market_slug,
                g.ticker,
                g.n_obs,
                g.start_ts,
                g.end_ts,
                CAST((EXTRACT(EPOCH FROM (g.end_ts - g.start_ts)) / 86400) AS INT) as days,
                g.sig_poly_to_eq,
                g.lag_poly_to_eq,
                g.p_poly_to_eq_corrected,
                g.sig_eq_to_poly,
                g.lag_eq_to_poly,
                g.p_eq_to_poly_corrected,
                g.poly_stationary,
                g.eq_stationary
            FROM mkt.main.granger_results g
            LEFT JOIN poly.main.tokens t ON g.token_id = t.token_id
            LEFT JOIN poly.main.markets m ON t.market_id = m.market_id
            WHERE (g.p_poly_to_eq_corrected <= ? OR g.p_eq_to_poly_corrected <= ?)
            {direction_filter}
            ORDER BY 
                CASE 
                    WHEN g.sig_poly_to_eq AND g.sig_eq_to_poly THEN 0  -- bidirectional first
                    WHEN g.sig_poly_to_eq THEN 1  -- poly->eq second
                    WHEN g.sig_eq_to_poly THEN 2  -- eq->poly third
                    ELSE 3
                END,
                LEAST(g.p_poly_to_eq_corrected, g.p_eq_to_poly_corrected) ASC
        """
        
        df = conn.execute(query, [min_sig_level, min_sig_level]).df()
        
        if df.empty:
            print(f"\nNo results found with significance level <= {min_sig_level}")
            return
        
        print("=" * 100)
        print(title)
        print("=" * 100)
        print(f"Significance Level (Bonferroni-corrected): p <= {min_sig_level}")
        print(f"Total Results: {len(df)}")
        print("=" * 100)
        
        for idx, row in df.iterrows():
            print(f"\n[{idx + 1}] {row['question'][:80]}")
            print(f"    Market: {row['market_slug'] or 'Unknown'}")
            print(f"    Token ID: {row['token_id'][:50]}...")
            print(f"    Ticker: {row['ticker']}")
            print()
            print(f"    Analysis Window:")
            print(f"      Start: {row['start_ts']}")
            print(f"      End:   {row['end_ts']}")
            print(f"      Duration: {row['days']} days ({row['n_obs']:,} observations)")
            print()
            
            # Polymarket -> Equity
            if row['sig_poly_to_eq']:
                print(f"    📊 Polymarket → Equity:")
                print(f"       Lag: {row['lag_poly_to_eq']} minutes")
                print(f"       p-value: {row['p_poly_to_eq_corrected']:.6f} ✓ SIGNIFICANT")
            else:
                print(f"    📊 Polymarket → Equity: Not significant (p={row['p_poly_to_eq_corrected']:.4f})")
            
            # Equity -> Polymarket
            if row['sig_eq_to_poly']:
                print(f"    📈 Equity → Polymarket:")
                print(f"       Lag: {row['lag_eq_to_poly']} minutes")
                print(f"       p-value: {row['p_eq_to_poly_corrected']:.6f} ✓ SIGNIFICANT")
            else:
                print(f"    📈 Equity → Polymarket: Not significant (p={row['p_eq_to_poly_corrected']:.4f})")
            
            # Stationarity
            print()
            stationary_status = "✓" if (row['poly_stationary'] and row['eq_stationary']) else "⚠️"
            print(f"    {stationary_status} Data Quality: Poly stationary={row['poly_stationary']}, Equity stationary={row['eq_stationary']}")
            
            print("-" * 100)
        
        # Summary statistics
        print("\n" + "=" * 100)
        print("SUMMARY STATISTICS")
        print("=" * 100)
        
        bidirectional = df[df['sig_poly_to_eq'] & df['sig_eq_to_poly']]
        poly_to_eq_only = df[df['sig_poly_to_eq'] & ~df['sig_eq_to_poly']]
        eq_to_poly_only = df[~df['sig_poly_to_eq'] & df['sig_eq_to_poly']]
        
        print(f"\nDirection Breakdown:")
        print(f"  Bidirectional (both ways): {len(bidirectional)}")
        print(f"  Polymarket → Equity only: {len(poly_to_eq_only)}")
        print(f"  Equity → Polymarket only: {len(eq_to_poly_only)}")
        
        if not bidirectional.empty:
            print(f"\nBidirectional Markets (Feedback Loops):")
            print(f"  Average lag Poly→Eq: {bidirectional['lag_poly_to_eq'].mean():.1f} minutes")
            print(f"  Average lag Eq→Poly: {bidirectional['lag_eq_to_poly'].mean():.1f} minutes")
        
        # Lag analysis
        sig_poly_to_eq = df[df['sig_poly_to_eq']]
        sig_eq_to_poly = df[df['sig_eq_to_poly']]
        
        if not sig_poly_to_eq.empty:
            print(f"\nPolymarket → Equity Timing:")
            print(f"  Mean lag: {sig_poly_to_eq['lag_poly_to_eq'].mean():.1f} minutes")
            print(f"  Median lag: {sig_poly_to_eq['lag_poly_to_eq'].median():.0f} minutes")
            print(f"  Range: {sig_poly_to_eq['lag_poly_to_eq'].min()}-{sig_poly_to_eq['lag_poly_to_eq'].max()} minutes")
        
        if not sig_eq_to_poly.empty:
            print(f"\nEquity → Polymarket Timing:")
            print(f"  Mean lag: {sig_eq_to_poly['lag_eq_to_poly'].mean():.1f} minutes")
            print(f"  Median lag: {sig_eq_to_poly['lag_eq_to_poly'].median():.0f} minutes")
            print(f"  Range: {sig_eq_to_poly['lag_eq_to_poly'].min()}-{sig_eq_to_poly['lag_eq_to_poly'].max()} minutes")
        
        # Data quality
        stationary = df[df['poly_stationary'] & df['eq_stationary']]
        print(f"\nData Quality:")
        print(f"  Stationary (both series): {len(stationary)} / {len(df)} ({100*len(stationary)/len(df):.1f}%)")
        
        print("=" * 100)
        
    finally:
        conn.close()

def export_to_csv(filename: str = "granger_results.csv", min_sig_level: float = 0.05):
    """Export results to CSV file."""
    conn = get_conn()
    
    try:
        query = """
            SELECT 
                g.token_id,
                m.question,
                m.slug as market_slug,
                g.ticker,
                g.n_obs,
                g.start_ts,
                g.end_ts,
                g.sig_poly_to_eq,
                g.lag_poly_to_eq,
                g.p_poly_to_eq_corrected,
                g.sig_eq_to_poly,
                g.lag_eq_to_poly,
                g.p_eq_to_poly_corrected,
                g.poly_stationary,
                g.eq_stationary
            FROM mkt.main.granger_results g
            LEFT JOIN poly.main.tokens t ON g.token_id = t.token_id
            LEFT JOIN poly.main.markets m ON t.market_id = m.market_id
            WHERE g.p_poly_to_eq_corrected <= ? OR g.p_eq_to_poly_corrected <= ?
            ORDER BY LEAST(g.p_poly_to_eq_corrected, g.p_eq_to_poly_corrected) ASC
        """
        
        df = conn.execute(query, [min_sig_level, min_sig_level]).df()
        df.to_csv(filename, index=False)
        print(f"Exported {len(df)} results to {filename}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Display Granger causality analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all significant results (any direction)
  python show_granger_results.py
  
  # Show only bidirectional causality
  python show_granger_results.py --direction both
  
  # Show only Polymarket → Equity
  python show_granger_results.py --direction poly_to_eq
  
  # Use stricter significance level
  python show_granger_results.py --sig-level 0.01
  
  # Export to CSV
  python show_granger_results.py --export results.csv
        """
    )
    
    parser.add_argument(
        "--sig-level",
        type=float,
        default=0.05,
        help="Significance level threshold (Bonferroni-corrected p-value, default: 0.05)"
    )
    
    parser.add_argument(
        "--direction",
        choices=["any", "both", "poly_to_eq", "eq_to_poly"],
        default="any",
        help="Filter by causality direction (default: any)"
    )
    
    parser.add_argument(
        "--export",
        type=str,
        metavar="FILENAME",
        help="Export results to CSV file"
    )
    
    args = parser.parse_args()
    
    if args.export:
        export_to_csv(args.export, args.sig_level)
    else:
        show_results(args.sig_level, args.direction)
