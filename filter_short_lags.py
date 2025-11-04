#!/usr/bin/env python3
"""
Create filtered Granger results excluding ultra-short lags (≤3 minutes).

These short lags are synchronous (not predictive) based on validation.
"""
import duckdb
import sys

DB_MKT = "./data/markets.duckdb"

def filter_results(min_lag=4):
    """Filter Granger results to exclude ultra-short lags."""
    conn = duckdb.connect(DB_MKT)
    
    try:
        # Count before filtering
        total_poly = conn.execute(
            "SELECT COUNT(*) FROM granger_results WHERE sig_poly_to_eq = TRUE"
        ).fetchone()[0]
        
        total_eq = conn.execute(
            "SELECT COUNT(*) FROM granger_results WHERE sig_eq_to_poly = TRUE"
        ).fetchone()[0]
        
        # Count ultra-short lags
        short_poly = conn.execute(
            f"SELECT COUNT(*) FROM granger_results WHERE sig_poly_to_eq = TRUE AND lag_poly_to_eq <= {min_lag - 1}"
        ).fetchone()[0]
        
        short_eq = conn.execute(
            f"SELECT COUNT(*) FROM granger_results WHERE sig_eq_to_poly = TRUE AND lag_eq_to_poly <= {min_lag - 1}"
        ).fetchone()[0]
        
        print("=" * 90)
        print("FILTERING GRANGER RESULTS - REMOVING ULTRA-SHORT LAGS")
        print("=" * 90)
        print(f"\nMinimum lag threshold: {min_lag} minutes")
        print(f"Rationale: Lags ≤{min_lag-1} min are synchronous (not predictive)\n")
        
        print("Before Filtering:")
        print(f"  Polymarket → Equity: {total_poly} significant")
        print(f"  Equity → Polymarket: {total_eq} significant")
        print(f"\nUltra-short lags to remove:")
        print(f"  Polymarket → Equity (≤{min_lag-1} min): {short_poly}")
        print(f"  Equity → Polymarket (≤{min_lag-1} min): {short_eq}")
        
        # Create filtered table
        conn.execute("DROP TABLE IF EXISTS granger_results_filtered")
        
        query = f"""
            CREATE TABLE granger_results_filtered AS
            SELECT *
            FROM granger_results
            WHERE (sig_poly_to_eq = FALSE OR lag_poly_to_eq >= {min_lag})
              AND (sig_eq_to_poly = FALSE OR lag_eq_to_poly >= {min_lag})
        """
        conn.execute(query)
        
        # Count after filtering
        filtered_poly = conn.execute(
            "SELECT COUNT(*) FROM granger_results_filtered WHERE sig_poly_to_eq = TRUE"
        ).fetchone()[0]
        
        filtered_eq = conn.execute(
            "SELECT COUNT(*) FROM granger_results_filtered WHERE sig_eq_to_poly = TRUE"
        ).fetchone()[0]
        
        filtered_both = conn.execute(
            "SELECT COUNT(*) FROM granger_results_filtered WHERE sig_poly_to_eq = TRUE AND sig_eq_to_poly = TRUE"
        ).fetchone()[0]
        
        print(f"\nAfter Filtering:")
        print(f"  Polymarket → Equity: {filtered_poly} significant ({100*filtered_poly/total_poly:.1f}% retained)")
        print(f"  Equity → Polymarket: {filtered_eq} significant ({100*filtered_eq/total_eq:.1f}% retained)")
        print(f"  Bidirectional: {filtered_both} markets")
        
        print(f"\n✓ Created table: granger_results_filtered")
        print(f"✓ Total markets: {conn.execute('SELECT COUNT(*) FROM granger_results_filtered').fetchone()[0]}")
        
        # Show lag distribution
        print(f"\n{'=' * 90}")
        print("LAG DISTRIBUTION (Filtered Results)")
        print(f"{'=' * 90}")
        
        print("\nPolymarket → Equity:")
        poly_lags = conn.execute("""
            SELECT 
                lag_poly_to_eq as lag,
                COUNT(*) as count
            FROM granger_results_filtered
            WHERE sig_poly_to_eq = TRUE
            GROUP BY lag_poly_to_eq
            ORDER BY lag_poly_to_eq
        """).fetchall()
        
        for lag, count in poly_lags:
            print(f"  {lag:2d} min: {count:2d} markets")
        
        print("\nEquity → Polymarket:")
        eq_lags = conn.execute("""
            SELECT 
                lag_eq_to_poly as lag,
                COUNT(*) as count
            FROM granger_results_filtered
            WHERE sig_eq_to_poly = TRUE
            GROUP BY lag_eq_to_poly
            ORDER BY lag_eq_to_poly
        """).fetchall()
        
        for lag, count in eq_lags:
            print(f"  {lag:2d} min: {count:2d} markets")
        
        # Statistics
        poly_stats = conn.execute("""
            SELECT 
                AVG(lag_poly_to_eq) as mean_lag,
                MEDIAN(lag_poly_to_eq) as median_lag,
                MIN(lag_poly_to_eq) as min_lag,
                MAX(lag_poly_to_eq) as max_lag
            FROM granger_results_filtered
            WHERE sig_poly_to_eq = TRUE
        """).fetchone()
        
        eq_stats = conn.execute("""
            SELECT 
                AVG(lag_eq_to_poly) as mean_lag,
                MEDIAN(lag_eq_to_poly) as median_lag,
                MIN(lag_eq_to_poly) as min_lag,
                MAX(lag_eq_to_poly) as max_lag
            FROM granger_results_filtered
            WHERE sig_eq_to_poly = TRUE
        """).fetchone()
        
        print(f"\n{'=' * 90}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 90}")
        
        print(f"\nPolymarket → Equity Timing:")
        print(f"  Mean lag: {poly_stats[0]:.1f} minutes")
        print(f"  Median lag: {poly_stats[1]:.0f} minutes")
        print(f"  Range: {poly_stats[2]}-{poly_stats[3]} minutes")
        
        print(f"\nEquity → Polymarket Timing:")
        print(f"  Mean lag: {eq_stats[0]:.1f} minutes")
        print(f"  Median lag: {eq_stats[1]:.0f} minutes")
        print(f"  Range: {eq_stats[2]}-{eq_stats[3]} minutes")
        
        print(f"\n{'=' * 90}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    min_lag = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    filter_results(min_lag)
