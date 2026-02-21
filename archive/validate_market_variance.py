#!/usr/bin/env python3
"""
Validate Step 1: Check if significant markets have meaningful price variation.
Markets stuck at constant probabilities might show spurious correlations.
"""
import os
import duckdb
import pandas as pd
import numpy as np

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

def analyze_market_variance():
    """
    Analyze price variation for all tokens with significant Granger causality.
    Shows if markets have meaningful movement or are stuck at fixed probabilities.
    """
    conn = get_conn()
    
    try:
        # Get all tokens with significant bidirectional causality
        query = """
            SELECT DISTINCT
                g.token_id,
                m.question,
                m.slug,
                g.start_ts,
                g.end_ts,
                g.n_obs
            FROM mkt.main.granger_results g
            LEFT JOIN poly.main.tokens t ON g.token_id = t.token_id
            LEFT JOIN poly.main.markets m ON t.market_id = m.market_id
            WHERE g.sig_poly_to_eq = TRUE 
            AND g.sig_eq_to_poly = TRUE
            ORDER BY m.question
        """
        
        tokens_df = conn.execute(query).df()
        
        print("=" * 100)
        print("MARKET PRICE VARIATION ANALYSIS")
        print("=" * 100)
        print(f"Analyzing {len(tokens_df)} tokens with bidirectional Granger causality\n")
        
        results = []
        
        for idx, row in tokens_df.iterrows():
            token_id = row['token_id']
            question = row['question']
            start_ts = row['start_ts']
            end_ts = row['end_ts']
            
            # Get price statistics for this token during the analysis window
            price_stats = conn.execute("""
                SELECT 
                    COUNT(*) as n_prices,
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    AVG(price) as avg_price,
                    STDDEV(price) as stddev_price,
                    MEDIAN(price) as median_price,
                    -- Price percentiles
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) as p25,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) as p75,
                    -- Count distinct price levels (rounded to 2 decimals)
                    COUNT(DISTINCT ROUND(price, 2)) as n_distinct_prices
                FROM poly.main.prices
                WHERE token_id = ?
                AND to_timestamp(ts) BETWEEN ? AND ?
            """, [token_id, start_ts, end_ts]).df()
            
            if not price_stats.empty:
                stats = price_stats.iloc[0]
                price_range = stats['max_price'] - stats['min_price']
                iqr = stats['p75'] - stats['p25']
                cv = stats['stddev_price'] / stats['avg_price'] if stats['avg_price'] > 0 else 0
                
                # Classify variation level
                if price_range < 0.05:
                    variation_level = "🔴 VERY LOW (< 5% range)"
                elif price_range < 0.15:
                    variation_level = "🟡 LOW (5-15% range)"
                elif price_range < 0.30:
                    variation_level = "🟢 MODERATE (15-30% range)"
                else:
                    variation_level = "🟢 HIGH (> 30% range)"
                
                result = {
                    'question': question[:60],
                    'n_prices': stats['n_prices'],
                    'min': stats['min_price'],
                    'max': stats['max_price'],
                    'range': price_range,
                    'mean': stats['avg_price'],
                    'median': stats['median_price'],
                    'stddev': stats['stddev_price'],
                    'cv': cv,
                    'iqr': iqr,
                    'n_distinct': stats['n_distinct_prices'],
                    'variation_level': variation_level
                }
                results.append(result)
                
                print(f"\n[{idx + 1}] {question[:70]}")
                print(f"    Price Range: {stats['min_price']:.3f} - {stats['max_price']:.3f} (range: {price_range:.3f})")
                print(f"    Mean: {stats['avg_price']:.3f}, Median: {stats['median_price']:.3f}, StdDev: {stats['stddev_price']:.4f}")
                print(f"    Coefficient of Variation: {cv:.4f}")
                print(f"    IQR (25th-75th): {iqr:.3f}")
                print(f"    Distinct price levels: {stats['n_distinct_prices']}")
                print(f"    {variation_level}")
        
        # Summary statistics
        print("\n" + "=" * 100)
        print("SUMMARY STATISTICS")
        print("=" * 100)
        
        df_results = pd.DataFrame(results)
        
        # Categorize by variation level
        very_low = df_results[df_results['range'] < 0.05]
        low = df_results[(df_results['range'] >= 0.05) & (df_results['range'] < 0.15)]
        moderate = df_results[(df_results['range'] >= 0.15) & (df_results['range'] < 0.30)]
        high = df_results[df_results['range'] >= 0.30]
        
        print(f"\nVariation Distribution:")
        print(f"  🔴 Very Low (<5% range): {len(very_low)} markets")
        print(f"  🟡 Low (5-15% range): {len(low)} markets")
        print(f"  🟢 Moderate (15-30% range): {len(moderate)} markets")
        print(f"  🟢 High (>30% range): {len(high)} markets")
        
        print(f"\nOverall Statistics:")
        print(f"  Mean price range: {df_results['range'].mean():.3f}")
        print(f"  Median price range: {df_results['range'].median():.3f}")
        print(f"  Mean coefficient of variation: {df_results['cv'].mean():.4f}")
        print(f"  Mean distinct price levels: {df_results['n_distinct'].mean():.0f}")
        
        # Flag potentially problematic markets
        problematic = df_results[df_results['range'] < 0.10]
        if not problematic.empty:
            print(f"\n⚠️  POTENTIAL CONCERNS ({len(problematic)} markets with <10% range):")
            for idx, row in problematic.iterrows():
                print(f"    • {row['question']}")
                print(f"      Range: {row['range']:.3f}, StdDev: {row['stddev']:.4f}")
        else:
            print(f"\n✅ All markets show healthy variation (>10% range)")
        
        # Check for extreme probabilities (stuck near 0 or 1)
        extreme_low = df_results[df_results['mean'] < 0.10]
        extreme_high = df_results[df_results['mean'] > 0.90]
        
        if not extreme_low.empty:
            print(f"\n📊 Markets with very low probability (mean < 10%):")
            for idx, row in extreme_low.iterrows():
                print(f"    • {row['question']} (mean: {row['mean']:.3f})")
        
        if not extreme_high.empty:
            print(f"\n📊 Markets with very high probability (mean > 90%):")
            for idx, row in extreme_high.iterrows():
                print(f"    • {row['question']} (mean: {row['mean']:.3f})")
        
        print("\n" + "=" * 100)
        print("INTERPRETATION GUIDE:")
        print("=" * 100)
        print("""
✅ GOOD: High variation (>30% range) + moderate mean (0.3-0.7)
   → Market has real price discovery, correlations likely meaningful

⚠️  CAUTION: Low variation (<10% range) OR extreme mean (<0.1 or >0.9)
   → Market may be stuck, correlations could be spurious

🔍 EXAMINE: Check if low-variation markets cluster around specific themes
   → If all circuit breaker markets have low variation, might indicate
      market consensus rather than lack of information
        """)
        
        return df_results
        
    finally:
        conn.close()

if __name__ == "__main__":
    df = analyze_market_variance()
    
    # Optional: export to CSV
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        filename = sys.argv[2] if len(sys.argv) > 2 else "market_variance.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ Exported variance analysis to {filename}")
