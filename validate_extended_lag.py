#!/usr/bin/env python3
"""
Validate Step 2: Re-run Granger analysis with extended maxlag=60 minutes.
Check if relationships hitting maxlag=30 actually extend further.
"""
import os
import sys
import duckdb
import pandas as pd

# Add parent directory to path to import from Granger.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Granger import (
    get_conn, granger_summary, overlap_window,
    load_polymarket, load_equity, merge_poly_with_equity, 
    make_returns, test_stationarity, bonferroni_correction
)
from statsmodels.tsa.stattools import grangercausalitytests

def extended_lag_analysis(token_ids: list = None, maxlag: int = 60):
    """
    Re-run Granger causality analysis with extended lag for specific tokens.
    Compare results with original maxlag=30 analysis.
    
    Args:
        token_ids: List of token IDs to analyze. If None, uses top markets hitting maxlag=30
        maxlag: Maximum lag to test (default: 60 minutes)
    """
    conn = get_conn()
    
    try:
        # If no token_ids provided, get tokens with high lags (28-30 min) from original analysis
        if token_ids is None:
            query = """
                SELECT DISTINCT
                    g.token_id,
                    m.question,
                    g.lag_poly_to_eq,
                    g.lag_eq_to_poly,
                    g.p_poly_to_eq_corrected,
                    g.p_eq_to_poly_corrected
                FROM mkt.main.granger_results g
                LEFT JOIN poly.main.tokens t ON g.token_id = t.token_id
                LEFT JOIN poly.main.markets m ON t.market_id = m.market_id
                WHERE (g.lag_poly_to_eq >= 28 OR g.lag_eq_to_poly >= 28)
                AND g.sig_poly_to_eq = TRUE 
                AND g.sig_eq_to_poly = TRUE
                ORDER BY LEAST(g.p_poly_to_eq_corrected, g.p_eq_to_poly_corrected)
                LIMIT 10
            """
            df = conn.execute(query).df()
            token_ids = df['token_id'].tolist()
            
            print("=" * 100)
            print("EXTENDED LAG ANALYSIS (maxlag=60)")
            print("=" * 100)
            print(f"Re-analyzing {len(token_ids)} tokens that hit maxlag boundaries (28-30 min)\n")
        
        results = []
        
        for idx, token_id in enumerate(token_ids):
            # Get market question
            market_info = conn.execute("""
                SELECT m.question, m.slug
                FROM poly.main.tokens t
                LEFT JOIN poly.main.markets m ON t.market_id = m.market_id
                WHERE t.token_id = ?
            """, [token_id]).df()
            
            if market_info.empty:
                continue
            
            question = market_info['question'].iloc[0]
            print(f"\n[{idx + 1}/{len(token_ids)}] {question[:70]}")
            print(f"Token: {token_id[:50]}...")
            
            # Get overlap window
            ow = overlap_window(conn, token_id, "SPY")
            if not ow:
                print("  → No overlap with equity data")
                continue
            
            start, end = ow
            
            # Load and merge data
            poly = load_polymarket(conn, token_id, start, end)
            eq = load_equity(conn, "SPY", start, end)
            df = merge_poly_with_equity(poly, eq)
            
            if df.empty:
                print("  → No merged data")
                continue
            
            # Calculate returns
            r = pd.DataFrame({
                'poly': make_returns(df['poly'], is_probability=True),
                'eq': make_returns(df['eq'], is_probability=False)
            }).dropna()
            
            if len(r) < maxlag * 3:
                print(f"  → Insufficient observations: {len(r)}")
                continue
            
            print(f"  Observations: {len(r):,}")
            
            # Run Granger tests with extended lag
            try:
                res1 = grangercausalitytests(r[["poly", "eq"]], maxlag=maxlag, verbose=False)
                pvals1 = {lag: res[0]["ssr_ftest"][1] for lag, res in res1.items()}
                corrected1 = bonferroni_correction(pvals1)
                
                res2 = grangercausalitytests(r[["eq", "poly"]], maxlag=maxlag, verbose=False)
                pvals2 = {lag: res[0]["ssr_ftest"][1] for lag, res in res2.items()}
                corrected2 = bonferroni_correction(pvals2)
                
                # Get original results (maxlag=30) for comparison
                original = conn.execute("""
                    SELECT 
                        lag_poly_to_eq as orig_lag_pty,
                        p_poly_to_eq_corrected as orig_p_pty,
                        lag_eq_to_poly as orig_lag_eqy,
                        p_eq_to_poly_corrected as orig_p_eqy
                    FROM mkt.main.granger_results
                    WHERE token_id = ?
                    LIMIT 1
                """, [token_id]).df()
                
                result = {
                    'token_id': token_id,
                    'question': question[:60],
                    'n_obs': len(r),
                    # New results (maxlag=60)
                    'new_lag_pty': corrected1['min_lag'],
                    'new_p_pty': corrected1['corrected_pvalue'],
                    'new_sig_pty': corrected1['is_significant'],
                    'new_lag_eqy': corrected2['min_lag'],
                    'new_p_eqy': corrected2['corrected_pvalue'],
                    'new_sig_eqy': corrected2['is_significant'],
                }
                
                # Add original results if available
                if not original.empty:
                    result.update({
                        'orig_lag_pty': int(original['orig_lag_pty'].iloc[0]),
                        'orig_p_pty': float(original['orig_p_pty'].iloc[0]),
                        'orig_lag_eqy': int(original['orig_lag_eqy'].iloc[0]),
                        'orig_p_eqy': float(original['orig_p_eqy'].iloc[0]),
                    })
                
                results.append(result)
                
                # Print comparison
                print(f"\n  📊 Polymarket → Equity:")
                if not original.empty:
                    print(f"     Original (maxlag=30): lag={result['orig_lag_pty']} min, p={result['orig_p_pty']:.6f}")
                print(f"     Extended (maxlag=60): lag={result['new_lag_pty']} min, p={result['new_p_pty']:.6f}")
                
                if result['new_lag_pty'] > 30:
                    print(f"     ⚠️  EXTENDS BEYOND 30 MIN! True optimal lag is {result['new_lag_pty']} min")
                elif result['new_lag_pty'] == result.get('orig_lag_pty'):
                    print(f"     ✓ Confirmed: Optimal lag is {result['new_lag_pty']} min (same as original)")
                else:
                    print(f"     🔄 Changed: {result.get('orig_lag_pty')} → {result['new_lag_pty']} min")
                
                print(f"\n  📈 Equity → Polymarket:")
                if not original.empty:
                    print(f"     Original (maxlag=30): lag={result['orig_lag_eqy']} min, p={result['orig_p_eqy']:.6f}")
                print(f"     Extended (maxlag=60): lag={result['new_lag_eqy']} min, p={result['new_p_eqy']:.6f}")
                
                if result['new_lag_eqy'] > 30:
                    print(f"     ⚠️  EXTENDS BEYOND 30 MIN! True optimal lag is {result['new_lag_eqy']} min")
                elif result['new_lag_eqy'] == result.get('orig_lag_eqy'):
                    print(f"     ✓ Confirmed: Optimal lag is {result['new_lag_eqy']} min (same as original)")
                else:
                    print(f"     🔄 Changed: {result.get('orig_lag_eqy')} → {result['new_lag_eqy']} min")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:80]}")
                continue
        
        # Summary
        if results:
            df_results = pd.DataFrame(results)
            
            print("\n" + "=" * 100)
            print("SUMMARY: LAG EXTENSION FINDINGS")
            print("=" * 100)
            
            # Check how many extend beyond 30
            pty_extends = df_results[df_results['new_lag_pty'] > 30]
            eqy_extends = df_results[df_results['new_lag_eqy'] > 30]
            
            print(f"\nMarkets where relationship extends beyond 30 minutes:")
            print(f"  Polymarket → Equity: {len(pty_extends)} / {len(df_results)}")
            if not pty_extends.empty:
                for _, row in pty_extends.iterrows():
                    print(f"    • {row['question']} (lag: {row['new_lag_pty']} min)")
            
            print(f"\n  Equity → Polymarket: {len(eqy_extends)} / {len(df_results)}")
            if not eqy_extends.empty:
                for _, row in eqy_extends.iterrows():
                    print(f"    • {row['question']} (lag: {row['new_lag_eqy']} min)")
            
            # Check if lags changed (even within 30)
            if 'orig_lag_pty' in df_results.columns:
                pty_changed = df_results[df_results['new_lag_pty'] != df_results['orig_lag_pty']]
                eqy_changed = df_results[df_results['new_lag_eqy'] != df_results['orig_lag_eqy']]
                
                print(f"\nLag changes (even within 30 min):")
                print(f"  Polymarket → Equity: {len(pty_changed)} changed")
                print(f"  Equity → Polymarket: {len(eqy_changed)} changed")
            
            print(f"\nNew lag distribution:")
            print(f"  Polymarket → Equity:")
            print(f"    Mean: {df_results['new_lag_pty'].mean():.1f} min")
            print(f"    Median: {df_results['new_lag_pty'].median():.0f} min")
            print(f"    Range: {df_results['new_lag_pty'].min()}-{df_results['new_lag_pty'].max()} min")
            
            print(f"\n  Equity → Polymarket:")
            print(f"    Mean: {df_results['new_lag_eqy'].mean():.1f} min")
            print(f"    Median: {df_results['new_lag_eqy'].median():.0f} min")
            print(f"    Range: {df_results['new_lag_eqy'].min()}-{df_results['new_lag_eqy'].max()} min")
            
            print("\n" + "=" * 100)
            print("INTERPRETATION:")
            print("=" * 100)
            print("""
If many relationships extend beyond 30 min:
  → Original analysis was TRUNCATED
  → Re-run full analysis with maxlag=60

If most relationships stay within 30 min:
  → Original maxlag=30 was APPROPRIATE
  → Results are robust
            """)
            
            return df_results
        else:
            print("\n⚠️  No results generated")
            return pd.DataFrame()
            
    finally:
        conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-run Granger analysis with extended lag")
    parser.add_argument("--maxlag", type=int, default=60, help="Maximum lag to test (default: 60)")
    parser.add_argument("--tokens", nargs="+", help="Specific token IDs to analyze")
    parser.add_argument("--export", type=str, help="Export results to CSV")
    
    args = parser.parse_args()
    
    df = extended_lag_analysis(token_ids=args.tokens, maxlag=args.maxlag)
    
    if args.export and not df.empty:
        df.to_csv(args.export, index=False)
        print(f"\n✅ Exported extended lag analysis to {args.export}")
