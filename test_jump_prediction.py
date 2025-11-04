"""
Test if large Polymarket price jumps predict subsequent SPY movements.

Hypothesis: If insider trading occurs on Polymarket, we should see:
1. Large Polymarket jumps followed by SPY moves in same direction
2. Stronger relationship than SPY jumps → Polymarket moves
3. Effect stronger when Polymarket jump happens during low-information periods
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats

def load_data_for_token(token_address, db_path='data/research.duckdb', spy_db='data/markets.duckdb'):
    """Load aligned Polymarket and SPY data for a token."""
    conn = duckdb.connect(db_path, read_only=True)
    spy_conn = duckdb.connect(spy_db, read_only=True)
    
    # Get Polymarket prices (rounded to minute)
    poly_df = conn.execute(f"""
        SELECT 
            DATE_TRUNC('minute', to_timestamp(ts)) as timestamp,
            price
        FROM prices
        WHERE token_id = '{token_address}'
        ORDER BY ts
    """).df()
    
    # Deduplicate - keep last price per minute
    poly_df = poly_df.groupby('timestamp')['price'].last().reset_index()
    
    # Get SPY prices
    spy_df = spy_conn.execute("""
        SELECT 
            ts_utc as timestamp,
            (bid_px + ask_px) / 2.0 as price
        FROM security_bbo_1m
        ORDER BY ts_utc
    """).df()
    
    conn.close()
    spy_conn.close()
    
    # Remove timezone info to allow merging
    poly_df['timestamp'] = pd.to_datetime(poly_df['timestamp']).dt.tz_localize(None)
    spy_df['timestamp'] = pd.to_datetime(spy_df['timestamp']).dt.tz_localize(None)
    
    # Merge on timestamp
    merged = pd.merge(poly_df, spy_df, on='timestamp', how='inner', suffixes=('_poly', '_spy'))
    merged = merged.sort_values('timestamp')
    
    # Calculate returns (percentage changes)
    merged['poly_return'] = merged['price_poly'].pct_change() * 100  # Convert to percentage
    merged['spy_return'] = merged['price_spy'].pct_change() * 100
    
    return merged

def identify_jumps(series, threshold_std=2.0, min_pct_change=3.0):
    """
    Identify significant jumps in a time series.
    
    Args:
        series: pandas Series of returns
        threshold_std: Number of standard deviations to classify as jump
        min_pct_change: Minimum percentage change to classify as jump
    
    Returns:
        Boolean series indicating jumps
    """
    # Calculate rolling statistics (exclude current value)
    rolling_mean = series.rolling(window=60, min_periods=30).mean().shift(1)
    rolling_std = series.rolling(window=60, min_periods=30).std().shift(1)
    
    # Identify jumps using both criteria
    z_score = np.abs((series - rolling_mean) / rolling_std)
    std_criterion = z_score > threshold_std
    pct_criterion = np.abs(series) > min_pct_change
    
    jumps = std_criterion & pct_criterion
    return jumps

def analyze_post_jump_behavior(df, jump_col, target_col, windows=[1, 5, 10, 15, 30, 60]):
    """
    Analyze target variable behavior after jumps.
    
    Args:
        df: DataFrame with jump indicators and target variable
        jump_col: Column name indicating jumps
        target_col: Column name of variable to analyze after jumps
        windows: List of forward-looking windows (in minutes)
    
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    jump_indices = df[df[jump_col]].index
    
    for window in windows:
        future_returns = []
        jump_directions = []
        
        for idx in jump_indices:
            if idx + window < len(df):
                # Cumulative return over next 'window' minutes
                future_ret = df[target_col].iloc[idx+1:idx+window+1].sum()
                future_returns.append(future_ret)
                
                # Direction of original jump
                jump_dir = np.sign(df[jump_col.replace('_jump', '_return')].iloc[idx])
                jump_directions.append(jump_dir)
        
        if len(future_returns) > 0:
            future_returns = np.array(future_returns)
            jump_directions = np.array(jump_directions)
            
            # Test if future returns are significantly different from zero
            t_stat, p_value = stats.ttest_1samp(future_returns, 0)
            
            # Test if future returns align with jump direction
            directional_returns = future_returns * jump_directions
            dir_t_stat, dir_p_value = stats.ttest_1samp(directional_returns, 0)
            
            # Percentage that moved in same direction
            same_direction = np.sum((future_returns * jump_directions) > 0) / len(future_returns) * 100
            
            results[window] = {
                'n_jumps': len(future_returns),
                'mean_future_return': np.mean(future_returns),
                'std_future_return': np.std(future_returns),
                't_stat': t_stat,
                'p_value': p_value,
                'mean_directional_return': np.mean(directional_returns),
                'dir_t_stat': dir_t_stat,
                'dir_p_value': dir_p_value,
                'pct_same_direction': same_direction
            }
    
    return results

def test_token_jumps(token_address, token_name):
    """Test jump prediction for a single token."""
    print(f"\n{'='*80}")
    print(f"Testing: {token_name}")
    print(f"Token: {token_address}")
    print(f"{'='*80}")
    
    # Load data
    df = load_data_for_token(token_address)
    
    if len(df) < 1000:
        print(f"Insufficient data: {len(df)} observations")
        return None
    
    print(f"Data points: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Identify jumps
    df['poly_jump'] = identify_jumps(df['poly_return'], threshold_std=2.0, min_pct_change=3.0)
    df['spy_jump'] = identify_jumps(df['spy_return'], threshold_std=2.0, min_pct_change=2.0)
    
    n_poly_jumps = df['poly_jump'].sum()
    n_spy_jumps = df['spy_jump'].sum()
    
    print(f"\nJumps identified:")
    print(f"  Polymarket: {n_poly_jumps} jumps ({n_poly_jumps/len(df)*100:.2f}% of observations)")
    print(f"  SPY: {n_spy_jumps} jumps ({n_spy_jumps/len(df)*100:.2f}% of observations)")
    
    if n_poly_jumps < 10:
        print("Too few Polymarket jumps for analysis")
        return None
    
    # Analyze Polymarket jumps → SPY movements
    print(f"\n{'─'*80}")
    print("HYPOTHESIS TEST: Do Polymarket jumps predict SPY movements?")
    print(f"{'─'*80}")
    
    poly_results = analyze_post_jump_behavior(
        df, 'poly_jump', 'spy_return', 
        windows=[1, 5, 10, 15, 30, 60]
    )
    
    print("\nSPY Movement After Polymarket Jumps:")
    print(f"{'Window':<10} {'N Jumps':<10} {'Mean Return':<15} {'T-stat':<10} {'P-value':<12} {'Same Dir %':<12} {'Dir P-val'}")
    print("─" * 100)
    
    for window, res in poly_results.items():
        sig = "***" if res['dir_p_value'] < 0.01 else ("**" if res['dir_p_value'] < 0.05 else ("*" if res['dir_p_value'] < 0.10 else ""))
        print(f"{window:>2} min     {res['n_jumps']:<10} {res['mean_future_return']:>+7.4f}%       "
              f"{res['t_stat']:>+7.3f}   {res['p_value']:<12.4f} {res['pct_same_direction']:>6.1f}%      "
              f"{res['dir_p_value']:<10.4f} {sig}")
    
    # Analyze SPY jumps → Polymarket movements (control)
    print(f"\n{'─'*80}")
    print("CONTROL TEST: Do SPY jumps predict Polymarket movements?")
    print(f"{'─'*80}")
    
    spy_results = analyze_post_jump_behavior(
        df, 'spy_jump', 'poly_return',
        windows=[1, 5, 10, 15, 30, 60]
    )
    
    print("\nPolymarket Movement After SPY Jumps:")
    print(f"{'Window':<10} {'N Jumps':<10} {'Mean Return':<15} {'T-stat':<10} {'P-value':<12} {'Same Dir %':<12} {'Dir P-val'}")
    print("─" * 100)
    
    for window, res in spy_results.items():
        sig = "***" if res['dir_p_value'] < 0.01 else ("**" if res['dir_p_value'] < 0.05 else ("*" if res['dir_p_value'] < 0.10 else ""))
        print(f"{window:>2} min     {res['n_jumps']:<10} {res['mean_future_return']:>+7.4f}%       "
              f"{res['t_stat']:>+7.3f}   {res['p_value']:<12.4f} {res['pct_same_direction']:>6.1f}%      "
              f"{res['dir_p_value']:<10.4f} {sig}")
    
    # Compare asymmetry
    print(f"\n{'─'*80}")
    print("ASYMMETRY TEST: Is one direction stronger?")
    print(f"{'─'*80}")
    
    for window in [5, 15, 30]:
        if window in poly_results and window in spy_results:
            poly_strength = poly_results[window]['mean_directional_return']
            spy_strength = spy_results[window]['mean_directional_return']
            
            print(f"\n{window}-minute window:")
            print(f"  Poly → SPY strength: {poly_strength:+.4f}% (p={poly_results[window]['dir_p_value']:.4f})")
            print(f"  SPY → Poly strength: {spy_strength:+.4f}% (p={spy_results[window]['dir_p_value']:.4f})")
            
            if poly_results[window]['dir_p_value'] < 0.05 and spy_results[window]['dir_p_value'] >= 0.05:
                print(f"  → Polymarket appears to LEAD (insider trading hypothesis supported)")
            elif spy_results[window]['dir_p_value'] < 0.05 and poly_results[window]['dir_p_value'] >= 0.05:
                print(f"  → SPY appears to LEAD (insider trading hypothesis rejected)")
            elif poly_results[window]['dir_p_value'] < 0.05 and spy_results[window]['dir_p_value'] < 0.05:
                print(f"  → Bidirectional relationship (suggests common information)")
            else:
                print(f"  → No significant relationship in either direction")
    
    return {
        'token_address': token_address,
        'token_name': token_name,
        'n_observations': len(df),
        'n_poly_jumps': n_poly_jumps,
        'n_spy_jumps': n_spy_jumps,
        'poly_to_spy': poly_results,
        'spy_to_poly': spy_results
    }

def main():
    """Test jump prediction hypothesis on significant Granger markets."""
    
    # Connect to both databases
    poly_conn = duckdb.connect('data/research.duckdb', read_only=True)
    mkt_conn = duckdb.connect('data/markets.duckdb', read_only=True)
    
    # Get markets with significant Granger causality
    granger_df = mkt_conn.execute("""
        SELECT DISTINCT 
            token_id,
            lag_poly_to_eq as lag,
            p_poly_to_eq_corrected as p_value_corrected
        FROM granger_results
        WHERE sig_poly_to_eq = TRUE
        ORDER BY p_poly_to_eq_corrected
        LIMIT 10
    """).df()
    
    # Get market questions
    markets_df = poly_conn.execute("""
        SELECT market_id, question
        FROM markets
    """).df()
    
    # Merge to get questions
    markets = granger_df.merge(
        markets_df,
        left_on='token_id',
        right_on='market_id',
        how='left'
    )
    
    poly_conn.close()
    mkt_conn.close()
    
    print(f"\nTesting {len(markets)} markets with significant Granger causality")
    print("=" * 100)
    
    all_results = []
    
    for idx, row in markets.iterrows():
        result = test_token_jumps(row['token_id'], row['question'])
        if result:
            all_results.append(result)
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY: Evidence of Insider Trading on Polymarket?")
    print(f"{'='*100}")
    
    if len(all_results) == 0:
        print("No markets had sufficient jumps for analysis")
        return
    
    # Count markets with significant predictive jumps
    poly_leads_15min = sum(1 for r in all_results if 15 in r['poly_to_spy'] and r['poly_to_spy'][15]['dir_p_value'] < 0.05)
    spy_leads_15min = sum(1 for r in all_results if 15 in r['spy_to_poly'] and r['spy_to_poly'][15]['dir_p_value'] < 0.05)
    
    print(f"\nAt 15-minute horizon:")
    print(f"  Markets where Poly jumps → SPY moves: {poly_leads_15min}/{len(all_results)} ({poly_leads_15min/len(all_results)*100:.1f}%)")
    print(f"  Markets where SPY jumps → Poly moves: {spy_leads_15min}/{len(all_results)} ({spy_leads_15min/len(all_results)*100:.1f}%)")
    
    if poly_leads_15min > spy_leads_15min:
        print(f"\n✓ EVIDENCE FOR insider trading: Polymarket jumps are more predictive")
    elif spy_leads_15min > poly_leads_15min:
        print(f"\n✗ EVIDENCE AGAINST insider trading: SPY jumps are more predictive")
    else:
        print(f"\n~ MIXED EVIDENCE: Both directions show similar predictive power")
    
    print("\nNote: True insider trading would show:")
    print("  1. Polymarket jumps BEFORE news is public")
    print("  2. SPY follows consistently in same direction")
    print("  3. Asymmetric relationship (Poly→SPY stronger than SPY→Poly)")
    print("  4. Effect concentrated at specific lags (information diffusion time)")

if __name__ == '__main__':
    main()
