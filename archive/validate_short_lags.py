#!/usr/bin/env python3
"""
Validate ultra-short lag Granger causality results (1-2 minutes).

Tests whether markets with very short lags are truly predictive or just
highly correlated. Uses lead-lag cross-correlation analysis to:
1. Check if Polymarket actually leads SPY
2. Measure correlation at different time lags
3. Compare to random baseline

If max correlation is at lag=0, markets move together (not predictive).
If max correlation is at positive lag, Polymarket leads (good!).
"""
import os
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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

def get_short_lag_markets(max_lag=3):
    """Get markets with ultra-short lags (1-3 minutes)."""
    conn = get_conn()
    
    try:
        query = f"""
            SELECT 
                g.token_id,
                m.question,
                g.start_ts,
                g.end_ts,
                g.n_obs,
                g.lag_poly_to_eq,
                g.p_poly_to_eq_corrected,
                g.lag_eq_to_poly,
                g.p_eq_to_poly_corrected,
                g.sig_poly_to_eq,
                g.sig_eq_to_poly
            FROM mkt.main.granger_results g
            LEFT JOIN poly.main.tokens t ON g.token_id = t.token_id
            LEFT JOIN poly.main.markets m ON t.market_id = m.market_id
            WHERE g.sig_poly_to_eq = TRUE
                AND g.lag_poly_to_eq <= ?
            ORDER BY g.lag_poly_to_eq, g.p_poly_to_eq_corrected
        """
        
        return conn.execute(query, [max_lag]).df()
    finally:
        conn.close()

def load_aligned_data(token_id, start_ts, end_ts):
    """Load Polymarket and SPY data, aligned by timestamp."""
    conn = get_conn()
    
    try:
        # Load Polymarket prices
        poly_query = """
            SELECT 
                to_timestamp(ts) as timestamp,
                price
            FROM poly.main.prices
            WHERE token_id = ?
            AND to_timestamp(ts) BETWEEN ? AND ?
            ORDER BY ts
        """
        poly_df = conn.execute(poly_query, [token_id, start_ts, end_ts]).df()
        poly_df['timestamp'] = pd.to_datetime(poly_df['timestamp'], utc=True)
        
        # Round to nearest minute and deduplicate (keep last)
        poly_df['timestamp'] = poly_df['timestamp'].dt.round('1min')
        poly_df = poly_df.groupby('timestamp')['price'].last().reset_index()
        poly_df = poly_df.set_index('timestamp')
        
        # Load SPY prices (already at 1-minute resolution)
        spy_query = """
            SELECT 
                ts_utc as timestamp,
                mid_px as price
            FROM mkt.main.security_bbo_1m
            WHERE ticker = 'SPY'
            AND ts_utc BETWEEN ? AND ?
            ORDER BY ts_utc
        """
        spy_df = conn.execute(spy_query, [start_ts, end_ts]).df()
        spy_df['timestamp'] = pd.to_datetime(spy_df['timestamp'], utc=True)
        spy_df = spy_df.set_index('timestamp')
        
        # Merge on timestamp (inner join to get overlapping data only)
        merged = pd.merge(
            poly_df, spy_df, 
            left_index=True, right_index=True, 
            how='inner', 
            suffixes=('_poly', '_spy')
        )
        
        if len(merged) < 100:
            return pd.DataFrame(), pd.DataFrame()
        
        # Return as separate series but with matching index
        poly_aligned = merged['price_poly']
        spy_aligned = merged['price_spy']
        
        return pd.DataFrame({'price': poly_aligned}), pd.DataFrame({'price': spy_aligned})
        
    finally:
        conn.close()

def lead_lag_correlation(poly_series, spy_series, max_lag_minutes=30):
    """
    Compute lead-lag cross-correlation.
    
    Positive lag: Polymarket leads SPY (what we want)
    Negative lag: SPY leads Polymarket
    Lag = 0: Synchronous movement (not predictive)
    
    Returns:
        lags: Array of lag values in minutes
        correlations: Correlation at each lag
        max_lag: Lag with maximum correlation
        max_corr: Maximum correlation value
    """
    # Align the series (inner join on timestamp)
    df = pd.DataFrame({
        'poly': poly_series,
        'spy': spy_series
    }).dropna()
    
    if len(df) < 100:
        return None, None, None, None
    
    # Calculate returns to remove common trends
    df['poly_ret'] = df['poly'].diff()
    df['spy_ret'] = df['spy'].diff()
    df = df.dropna()
    
    lags = list(range(-max_lag_minutes, max_lag_minutes + 1))
    correlations = []
    
    for lag in lags:
        if lag == 0:
            corr = df['poly_ret'].corr(df['spy_ret'])
        elif lag > 0:
            # Positive lag: shift Polymarket back (Poly leads SPY)
            corr = df['poly_ret'].shift(lag).corr(df['spy_ret'])
        else:
            # Negative lag: shift SPY back (SPY leads Poly)
            corr = df['poly_ret'].corr(df['spy_ret'].shift(-lag))
        
        correlations.append(corr)
    
    correlations = np.array(correlations)
    max_idx = np.nanargmax(np.abs(correlations))
    max_lag = lags[max_idx]
    max_corr = correlations[max_idx]
    
    return np.array(lags), correlations, max_lag, max_corr

def test_significance(poly_series, spy_series, lag_minutes):
    """
    Test if correlation at specific lag is statistically significant.
    
    Uses permutation test: shuffle one series many times and compute
    correlation distribution under null hypothesis of no relationship.
    """
    # Align series
    df = pd.DataFrame({
        'poly': poly_series,
        'spy': spy_series
    }).dropna()
    
    # Calculate returns
    df['poly_ret'] = df['poly'].diff()
    df['spy_ret'] = df['spy'].diff()
    df = df.dropna()
    
    # Observed correlation at specified lag
    if lag_minutes > 0:
        obs_corr = df['poly_ret'].shift(lag_minutes).corr(df['spy_ret'])
    elif lag_minutes < 0:
        obs_corr = df['poly_ret'].corr(df['spy_ret'].shift(-lag_minutes))
    else:
        obs_corr = df['poly_ret'].corr(df['spy_ret'])
    
    # Permutation test
    n_perms = 1000
    perm_corrs = []
    
    for _ in range(n_perms):
        shuffled = df['poly_ret'].sample(frac=1.0).values
        if lag_minutes > 0:
            perm_corr = pd.Series(shuffled, index=df.index).shift(lag_minutes).corr(df['spy_ret'])
        elif lag_minutes < 0:
            perm_corr = pd.Series(shuffled, index=df.index).corr(df['spy_ret'].shift(-lag_minutes))
        else:
            perm_corr = pd.Series(shuffled, index=df.index).corr(df['spy_ret'])
        perm_corrs.append(perm_corr)
    
    perm_corrs = np.array(perm_corrs)
    p_value = np.mean(np.abs(perm_corrs) >= np.abs(obs_corr))
    
    return obs_corr, p_value

def plot_lead_lag(token_id, question, lags, correlations, granger_lag, max_corr_lag):
    """Plot lead-lag correlation profile."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot correlation vs lag
    ax.plot(lags, correlations, 'b-', linewidth=2, label='Cross-correlation')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, label='Synchronous (lag=0)')
    
    # Mark Granger lag
    granger_corr = correlations[np.where(lags == granger_lag)[0][0]]
    ax.axvline(x=granger_lag, color='red', linestyle='--', alpha=0.5, 
               label=f'Granger lag ({granger_lag} min)')
    ax.scatter([granger_lag], [granger_corr], color='red', s=100, zorder=5)
    
    # Mark maximum correlation lag
    max_corr = np.max(np.abs(correlations))
    max_corr_val = correlations[np.where(lags == max_corr_lag)[0][0]]
    ax.scatter([max_corr_lag], [max_corr_val], color='green', s=150, 
               marker='*', zorder=5, label=f'Max corr lag ({max_corr_lag} min)')
    
    # Shading for interpretation
    ax.axvspan(-30, 0, alpha=0.1, color='orange', label='SPY leads')
    ax.axvspan(0, 30, alpha=0.1, color='green', label='Poly leads')
    
    ax.set_xlabel('Lag (minutes)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title(f'{question[:60]}\nLead-Lag Cross-Correlation Analysis', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    safe_name = question[:50].replace('?', '').replace('/', '_')
    os.makedirs('plots/validation', exist_ok=True)
    plt.savefig(f'plots/validation/leadlag_{token_id[:15]}_{safe_name}.png', dpi=150)
    plt.close()

def main():
    print("=" * 90)
    print("VALIDATING ULTRA-SHORT LAG GRANGER CAUSALITY RESULTS")
    print("=" * 90)
    print("Testing markets with lag ≤ 3 minutes")
    print("=" * 90)
    print()
    
    # Get markets with short lags
    markets = get_short_lag_markets(max_lag=3)
    
    if markets.empty:
        print("No markets found with lag ≤ 3 minutes")
        return
    
    print(f"Found {len(markets)} markets to validate\n")
    
    results = []
    
    for idx, row in markets.iterrows():
        token_id = row['token_id']
        question = row['question']
        granger_lag = row['lag_poly_to_eq']
        p_value = row['p_poly_to_eq_corrected']
        
        print(f"[{idx + 1}/{len(markets)}] {question[:70]}")
        print(f"  Granger lag: {granger_lag} min, p-value: {p_value:.6f}")
        
        # Load data
        poly_df, spy_df = load_aligned_data(token_id, row['start_ts'], row['end_ts'])
        
        if poly_df.empty or spy_df.empty:
            print(f"  ⚠️  No data available")
            print()
            continue
        
        # Compute lead-lag correlation
        lags, correlations, max_lag, max_corr = lead_lag_correlation(
            poly_df['price'], spy_df['price'], max_lag_minutes=30
        )
        
        if lags is None:
            print(f"  ⚠️  Insufficient data for correlation analysis")
            print()
            continue
        
        # Test significance at Granger lag
        corr_at_granger, perm_p = test_significance(
            poly_df['price'], spy_df['price'], granger_lag
        )
        
        # Test significance at lag=0 (synchronous)
        corr_at_zero, perm_p_zero = test_significance(
            poly_df['price'], spy_df['price'], 0
        )
        
        print(f"  📊 Lead-Lag Analysis:")
        print(f"     Max correlation: {max_corr:.4f} at lag={max_lag} min")
        print(f"     Correlation at Granger lag ({granger_lag} min): {corr_at_granger:.4f} (p={perm_p:.4f})")
        print(f"     Correlation at lag=0 (synchronous): {corr_at_zero:.4f} (p={perm_p_zero:.4f})")
        
        # Interpretation
        if max_lag == 0:
            print(f"  ❌ SYNCHRONOUS: Markets move together (not predictive)")
            verdict = "SYNCHRONOUS"
        elif max_lag == granger_lag:
            print(f"  ✅ CONFIRMED: Granger lag matches max correlation lag")
            verdict = "CONFIRMED"
        elif max_lag > 0 and abs(max_lag - granger_lag) <= 2:
            print(f"  ✅ PLAUSIBLE: Max corr near Granger lag (Poly leads)")
            verdict = "PLAUSIBLE"
        elif max_lag < 0:
            print(f"  ⚠️  REVERSED: SPY leads Polymarket (opposite direction)")
            verdict = "REVERSED"
        else:
            print(f"  ⚠️  MISMATCH: Max corr at {max_lag} min vs Granger {granger_lag} min")
            verdict = "MISMATCH"
        
        print()
        
        # Store results
        results.append({
            'token_id': token_id,
            'question': question,
            'granger_lag': granger_lag,
            'granger_p': p_value,
            'max_corr_lag': max_lag,
            'max_corr': max_corr,
            'corr_at_granger': corr_at_granger,
            'perm_p': perm_p,
            'corr_at_zero': corr_at_zero,
            'perm_p_zero': perm_p_zero,
            'verdict': verdict,
            'n_obs': row['n_obs']
        })
        
        # Plot
        plot_lead_lag(token_id, question, lags, correlations, granger_lag, max_lag)
    
    # Summary
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    if not results:
        print("\n⚠️  No markets had sufficient data for validation")
        print("This suggests Polymarket price updates are too sparse for minute-level analysis")
        return
    
    results_df = pd.DataFrame(results)
    
    print(f"\nTotal markets tested: {len(results_df)}")
    print(f"\nVerdict breakdown:")
    for verdict in ['CONFIRMED', 'PLAUSIBLE', 'SYNCHRONOUS', 'REVERSED', 'MISMATCH']:
        count = len(results_df[results_df['verdict'] == verdict])
        pct = 100 * count / len(results_df) if len(results_df) > 0 else 0
        print(f"  {verdict:12s}: {count:2d} ({pct:5.1f}%)")
    
    # Save detailed results
    results_df.to_csv('validation_short_lags.csv', index=False)
    print(f"\n✓ Detailed results saved to: validation_short_lags.csv")
    print(f"✓ Lead-lag plots saved to: plots/validation/")
    
    # Show suspicious markets
    suspicious = results_df[results_df['verdict'].isin(['SYNCHRONOUS', 'REVERSED'])]
    if not suspicious.empty:
        print(f"\n⚠️  SUSPICIOUS MARKETS (likely spurious):")
        print(f"=" * 90)
        for _, row in suspicious.iterrows():
            print(f"  {row['question'][:70]}")
            print(f"    Granger lag: {row['granger_lag']} min")
            print(f"    Max corr lag: {row['max_corr_lag']} min")
            print(f"    Verdict: {row['verdict']}")
            print()

if __name__ == "__main__":
    main()
