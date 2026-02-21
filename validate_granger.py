"""
Consolidated validation module for Granger causality results.

This module performs lead-lag cross-correlation analysis to validate
whether Granger causality results represent true predictive relationships
or spurious correlations from common information shocks.

Usage:
    # Validate specific lag ranges
    python validate_granger.py --direction poly_to_eq --lag-range short
    python validate_granger.py --direction poly_to_eq --lag-range medium
    python validate_granger.py --direction poly_to_eq --lag-range long
    python validate_granger.py --direction eq_to_poly --limit 30
    
    # Validate all
    python validate_granger.py --direction poly_to_eq --lag-range all
"""

import os
import sys
import argparse
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

# Database paths
DB_MKT = os.getenv("MKT_DB", "./data/markets.duckdb")
DB_POLY = os.getenv("POLY_DB", "./data/research.duckdb")


def get_conn():
    """Create in-memory connection with both databases attached."""
    conn = duckdb.connect()
    conn.execute(f"ATTACH DATABASE '{os.path.abspath(DB_MKT)}' AS mkt")
    conn.execute(f"ATTACH DATABASE '{os.path.abspath(DB_POLY)}' AS poly")
    return conn


def get_markets_for_validation(direction='poly_to_eq', lag_range=None, limit=None):
    """
    Get markets with significant Granger causality results for validation.
    
    Args:
        direction: 'poly_to_eq' or 'eq_to_poly'
        lag_range: 'short' (1-3), 'medium' (5-15), 'long' (16-30), or None for all
        limit: Maximum number of markets to return
        
    Returns:
        DataFrame with token_id, question, lag, p_value, etc.
    """
    conn = get_conn()
    
    if direction == 'poly_to_eq':
        where_clause = "sig_poly_to_eq = TRUE"
        lag_col = "lag_poly_to_eq"
        p_col = "p_poly_to_eq_corrected"
    else:  # eq_to_poly
        where_clause = "sig_eq_to_poly = TRUE"
        lag_col = "lag_eq_to_poly"
        p_col = "p_eq_to_poly_corrected"
    
    # Add lag range filter if specified
    if lag_range == 'short':
        where_clause += f" AND {lag_col} BETWEEN 1 AND 3"
    elif lag_range == 'medium':
        where_clause += f" AND {lag_col} BETWEEN 5 AND 15"
    elif lag_range == 'long':
        where_clause += f" AND {lag_col} BETWEEN 16 AND 30"
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query = f"""
        SELECT 
            gr.token_id,
            m.question,
            {lag_col} as lag,
            {p_col} as p_value,
            gr.start,
            gr.end,
            gr.n_obs
        FROM mkt.main.granger_results gr
        LEFT JOIN poly.main.markets m ON gr.token_id = m.market_id
        WHERE {where_clause}
        ORDER BY {p_col}
        {limit_clause}
    """
    
    df = conn.execute(query).df()
    conn.close()
    return df


def load_aligned_data(token_id, start_ts, end_ts):
    """
    Load and align Polymarket and SPY data for a specific token and time range.
    
    Args:
        token_id: Token ID to load
        start_ts: Start timestamp
        end_ts: End timestamp
        
    Returns:
        Tuple of (poly_series, spy_series) aligned by timestamp
    """
    conn = get_conn()
    
    # Load Polymarket data
    poly_df = conn.execute(f"""
        SELECT 
            DATE_TRUNC('minute', to_timestamp(ts)) as timestamp,
            price
        FROM poly.main.prices
        WHERE token_id = '{token_id}'
            AND to_timestamp(ts) BETWEEN '{start_ts}' AND '{end_ts}'
        ORDER BY ts
    """).df()
    
    if len(poly_df) == 0:
        conn.close()
        return None, None
    
    # Deduplicate: keep last price per minute
    poly_df = poly_df.groupby('timestamp')['price'].last().reset_index()
    
    # Load SPY data
    spy_df = conn.execute(f"""
        SELECT 
            ts_utc as timestamp,
            (bid_px + ask_px) / 2.0 as price
        FROM mkt.main.security_bbo_1m
        WHERE ts_utc BETWEEN '{start_ts}' AND '{end_ts}'
        ORDER BY ts_utc
    """).df()
    
    conn.close()
    
    if len(spy_df) == 0:
        return None, None
    
    # Remove timezone info for merging
    poly_df['timestamp'] = pd.to_datetime(poly_df['timestamp']).dt.tz_localize(None)
    spy_df['timestamp'] = pd.to_datetime(spy_df['timestamp']).dt.tz_localize(None)
    
    # Inner join on timestamp
    merged = pd.merge(poly_df, spy_df, on='timestamp', how='inner', suffixes=('_poly', '_spy'))
    
    if len(merged) < 100:
        return None, None
    
    # Create aligned series
    poly_series = pd.Series(merged['price_poly'].values, index=merged['timestamp'])
    spy_series = pd.Series(merged['price_spy'].values, index=merged['timestamp'])
    
    return poly_series, spy_series


def compute_lead_lag_correlation(poly_series, spy_series, max_lag_minutes=30):
    """
    Compute cross-correlation at different lags.
    
    Positive lag: Polymarket leads SPY
    Negative lag: SPY leads Polymarket
    Lag=0: Synchronous
    
    Args:
        poly_series: Polymarket price series
        spy_series: SPY price series
        max_lag_minutes: Maximum lag to test in minutes
        
    Returns:
        Tuple of (lags, correlations, max_corr_lag)
    """
    # Calculate returns (first differences)
    poly_returns = poly_series.diff().dropna()
    spy_returns = spy_series.diff().dropna()
    
    # Align the series
    common_idx = poly_returns.index.intersection(spy_returns.index)
    poly_returns = poly_returns.loc[common_idx]
    spy_returns = spy_returns.loc[common_idx]
    
    lags = range(-max_lag_minutes, max_lag_minutes + 1)
    correlations = []
    
    for lag in lags:
        if lag > 0:
            # Positive lag: Poly leads SPY
            corr = poly_returns.iloc[:-lag].corr(spy_returns.iloc[lag:]) if lag < len(poly_returns) else np.nan
        elif lag < 0:
            # Negative lag: SPY leads Poly
            corr = poly_returns.iloc[-lag:].corr(spy_returns.iloc[:lag]) if -lag < len(spy_returns) else np.nan
        else:
            # lag = 0: Synchronous
            corr = poly_returns.corr(spy_returns)
        
        correlations.append(corr)
    
    correlations = np.array(correlations)
    
    # Find lag with maximum absolute correlation
    max_idx = np.nanargmax(np.abs(correlations))
    max_corr_lag = lags[max_idx]
    
    return list(lags), list(correlations), max_corr_lag


def permutation_test(poly_series, spy_series, lag_minutes, n_permutations=1000):
    """
    Test if correlation at specific lag is statistically significant.
    
    Args:
        poly_series: Polymarket price series
        spy_series: SPY price series  
        lag_minutes: Lag to test
        n_permutations: Number of random shuffles for null distribution
        
    Returns:
        Tuple of (observed_corr, p_value)
    """
    # Calculate returns
    poly_returns = poly_series.diff().dropna()
    spy_returns = spy_series.diff().dropna()
    
    # Align series
    common_idx = poly_returns.index.intersection(spy_returns.index)
    poly_returns = poly_returns.loc[common_idx].values
    spy_returns = spy_returns.loc[common_idx].values
    
    # Observed correlation
    if lag_minutes > 0:
        if lag_minutes >= len(poly_returns):
            return np.nan, 1.0
        obs_corr = np.corrcoef(poly_returns[:-lag_minutes], spy_returns[lag_minutes:])[0, 1]
    elif lag_minutes < 0:
        if -lag_minutes >= len(spy_returns):
            return np.nan, 1.0
        obs_corr = np.corrcoef(poly_returns[-lag_minutes:], spy_returns[:lag_minutes])[0, 1]
    else:
        obs_corr = np.corrcoef(poly_returns, spy_returns)[0, 1]
    
    # Permutation test
    perm_corrs = []
    for _ in range(n_permutations):
        shuffled_poly = np.random.permutation(poly_returns)
        
        if lag_minutes > 0:
            perm_corr = np.corrcoef(shuffled_poly[:-lag_minutes], spy_returns[lag_minutes:])[0, 1]
        elif lag_minutes < 0:
            perm_corr = np.corrcoef(shuffled_poly[-lag_minutes:], spy_returns[:lag_minutes])[0, 1]
        else:
            perm_corr = np.corrcoef(shuffled_poly, spy_returns)[0, 1]
        
        perm_corrs.append(perm_corr)
    
    # P-value: proportion of permutations with equal or stronger correlation
    p_value = np.mean(np.abs(perm_corrs) >= np.abs(obs_corr))
    
    return obs_corr, p_value


def classify_result(granger_lag, max_corr_lag, corr_at_granger, corr_at_max):
    """
    Classify validation result.
    
    Returns:
        String classification: CONFIRMED, PLAUSIBLE, NEAR, SYNCHRONOUS, REVERSED, MISMATCH
    """
    if max_corr_lag == granger_lag:
        return "CONFIRMED"
    elif abs(max_corr_lag - granger_lag) <= 3:
        return "PLAUSIBLE"
    elif abs(max_corr_lag - granger_lag) <= 5:
        return "NEAR"
    elif max_corr_lag == 0:
        return "SYNCHRONOUS"
    elif (granger_lag > 0 and max_corr_lag < 0) or (granger_lag < 0 and max_corr_lag > 0):
        return "REVERSED"
    else:
        return "MISMATCH"


def plot_lead_lag(token_id, question, lags, correlations, granger_lag, max_corr_lag, 
                  direction, output_dir='plots/validation'):
    """
    Plot lead-lag correlation with markers for Granger and maximum correlation lags.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot correlation curve
    ax.plot(lags, correlations, 'b-', linewidth=2, label='Cross-correlation')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Synchronous (lag=0)')
    
    # Mark Granger lag
    granger_corr = correlations[lags.index(granger_lag)] if granger_lag in lags else np.nan
    ax.axvline(x=granger_lag, color='red', linestyle='--', linewidth=2, 
               label=f'Granger lag ({granger_lag} min, r={granger_corr:.3f})')
    ax.plot([granger_lag], [granger_corr], 'ro', markersize=10)
    
    # Mark maximum correlation lag
    max_corr = correlations[lags.index(max_corr_lag)] if max_corr_lag in lags else np.nan
    ax.axvline(x=max_corr_lag, color='green', linestyle='--', linewidth=2,
               label=f'Max correlation ({max_corr_lag} min, r={max_corr:.3f})')
    ax.plot([max_corr_lag], [max_corr], 'go', markersize=10)
    
    ax.set_xlabel('Lag (minutes)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    
    direction_label = "Polymarket → SPY" if direction == 'poly_to_eq' else "SPY → Polymarket"
    title = f'{direction_label}: {question[:80]}'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save
    safe_filename = "".join(c for c in question[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_filename = safe_filename.replace(' ', '_')
    filepath = os.path.join(output_dir, f'{safe_filename}_{token_id[:8]}.png')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def validate_market(row, direction='poly_to_eq', output_dir='plots/validation', verbose=True):
    """
    Validate a single market's Granger causality result.
    
    Args:
        row: DataFrame row with token_id, question, lag, etc.
        direction: 'poly_to_eq' or 'eq_to_poly'
        output_dir: Directory for plots
        verbose: Print progress
        
    Returns:
        Dictionary with validation results
    """
    token_id = row['token_id']
    question = row['question'] if pd.notna(row['question']) else f"Token {token_id[:16]}"
    granger_lag = int(row['lag'])
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Validating: {question}")
        print(f"Token: {token_id}")
        print(f"Granger lag: {granger_lag} minutes")
        print(f"{'='*80}")
    
    # Load data
    poly_series, spy_series = load_aligned_data(token_id, row['start'], row['end'])
    
    if poly_series is None or spy_series is None:
        if verbose:
            print("❌ Failed to load data")
        return None
    
    if verbose:
        print(f"Loaded {len(poly_series):,} aligned observations")
    
    # Compute lead-lag correlation
    lags, correlations, max_corr_lag = compute_lead_lag_correlation(poly_series, spy_series)
    
    # Get correlations at key lags
    corr_at_granger = correlations[lags.index(granger_lag)] if granger_lag in lags else np.nan
    corr_at_zero = correlations[lags.index(0)]
    corr_at_max = correlations[lags.index(max_corr_lag)]
    
    # Test significance at Granger lag and at lag=0
    granger_corr_obs, granger_p = permutation_test(poly_series, spy_series, granger_lag)
    zero_corr_obs, zero_p = permutation_test(poly_series, spy_series, 0)
    
    # Classify result
    verdict = classify_result(granger_lag, max_corr_lag, corr_at_granger, corr_at_max)
    
    if verbose:
        print(f"\n📊 Lead-Lag Correlation Results:")
        print(f"  Maximum correlation: {corr_at_max:.4f} at lag={max_corr_lag} min")
        print(f"  Correlation at Granger lag ({granger_lag} min): {corr_at_granger:.4f} (p={granger_p:.4f})")
        print(f"  Correlation at lag=0: {corr_at_zero:.4f} (p={zero_p:.4f})")
        print(f"  Ratio (lag=0 / Granger lag): {abs(corr_at_zero / corr_at_granger) if corr_at_granger != 0 else np.inf:.2f}x")
        print(f"\n🎯 Verdict: {verdict}")
        
        if verdict == "CONFIRMED":
            print("  ✅ Maximum correlation at Granger lag - VALIDATED")
        elif verdict in ["PLAUSIBLE", "NEAR"]:
            print(f"  ⚠️  Maximum correlation within {abs(max_corr_lag - granger_lag)} min of Granger lag")
        elif verdict == "SYNCHRONOUS":
            print("  ❌ Maximum correlation at lag=0 - markets move together, not predictively")
        elif verdict == "REVERSED":
            print("  ❌ Maximum correlation in opposite direction - Granger claim reversed")
        else:
            print("  ❌ Maximum correlation at inconsistent lag - Granger claim not supported")
    
    # Generate plot
    plot_path = plot_lead_lag(token_id, question, lags, correlations, granger_lag, 
                              max_corr_lag, direction, output_dir)
    
    if verbose:
        print(f"\n📈 Plot saved: {plot_path}")
    
    return {
        'token_id': token_id,
        'question': question,
        'granger_lag': granger_lag,
        'max_corr_lag': max_corr_lag,
        'corr_at_granger': corr_at_granger,
        'corr_at_zero': corr_at_zero,
        'corr_at_max': corr_at_max,
        'granger_p_value': granger_p,
        'zero_p_value': zero_p,
        'verdict': verdict,
        'n_observations': len(poly_series),
        'plot_path': plot_path
    }


def print_summary(results, direction='poly_to_eq'):
    """Print summary statistics of validation results."""
    df = pd.DataFrame(results)
    
    print(f"\n{'='*100}")
    print(f"VALIDATION SUMMARY: {direction.upper()}")
    print(f"{'='*100}")
    print(f"\nTotal markets validated: {len(df)}")
    
    # Verdict distribution
    print(f"\n📋 Verdict Distribution:")
    verdict_counts = df['verdict'].value_counts()
    for verdict, count in verdict_counts.items():
        pct = count / len(df) * 100
        print(f"  {verdict:<15} {count:>3} ({pct:>5.1f}%)")
    
    # Statistical summary
    print(f"\n📊 Correlation Statistics:")
    print(f"  Mean |correlation| at Granger lag: {df['corr_at_granger'].abs().mean():.4f}")
    print(f"  Mean |correlation| at lag=0:       {df['corr_at_zero'].abs().mean():.4f}")
    print(f"  Mean |correlation| at max lag:     {df['corr_at_max'].abs().mean():.4f}")
    
    ratio = df['corr_at_zero'].abs() / df['corr_at_granger'].abs().replace(0, np.nan)
    print(f"  Mean ratio (lag=0 / Granger lag):  {ratio.mean():.2f}x")
    
    # Significance tests
    granger_sig = (df['granger_p_value'] < 0.05).sum()
    zero_sig = (df['zero_p_value'] < 0.05).sum()
    
    print(f"\n🔬 Statistical Significance (p<0.05):")
    print(f"  Significant at Granger lag: {granger_sig}/{len(df)} ({granger_sig/len(df)*100:.1f}%)")
    print(f"  Significant at lag=0:       {zero_sig}/{len(df)} ({zero_sig/len(df)*100:.1f}%)")
    
    # Key finding
    synchronous_pct = (df['verdict'] == 'SYNCHRONOUS').sum() / len(df) * 100
    confirmed_pct = (df['verdict'] == 'CONFIRMED').sum() / len(df) * 100
    
    print(f"\n🎯 Key Finding:")
    print(f"  {synchronous_pct:.1f}% of markets show SYNCHRONOUS movement (max correlation at lag=0)")
    print(f"  {confirmed_pct:.1f}% of markets CONFIRM Granger lag (max correlation at Granger lag)")
    
    if synchronous_pct > 70:
        print(f"\n  ⚠️  CONCLUSION: Overwhelming evidence of synchronous movement.")
        print(f"      Markets react to same information simultaneously, not predictively.")
    elif confirmed_pct > 70:
        print(f"\n  ✅ CONCLUSION: Strong validation of Granger causality results.")
    else:
        print(f"\n  ⚠️  CONCLUSION: Mixed results. Granger causality partially validated.")


def main():
    parser = argparse.ArgumentParser(description='Validate Granger causality results using lead-lag correlation')
    parser.add_argument('--direction', choices=['poly_to_eq', 'eq_to_poly'], default='poly_to_eq',
                        help='Direction of causality to test')
    parser.add_argument('--lag-range', choices=['short', 'medium', 'long', 'all'], default='all',
                        help='Lag range to validate: short (1-3), medium (5-15), long (16-30), or all')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of markets to validate')
    parser.add_argument('--output-dir', default='plots/validation',
                        help='Directory for output plots')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Determine lag range for display
    if args.lag_range == 'all':
        lag_range_display = "all lags"
        lag_range_param = None
    else:
        lag_range_display = f"{args.lag_range} lags"
        lag_range_param = args.lag_range
    
    print(f"\n{'='*100}")
    print(f"GRANGER CAUSALITY VALIDATION")
    print(f"{'='*100}")
    print(f"Direction: {args.direction}")
    print(f"Lag range: {lag_range_display}")
    print(f"Limit: {args.limit if args.limit else 'None (all markets)'}")
    print(f"Output directory: {args.output_dir}")
    
    # Get markets to validate
    markets = get_markets_for_validation(args.direction, lag_range_param, args.limit)
    
    if len(markets) == 0:
        print(f"\n❌ No markets found matching criteria")
        return
    
    print(f"\n✅ Found {len(markets)} markets to validate")
    
    # Validate each market
    results = []
    for idx, row in markets.iterrows():
        result = validate_market(row, args.direction, args.output_dir, verbose=True)
        if result:
            results.append(result)
    
    # Print summary
    if results:
        print_summary(results, args.direction)
        
        # Save CSV if requested
        if args.save_csv:
            csv_filename = f"validation_{args.direction}"
            if lag_range_param:
                csv_filename += f"_{args.lag_range}"
            csv_filename += ".csv"
            
            df = pd.DataFrame(results)
            df.to_csv(csv_filename, index=False)
            print(f"\n💾 Results saved to {csv_filename}")
    else:
        print(f"\n❌ No markets successfully validated")


if __name__ == '__main__':
    main()
