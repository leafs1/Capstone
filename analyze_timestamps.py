#!/usr/bin/env python3
"""
Analyze timestamp distributions for Polymarket tokens during market hours.
Shows the range of times and frequency of updates for each token.
"""

import os
import duckdb
import pandas as pd
from datetime import datetime

DB_MKT = os.getenv("MKT_DB", "./data/markets.duckdb")
DB_POLY = os.getenv("POLY_DB", "./data/research.duckdb")

_MKT_PATH = os.path.abspath(DB_MKT)
_POLY_PATH = os.path.abspath(DB_POLY)

POLY_TABLE = "poly.main.prices"
EQUITY_TABLE = "mkt.main.security_bbo_1m"

def get_conn():
    conn = duckdb.connect()
    conn.execute(f"ATTACH DATABASE '{_MKT_PATH}' AS mkt")
    if _POLY_PATH != _MKT_PATH:
        conn.execute(f"ATTACH DATABASE '{_POLY_PATH}' AS poly")
    else:
        conn.execute("CREATE SCHEMA IF NOT EXISTS poly")
    return conn

def analyze_token_timestamps(token_id: str, conn) -> dict:
    """Analyze timestamp distribution for a single token."""
    
    # Get all timestamps for this token
    q = f"""
        SELECT 
            to_timestamp(ts)::TIMESTAMP AS ts_utc,
            price
        FROM {POLY_TABLE}
        WHERE token_id = ?
        ORDER BY ts
    """
    df = conn.execute(q, [token_id]).df()
    
    if df.empty:
        return None
    
    # Convert to datetime
    df['ts_utc'] = pd.to_datetime(df['ts_utc'], utc=True)
    df_et = df.copy()
    df_et['ts_et'] = df_et['ts_utc'].dt.tz_convert('US/Eastern')
    
    # Calculate market hours mask
    df_et['hour'] = df_et['ts_et'].dt.hour
    df_et['minute'] = df_et['ts_et'].dt.minute
    df_et['weekday'] = df_et['ts_et'].dt.weekday
    df_et['time_minutes'] = df_et['hour'] * 60 + df_et['minute']
    
    df_et['market_hours'] = (
        (df_et['weekday'] < 5) &  # Monday-Friday
        (df_et['time_minutes'] >= 570) &  # >= 9:30 AM
        (df_et['time_minutes'] < 960)  # < 4:00 PM
    )
    
    # Statistics
    total_updates = len(df)
    market_hours_updates = df_et['market_hours'].sum()
    
    # Time range
    start_time = df['ts_utc'].min()
    end_time = df['ts_utc'].max()
    duration_days = (end_time - start_time).days
    
    # Market hours time distribution
    market_df = df_et[df_et['market_hours']]
    
    if len(market_df) > 0:
        # Calculate time gaps (in minutes)
        market_df = market_df.sort_values('ts_utc')
        time_diffs = market_df['ts_utc'].diff().dt.total_seconds() / 60
        
        avg_gap_minutes = time_diffs.mean()
        median_gap_minutes = time_diffs.median()
        max_gap_minutes = time_diffs.max()
        
        # Hour distribution
        hour_dist = market_df['hour'].value_counts().sort_index()
        most_active_hour = hour_dist.idxmax()
        
        # Day of week distribution
        dow_dist = market_df['weekday'].value_counts().sort_index()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        
        return {
            'token_id': token_id[:20] + '...',
            'total_updates': total_updates,
            'market_hours_updates': market_hours_updates,
            'market_hours_pct': 100 * market_hours_updates / total_updates,
            'start_time': start_time,
            'end_time': end_time,
            'duration_days': duration_days,
            'avg_gap_min': avg_gap_minutes,
            'median_gap_min': median_gap_minutes,
            'max_gap_min': max_gap_minutes,
            'most_active_hour': most_active_hour,
            'updates_per_day': market_hours_updates / max(1, duration_days),
            'hour_distribution': hour_dist.to_dict(),
            'dow_distribution': {dow_names[k]: v for k, v in dow_dist.items()}
        }
    else:
        return {
            'token_id': token_id[:20] + '...',
            'total_updates': total_updates,
            'market_hours_updates': 0,
            'market_hours_pct': 0,
            'start_time': start_time,
            'end_time': end_time,
            'duration_days': duration_days,
            'avg_gap_min': None,
            'median_gap_min': None,
            'max_gap_min': None,
            'most_active_hour': None,
            'updates_per_day': 0
        }

def main():
    conn = get_conn()
    
    # Get tokens with sufficient data
    q = f"""
        SELECT token_id, COUNT(*) as n
        FROM {POLY_TABLE}
        GROUP BY token_id
        HAVING COUNT(*) >= 200
        ORDER BY n DESC
        LIMIT 20
    """
    tokens = conn.execute(q).df()
    
    print("="*90)
    print("Polymarket Token Timestamp Analysis")
    print("="*90)
    print(f"Analyzing top 20 tokens with most data points")
    print(f"Market hours: Monday-Friday, 9:30 AM - 4:00 PM ET")
    print("="*90)
    
    results = []
    for idx, row in tokens.iterrows():
        token_id = row['token_id']
        print(f"\nAnalyzing token {idx+1}/20: {token_id[:30]}...")
        
        result = analyze_token_timestamps(token_id, conn)
        if result:
            results.append(result)
            print(f"  Total updates: {result['total_updates']:,}")
            print(f"  Market hours updates: {result['market_hours_updates']:,} ({result['market_hours_pct']:.1f}%)")
            print(f"  Date range: {result['start_time'].date()} to {result['end_time'].date()} ({result['duration_days']} days)")
            if result['avg_gap_min']:
                print(f"  Avg gap between updates: {result['avg_gap_min']:.1f} min")
                print(f"  Median gap: {result['median_gap_min']:.1f} min")
                print(f"  Updates per trading day: {result['updates_per_day']:.1f}")
                print(f"  Most active hour: {result['most_active_hour']}:00")
    
    # Summary statistics
    if results:
        df_results = pd.DataFrame(results)
        
        print("\n" + "="*90)
        print("SUMMARY STATISTICS")
        print("="*90)
        print(f"\nMarket Hours Coverage:")
        print(f"  Average: {df_results['market_hours_pct'].mean():.1f}% of updates during market hours")
        print(f"  Min: {df_results['market_hours_pct'].min():.1f}%")
        print(f"  Max: {df_results['market_hours_pct'].max():.1f}%")
        
        print(f"\nUpdate Frequency (during market hours):")
        valid_gaps = df_results[df_results['avg_gap_min'].notna()]
        if len(valid_gaps) > 0:
            print(f"  Average gap: {valid_gaps['avg_gap_min'].mean():.1f} minutes")
            print(f"  Median gap: {valid_gaps['median_gap_min'].mean():.1f} minutes")
            print(f"  Average updates/day: {valid_gaps['updates_per_day'].mean():.1f}")
        
        print(f"\nData Duration:")
        print(f"  Average: {df_results['duration_days'].mean():.0f} days")
        print(f"  Min: {df_results['duration_days'].min()} days")
        print(f"  Max: {df_results['duration_days'].max()} days")
        
        # Most active hours
        all_hours = {}
        for result in results:
            if 'hour_distribution' in result and result['hour_distribution']:
                for hour, count in result['hour_distribution'].items():
                    all_hours[hour] = all_hours.get(hour, 0) + count
        
        if all_hours:
            print(f"\nMost Active Trading Hours (ET):")
            sorted_hours = sorted(all_hours.items(), key=lambda x: x[1], reverse=True)[:5]
            for hour, count in sorted_hours:
                print(f"  {hour}:00 - {count:,} updates")
        
        # Day of week analysis
        all_dow = {'Mon': 0, 'Tue': 0, 'Wed': 0, 'Thu': 0, 'Fri': 0}
        for result in results:
            if 'dow_distribution' in result and result['dow_distribution']:
                for dow, count in result['dow_distribution'].items():
                    all_dow[dow] += count
        
        print(f"\nUpdates by Day of Week:")
        for dow in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
            print(f"  {dow}: {all_dow[dow]:,} updates")
        
        print("\n" + "="*90)
        
        # Create detailed CSV
        csv_data = df_results[['token_id', 'total_updates', 'market_hours_updates', 
                                'market_hours_pct', 'duration_days', 'avg_gap_min', 
                                'updates_per_day']].copy()
        csv_data.to_csv('token_timestamp_analysis.csv', index=False)
        print("\nDetailed results saved to: token_timestamp_analysis.csv")
    
    conn.close()

if __name__ == "__main__":
    main()
