#!/usr/bin/env python3
"""
Plot Polymarket contract prices vs SPY prices for markets with significant Granger causality.
Shows visual relationships between prediction markets and equity prices.
"""
import os
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

def get_significant_markets(direction="both", limit=1000):
    """Get markets with significant Granger causality."""
    conn = get_conn()
    
    try:
        if direction == "both":
            direction_filter = "AND g.sig_poly_to_eq = TRUE AND g.sig_eq_to_poly = TRUE"
        elif direction == "poly_to_eq":
            direction_filter = "AND g.sig_poly_to_eq = TRUE"
        elif direction == "eq_to_poly":
            direction_filter = "AND g.sig_eq_to_poly = TRUE"
        else:
            direction_filter = "AND (g.sig_poly_to_eq = TRUE OR g.sig_eq_to_poly = TRUE)"
        
        query = f"""
            SELECT DISTINCT
                g.token_id,
                m.question,
                m.slug,
                g.start_ts,
                g.end_ts,
                g.n_obs,
                g.lag_poly_to_eq,
                g.lag_eq_to_poly,
                g.p_poly_to_eq_corrected,
                g.p_eq_to_poly_corrected
            FROM mkt.main.granger_results g
            LEFT JOIN poly.main.tokens t ON g.token_id = t.token_id
            LEFT JOIN poly.main.markets m ON t.market_id = m.market_id
            WHERE 1=1 {direction_filter}
            ORDER BY LEAST(g.p_poly_to_eq_corrected, g.p_eq_to_poly_corrected)
            LIMIT ?
        """
        
        return conn.execute(query, [limit]).df()
    finally:
        conn.close()

def load_market_data(token_id, start_ts, end_ts):
    """Load Polymarket and SPY data for a specific token."""
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
        poly_df = poly_df.set_index('timestamp')
        
        # Use minute-level data for accurate visualization
        poly_data = poly_df['price']
        
        # Load SPY prices
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
        
        # Use minute-level data for accurate visualization
        spy_data = spy_df['price']
        
        return poly_data, spy_data
        
    finally:
        conn.close()

def plot_market(token_id, question, start_ts, end_ts, lag_pty, lag_eqy, p_pty, p_eqy, 
                output_dir="plots"):
    """Create dual-axis plot showing Polymarket contract vs SPY price."""
    
    # Load data
    poly_prices, spy_prices = load_market_data(token_id, start_ts, end_ts)
    
    if poly_prices.empty or spy_prices.empty:
        print(f"  ⚠️  No data for plotting")
        return None
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot Polymarket price on left axis
    color = 'tab:blue'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Polymarket Probability', color=color, fontsize=12)
    ax1.plot(poly_prices.index, poly_prices.values, color=color, linewidth=1.5, 
             label='Polymarket', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Plot SPY price on right axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('SPY Price ($)', color=color, fontsize=12)
    ax2.plot(spy_prices.index, spy_prices.values, color=color, linewidth=1.5, 
             label='SPY', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Format title with Granger info
    title = f"{question[:80]}\n"
    title += f"Poly→Eq: lag={lag_pty}min, p={p_pty:.6f} | "
    title += f"Eq→Poly: lag={lag_eqy}min, p={p_eqy:.6f}"
    plt.title(title, fontsize=11, pad=20)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                           for c in question[:50])
    filename = f"{output_dir}/{safe_filename}_{token_id[:16]}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()
    
    return filename

def plot_all_significant_markets(direction="both", limit=1000, output_dir="plots"):
    """Plot all significant markets."""
    
    print("=" * 100)
    print("PLOTTING POLYMARKET CONTRACTS vs SPY PRICES")
    print("=" * 100)
    print(f"Direction filter: {direction}")
    print(f"Number of markets: {limit}")
    print(f"Output directory: {output_dir}")
    print("=" * 100)
    
    markets = get_significant_markets(direction=direction, limit=limit)
    
    if markets.empty:
        print("\n⚠️  No markets found with significant Granger causality")
        return
    
    print(f"\nFound {len(markets)} markets to plot\n")
    
    successful_plots = []
    
    for idx, row in markets.iterrows():
        print(f"[{idx+1}/{len(markets)}] {row['question'][:70]}")
        
        try:
            filename = plot_market(
                token_id=row['token_id'],
                question=row['question'],
                start_ts=row['start_ts'],
                end_ts=row['end_ts'],
                lag_pty=row['lag_poly_to_eq'],
                lag_eqy=row['lag_eq_to_poly'],
                p_pty=row['p_poly_to_eq_corrected'],
                p_eqy=row['p_eq_to_poly_corrected'],
                output_dir=output_dir
            )
            
            if filename:
                successful_plots.append(filename)
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:80]}")
    
    print("\n" + "=" * 100)
    print(f"✓ Successfully created {len(successful_plots)} plots")
    print(f"✓ Saved to: {output_dir}/")
    print("=" * 100)
    
    return successful_plots

def plot_correlation_scatter(token_id, question, start_ts, end_ts, output_dir="plots"):
    """Create scatter plot showing correlation between Polymarket and SPY returns."""
    
    # Load data
    poly_prices, spy_prices = load_market_data(token_id, start_ts, end_ts)
    
    if poly_prices.empty or spy_prices.empty:
        return None
    
    # Calculate returns
    poly_returns = poly_prices.diff().dropna()
    spy_returns = spy_prices.pct_change().dropna()
    
    # Align by index
    combined = pd.DataFrame({
        'poly_returns': poly_returns,
        'spy_returns': spy_returns
    }).dropna()
    
    if len(combined) < 10:
        return None
    
    # Calculate correlation
    correlation = combined['poly_returns'].corr(combined['spy_returns'])
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(combined['spy_returns'], combined['poly_returns'], 
              alpha=0.5, s=20, edgecolors='none')
    
    # Add regression line
    z = np.polyfit(combined['spy_returns'], combined['poly_returns'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(combined['spy_returns'].min(), combined['spy_returns'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
           label=f'Linear fit (R={correlation:.3f})')
    
    ax.set_xlabel('SPY Hourly Returns', fontsize=12)
    ax.set_ylabel('Polymarket Hourly Price Change', fontsize=12)
    ax.set_title(f"{question[:80]}\nCorrelation: {correlation:.4f}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                           for c in question[:50])
    filename = f"{output_dir}/scatter_{safe_filename}_{token_id[:16]}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot Polymarket contracts vs SPY prices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot top 10 bidirectional markets
  python plot_granger_markets.py --direction both --limit 10
  
  # Plot all significant markets (any direction)
  python plot_granger_markets.py --direction any --limit 20
  
  # Include scatter plots showing correlation
  python plot_granger_markets.py --scatter
  
  # Custom output directory
  python plot_granger_markets.py --output my_plots/
        """
    )
    
    parser.add_argument(
        "--direction",
        choices=["any", "both", "poly_to_eq", "eq_to_poly"],
        default="both",
        help="Filter by causality direction (default: both)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of markets to plot (default: 10)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots/)"
    )
    
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Also create scatter plots showing correlation"
    )
    
    args = parser.parse_args()
    
    # Create time series plots
    plot_files = plot_all_significant_markets(
        direction=args.direction,
        limit=args.limit,
        output_dir=args.output
    )
    
    # Create scatter plots if requested
    if args.scatter and plot_files:
        print("\n" + "=" * 100)
        print("CREATING CORRELATION SCATTER PLOTS")
        print("=" * 100)
        
        markets = get_significant_markets(direction=args.direction, limit=args.limit)
        
        for idx, row in markets.iterrows():
            print(f"[{idx+1}/{len(markets)}] Creating scatter plot...")
            try:
                plot_correlation_scatter(
                    token_id=row['token_id'],
                    question=row['question'],
                    start_ts=row['start_ts'],
                    end_ts=row['end_ts'],
                    output_dir=args.output
                )
                print(f"  ✓ Saved scatter plot")
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:80]}")
        
        print("=" * 100)
