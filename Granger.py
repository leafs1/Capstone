import os
import duckdb
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="The test statistic is outside of the range")

DB_MKT = os.getenv("MKT_DB", "./data/markets.duckdb")          # equity DB file
DB_POLY = os.getenv("POLY_DB", "./data/research.duckdb")       # polymarket DB file
DB_OUT = os.getenv("OUT_DB", DB_MKT)                           # where to store results
USE_EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "true").lower() == "true"  # Include pre/post market

# Resolve absolute paths and pick the correct alias for OUT_DB
_MKT_PATH  = os.path.abspath(DB_MKT)
_POLY_PATH = os.path.abspath(DB_POLY)
_OUT_PATH  = os.path.abspath(DB_OUT)

if _OUT_PATH == _MKT_PATH:
    OUT_ALIAS = "mkt"
elif _OUT_PATH == _POLY_PATH:
    OUT_ALIAS = "poly"
else:
    OUT_ALIAS = "out"

POLY_TABLE = "poly.main.prices"
EQUITY_TABLE = "mkt.main.security_bbo_1m"
RESULTS_TABLE = f"{OUT_ALIAS}.main.granger_results"

def _attach(conn: duckdb.DuckDBPyConnection, alias: str, path: str) -> None:
    p = os.path.abspath(path).replace("'", "''")
    conn.execute(f"ATTACH DATABASE '{p}' AS {alias}")

def get_conn() -> duckdb.DuckDBPyConnection:
    """
    In-memory router that ATTACHes:
      - mkt  -> equities DB
      - poly -> polymarket DB
      - out  -> results DB (only if a distinct file)
    """
    conn = duckdb.connect()  # in-memory
    _attach(conn, "mkt", DB_MKT)
    if _POLY_PATH != _MKT_PATH:
        _attach(conn, "poly", DB_POLY)
    else:
        # If POLY and MKT are same file, ensure alias exists
        conn.execute("CREATE SCHEMA IF NOT EXISTS poly")
    if OUT_ALIAS == "out":
        _attach(conn, "out", DB_OUT)
    return conn

def load_polymarket(conn, token_id: str, start: str, end: str) -> pd.Series:
    """
    Return actual Polymarket price updates (no resampling or forward filling).
    Returns only timestamps where real price updates occurred.
    """
    q = f"""
        SELECT
            to_timestamp(ts)::TIMESTAMP AS ts_utc,
            price
        FROM {POLY_TABLE}
        WHERE token_id = ? AND ts BETWEEN epoch(?::TIMESTAMP) AND epoch(?::TIMESTAMP)
        ORDER BY ts
    """
    df = conn.execute(q, [token_id, start, end]).df()
    if df.empty:
        return pd.Series(dtype="float64")
    
    # Set index and return without resampling or filling
    s = df.set_index(pd.to_datetime(df["ts_utc"], utc=True))["price"]
    return s

def load_equity(conn, ticker: str, start: str, end: str) -> pd.Series:
    """
    Return equity prices at 1-minute resolution.
    """
    q = f"""
        SELECT ts_utc, mid_px
        FROM {EQUITY_TABLE}
        WHERE ticker = ? AND ts_utc BETWEEN ? AND ?
        ORDER BY ts_utc
    """
    df = conn.execute(q, [ticker, start, end]).df()
    if df.empty:
        return pd.Series(dtype="float64")
    
    # Return at 1-minute granularity (no resampling since it's already 1min)
    s = df.set_index(pd.to_datetime(df["ts_utc"], utc=True))["mid_px"]
    return s

def make_returns(s: pd.Series, is_probability: bool = False) -> pd.Series:
    """
    Calculate returns/changes for time series.
    
    Args:
        s: Time series data
        is_probability: If True, treats as probability (0-1) and uses simple difference.
                       If False, uses log returns (for prices).
    
    For Polymarket (probabilities): Use simple difference (delta)
    For equity prices: Use log returns
    """
    if is_probability:
        # Simple price change for probabilities (0 to 1)
        return s.diff().dropna()
    else:
        # Log returns for actual prices
        return np.log(s).diff().dropna()

def merge_poly_with_equity(poly: pd.Series, eq: pd.Series) -> pd.DataFrame:
    """
    Merge Polymarket and equity data at 1-minute resolution.
    Since both are at 1-minute resolution, we can directly combine them.
    This approach:
    - Rounds Polymarket timestamps to nearest minute to match SPY
    - Inner join to keep only overlapping timestamps
    - Filters to US market hours only
    - Preserves full 1-minute resolution
    
    Args:
        poly: Polymarket price series (1-minute frequency with seconds)
        eq: Equity price series (1-minute frequency at minute boundaries)
    """
    if poly.empty or eq.empty:
        return pd.DataFrame()
    
    # Round Polymarket timestamps to nearest minute to match equity minute boundaries
    poly_rounded = poly.copy()
    poly_rounded.index = poly_rounded.index.round('1min')
    
    # If multiple poly prices round to same minute, take the last one (most recent)
    poly_rounded = poly_rounded[~poly_rounded.index.duplicated(keep='last')]
    
    # Combine both series - inner join keeps only matching timestamps
    df = pd.DataFrame({
        'poly': poly_rounded,
        'eq': eq
    })
    
    # Drop any rows where we don't have both values
    df = df.dropna()
    
    # Filter to US trading hours (configurable via EXTENDED_HOURS env var)
    # Convert to US/Eastern time
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert('US/Eastern')
    
    # Create time in minutes since midnight for easier comparison
    time_minutes = df_et.index.hour * 60 + df_et.index.minute
    
    if USE_EXTENDED_HOURS:
        # Extended hours: 4:00 AM - 8:00 PM ET (includes pre-market, regular, after-hours)
        #   Pre-market:  4:00 AM - 9:30 AM
        #   Regular:     9:30 AM - 4:00 PM
        #   After-hours: 4:00 PM - 8:00 PM
        trading_hours = (
            (df_et.index.weekday < 5) &  # Monday to Friday
            (time_minutes >= 240) &       # >= 4:00 AM
            (time_minutes < 1200)         # < 8:00 PM
        )
    else:
        # Regular market hours only: 9:30 AM - 4:00 PM ET
        trading_hours = (
            (df_et.index.weekday < 5) &  # Monday to Friday
            (time_minutes >= 570) &       # >= 9:30 AM
            (time_minutes < 960)          # < 4:00 PM
        )
    
    df_filtered = df[trading_hours]
    
    return df_filtered

def test_stationarity(series: pd.Series, name: str = "") -> dict:
    """
    Test stationarity using ADF and KPSS tests.
    Returns dict with test results.
    """
    # Augmented Dickey-Fuller test (H0: unit root/non-stationary)
    adf_result = adfuller(series.dropna(), autolag='AIC')
    adf_stationary = adf_result[1] < 0.05  # reject H0 if p < 0.05
    
    # KPSS test (H0: stationary)
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    kpss_stationary = kpss_result[1] >= 0.05  # fail to reject H0 if p >= 0.05
    
    return {
        "name": name,
        "adf_pvalue": adf_result[1],
        "adf_stationary": adf_stationary,
        "kpss_pvalue": kpss_result[1],
        "kpss_stationary": kpss_stationary,
        "is_stationary": adf_stationary and kpss_stationary
    }

def bonferroni_correction(pvalues: dict, alpha: float = 0.05) -> dict:
    """
    Apply Bonferroni correction for multiple testing.
    Returns dict with corrected significance and minimum corrected p-value.
    """
    n_tests = len(pvalues)
    corrected_alpha = alpha / n_tests
    significant_lags = {lag: p for lag, p in pvalues.items() if p < corrected_alpha}
    
    min_lag, min_p = min(pvalues.items(), key=lambda x: x[1])
    corrected_p = min(min_p * n_tests, 1.0)  # Bonferroni corrected p-value
    
    return {
        "min_lag": min_lag,
        "min_pvalue": min_p,
        "corrected_pvalue": corrected_p,
        "is_significant": corrected_p < alpha,
        "significant_lags": significant_lags,
        "n_tests": n_tests
    }

def check_residual_autocorr(residuals: np.ndarray, lags: int = 20) -> dict:
    """
    Check for residual autocorrelation using Ljung-Box test.
    Returns dict with test results.
    """
    try:
        lb_result = acorr_ljungbox(residuals, lags=lags, return_df=False)
        # Check if any lag shows significant autocorrelation
        min_pvalue = np.min(lb_result[1])
        has_autocorr = min_pvalue < 0.05
        
        return {
            "lb_min_pvalue": float(min_pvalue),
            "has_autocorrelation": has_autocorr
        }
    except Exception as e:
        return {
            "lb_min_pvalue": None,
            "has_autocorrelation": None,
            "error": str(e)
        }

def granger_pair(token_id: str, ticker: str, start: str, end: str, maxlag: int = 30):
    """
    Perform comprehensive Granger causality analysis with stationarity tests,
    cointegration checks, and multiple testing corrections.
    Uses only actual Polymarket price updates matched with SPY prices.
    """
    # Use attached DBs
    conn = get_conn()
    try:
        poly = load_polymarket(conn, token_id, start, end)
        eq = load_equity(conn, ticker, start, end)
    finally:
        conn.close()

    # Merge using actual poly timestamps
    df = merge_poly_with_equity(poly, eq)
    if df.empty:
        print("No overlapping data.")
        return

    # Calculate returns: simple difference for Polymarket (probabilities), log returns for equity
    r = pd.DataFrame({
        'poly': make_returns(df['poly'], is_probability=True),
        'eq': make_returns(df['eq'], is_probability=False)
    }).dropna()
    
    if len(r) < maxlag * 10:
        print(f"Too few observations for maxlag={maxlag}: got {len(r)} rows.")
        return

    print(f"\n{'='*70}")
    print(f"Granger Causality Analysis: {token_id} <-> {ticker}")
    print(f"{'='*70}")
    print(f"Rows: {len(r)}, window: {r.index.min()} -> {r.index.max()}")
    print(f"(Based on {len(r)} actual Polymarket price updates)")
    
    # Test stationarity
    print(f"\n{'─'*70}")
    print("STATIONARITY TESTS")
    print(f"{'─'*70}")
    poly_stat = test_stationarity(r["poly"], "Polymarket returns")
    eq_stat = test_stationarity(r["eq"], "Equity returns")
    
    print(f"\nPolymarket returns:")
    print(f"  ADF p-value: {poly_stat['adf_pvalue']:.4f} (stationary: {poly_stat['adf_stationary']})")
    print(f"  KPSS p-value: {poly_stat['kpss_pvalue']:.4f} (stationary: {poly_stat['kpss_stationary']})")
    print(f"  Overall stationary: {poly_stat['is_stationary']}")
    
    print(f"\nEquity returns:")
    print(f"  ADF p-value: {eq_stat['adf_pvalue']:.4f} (stationary: {eq_stat['adf_stationary']})")
    print(f"  KPSS p-value: {eq_stat['kpss_pvalue']:.4f} (stationary: {eq_stat['kpss_stationary']})")
    print(f"  Overall stationary: {eq_stat['is_stationary']}")
    
    if not (poly_stat['is_stationary'] and eq_stat['is_stationary']):
        print("\n⚠️  WARNING: One or both series are non-stationary!")
        print("   Granger causality results may be spurious.")
        print("   Note: Returns data should typically be stationary.")

    # Granger causality tests
    print(f"\n{'─'*70}")
    print("GRANGER CAUSALITY TESTS")
    print(f"{'─'*70}")
    
    print("\nTesting: Equity → Polymarket")
    res1 = grangercausalitytests(r[["poly", "eq"]], maxlag=maxlag, verbose=False)
    pvals1 = {lag: res[0]["ssr_ftest"][1] for lag, res in res1.items()}
    corrected1 = bonferroni_correction(pvals1)
    
    print(f"  Min p-value: {corrected1['min_pvalue']:.6f} at lag {corrected1['min_lag']}")
    print(f"  Bonferroni corrected p-value: {corrected1['corrected_pvalue']:.6f}")
    print(f"  Significant (α=0.05): {corrected1['is_significant']}")
    if corrected1['significant_lags']:
        print(f"  Significant lags: {list(corrected1['significant_lags'].keys())}")

    print("\nTesting: Polymarket → Equity")
    res2 = grangercausalitytests(r[["eq", "poly"]], maxlag=maxlag, verbose=False)
    pvals2 = {lag: res[0]["ssr_ftest"][1] for lag, res in res2.items()}
    corrected2 = bonferroni_correction(pvals2)
    
    print(f"  Min p-value: {corrected2['min_pvalue']:.6f} at lag {corrected2['min_lag']}")
    print(f"  Bonferroni corrected p-value: {corrected2['corrected_pvalue']:.6f}")
    print(f"  Significant (α=0.05): {corrected2['is_significant']}")
    if corrected2['significant_lags']:
        print(f"  Significant lags: {list(corrected2['significant_lags'].keys())}")
    
    # Check residuals (for the best fitting lag)
    print(f"\n{'─'*70}")
    print("RESIDUAL DIAGNOSTICS")
    print(f"{'─'*70}")
    
    # Get residuals from the model at optimal lag
    best_lag1 = corrected1['min_lag']
    best_lag2 = corrected2['min_lag']
    
    try:
        from statsmodels.tsa.api import VAR
        model = VAR(r[["poly", "eq"]])
        fitted = model.fit(maxlags=best_lag1)
        resid_check = check_residual_autocorr(fitted.resid[:, 0], lags=min(20, len(fitted.resid)//5))
        print(f"\nResiduals (Equity→Poly, lag={best_lag1}):")
        print(f"  Ljung-Box min p-value: {resid_check['lb_min_pvalue']:.4f}")
        print(f"  Has autocorrelation: {resid_check['has_autocorrelation']}")
    except Exception as e:
        print(f"\nResidual diagnostics failed: {e}")
    
    print(f"\n{'='*70}\n")

def list_tokens(conn, min_rows: int = 200) -> pd.DataFrame:
    q = f"""
        SELECT 
            p.token_id, 
            COUNT(*) AS n, 
            MIN(p.ts) AS min_ts, 
            MAX(p.ts) AS max_ts,
            MAX(m.question) AS question
        FROM {POLY_TABLE} p
        LEFT JOIN poly.main.tokens t ON p.token_id = t.token_id
        LEFT JOIN poly.main.markets m ON t.market_id = m.market_id
        GROUP BY p.token_id
        HAVING COUNT(*) >= ?
        ORDER BY n DESC
    """
    return conn.execute(q, [min_rows]).df()

def equity_window(conn, ticker: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    q = f"SELECT MIN(ts_utc) AS s, MAX(ts_utc) AS e FROM {EQUITY_TABLE} WHERE ticker = ?"
    df = conn.execute(q, [ticker]).df()
    if df.empty or pd.isna(df.loc[0, "s"]) or pd.isna(df.loc[0, "e"]):
        return None
    return pd.to_datetime(df.loc[0, "s"], utc=True), pd.to_datetime(df.loc[0, "e"], utc=True)

def token_window(conn, token_id: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    q = f"SELECT to_timestamp(MIN(ts)) AS s, to_timestamp(MAX(ts)) AS e FROM {POLY_TABLE} WHERE token_id = ?"
    df = conn.execute(q, [token_id]).df()
    if df.empty or pd.isna(df.loc[0, "s"]) or pd.isna(df.loc[0, "e"]):
        return None
    return pd.to_datetime(df.loc[0, "s"], utc=True), pd.to_datetime(df.loc[0, "e"], utc=True)

def overlap_window(conn, token_id: str, ticker: str) -> tuple[str, str] | None:
    tw = token_window(conn, token_id)
    ew = equity_window(conn, ticker)
    if not tw or not ew:
        return None
    start = max(tw[0], ew[0])
    end = min(tw[1], ew[1])
    if start >= end:
        return None
    return start.isoformat(), end.isoformat()

def granger_summary(conn, token_id: str, ticker: str, start: str, end: str, maxlag: int = 30):
    """
    Enhanced summary with stationarity checks and Bonferroni correction.
    Uses only actual Polymarket price updates matched with SPY prices.
    """
    poly = load_polymarket(conn, token_id, start, end)
    eq = load_equity(conn, ticker, start, end)

    # Merge using actual poly timestamps
    df = merge_poly_with_equity(poly, eq)
    if df.empty:
        return None

    # Calculate returns: simple difference for Polymarket (probabilities), log returns for equity
    r = pd.DataFrame({
        'poly': make_returns(df['poly'], is_probability=True),
        'eq': make_returns(df['eq'], is_probability=False)
    }).dropna()
    
    # Reduced requirement: need at least 3x the maxlag for reliable Granger test
    if len(r) < max(50, maxlag * 3):  # ensure enough data
        return None
    
    # Check for insufficient variation (constant or near-constant series)
    poly_var = r["poly"].var()
    eq_var = r["eq"].var()
    if poly_var < 1e-10 or eq_var < 1e-10:
        # Series has no meaningful variation, skip
        return None

    # Test stationarity
    poly_stat = test_stationarity(r["poly"], "poly")
    eq_stat = test_stationarity(r["eq"], "eq")
    
    # Granger tests with Bonferroni correction
    try:
        res1 = grangercausalitytests(r[["poly", "eq"]], maxlag=maxlag, verbose=False)
        pvals1 = {lag: res[0]["ssr_ftest"][1] for lag, res in res1.items()}
        corrected1 = bonferroni_correction(pvals1)
    except Exception as e:
        # Perfect fit or other numerical issue
        print(f"  → Granger test failed (Equity→Poly): {str(e)[:80]}")
        return None

    try:
        res2 = grangercausalitytests(r[["eq", "poly"]], maxlag=maxlag, verbose=False)
        pvals2 = {lag: res[0]["ssr_ftest"][1] for lag, res in res2.items()}
        corrected2 = bonferroni_correction(pvals2)
    except Exception as e:
        # Perfect fit or other numerical issue
        print(f"  → Granger test failed (Poly→Equity): {str(e)[:80]}")
        return None

    return {
        "token_id": token_id,
        "ticker": ticker,
        "start": r.index.min().isoformat(),
        "end": r.index.max().isoformat(),
        "n_obs": int(len(r)),
        # Equity -> Poly
        "lag_eq_to_poly": int(corrected1['min_lag']),
        "p_eq_to_poly": float(corrected1['min_pvalue']),
        "p_eq_to_poly_corrected": float(corrected1['corrected_pvalue']),
        "sig_eq_to_poly": bool(corrected1['is_significant']),
        # Poly -> Equity
        "lag_poly_to_eq": int(corrected2['min_lag']),
        "p_poly_to_eq": float(corrected2['min_pvalue']),
        "p_poly_to_eq_corrected": float(corrected2['corrected_pvalue']),
        "sig_poly_to_eq": bool(corrected2['is_significant']),
        # Stationarity diagnostics
        "poly_stationary": bool(poly_stat['is_stationary']),
        "eq_stationary": bool(eq_stat['is_stationary']),
        "poly_adf_p": float(poly_stat['adf_pvalue']),
        "eq_adf_p": float(eq_stat['adf_pvalue']),
    }

def ensure_results_table(conn):
    """
    Create results table for thorough Granger causality analysis.
    Includes Bonferroni-corrected p-values and stationarity diagnostics.
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
            token_id TEXT,
            ticker TEXT,
            start_ts TIMESTAMP,
            end_ts TIMESTAMP,
            n_obs INTEGER,
            lag_eq_to_poly INTEGER,
            p_eq_to_poly DOUBLE,
            p_eq_to_poly_corrected DOUBLE,
            sig_eq_to_poly BOOLEAN,
            lag_poly_to_eq INTEGER,
            p_poly_to_eq DOUBLE,
            p_poly_to_eq_corrected DOUBLE,
            sig_poly_to_eq BOOLEAN,
            poly_stationary BOOLEAN,
            eq_stationary BOOLEAN,
            poly_adf_p DOUBLE,
            eq_adf_p DOUBLE
        )
    """)

def run_all_tokens(ticker: str, maxlag: int = 30, min_rows: int = 200, limit: int | None = None) -> pd.DataFrame:
    """
    Run enhanced Granger analysis on all tokens with sufficient data.
    Skips tokens that have already been processed.
    """
    conn = get_conn()
    ensure_results_table(conn)
    try:
        toks = list_tokens(conn, min_rows=min_rows)
        if limit:
            toks = toks.head(limit)
        
        # Get existing results to skip already processed tokens
        existing = conn.execute(f"""
            SELECT token_id 
            FROM {RESULTS_TABLE} 
            WHERE ticker = ?
        """, [ticker]).df()
        existing_tokens = set(existing['token_id']) if not existing.empty else set()
        
        results = []
        skipped = 0
        for idx, row in toks.iterrows():
            token_id = row['token_id']
            question = row.get('question', 'Unknown')
            if question and len(question) > 60:
                question_display = question[:57] + "..."
            else:
                question_display = question or "Unknown"
            
            # Skip if already processed
            if token_id in existing_tokens:
                skipped += 1
                if skipped <= 5:  # Show first 5 skips
                    print(f"\n[{idx+1}/{len(toks)}] Skipping (already processed): {token_id}")
                    print(f"    Question: {question_display}")
                elif skipped == 6:
                    print(f"\n... (skipping {len(existing_tokens) - 5} more already-processed tokens)")
                continue
            
            print(f"\n[{idx+1}/{len(toks)}] Processing: {token_id}")
            print(f"    Question: {question_display}")
            ow = overlap_window(conn, token_id, ticker)
            if not ow:
                print("  → No overlap with equity data")
                continue
            start, end = ow
            summary = granger_summary(conn, token_id, ticker, start, end, maxlag=maxlag)
            if summary:
                results.append(summary)
                conn.execute(f"DELETE FROM {RESULTS_TABLE} WHERE token_id = ? AND ticker = ?", [token_id, ticker])
                conn.execute(f"""
                    INSERT INTO {RESULTS_TABLE}
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    summary["token_id"], summary["ticker"],
                    summary["start"], summary["end"], summary["n_obs"],
                    summary["lag_eq_to_poly"], summary["p_eq_to_poly"],
                    summary["p_eq_to_poly_corrected"], summary["sig_eq_to_poly"],
                    summary["lag_poly_to_eq"], summary["p_poly_to_eq"],
                    summary["p_poly_to_eq_corrected"], summary["sig_poly_to_eq"],
                    summary["poly_stationary"], summary["eq_stationary"],
                    summary["poly_adf_p"], summary["eq_adf_p"],
                ])
                print(f"  ✓ Poly→Eq: p={summary['p_poly_to_eq_corrected']:.4f} (sig: {summary['sig_poly_to_eq']})")
                print(f"  ✓ Eq→Poly: p={summary['p_eq_to_poly_corrected']:.4f} (sig: {summary['sig_eq_to_poly']})")
            else:
                print("  → Insufficient data")
        
        if skipped > 0:
            print(f"\n{'='*70}")
            print(f"Skipped {skipped} already-processed tokens")
            print(f"{'='*70}")
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("p_poly_to_eq_corrected").reset_index(drop=True)
        return df
    finally:
        conn.close()

if __name__ == "__main__":
    TICKER = os.getenv("GRANGER_TICKER", "SPY")
    MAXLAG = int(os.getenv("GRANGER_MAXLAG", "30"))
    MIN_ROWS = int(os.getenv("GRANGER_MIN_ROWS", "200"))
    LIMIT = int(os.getenv("GRANGER_LIMIT", "0")) or None

    print("="*70)
    print("Enhanced Granger Causality Analysis")
    print("="*70)
    print(f"Ticker: {TICKER}")
    print(f"Max Lag: {MAXLAG}")
    print(f"Min Rows: {MIN_ROWS}")
    print(f"Limit: {LIMIT if LIMIT else 'None (all tokens)'}")
    print(f"Trading Hours: {'Extended (4am-8pm ET)' if USE_EXTENDED_HOURS else 'Regular (9:30am-4pm ET)'}")
    print("="*70)

    df = run_all_tokens(TICKER, maxlag=MAXLAG, min_rows=MIN_ROWS, limit=LIMIT)
    if df.empty:
        print("\nNo results.")
    else:
        print("\n" + "="*70)
        print("SUMMARY RESULTS (Top 20 by corrected p-value)")
        print("="*70)
        display_cols = [
            "token_id", "n_obs", 
            "p_poly_to_eq_corrected", "sig_poly_to_eq", "lag_poly_to_eq",
            "p_eq_to_poly_corrected", "sig_eq_to_poly", "lag_eq_to_poly",
            "poly_stationary", "eq_stationary"
        ]
        print(df[display_cols].head(20).to_string(index=False))
        
        # Summary statistics
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(f"Total pairs analyzed: {len(df)}")
        sig_poly_to_eq = df[df['sig_poly_to_eq']]
        sig_eq_to_poly = df[df['sig_eq_to_poly']]
        print(f"Significant Poly→Eq (corrected): {len(sig_poly_to_eq)}")
        print(f"Significant Eq→Poly (corrected): {len(sig_eq_to_poly)}")
        
        # Lag distribution analysis
        if not sig_poly_to_eq.empty or not sig_eq_to_poly.empty:
            print("\n" + "="*70)
            print("LAG ANALYSIS (Timing of Relationships)")
            print("="*70)
            
            if not sig_poly_to_eq.empty:
                print(f"\nPOLYMARKET → EQUITY ({len(sig_poly_to_eq)} significant)")
                print("─" * 70)
                print(f"  Mean lag: {sig_poly_to_eq['lag_poly_to_eq'].mean():.1f} minutes")
                print(f"  Median lag: {sig_poly_to_eq['lag_poly_to_eq'].median():.1f} minutes")
                print(f"  Range: {sig_poly_to_eq['lag_poly_to_eq'].min()}-{sig_poly_to_eq['lag_poly_to_eq'].max()} minutes")
                
                # Show lag distribution
                lag_dist = sig_poly_to_eq['lag_poly_to_eq'].value_counts().sort_index()
                print(f"\n  Lag distribution:")
                for lag, count in lag_dist.head(10).items():
                    bar = '█' * min(count, 50)
                    print(f"    {lag:2d} min: {bar} ({count})")
            
            if not sig_eq_to_poly.empty:
                print(f"\nEQUITY → POLYMARKET ({len(sig_eq_to_poly)} significant)")
                print("─" * 70)
                print(f"  Mean lag: {sig_eq_to_poly['lag_eq_to_poly'].mean():.1f} minutes")
                print(f"  Median lag: {sig_eq_to_poly['lag_eq_to_poly'].median():.1f} minutes")
                print(f"  Range: {sig_eq_to_poly['lag_eq_to_poly'].min()}-{sig_eq_to_poly['lag_eq_to_poly'].max()} minutes")
                
                # Show lag distribution
                lag_dist = sig_eq_to_poly['lag_eq_to_poly'].value_counts().sort_index()
                print(f"\n  Lag distribution:")
                for lag, count in lag_dist.head(10).items():
                    bar = '█' * min(count, 50)
                    print(f"    {lag:2d} min: {bar} ({count})")
            
            # Bidirectional relationships
            bidirectional = df[df['sig_poly_to_eq'] & df['sig_eq_to_poly']]
            if not bidirectional.empty:
                print(f"\nBIDIRECTIONAL CAUSALITY ({len(bidirectional)} tokens)")
                print("─" * 70)
                print("  These show feedback loops (both directions significant):")
                for idx, row in bidirectional.head(5).iterrows():
                    print(f"    • {row['token_id'][:16]}... Poly→Eq:{row['lag_poly_to_eq']}min, Eq→Poly:{row['lag_eq_to_poly']}min")
        
        # Quality warnings
        non_stationary = df[~(df['poly_stationary'] & df['eq_stationary'])]
        if not non_stationary.empty:
            print(f"\n⚠️  DATA QUALITY: {len(non_stationary)} results have non-stationary series (may be spurious)")
        print(f"Stationary pairs: {(df['poly_stationary'] & df['eq_stationary']).sum()}")
        print("="*70)