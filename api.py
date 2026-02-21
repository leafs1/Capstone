"""
Flask API backend for the Research Dashboard.
Wraps existing analysis modules (Granger, FinData, PolyMarket, DbController)
and serves JSON endpoints + generated plots to the React frontend.
"""
import os
import io
import json
import base64
import traceback
import threading
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Project imports ──────────────────────────────────────────────
import Granger
from DbController import DuckDBController, DuckDBConfig

# ── Configuration ────────────────────────────────────────────────
DB_MKT  = os.getenv("MKT_DB",  "./data/markets.duckdb")
DB_POLY = os.getenv("POLY_DB", "./data/research.duckdb")

app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler — return JSON instead of HTML for all errors."""
    import traceback
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

# ── Helpers ──────────────────────────────────────────────────────

# Single persistent read-only connection shared across all read requests.
# DuckDB does not allow multiple processes/connections to attach the same
# database file simultaneously, so we reuse one connection + a lock.
_read_conn = None
_read_lock = threading.Lock()
_write_lock = threading.Lock()


def _init_read_conn():
    """Initialise the shared read-only connection (called once at startup)."""
    global _read_conn
    _read_conn = duckdb.connect()
    mkt  = os.path.abspath(DB_MKT).replace("'", "''")
    poly = os.path.abspath(DB_POLY).replace("'", "''")
    _read_conn.execute(f"ATTACH DATABASE '{mkt}' AS mkt (READ_ONLY)")
    if os.path.abspath(DB_MKT) != os.path.abspath(DB_POLY):
        _read_conn.execute(f"ATTACH DATABASE '{poly}' AS poly (READ_ONLY)")


def _get_conn():
    """Return the shared read-only connection (caller must hold _read_lock)."""
    return _read_conn


def _get_write_conn():
    """Create a fresh writable connection (caller must hold _write_lock).

    We detach the read-only databases first, open writable, then re-attach
    read-only afterwards.  In practice this is only used by background
    analysis jobs.
    """
    conn = duckdb.connect()
    mkt  = os.path.abspath(DB_MKT).replace("'", "''")
    poly = os.path.abspath(DB_POLY).replace("'", "''")
    conn.execute(f"ATTACH DATABASE '{mkt}' AS mkt")
    if os.path.abspath(DB_MKT) != os.path.abspath(DB_POLY):
        conn.execute(f"ATTACH DATABASE '{poly}' AS poly (READ_ONLY)")
    return conn


def _safe(val):
    """Make a value JSON-safe (handle NaN, Inf, numpy types)."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
    return val


def _df_to_records(df):
    """Convert DataFrame to list of dicts with JSON-safe values."""
    records = []
    for _, row in df.iterrows():
        records.append({k: _safe(v) for k, v in row.items()})
    return records


# In-memory job tracking for long-running analyses
_jobs = {}
_jobs_lock = threading.Lock()


# ═════════════════════════════════════════════════════════════════
# DATASETS
# ═════════════════════════════════════════════════════════════════

@app.route("/api/datasets/summary")
def datasets_summary():
    """Summary counts for all dataset tables."""
    with _read_lock:
        conn = _get_conn()
        # Polymarket data
        poly_markets = conn.execute("SELECT COUNT(*) FROM poly.main.markets").fetchone()[0]
        poly_tokens  = conn.execute("SELECT COUNT(*) FROM poly.main.tokens").fetchone()[0]
        poly_prices  = conn.execute("SELECT COUNT(*) FROM poly.main.prices").fetchone()[0]

        # Equity data
        try:
            eq_rows   = conn.execute("SELECT COUNT(*) FROM mkt.main.security_bbo_1m").fetchone()[0]
            eq_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM mkt.main.security_bbo_1m").fetchone()[0]
            eq_range  = conn.execute("SELECT MIN(ts_utc)::TEXT, MAX(ts_utc)::TEXT FROM mkt.main.security_bbo_1m").fetchone()
        except Exception:
            eq_rows, eq_tickers, eq_range = 0, 0, (None, None)

        # Granger results
        try:
            gr_total = conn.execute("SELECT COUNT(*) FROM mkt.main.granger_results").fetchone()[0]
            gr_sig_pe = conn.execute("SELECT COUNT(*) FROM mkt.main.granger_results WHERE sig_poly_to_eq=TRUE").fetchone()[0]
            gr_sig_ep = conn.execute("SELECT COUNT(*) FROM mkt.main.granger_results WHERE sig_eq_to_poly=TRUE").fetchone()[0]
        except Exception:
            gr_total, gr_sig_pe, gr_sig_ep = 0, 0, 0

    return jsonify({
        "polymarket": {
            "markets": poly_markets,
            "tokens": poly_tokens,
            "prices": poly_prices,
        },
        "equity": {
            "rows": eq_rows,
            "tickers": eq_tickers,
            "date_range": {"start": eq_range[0], "end": eq_range[1]} if eq_range[0] else None,
        },
        "granger": {
            "total": gr_total,
            "sig_poly_to_eq": gr_sig_pe,
            "sig_eq_to_poly": gr_sig_ep,
        }
    })


@app.route("/api/datasets/markets")
def datasets_markets():
    """List Polymarket markets with optional theme filter."""
    theme = request.args.get("theme")
    limit = int(request.args.get("limit", 200))
    offset = int(request.args.get("offset", 0))

    where = "WHERE m.theme = ?" if theme else ""
    params = [theme] if theme else []

    q = f"""
        SELECT m.market_id, m.question, m.theme, m.active, m.closed,
               m.startDateIso, m.endDateIso, m.liquidityNum, m.volumeNum,
               COUNT(DISTINCT t.token_id) as token_count,
               COALESCE(SUM(pc.price_count), 0) as total_prices
        FROM poly.main.markets m
        LEFT JOIN poly.main.tokens t ON m.market_id = t.market_id
        LEFT JOIN (
            SELECT token_id, COUNT(*) as price_count
            FROM poly.main.prices
            GROUP BY token_id
        ) pc ON t.token_id = pc.token_id
        {where}
        GROUP BY m.market_id, m.question, m.theme, m.active, m.closed,
                 m.startDateIso, m.endDateIso, m.liquidityNum, m.volumeNum
        ORDER BY m.volumeNum DESC NULLS LAST
        LIMIT ? OFFSET ?
    """
    params += [limit, offset]

    with _read_lock:
        conn = _get_conn()
        df = conn.execute(q, params).df()
        total_q = f"SELECT COUNT(*) FROM poly.main.markets m {where}"
        total = conn.execute(total_q, [theme] if theme else []).fetchone()[0]

    return jsonify({"data": _df_to_records(df), "total": total})


@app.route("/api/datasets/themes")
def datasets_themes():
    """Get list of all themes and their counts."""
    with _read_lock:
        conn = _get_conn()
        df = conn.execute("""
            SELECT theme, COUNT(*) as count
            FROM poly.main.markets
            WHERE theme IS NOT NULL
            GROUP BY theme
            ORDER BY count DESC
        """).df()
    return jsonify(_df_to_records(df))


@app.route("/api/datasets/equity/tickers")
def datasets_equity_tickers():
    """Get available equity tickers."""
    try:
        with _read_lock:
            conn = _get_conn()
            df = conn.execute("""
                SELECT ticker, COUNT(*) as rows,
                       MIN(ts_utc)::TEXT as start_date,
                       MAX(ts_utc)::TEXT as end_date
                FROM mkt.main.security_bbo_1m
                GROUP BY ticker
                ORDER BY ticker
            """).df()
        return jsonify(_df_to_records(df))
    except Exception:
        return jsonify([])


@app.route("/api/datasets/tokens")
def datasets_tokens():
    """List tokens with price data summary."""
    limit = int(request.args.get("limit", 100))
    min_rows = int(request.args.get("min_rows", 50))

    with _read_lock:
        conn = _get_conn()
        df = conn.execute(f"""
            SELECT p.token_id,
                   COUNT(*) AS n_prices,
                   MIN(p.ts) AS min_ts,
                   MAX(p.ts) AS max_ts,
                   MAX(m.question) AS question,
                   MAX(m.theme) AS theme
            FROM poly.main.prices p
            LEFT JOIN poly.main.tokens t ON p.token_id = t.token_id
            LEFT JOIN poly.main.markets m ON t.market_id = m.market_id
            GROUP BY p.token_id
            HAVING COUNT(*) >= ?
            ORDER BY n_prices DESC
            LIMIT ?
        """, [min_rows, limit]).df()
    return jsonify(_df_to_records(df))


# ═════════════════════════════════════════════════════════════════
# GRANGER RESULTS
# ═════════════════════════════════════════════════════════════════

@app.route("/api/results")
def granger_results():
    """Get Granger causality results with market metadata."""
    direction = request.args.get("direction", "any")
    sig_level = float(request.args.get("sig_level", 0.05))
    limit = int(request.args.get("limit", 200))
    sort_by = request.args.get("sort_by", "p_value")

    if direction == "both":
        dir_filter = "AND g.sig_poly_to_eq = TRUE AND g.sig_eq_to_poly = TRUE"
    elif direction == "poly_to_eq":
        dir_filter = "AND g.sig_poly_to_eq = TRUE"
    elif direction == "eq_to_poly":
        dir_filter = "AND g.sig_eq_to_poly = TRUE"
    else:
        dir_filter = "AND (g.sig_poly_to_eq = TRUE OR g.sig_eq_to_poly = TRUE)"

    q = f"""
        SELECT
            g.token_id,
            COALESCE(m.question, 'Unknown') as question,
            COALESCE(m.theme, 'unknown') as theme,
            g.ticker,
            g.n_obs,
            g.start_ts::TEXT as start_ts,
            g.end_ts::TEXT as end_ts,
            g.sig_poly_to_eq,
            g.lag_poly_to_eq,
            g.p_poly_to_eq AS p_poly_to_eq_raw,
            g.p_poly_to_eq_corrected,
            g.sig_eq_to_poly,
            g.lag_eq_to_poly,
            g.p_eq_to_poly AS p_eq_to_poly_raw,
            g.p_eq_to_poly_corrected,
            g.poly_stationary,
            g.eq_stationary,
            g.poly_adf_p,
            g.eq_adf_p
        FROM mkt.main.granger_results g
        LEFT JOIN poly.main.tokens t ON g.token_id = t.token_id
        LEFT JOIN poly.main.markets m ON t.market_id = m.market_id
        WHERE (g.p_poly_to_eq_corrected <= {sig_level}
               OR g.p_eq_to_poly_corrected <= {sig_level})
        {dir_filter}
        ORDER BY LEAST(g.p_poly_to_eq_corrected, g.p_eq_to_poly_corrected) ASC
        LIMIT {limit}
    """

    try:
        with _read_lock:
            conn = _get_conn()
            df = conn.execute(q).df()
            all_results = conn.execute("SELECT COUNT(*) FROM mkt.main.granger_results").fetchone()[0]

        return jsonify({
            "data": _df_to_records(df),
            "total_analyzed": all_results,
            "total_filtered": len(df),
        })
    except Exception as e:
        return jsonify({"data": [], "total_analyzed": 0, "total_filtered": 0, "error": str(e)})


@app.route("/api/results/stats")
def results_stats():
    """Aggregate statistics about Granger results."""
    try:
        with _read_lock:
            conn = _get_conn()
            total = conn.execute("SELECT COUNT(*) FROM mkt.main.granger_results").fetchone()[0]
            sig_pe = conn.execute("SELECT COUNT(*) FROM mkt.main.granger_results WHERE sig_poly_to_eq=TRUE").fetchone()[0]
            sig_ep = conn.execute("SELECT COUNT(*) FROM mkt.main.granger_results WHERE sig_eq_to_poly=TRUE").fetchone()[0]
            sig_both = conn.execute("SELECT COUNT(*) FROM mkt.main.granger_results WHERE sig_poly_to_eq=TRUE AND sig_eq_to_poly=TRUE").fetchone()[0]
            stationary = conn.execute("SELECT COUNT(*) FROM mkt.main.granger_results WHERE poly_stationary=TRUE AND eq_stationary=TRUE").fetchone()[0]

            lag_pe = conn.execute("""
                SELECT lag_poly_to_eq as lag, COUNT(*) as count
                FROM mkt.main.granger_results WHERE sig_poly_to_eq=TRUE
                GROUP BY lag_poly_to_eq ORDER BY lag_poly_to_eq
            """).df()
            lag_ep = conn.execute("""
                SELECT lag_eq_to_poly as lag, COUNT(*) as count
                FROM mkt.main.granger_results WHERE sig_eq_to_poly=TRUE
                GROUP BY lag_eq_to_poly ORDER BY lag_eq_to_poly
            """).df()

            pval_dist = conn.execute("""
                SELECT
                    CASE
                        WHEN LEAST(p_poly_to_eq_corrected, p_eq_to_poly_corrected) < 0.001 THEN '< 0.001'
                        WHEN LEAST(p_poly_to_eq_corrected, p_eq_to_poly_corrected) < 0.01 THEN '0.001-0.01'
                        WHEN LEAST(p_poly_to_eq_corrected, p_eq_to_poly_corrected) < 0.05 THEN '0.01-0.05'
                        ELSE '> 0.05'
                    END as bucket,
                    COUNT(*) as count
                FROM mkt.main.granger_results
                GROUP BY bucket
                ORDER BY bucket
            """).df()

            theme_df = conn.execute("""
                SELECT COALESCE(m.theme, 'unknown') as theme,
                       COUNT(*) as total,
                       SUM(CASE WHEN g.sig_poly_to_eq THEN 1 ELSE 0 END) as sig_poly_to_eq,
                       SUM(CASE WHEN g.sig_eq_to_poly THEN 1 ELSE 0 END) as sig_eq_to_poly
                FROM mkt.main.granger_results g
                LEFT JOIN poly.main.tokens t ON g.token_id = t.token_id
                LEFT JOIN poly.main.markets m ON t.market_id = m.market_id
                GROUP BY theme
                ORDER BY total DESC
            """).df()

            avg_obs = conn.execute("SELECT AVG(n_obs)::INT FROM mkt.main.granger_results").fetchone()[0]

        return jsonify({
            "total": total,
            "sig_poly_to_eq": sig_pe,
            "sig_eq_to_poly": sig_ep,
            "sig_both": sig_both,
            "stationary": stationary,
            "avg_observations": avg_obs,
            "lag_dist_poly_to_eq": _df_to_records(lag_pe),
            "lag_dist_eq_to_poly": _df_to_records(lag_ep),
            "pvalue_distribution": _df_to_records(pval_dist),
            "theme_breakdown": _df_to_records(theme_df),
        })
    except Exception as e:
        return jsonify({"error": str(e), "total": 0})


# ═════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════

@app.route("/api/plot/timeseries/<token_id>")
def plot_timeseries(token_id):
    """Return time series data (poly + equity) for a token as JSON for Recharts."""
    with _read_lock:
        conn = _get_conn()
        meta = conn.execute("""
            SELECT start_ts::TEXT, end_ts::TEXT, ticker
            FROM mkt.main.granger_results
            WHERE token_id = ?
            LIMIT 1
        """, [token_id]).fetchone()
        if not meta:
            return jsonify({"error": "Token not found in results"}), 404

        start_ts, end_ts, ticker = meta

        poly_df = conn.execute("""
            SELECT to_timestamp(ts)::TEXT as timestamp, price
            FROM poly.main.prices
            WHERE token_id = ?
            AND to_timestamp(ts) BETWEEN ?::TIMESTAMP AND ?::TIMESTAMP
            ORDER BY ts
        """, [token_id, start_ts, end_ts]).df()

        eq_df = conn.execute("""
            SELECT ts_utc::TEXT as timestamp, mid_px as price
            FROM mkt.main.security_bbo_1m
            WHERE ticker = ?
            AND ts_utc BETWEEN ?::TIMESTAMP AND ?::TIMESTAMP
            ORDER BY ts_utc
        """, [ticker, start_ts, end_ts]).df()

    # Data processing outside the lock
    if not poly_df.empty:
        poly_df['timestamp'] = pd.to_datetime(poly_df['timestamp'])
        poly_df = poly_df.set_index('timestamp').resample('1h').last().dropna().reset_index()
        poly_df['timestamp'] = poly_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

    if not eq_df.empty:
        eq_df['timestamp'] = pd.to_datetime(eq_df['timestamp'])
        eq_df = eq_df.set_index('timestamp').resample('1h').last().dropna().reset_index()
        eq_df['timestamp'] = eq_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

    if not poly_df.empty and not eq_df.empty:
        merged = pd.merge(poly_df, eq_df, on='timestamp', how='outer', suffixes=('_poly', '_eq'))
        merged = merged.sort_values('timestamp').fillna(method='ffill')
        merged = merged.rename(columns={'price_poly': 'poly', 'price_eq': 'equity'})
        data = _df_to_records(merged[['timestamp', 'poly', 'equity']].dropna())
    else:
        data = []

    return jsonify({
        "data": data,
        "ticker": ticker,
        "token_id": token_id,
    })


@app.route("/api/plot/scatter/<token_id>")
def plot_scatter(token_id):
    """Return scatter data (poly returns vs equity returns) for a token."""
    with _read_lock:
        conn = _get_conn()
        meta = conn.execute("""
            SELECT start_ts::TEXT, end_ts::TEXT, ticker
            FROM mkt.main.granger_results WHERE token_id = ? LIMIT 1
        """, [token_id]).fetchone()
        if not meta:
            return jsonify({"error": "Not found"}), 404

        start_ts, end_ts, ticker = meta
        poly = Granger.load_polymarket(conn, token_id, start_ts, end_ts)
        eq = Granger.load_equity(conn, ticker, start_ts, end_ts)

    df = Granger.merge_poly_with_equity(poly, eq)

    if df.empty:
        return jsonify({"data": [], "correlation": None})

    r = pd.DataFrame({
        'poly_return': Granger.make_returns(df['poly'], is_probability=True),
        'eq_return': Granger.make_returns(df['eq'], is_probability=False)
    }).dropna()

    if len(r) > 2000:
        r = r.sample(2000, random_state=42)

    corr = float(r['poly_return'].corr(r['eq_return']))

    return jsonify({
        "data": _df_to_records(r.reset_index(drop=True)),
        "correlation": _safe(corr),
        "n_points": len(r),
    })


@app.route("/api/plot/png/<token_id>")
def plot_png(token_id):
    """Generate and return a dual-axis PNG plot as base64."""
    with _read_lock:
        conn = _get_conn()
        meta = conn.execute("""
            SELECT g.start_ts, g.end_ts, g.ticker,
                   g.lag_poly_to_eq, g.lag_eq_to_poly,
                   g.p_poly_to_eq_corrected, g.p_eq_to_poly_corrected,
                   COALESCE(m.question, 'Unknown') as question
            FROM mkt.main.granger_results g
            LEFT JOIN poly.main.tokens t ON g.token_id = t.token_id
            LEFT JOIN poly.main.markets m ON t.market_id = m.market_id
            WHERE g.token_id = ?
            LIMIT 1
        """, [token_id]).fetchone()

        if not meta:
            return jsonify({"error": "Not found"}), 404

        start_ts, end_ts, ticker, lag_pe, lag_ep, p_pe, p_ep, question = meta

        poly_df = conn.execute("""
            SELECT to_timestamp(ts) as timestamp, price
            FROM poly.main.prices
            WHERE token_id = ? AND to_timestamp(ts) BETWEEN ? AND ?
            ORDER BY ts
        """, [token_id, start_ts, end_ts]).df()

        eq_df = conn.execute("""
            SELECT ts_utc as timestamp, mid_px as price
            FROM mkt.main.security_bbo_1m
            WHERE ticker = ? AND ts_utc BETWEEN ? AND ?
            ORDER BY ts_utc
        """, [ticker, start_ts, end_ts]).df()

    if poly_df.empty or eq_df.empty:
        return jsonify({"error": "No data"}), 404

    poly_df['timestamp'] = pd.to_datetime(poly_df['timestamp'], utc=True)
    eq_df['timestamp'] = pd.to_datetime(eq_df['timestamp'], utc=True)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Polymarket Probability', color='tab:blue')
    ax1.plot(poly_df['timestamp'], poly_df['price'], color='tab:blue', linewidth=1.2, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel(f'{ticker} Price ($)', color='tab:red')
    ax2.plot(eq_df['timestamp'], eq_df['price'], color='tab:red', linewidth=1.2, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    title = f"{question[:80]}\n"
    title += f"Poly→Eq: lag={lag_pe}min, p={p_pe:.6f} | Eq→Poly: lag={lag_ep}min, p={p_ep:.6f}"
    plt.title(title, fontsize=11, pad=20)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close()
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()

    return jsonify({"image": f"data:image/png;base64,{b64}"})


# ═════════════════════════════════════════════════════════════════
# ANALYSIS (Run Granger)
# ═════════════════════════════════════════════════════════════════

@app.route("/api/analysis/run", methods=["POST"])
def analysis_run():
    """Start a Granger causality analysis run (background thread)."""
    body = request.json or {}
    ticker = body.get("ticker", "SPY")
    maxlag = int(body.get("maxlag", 30))
    min_rows = int(body.get("min_rows", 200))
    limit = body.get("limit")
    if limit:
        limit = int(limit)

    job_id = f"granger_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "running",
            "progress": 0,
            "total": 0,
            "processed": 0,
            "results_count": 0,
            "started": datetime.now().isoformat(),
            "error": None,
        }

    def _run():
        try:
            conn = Granger.get_conn()
            Granger.ensure_results_table(conn)
            toks = Granger.list_tokens(conn, min_rows=min_rows)
            if limit:
                toks = toks.head(limit)

            with _jobs_lock:
                _jobs[job_id]["total"] = len(toks)

            results = []
            for idx, row in toks.iterrows():
                tid = row['token_id']
                ow = Granger.overlap_window(conn, tid, ticker)
                if not ow:
                    with _jobs_lock:
                        _jobs[job_id]["processed"] = idx + 1
                    continue

                start, end = ow
                summary = Granger.granger_summary(conn, tid, ticker, start, end, maxlag=maxlag)
                if summary:
                    results.append(summary)
                    conn.execute(f"DELETE FROM {Granger.RESULTS_TABLE} WHERE token_id = ? AND ticker = ?", [tid, ticker])
                    conn.execute(f"""
                        INSERT INTO {Granger.RESULTS_TABLE}
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

                with _jobs_lock:
                    _jobs[job_id]["processed"] = idx + 1
                    _jobs[job_id]["results_count"] = len(results)
                    _jobs[job_id]["progress"] = int((idx + 1) / len(toks) * 100)

            conn.close()
            with _jobs_lock:
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["progress"] = 100
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/api/analysis/run_single", methods=["POST"])
def analysis_run_single():
    """Run Granger analysis on a single token (synchronous)."""
    body = request.json or {}
    token_id = body.get("token_id")
    ticker = body.get("ticker", "SPY")
    maxlag = int(body.get("maxlag", 30))

    if not token_id:
        return jsonify({"error": "token_id is required"}), 400

    try:
        conn = Granger.get_conn()
        Granger.ensure_results_table(conn)
        ow = Granger.overlap_window(conn, token_id, ticker)
        if not ow:
            conn.close()
            return jsonify({"error": "No overlapping data window"}), 400

        start, end = ow
        summary = Granger.granger_summary(conn, token_id, ticker, start, end, maxlag=maxlag)
        if summary:
            conn.execute(f"DELETE FROM {Granger.RESULTS_TABLE} WHERE token_id = ? AND ticker = ?", [token_id, ticker])
            conn.execute(f"""
                INSERT INTO {Granger.RESULTS_TABLE}
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
            conn.close()
            return jsonify({"result": summary})
        else:
            conn.close()
            return jsonify({"error": "Insufficient data for analysis"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analysis/status/<job_id>")
def analysis_status(job_id):
    """Check status of a running analysis job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/analysis/jobs")
def analysis_jobs():
    """List all analysis jobs."""
    with _jobs_lock:
        return jsonify(list(_jobs.values()))


# ═════════════════════════════════════════════════════════════════
# HEALTH
# ═════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    """Health check."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    _init_read_conn()
    print("=" * 60)
    print("  Research Dashboard API")
    print(f"  Markets DB: {os.path.abspath(DB_MKT)}")
    print(f"  Poly DB:    {os.path.abspath(DB_POLY)}")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5001, debug=False)
