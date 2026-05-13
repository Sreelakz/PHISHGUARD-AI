"""
performance_monitor.py — PhishGuard AI Performance Tracking
============================================================

PURPOSE:
    Measure, store, and visualise latency for every stage of
    the 7-step analysis pipeline. Answers:
      • "How long does ML inference take on average?"
      • "Which API is slowing us down?"
      • "Is the system degrading over time?"

ARCHITECTURE:
    track_performance(stage, ms)  ← call after each pipeline stage
    PerformanceMonitor.record()   ← object-oriented interface
    get_stats(stage)              ← retrieve averages + percentiles
    render_dashboard()            ← Streamlit widget

STORAGE:
    SQLite: performance/perf.db   (persistent, queryable)
    In-memory deque (1000 entries per stage, for fast percentile calc)

PIPELINE INTEGRATION:
    In analyzer.py, after each stage just add:
        from performance_monitor import track_performance
        track_performance("ml_prediction", timings["ml_prediction_ms"])
"""

from __future__ import annotations

import os
import sys
import time
import sqlite3
import statistics
import threading
import logging
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# ── Storage config ────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
PERF_DIR = os.path.join(_HERE, "performance")
PERF_DB  = os.path.join(PERF_DIR, "perf.db")
os.makedirs(PERF_DIR, exist_ok=True)

# ── Pipeline stage names (canonical) ─────────────────────────────────────
STAGES = [
    "feature_extraction",
    "auxiliary_intel",
    "rules_engine",
    "ml_prediction",
    "safe_browsing",
    "virustotal",
    "risk_xai",
    "total",
]

# ── In-memory ring buffer (fast stats, no DB needed for percentiles) ──────
_MEMORY_WINDOW = 1000   # last N measurements per stage
_memory: Dict[str, deque] = defaultdict(lambda: deque(maxlen=_MEMORY_WINDOW))
_lock = threading.RLock()


# ═══════════════════════════════════════════════════════════════════════════
#  DATABASE SETUP
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def _db():
    conn = sqlite3.connect(PERF_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


def _init_db():
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS perf_logs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                stage      TEXT    NOT NULL,
                elapsed_ms REAL    NOT NULL,
                timestamp  TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_stage ON perf_logs(stage)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_ts    ON perf_logs(timestamp DESC)")
        conn.commit()


_init_db()


# ═══════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION — called from analyzer.py (or anywhere)
# ═══════════════════════════════════════════════════════════════════════════

def track_performance(stage_name: str, time_taken_ms: float) -> None:
    """
    Record a single stage timing measurement.

    Call this immediately after each pipeline stage completes.

    Args:
        stage_name:    Name of the pipeline stage (use STAGES constants)
        time_taken_ms: Elapsed time in milliseconds (int or float)

    Usage:
        from performance_monitor import track_performance
        t0 = time.perf_counter()
        result = run_ml_model(features)
        track_performance("ml_prediction", (time.perf_counter() - t0) * 1000)
    """
    if time_taken_ms < 0:
        return

    ms = float(time_taken_ms)

    # ── In-memory update (fast) ───────────────────────────────────────
    with _lock:
        _memory[stage_name].append(ms)

    # ── DB persist (async-ish: fire and don't block) ──────────────────
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO perf_logs (stage, elapsed_ms) VALUES (?,?)",
                (stage_name, ms)
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Perf DB write failed: {e}")


def track_from_timings_dict(timings: Dict[str, int]) -> None:
    """
    Convenience: bulk-load a timings dict (as returned by analyzer.py).

    Usage:
        # analyzer.py already builds timings_ms dict — just pass it in:
        from performance_monitor import track_from_timings_dict
        track_from_timings_dict(result["timings_ms"])
    """
    stage_map = {
        "feature_extraction_ms": "feature_extraction",
        "auxiliary_intel_ms":    "auxiliary_intel",
        "rules_engine_ms":       "rules_engine",
        "ml_prediction_ms":      "ml_prediction",
        "threat_intel_ms":       "safe_browsing",  # combined GSB+VT
        "risk_xai_ms":           "risk_xai",
        "total_ms":              "total",
    }
    for key, stage in stage_map.items():
        if key in timings and timings[key] is not None:
            track_performance(stage, timings[key])


# ═══════════════════════════════════════════════════════════════════════════
#  OBJECT-ORIENTED INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

class PerformanceMonitor:
    """
    Context-manager + decorator interface for performance tracking.

    Usage (context manager):
        with PerformanceMonitor("ml_prediction") as pm:
            result = ml_model.predict(features)
        # Timing recorded automatically

    Usage (decorator):
        @PerformanceMonitor.wrap("feature_extraction")
        def extract_features(url):
            ...
    """

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self._t0: Optional[float] = None
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000
        track_performance(self.stage_name, self.elapsed_ms)

    @staticmethod
    def wrap(stage_name: str):
        """Decorator to auto-track a function's execution time."""
        def decorator(fn: Callable):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                with PerformanceMonitor(stage_name):
                    return fn(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def time_it(fn: Callable, stage_name: str, *args, **kwargs):
        """One-line timed function call."""
        with PerformanceMonitor(stage_name):
            return fn(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
#  STATS RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

def get_stats(
    stage: Optional[str] = None,
    lookback_hours: int   = 24,
) -> Dict:
    """
    Get performance statistics for one or all stages.

    Args:
        stage:          Specific stage name, or None for all stages
        lookback_hours: Only consider measurements from last N hours

    Returns:
        {
            "ml_prediction": {
                "count":  150,
                "avg_ms": 42.3,
                "min_ms": 12.1,
                "max_ms": 310.5,
                "p50_ms": 38.0,
                "p95_ms": 120.0,
                "p99_ms": 280.0,
                "slow_count": 5,   # > 200ms
                "trend": "stable"  # "improving" | "degrading" | "stable"
            },
            ...
        }
    """
    cutoff = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()

    # Pull from DB for the time window
    with _db() as conn:
        if stage:
            rows = conn.execute(
                "SELECT stage, elapsed_ms FROM perf_logs "
                "WHERE stage=? AND timestamp>=? ORDER BY timestamp DESC LIMIT 10000",
                (stage, cutoff)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT stage, elapsed_ms FROM perf_logs "
                "WHERE timestamp>=? ORDER BY timestamp DESC LIMIT 50000",
                (cutoff,)
            ).fetchall()

    # Group by stage
    by_stage: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        by_stage[row["stage"]].append(row["elapsed_ms"])

    result = {}
    for s, values in by_stage.items():
        if not values:
            continue
        sorted_v = sorted(values)
        n        = len(sorted_v)
        avg      = statistics.mean(values)
        slow     = sum(1 for v in values if v > 200)

        # Simple trend: compare first half avg vs second half avg
        if n >= 10:
            first_half  = statistics.mean(sorted_v[:n//2])
            second_half = statistics.mean(sorted_v[n//2:])
            ratio       = second_half / first_half if first_half > 0 else 1.0
            trend = "degrading" if ratio > 1.2 else ("improving" if ratio < 0.8 else "stable")
        else:
            trend = "insufficient_data"

        result[s] = {
            "count":     n,
            "avg_ms":    round(avg, 1),
            "min_ms":    round(sorted_v[0], 1),
            "max_ms":    round(sorted_v[-1], 1),
            "p50_ms":    round(sorted_v[int(n * 0.50)], 1),
            "p95_ms":    round(sorted_v[int(n * 0.95)], 1),
            "p99_ms":    round(sorted_v[min(int(n * 0.99), n-1)], 1),
            "slow_count": slow,
            "slow_rate_pct": round(slow / n * 100, 1),
            "trend":     trend,
        }

    return result


def get_recent_totals(n: int = 20) -> List[Dict]:
    """Get the last N total scan timings (for Streamlit line chart)."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT elapsed_ms, timestamp FROM perf_logs "
            "WHERE stage='total' ORDER BY timestamp DESC LIMIT ?",
            (n,)
        ).fetchall()
    return [{"elapsed_ms": r["elapsed_ms"],
             "timestamp":  r["timestamp"]} for r in reversed(rows)]


def clear_old_logs(older_than_days: int = 30) -> int:
    """Prune old performance logs to keep DB size manageable."""
    cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
    with _db() as conn:
        cur = conn.execute("DELETE FROM perf_logs WHERE timestamp<?", (cutoff,))
        conn.commit()
        return cur.rowcount


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT DASHBOARD WIDGET
# ═══════════════════════════════════════════════════════════════════════════

def render_dashboard(lookback_hours: int = 24) -> None:
    """
    Render a complete performance dashboard in Streamlit.

    Usage in admin page:
        from performance_monitor import render_dashboard
        render_dashboard(lookback_hours=24)
    """
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        print("Streamlit/Pandas not available — use get_stats() directly.")
        return

    st.subheader("⚡ Pipeline Performance Monitor")
    st.caption(f"Data from last {lookback_hours}h")

    stats = get_stats(lookback_hours=lookback_hours)

    if not stats:
        st.info("No performance data yet. Run some scans first.")
        return

    # ── Top-level metric cards ────────────────────────────────────────
    total = stats.get("total", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Total Time",  f"{total.get('avg_ms', 0):.0f} ms")
    col2.metric("P95 Latency",     f"{total.get('p95_ms', 0):.0f} ms")
    col3.metric("Scans Tracked",   total.get("count", 0))
    col4.metric("Slow Scans (>200ms)", total.get("slow_count", 0))

    st.divider()

    # ── Per-stage breakdown table ─────────────────────────────────────
    stage_order = [s for s in STAGES if s in stats]
    rows = []
    for s in stage_order:
        d = stats[s]
        trend_icon = {"degrading":"🔴","improving":"🟢","stable":"🟡"}.get(d["trend"],"⚪")
        rows.append({
            "Stage":        s.replace("_"," ").title(),
            "Count":        d["count"],
            "Avg (ms)":     d["avg_ms"],
            "P50 (ms)":     d["p50_ms"],
            "P95 (ms)":     d["p95_ms"],
            "Max (ms)":     d["max_ms"],
            "Slow >200ms":  d["slow_count"],
            "Trend":        f"{trend_icon} {d['trend']}",
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Recent total scan latency chart ───────────────────────────────
    st.subheader("📈 Recent Scan Latency (Total)")
    recents = get_recent_totals(50)
    if len(recents) >= 2:
        chart_data = pd.DataFrame(recents).set_index("timestamp")[["elapsed_ms"]]
        chart_data.index = pd.to_datetime(chart_data.index)
        st.line_chart(chart_data, y="elapsed_ms", height=200)
    else:
        st.caption("Not enough data for chart yet.")

    # ── Stage average bar chart ───────────────────────────────────────
    if rows:
        st.subheader("📊 Average Latency by Stage")
        bar_df = pd.DataFrame([
            {"Stage": r["Stage"], "Avg ms": r["Avg (ms)"]}
            for r in rows if r["Stage"].lower() != "total"
        ])
        st.bar_chart(bar_df.set_index("Stage"), height=220)

    # ── Slow scan alert ───────────────────────────────────────────────
    slow_stages = [s for s, d in stats.items() if d.get("slow_rate_pct", 0) > 20]
    if slow_stages:
        st.warning(f"⚠️ High slow-scan rate in: {', '.join(slow_stages)}. "
                   "Consider optimising these stages.")

    # ── Purge old logs ────────────────────────────────────────────────
    if st.button("🗑️ Purge Logs Older Than 30 Days"):
        removed = clear_old_logs(30)
        st.success(f"Removed {removed} old log entries.")


# ═══════════════════════════════════════════════════════════════════════════
#  INLINE INTEGRATION PATCH for analyzer.py
# ═══════════════════════════════════════════════════════════════════════════

def patch_analyzer_result(result: Dict) -> Dict:
    """
    Convenience: call this on any analyze_url() result to auto-record
    all timings without modifying analyzer.py.

    Usage (in app.py, after calling analyze_url):
        from performance_monitor import patch_analyzer_result
        result = analyze_url(url, ...)
        patch_analyzer_result(result)   # <-- add this line
    """
    if "timings_ms" in result:
        track_from_timings_dict(result["timings_ms"])
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  SELF TEST
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import random
    import pprint

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("🧪 PERFORMANCE MONITOR — SELF TEST")
    print("=" * 60)

    # Simulate 30 pipeline runs
    print("\n📊 Simulating 30 pipeline runs...")
    for i in range(30):
        track_performance("feature_extraction", random.uniform(20, 80))
        track_performance("auxiliary_intel",    random.uniform(50, 300))
        track_performance("rules_engine",       random.uniform(1, 10))
        track_performance("ml_prediction",      random.uniform(15, 60))
        track_performance("safe_browsing",      random.uniform(400, 1200))
        track_performance("virustotal",         random.uniform(1000, 4000) if random.random() > 0.5 else 0)
        track_performance("risk_xai",           random.uniform(5, 30))
        total = random.uniform(1500, 5500)
        track_performance("total",              total)

    # Context manager test
    print("\n⏱️  Context manager test:")
    with PerformanceMonitor("ml_prediction") as pm:
        time.sleep(0.025)
    print(f"   Recorded: {pm.elapsed_ms:.1f}ms")

    # Decorator test
    @PerformanceMonitor.wrap("feature_extraction")
    def fake_extraction():
        time.sleep(0.015)
        return {"url_length": 45}

    fake_extraction()
    print(f"   Decorator: OK")

    # Stats
    print("\n📈 Performance stats (all stages):")
    stats = get_stats()
    for stage, d in stats.items():
        print(f"\n   [{stage}]")
        print(f"     count={d['count']}  avg={d['avg_ms']}ms  "
              f"p50={d['p50_ms']}ms  p95={d['p95_ms']}ms  trend={d['trend']}")

    print("\n" + "=" * 60)
    print("✅ Module 4 test complete")
    print("=" * 60)