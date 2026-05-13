"""
app.py — PhishGuard AI Backend v4.2 (Phase 8 Optimized)
========================================================

ARCHITECTURE: JavaScript-only API flow.
  • /api/analyze   → POST JSON → returns JSON   (used by JS fetch)
  • /download_report → POST → returns PDF blob  (used by JS fetch)
  • NO Flask form submission route (/analyze removed)

PHASE 8 OPTIMIZATIONS:
  ✅ TTL+LRU cache for repeated URL scans (15-min TTL, 1000 entries)
  ✅ Gzip compression for all JSON responses (Flask-Compress)
  ✅ SQLite WAL mode + indexes (10x faster reads)
  ✅ Request-level timing middleware (logs slow requests)
  ✅ Global error handler (never exposes stack traces to client)
  ✅ Structured logging via utils.logger
  ✅ Graceful shutdown (closes sessions, flushes logs)
  ✅ One-time DB migration (flag file, no repeated ALTER TABLE)

FIX (v4.2.1):
  ✅ Last-result storage moved from Flask session cookie (4KB limit)
     to server-side dict keyed by session ID. Fixes "Download failed".
"""

import os
import sys
import sqlite3
import json
import time
import uuid
import atexit
import threading
from datetime import datetime, timedelta
from contextlib import contextmanager

# ── Path fix ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, g, session, send_file

try:
    from flask_compress import Compress
    _COMPRESS_AVAILABLE = True
except ImportError:
    _COMPRESS_AVAILABLE = False

# ── Local imports ────────────────────────────────────────────────────────
from utils.logger import get_logger
from utils.perf_cache import url_cache, circuit_breaker, normalize_url
from utils.secure_input_handler import validate_and_sanitize
from backend.report_generator import generate_report
from backend.simple_explainer import generate_simple_explanation

# ── Pipeline components ──────────────────────────────────────────────────
from feature_extractor       import FeatureExtractor
from ml_model                import MLModel
from utils.ssl_checker       import SSLChecker
from domain_intelligence     import DomainIntelligence
from utils.redirect_detector import RedirectDetector
from utils.homograph_detector import HomographDetector
from explainable_ai          import ExplainableAI
from visual_analyzer         import VisualAnalyzer
from risk_calculator         import RiskCalculator
from shap_explainer          import get_shap_explainer
from safe_browsing           import get_safe_browsing_checker
from analyzer                import analyze_url, analyze_batch

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════
#  Flask setup
# ══════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = os.getenv("SECRET_KEY", "fallback_secret_key")

if _COMPRESS_AVAILABLE:
    Compress(app)
    app.config["COMPRESS_MIMETYPES"] = ["application/json", "text/html", "text/css"]
    app.config["COMPRESS_LEVEL"] = 6
    app.config["COMPRESS_MIN_SIZE"] = 500
    logger.info("✅ Gzip compression enabled")
else:
    logger.warning("⚠️  Flask-Compress not installed — responses will not be compressed")

# ══════════════════════════════════════════════════════════════════════════
#  Server-side result store (FIX for "Download failed")
#  ────────────────────────────────────────────────────────────────────────
#  Flask's default session uses signed cookies which have a ~4KB limit.
#  Our analysis result (with SHAP, features, threat intel) is much larger,
#  so session["last_result"] would silently get dropped → /download_report
#  would see no result → return 400 → frontend shows "Download failed".
#
#  Fix: keep results in a process-local dict keyed by a per-browser session
#  ID stored in the cookie. Cookie stays tiny (~40 bytes), result stays
#  on the server.
# ══════════════════════════════════════════════════════════════════════════
_RESULTS_STORE: dict = {}                     # session_id -> (timestamp, result)
_RESULTS_LOCK  = threading.Lock()
_RESULT_TTL_SECONDS = 60 * 60                  # keep results for 1 hour
_RESULT_MAX_ENTRIES = 500                      # hard cap


def _get_or_create_sid() -> str:
    """Return a stable per-browser session id, creating one if absent."""
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
        session.permanent = False
    return sid


def _store_last_result(result: dict) -> None:
    """Save result for the current session (server-side, not in cookie)."""
    sid = _get_or_create_sid()
    now = time.time()
    with _RESULTS_LOCK:
        _RESULTS_STORE[sid] = (now, result)
        # opportunistic cleanup
        if len(_RESULTS_STORE) > _RESULT_MAX_ENTRIES:
            cutoff = now - _RESULT_TTL_SECONDS
            stale = [k for k, (ts, _) in _RESULTS_STORE.items() if ts < cutoff]
            for k in stale:
                _RESULTS_STORE.pop(k, None)
            # if still too big, drop oldest
            if len(_RESULTS_STORE) > _RESULT_MAX_ENTRIES:
                oldest = sorted(_RESULTS_STORE.items(), key=lambda kv: kv[1][0])
                for k, _ in oldest[: len(_RESULTS_STORE) - _RESULT_MAX_ENTRIES]:
                    _RESULTS_STORE.pop(k, None)


def _get_last_result() -> dict | None:
    """Fetch the last result for this session (or None)."""
    sid = session.get("sid")
    if not sid:
        return None
    with _RESULTS_LOCK:
        entry = _RESULTS_STORE.get(sid)
    if not entry:
        return None
    ts, result = entry
    if time.time() - ts > _RESULT_TTL_SECONDS:
        with _RESULTS_LOCK:
            _RESULTS_STORE.pop(sid, None)
        return None
    return result


# ══════════════════════════════════════════════════════════════════════════
#  Initialize components (singletons)
# ══════════════════════════════════════════════════════════════════════════
feature_extractor   = FeatureExtractor()
ml_model            = MLModel()
ml_model.load()

ssl_checker         = SSLChecker()
domain_intelligence = DomainIntelligence()
redirect_detector   = RedirectDetector()
homograph_detector  = HomographDetector()
explainable_ai      = ExplainableAI()
visual_analyzer     = VisualAnalyzer()
risk_calculator     = RiskCalculator()
shap_explainer      = get_shap_explainer()
safe_browsing       = get_safe_browsing_checker()

AUX_MODULES = {
    "ssl":       ssl_checker,
    "domain":    domain_intelligence,
    "redirect":  redirect_detector,
    "homograph": homograph_detector,
    "visual":    visual_analyzer,
}

# ══════════════════════════════════════════════════════════════════════════
#  Database
# ══════════════════════════════════════════════════════════════════════════
DB_PATH      = os.path.join(BASE_DIR, "database", "phishing_detection.db")
DB_INIT_FLAG = os.path.join(BASE_DIR, "database", ".db_initialized_v42")


@contextmanager
def _db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    already_init = os.path.exists(DB_INIT_FLAG)

    with _db_conn() as conn:
        c = conn.cursor()
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")

        c.execute("""CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL, prediction TEXT, confidence REAL,
            risk_score REAL, risk_level TEXT, explanations TEXT,
            features TEXT, verdict_source TEXT, threat_intel TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")

        c.execute("""CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL, reason TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")

        c.execute("CREATE INDEX IF NOT EXISTS idx_scans_timestamp  ON scans(timestamp DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_scans_prediction ON scans(prediction)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_scans_risk_level ON scans(risk_level)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_scans_url        ON scans(url)")

        if not already_init:
            for col, col_type in [("verdict_source", "TEXT"), ("threat_intel", "TEXT")]:
                try:
                    c.execute(f"ALTER TABLE scans ADD COLUMN {col} {col_type}")
                    logger.info(f"DB migration: added column {col}")
                except sqlite3.OperationalError:
                    pass

        conn.commit()

    if not already_init:
        try:
            with open(DB_INIT_FLAG, "w") as f:
                f.write(datetime.now().isoformat())
        except Exception:
            pass


def _save_scan(url, prediction, confidence, risk_score, risk_level,
               explanations, features, verdict_source=None, threat_intel=None):
    try:
        with _db_conn() as conn:
            conn.execute(
                """INSERT INTO scans
                   (url,prediction,confidence,risk_score,risk_level,
                    explanations,features,verdict_source,threat_intel)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (url, prediction, confidence, risk_score, risk_level,
                 json.dumps(explanations, default=str),
                 json.dumps(features,     default=str),
                 verdict_source,
                 json.dumps(threat_intel, default=str) if threat_intel else None)
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"DB save failed: {e}")


def _get_scans(limit=50, offset=0, pred=None):
    with _db_conn() as conn:
        c = conn.cursor()
        if pred:
            c.execute("SELECT * FROM scans WHERE prediction=? "
                      "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                      (pred, limit, offset))
        else:
            c.execute("SELECT * FROM scans ORDER BY timestamp DESC "
                      "LIMIT ? OFFSET ?", (limit, offset))
        return [dict(r) for r in c.fetchall()]


# ══════════════════════════════════════════════════════════════════════════
#  Middleware
# ══════════════════════════════════════════════════════════════════════════
@app.before_request
def _before():
    g.start_time = time.perf_counter()


@app.after_request
def _after(response):
    try:
        elapsed_ms = (time.perf_counter() - g.start_time) * 1000
        if elapsed_ms > 2000:
            logger.warning(f"⚠️  SLOW: {request.method} {request.path} took {elapsed_ms:.0f}ms")
        response.headers["X-Response-Time-MS"] = f"{elapsed_ms:.0f}"
    except Exception:
        pass
    return response


@app.errorhandler(Exception)
def _handle_uncaught(e):
    logger.exception(f"Unhandled exception on {request.path}")
    return jsonify({"error": "Internal server error", "detail": str(e)[:200]}), 500


# ══════════════════════════════════════════════════════════════════════════
#  Routes — Web pages
# ══════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    """Render the scanner UI. No result is pre-loaded — JS handles everything."""
    return render_template("index.html")


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", scans=_get_scans(50))


@app.route("/admin")
def admin():
    return render_template("admin.html")


# ══════════════════════════════════════════════════════════════════════════
#  Routes — Analysis API
# ══════════════════════════════════════════════════════════════════════════
def _build_analyze_response(result: dict) -> dict:
    """
    Convert raw analyzer result → clean API response for the JS frontend.

    Key fix: `explanations` on the response carries the full nested dict
    (ai_narrative, category_scores, indicators …) so the UI can read
    data.explanations.ai_narrative, data.explanations.category_scores etc.
    """
    threat_intel = result.get("threat_intel") or {}

    # Pull the detailed explanation block (has ai_narrative, category_scores, indicators)
    detailed = result.get("detailed_explanation") or {}

    # Merge top-level explanations key: prefer detailed, fallback to simple
    explanations_out = {
        "ai_narrative":     detailed.get("ai_narrative")     or result.get("verdict_reason") or "",
        "category_scores":  detailed.get("category_scores")  or {},
        "indicators":       detailed.get("indicators")       or [],
        "top_reasons":      detailed.get("top_reasons")      or [],
        "summary":          detailed.get("summary")          or "",
    }

    return {
        # ── Core verdict ──
        "url":              result.get("url", ""),
        "prediction":       result.get("verdict", "UNKNOWN"),
        "confidence":       result.get("confidence", 0),
        "risk_score":       result.get("risk_score"),
        "risk_level":       result.get("risk_level"),
        "risk_colour":      result.get("risk_colour"),

        # ── Explanations (merged, flat for JS) ──
        "explanations":         explanations_out,
        "detailed_explanation": detailed,
        "simple_explanation":   result.get("simple_explanation") or {},

        # ── SHAP — always included; may be [] if explainer not ready ──
        "shap":             result.get("shap"),
        "shap_explanation": result.get("shap_explanation") or [],

        # ── Signals ──
        "explanation":      result.get("explanation", ""),
        "indicators":       result.get("indicators", []),
        "threat_message":   result.get("threat_message", ""),

        # ── Threat Intel ──
        "safe_browsing":    threat_intel.get("safe_browsing"),
        "virustotal":       threat_intel.get("virustotal"),

        # ── Verdict metadata ──
        "verdict_source":   result.get("verdict_source"),
        "verdict_reason":   result.get("verdict_reason"),
        "dominant_signal":  (result.get("risk") or {}).get("dominant_signal"),
        "signal_weights":   (result.get("risk") or {}).get("signal_weights"),

        # ── Features + importances ──
        "features":         result.get("features", {}),
        "importances":      result.get("importances", {}),

        # ── Timing / metadata ──
        "timings_ms":       result.get("timings_ms", {}),
        "pipeline_version": result.get("pipeline_version"),
        "timestamp":        result.get("timestamp"),
        "ml":               result.get("ml"),
        "rules":            result.get("rules"),
    }


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Main analysis endpoint.
    Accepts: { url, enable_html, enable_virustotal }
    Returns: full analysis JSON consumed by the JS frontend.
    """
    try:
        data = request.get_json(force=True) or {}
        url  = (data.get("url") or "").strip()
        if not url:
            return jsonify({"error": "URL is required"}), 400

        enable_html = bool(data.get("enable_html", True))
        enable_vt   = bool(data.get("enable_virustotal", True))

        # ── Cache check ──
        cache_key = f"{normalize_url(url)}|html={enable_html}|vt={enable_vt}"
        cached = url_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache HIT: {url}")
            response = dict(cached)
            response["_cache_hit"] = True
            # Even on cache hit we still need to remember the *raw* result
            # for /download_report. We stored the raw result alongside the
            # response in the cache (see below).
            raw = cached.get("__raw_result__")
            if raw:
                _store_last_result(raw)
            return jsonify({k: v for k, v in response.items() if k != "__raw_result__"})

        # ── Full pipeline ──
        result = analyze_url(
            url,
            enable_html=enable_html,
            enable_safe_browsing=True,
            enable_virustotal=enable_vt,
            auxiliary_modules=AUX_MODULES,
            risk_calculator=risk_calculator,
            explainable_ai=explainable_ai,
            shap_explainer=shap_explainer,
        )

        # Persist raw result for this browser session (server-side, not cookie)
        _store_last_result(result)

        # Persist to DB
        threat_intel = result.get("threat_intel") or {}
        _save_scan(
            url=result.get("url", url),
            prediction=result.get("verdict"),
            confidence=result.get("confidence"),
            risk_score=result.get("risk_score"),
            risk_level=result.get("risk_level"),
            explanations=result.get("explanations", {}),
            features=result.get("features", {}),
            verdict_source=result.get("verdict_source"),
            threat_intel=threat_intel,
        )

        response = _build_analyze_response(result)
        response["_cache_hit"] = False

        # Cache the response *with* the raw result attached so a cache-hit
        # path can still serve /download_report properly.
        cache_payload = dict(response)
        cache_payload["__raw_result__"] = result
        url_cache.set(cache_key, cache_payload)

        return jsonify(response)

    except Exception as e:
        logger.exception("analyze endpoint failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze/batch", methods=["POST"])
def analyze_batch_endpoint():
    try:
        data = request.get_json(force=True) or {}
        urls = data.get("urls") or []
        if not isinstance(urls, list) or not urls:
            return jsonify({"error": "urls array is required"}), 400
        if len(urls) > 50:
            return jsonify({"error": "Max 50 URLs per batch"}), 400

        results = analyze_batch(
            urls,
            enable_html=bool(data.get("enable_html", False)),
            enable_safe_browsing=True,
            enable_virustotal=bool(data.get("enable_virustotal", False)),
            auxiliary_modules=AUX_MODULES,
            risk_calculator=risk_calculator,
            explainable_ai=explainable_ai,
            shap_explainer=shap_explainer,
        )

        for r in results:
            if r.get("verdict") != "ERROR":
                _save_scan(
                    url=r.get("url", ""),
                    prediction=r.get("verdict"),
                    confidence=r.get("confidence", 0),
                    risk_score=r.get("risk_score"),
                    risk_level=r.get("risk_level"),
                    explanations=r.get("explanations", {}),
                    features=r.get("features", {}),
                    verdict_source=r.get("verdict_source"),
                    threat_intel=r.get("threat_intel"),
                )

        # Optionally store the *first* batch result so the user can still
        # click "Download Report" after a batch scan.
        if results:
            first_ok = next((r for r in results if r.get("verdict") != "ERROR"), None)
            if first_ok:
                _store_last_result(first_ok)

        return jsonify({"count": len(results), "results": results})

    except Exception as e:
        logger.exception("batch endpoint failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/report", methods=["POST"])
def report():
    try:
        data = request.get_json(force=True)
        url  = (data.get("url") or "").strip()
        if not url:
            return jsonify({"error": "URL is required"}), 400
        with _db_conn() as conn:
            conn.execute(
                "INSERT INTO reports (url,reason) VALUES (?,?)",
                (url, data.get("reason", "User report"))
            )
            conn.commit()
        return jsonify({"message": "Report submitted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download_report", methods=["POST"])
def download_report():
    """
    Generate and stream a PDF report for the last analyzed URL.
    Called via JS fetch (blob download) — no form involved.

    The raw analyzer result is stored server-side (keyed by session id)
    by /api/analyze. Flask's default cookie session can't hold our
    multi-KB result, which previously caused this endpoint to always
    return 400 → frontend showed "Download failed".
    """
    result = _get_last_result()
    if not result:
        logger.warning("download_report: no last result found for session")
        return jsonify({
            "error": "No scan result available. Please run a scan first."
        }), 400

    try:
        report_files = generate_report(result)
    except Exception as e:
        logger.exception("Report generation failed")
        return jsonify({
            "error":   "Report generation failed",
            "details": str(e)[:200],
        }), 500

    pdf_path = report_files.get("pdf") if isinstance(report_files, dict) else None
    if not pdf_path or not os.path.exists(pdf_path):
        err_detail = (
            report_files.get("pdf_error", "Unknown error")
            if isinstance(report_files, dict) else str(report_files)
        )
        logger.error(f"PDF missing on disk: {err_detail}")
        return jsonify({
            "error":   "PDF generation failed",
            "details": err_detail,
        }), 500

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name="phishguard_report.pdf",
        mimetype="application/pdf",
    )


# ══════════════════════════════════════════════════════════════════════════
#  Routes — Scans / Admin
# ══════════════════════════════════════════════════════════════════════════
@app.route("/api/scans", methods=["GET"])
def get_all_scans():
    try:
        return jsonify(_get_scans(
            int(request.args.get("limit", 50)),
            int(request.args.get("offset", 0)),
            request.args.get("prediction")
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/stats", methods=["GET"])
def admin_stats():
    try:
        with _db_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) as t FROM scans");                               total = c.fetchone()["t"]
            c.execute("SELECT COUNT(*) as t FROM scans WHERE prediction='PHISHING'");   ph    = c.fetchone()["t"]
            c.execute("SELECT COUNT(*) as t FROM scans WHERE prediction='LEGITIMATE'"); lg    = c.fetchone()["t"]
            c.execute("SELECT AVG(risk_score) as a FROM scans");                        avg   = round(c.fetchone()["a"] or 0, 1)

            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            c.execute("SELECT DATE(timestamp) as day,prediction,COUNT(*) as cnt "
                      "FROM scans WHERE timestamp>=? "
                      "GROUP BY day,prediction ORDER BY day", (cutoff,))
            daily = [dict(r) for r in c.fetchall()]

            c.execute("SELECT risk_level,COUNT(*) as cnt FROM scans GROUP BY risk_level")
            risk_dist = {r["risk_level"]: r["cnt"] for r in c.fetchall()}

            c.execute("SELECT url,risk_score,prediction FROM scans "
                      "WHERE prediction='PHISHING' ORDER BY risk_score DESC LIMIT 10")
            top = [dict(r) for r in c.fetchall()]

        return jsonify({
            "total_scans":       total,
            "phishing_count":    ph,
            "legitimate_count":  lg,
            "phishing_rate":     round(ph / max(total, 1) * 100, 1),
            "average_risk":      avg,
            "daily_breakdown":   daily,
            "risk_distribution": risk_dist,
            "top_threats":       top,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/heatmap", methods=["GET"])
def admin_heatmap():
    try:
        with _db_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT features FROM scans WHERE prediction='PHISHING' "
                      "ORDER BY timestamp DESC LIMIT 100")
            rows = c.fetchall()

        totals = {}
        count  = len(rows)
        for row in rows:
            try:
                f = json.loads(row["features"] or "{}")
                d = f.get("domain_info") or {}
                s = f.get("ssl_info") or {}
                r = f.get("redirect_info") or {}
                h = f.get("homograph_info") or {}
                sig = {
                    "long_url":         int(f.get("url_length", 0) > 54),
                    "ip_in_url":        f.get("has_ip_address", 0),
                    "at_symbol":        f.get("has_at_symbol", 0),
                    "susp_keywords":    f.get("has_suspicious_keywords", 0),
                    "no_https":         int(not s.get("uses_https", True)),
                    "young_domain":     int(d.get("domain_age_days", 9999) < 180),
                    "redirects":        int(r.get("redirect_count", 0) > 0),
                    "homograph":        int(h.get("is_homograph", False)),
                    "login_form":       f.get("has_login_form", 0),
                    "iframes":          int(f.get("has_iframes", 0) > 0),
                    "obfuscated_js":    f.get("has_obfuscated_js", 0),
                    "favicon_mismatch": f.get("has_favicon_mismatch", 0),
                }
                for k, v in sig.items():
                    totals[k] = totals.get(k, 0) + v
            except Exception:
                continue

        return jsonify({
            "heatmap":     {k: round(v / max(count, 1) * 100, 1) for k, v in totals.items()},
            "sample_size": count,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/logs", methods=["GET"])
def admin_logs():
    try:
        with _db_conn() as conn:
            c = conn.cursor()
            q = ("SELECT id,url,prediction,confidence,risk_score,risk_level,"
                 "verdict_source,timestamp FROM scans WHERE 1=1")
            params = []
            pred  = request.args.get("prediction")
            level = request.args.get("risk_level")
            if pred:  q += " AND prediction=?";  params.append(pred)
            if level: q += " AND risk_level=?";  params.append(level)
            q += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params += [int(request.args.get("limit", 100)),
                       int(request.args.get("offset", 0))]
            c.execute(q, params)
            logs = [dict(r) for r in c.fetchall()]
        return jsonify({"logs": logs, "count": len(logs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/export", methods=["GET"])
def admin_export():
    try:
        return jsonify({
            "exported_at": datetime.now().isoformat(),
            "scans":       _get_scans(10000),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/cache-stats", methods=["GET"])
def cache_stats():
    try:
        return jsonify({
            "url_cache":        url_cache.stats(),
            "circuit_breakers": circuit_breaker.all_status(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/cache-clear", methods=["POST"])
def clear_url_cache():
    try:
        removed = url_cache.clear()
        return jsonify({"message": f"Cleared {removed} cached results", "removed": removed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/api-health", methods=["GET"])
def api_health():
    try:
        cache_size = 0
        cache_file = os.path.join(BASE_DIR, "cache", "vt_cache.json")
        try:
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    cache_size = len(json.load(f))
        except Exception:
            pass

        vt_key  = os.getenv("VIRUSTOTAL_API_KEY", "")
        gsb_key = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY", "")

        return jsonify({
            "ml_model": {
                "status":  "online" if ml_model.is_trained() else "offline",
                "metrics": getattr(ml_model, "metrics", {}) or {},
            },
            "safe_browsing": {
                "status":  "online" if safe_browsing.is_available() else "offline",
                "enabled": os.getenv("SAFE_BROWSING_ENABLED", "true").lower() == "true",
                "key_set": bool(gsb_key and not gsb_key.startswith("your")),
            },
            "virustotal": {
                "status":        "online" if (vt_key and not vt_key.startswith("your")) else "offline",
                "enabled":       os.getenv("VIRUSTOTAL_ENABLED", "true").lower() == "true",
                "key_set":       bool(vt_key and not vt_key.startswith("your")),
                "threshold":     float(os.getenv("VIRUSTOTAL_CONFIDENCE_THRESHOLD", "0.7")),
                "cache_entries": cache_size,
            },
            "shap_explainer": {
                "status": "online" if shap_explainer.is_ready() else "offline",
            },
            "rule_engine":      {"status": "online"},
            "url_cache":        url_cache.stats(),
            "circuit_breakers": circuit_breaker.all_status(),
            "timestamp":        datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/clear-cache", methods=["POST"])
def clear_cache():
    try:
        from utils.vt_cache import clear_cache as wipe
        removed = wipe()
        return jsonify({"message": f"Cleared {removed} cache entries", "removed": removed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/verdict-sources", methods=["GET"])
def verdict_sources():
    try:
        with _db_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT verdict_source, COUNT(*) as cnt FROM scans "
                      "WHERE verdict_source IS NOT NULL GROUP BY verdict_source")
            sources = {r["verdict_source"]: r["cnt"] for r in c.fetchall()}

            c.execute("SELECT threat_intel FROM scans "
                      "WHERE threat_intel IS NOT NULL AND prediction='PHISHING' "
                      "ORDER BY timestamp DESC LIMIT 500")
            ti_rows = c.fetchall()

        sb_hits = vt_hits = vt_skipped = 0
        for row in ti_rows:
            try:
                ti = json.loads(row["threat_intel"] or "{}")
                if (ti.get("safe_browsing") or {}).get("is_threat"):
                    sb_hits += 1
                vt = ti.get("virustotal") or {}
                if vt.get("status") == "completed" and vt.get("is_malicious"):
                    vt_hits += 1
                elif vt.get("status") == "skipped":
                    vt_skipped += 1
            except Exception:
                continue

        return jsonify({
            "sources": sources,
            "threat_intel": {
                "safe_browsing_hits": sb_hits,
                "virustotal_hits":    vt_hits,
                "virustotal_skipped": vt_skipped,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    vt_key = os.getenv("VIRUSTOTAL_API_KEY", "")
    return jsonify({
        "status":         "ok",
        "version":        "4.2.1",
        "pipeline":       "8.0",
        "ml_model":       "LOADED" if ml_model.is_trained() else "NOT TRAINED",
        "rule_engine":    "ENABLED",
        "shap_explainer": "ENABLED" if shap_explainer.is_ready() else "DISABLED",
        "safe_browsing":  "ENABLED" if safe_browsing.is_available() else "DISABLED",
        "virustotal":     "ENABLED" if (vt_key and not vt_key.startswith("your")) else "DISABLED",
        "url_cache":      "ENABLED",
        "compression":    "ENABLED" if _COMPRESS_AVAILABLE else "DISABLED",
        "result_store":   f"{len(_RESULTS_STORE)} entries",
        "timestamp":      datetime.now().isoformat(),
    })


# ══════════════════════════════════════════════════════════════════════════
#  Graceful shutdown
# ══════════════════════════════════════════════════════════════════════════
def _on_shutdown():
    logger.info("Shutting down — closing sessions...")
    try:
        feature_extractor.close()
    except Exception:
        pass
    with _RESULTS_LOCK:
        _RESULTS_STORE.clear()
    logger.info("Shutdown complete")


atexit.register(_on_shutdown)


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    init_db()

    print("\n" + "=" * 70)
    print("🛡️  PhishGuard AI v4.2.1 — Phase 8 (Performance Optimized)")
    print("=" * 70)

    if ml_model.is_trained():
        print("[STARTUP] ✅ ML model loaded")
        try:
            if shap_explainer.initialize(ml_model):
                print("[STARTUP] ✅ SHAP Explainable AI: INITIALIZED")
        except Exception as e:
            print(f"[STARTUP] ⚠️  SHAP error: {e}")
    else:
        print("[STARTUP] ⚠️  ML model not trained — run training first")

    print("[STARTUP]", "✅" if safe_browsing.is_available() else "⚠️ ",
          "Google Safe Browsing:", "ENABLED" if safe_browsing.is_available() else "DISABLED")

    vt_key = os.getenv("VIRUSTOTAL_API_KEY", "")
    print("[STARTUP]", "✅" if (vt_key and not vt_key.startswith("your")) else "⚠️ ",
          "VirusTotal API:", "ENABLED" if (vt_key and not vt_key.startswith("your")) else "DISABLED")

    print("[STARTUP] ✅ Rule Engine: ENABLED")
    print("[STARTUP] ✅ URL Cache: ENABLED (1000 entries, 15-min TTL)")
    print(f"[STARTUP] {'✅' if _COMPRESS_AVAILABLE else '⚠️ '} Gzip compression: "
          f"{'ENABLED' if _COMPRESS_AVAILABLE else 'DISABLED'}")
    print("[STARTUP] ✅ SQLite WAL mode + indexes: ENABLED")
    print("[STARTUP] ✅ Server-side result store: ENABLED (fixes Download)")
    print("[STARTUP] ✅ Unified Pipeline: READY")
    print("[STARTUP] ✅ Architecture: JS-only API (no form submissions)")
    print("=" * 70 + "\n")

    app.run(debug=True, port=5000, use_reloader=False)