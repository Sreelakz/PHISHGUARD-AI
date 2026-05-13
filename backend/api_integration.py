"""
backend/api_integration.py
---------------------------
External threat intelligence integrations — PHASE 8 OPTIMIZED.

APIs:
  • Google Safe Browsing (Phase 4) — fast, free, unlimited
  • VirusTotal (Phase 5)           — 70+ AV engines, rate-limited

PHASE 8 OPTIMIZATIONS:
  ✅ Persistent requests.Session() for each API (TCP connection reuse)
  ✅ Automatic retries on transient failures (502/503/504)
  ✅ Circuit breaker: auto-disables API after 5 consecutive failures for 5 min
  ✅ Structured logging via utils.logger
  ✅ 24h disk cache for VirusTotal (respects free-tier quota)
  ✅ Smart gating: VT only fires when ML confidence > threshold

BACKWARD COMPATIBLE: Same function signatures, same return format.
"""

import os
import sys
import time
import base64
from typing import Dict
from collections import deque
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────
#  Path setup (works from any directory)
# ──────────────────────────────────────────────────────────────────────
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load .env from project root
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# ── Local imports (Phase 8) ───────────────────────────────────────────
try:
    from utils.logger import get_logger
except ModuleNotFoundError:
    import logging
    def get_logger(name): return logging.getLogger(name)

try:
    from utils.perf_cache import circuit_breaker
except ModuleNotFoundError:
    # Graceful fallback: fake circuit breaker that never trips
    class _FakeCB:
        def is_open(self, s): return False
        def record_success(self, s): pass
        def record_failure(self, s): pass
        def status(self, s): return {"state": "DISABLED"}
    circuit_breaker = _FakeCB()

try:
    from utils.vt_cache import get_cached, set_cached
except ModuleNotFoundError:
    def get_cached(url): return None
    def set_cached(url, result): pass

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
# Google Safe Browsing
GSB_API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY", "").strip()
GSB_ENABLED = os.getenv("SAFE_BROWSING_ENABLED", "true").lower() == "true"
GSB_URL = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

# VirusTotal
VT_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "").strip()
VT_ENABLED = os.getenv("VIRUSTOTAL_ENABLED", "true").lower() == "true"
VT_THRESHOLD = float(os.getenv("VIRUSTOTAL_CONFIDENCE_THRESHOLD", "0.7"))
VT_BASE_URL = "https://www.virustotal.com/api/v3"

# Rate limiting (VirusTotal free tier = 4 req/min)
VT_MAX_REQUESTS_PER_MIN = 4
VT_WINDOW_SECONDS = 60
_vt_request_timestamps: deque = deque(maxlen=VT_MAX_REQUESTS_PER_MIN)

# Circuit breaker service names
CB_GSB = "safe_browsing"
CB_VT = "virustotal"


# ══════════════════════════════════════════════════════════════════════
#  Session factory (connection pooling + retries)
# ══════════════════════════════════════════════════════════════════════
def _build_api_session() -> requests.Session:
    """Build a session with retries + connection pooling for API calls."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,                      # 1s, 2s, 4s
        status_forcelist=[502, 503, 504],        # don't retry 429 (rate limit)
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=retry,
    )
    session.mount("http://",  adapter)
    session.mount("https://", adapter)
    return session


# Singleton sessions (reused across calls — massive perf win)
_gsb_session = _build_api_session()
_vt_session = _build_api_session()


# ══════════════════════════════════════════════════════════════════════
#  PHASE 4: GOOGLE SAFE BROWSING
# ══════════════════════════════════════════════════════════════════════
def check_safe_browsing(url: str) -> Dict:
    """
    Query Google Safe Browsing v4 for threat matches.

    Returns:
        {
            "status": "safe" | "unsafe" | "error" | "disabled" | "circuit_open",
            "is_threat": bool,
            "threat_types": [str, ...],
            "message": str,
        }
    """
    if not GSB_ENABLED:
        return {"status": "disabled", "is_threat": False,
                "threat_types": [], "message": "Safe Browsing disabled in .env"}

    if not GSB_API_KEY or GSB_API_KEY.startswith("your"):
        return {"status": "error", "is_threat": False,
                "threat_types": [], "message": "API key missing"}

    # Circuit breaker check
    if circuit_breaker.is_open(CB_GSB):
        cb_status = circuit_breaker.status(CB_GSB)
        remaining = cb_status.get("cooldown_remaining_sec", 0)
        return {
            "status": "circuit_open", "is_threat": False,
            "threat_types": [],
            "message": f"⚠️ Safe Browsing temporarily disabled (circuit breaker) — "
                       f"retrying in {remaining}s",
        }

    payload = {
        "client": {"clientId": "phishing-detector", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": [
                "MALWARE", "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }

    t0 = time.perf_counter()
    try:
        resp = _gsb_session.post(
            GSB_URL, params={"key": GSB_API_KEY},
            json=payload, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Safe Browsing took {elapsed_ms:.0f}ms")

        circuit_breaker.record_success(CB_GSB)

        matches = data.get("matches", [])
        if matches:
            threats = list({m.get("threatType", "UNKNOWN") for m in matches})
            return {
                "status": "unsafe", "is_threat": True,
                "threat_types": threats,
                "message": f"⚠️ Threat detected: {', '.join(threats)}",
            }
        return {"status": "safe", "is_threat": False,
                "threat_types": [], "message": "✅ No threats in Google's database"}

    except requests.exceptions.Timeout:
        circuit_breaker.record_failure(CB_GSB)
        logger.warning(f"Safe Browsing timeout for {url}")
        return {"status": "error", "is_threat": False,
                "threat_types": [], "message": "Timeout"}
    except requests.exceptions.RequestException as e:
        circuit_breaker.record_failure(CB_GSB)
        logger.error(f"Safe Browsing error: {e}")
        return {"status": "error", "is_threat": False,
                "threat_types": [], "message": f"Network error: {str(e)[:80]}"}


# ══════════════════════════════════════════════════════════════════════
#  PHASE 5: VIRUSTOTAL (SMART USAGE)
# ══════════════════════════════════════════════════════════════════════
def _vt_rate_limit_wait() -> float:
    """Enforce 4 req/min limit using a sliding window."""
    now = time.time()

    while _vt_request_timestamps and (now - _vt_request_timestamps[0]) > VT_WINDOW_SECONDS:
        _vt_request_timestamps.popleft()

    waited = 0.0
    if len(_vt_request_timestamps) >= VT_MAX_REQUESTS_PER_MIN:
        oldest = _vt_request_timestamps[0]
        wait_time = VT_WINDOW_SECONDS - (now - oldest) + 0.5
        if wait_time > 0:
            logger.warning(f"⏳ VT rate limit: sleeping {wait_time:.1f}s")
            time.sleep(wait_time)
            waited = wait_time

    _vt_request_timestamps.append(time.time())
    return waited


def _vt_url_id(url: str) -> str:
    """VT v3 URL identifier = base64url(no padding) of URL."""
    return base64.urlsafe_b64encode(url.encode()).decode().strip("=")


def check_virustotal(url: str, ml_confidence: float = 1.0) -> Dict:
    """
    Smart VirusTotal scan.

    Args:
        url: URL to scan
        ml_confidence: phishing probability from ML model (0.0–1.0)

    Returns:
        Dict with status, malicious_count, detection_ratio, permalink, etc.
    """
    # ── Gate 1: Disabled ───────────────────────────────────────────
    if not VT_ENABLED:
        return _vt_skip_response("VirusTotal disabled in .env", status="disabled")

    # ── Gate 2: Missing key ────────────────────────────────────────
    if not VT_API_KEY or VT_API_KEY.startswith("your"):
        return _vt_skip_response("VirusTotal API key missing", status="error")

    # ── Gate 3: Smart confidence filter (THE KEY FEATURE) ──────────
    if ml_confidence < VT_THRESHOLD:
        return _vt_skip_response(
            f"ML confidence {ml_confidence:.2f} < threshold {VT_THRESHOLD} — "
            f"skipped to save API quota",
            status="skipped",
        )

    # ── Gate 4: Circuit breaker ────────────────────────────────────
    if circuit_breaker.is_open(CB_VT):
        cb_status = circuit_breaker.status(CB_VT)
        remaining = cb_status.get("cooldown_remaining_sec", 0)
        return _vt_skip_response(
            f"VirusTotal temporarily disabled (circuit breaker) — "
            f"retrying in {remaining}s",
            status="circuit_open",
        )

    # ── Gate 5: Cache check ────────────────────────────────────────
    cached = get_cached(url)
    if cached is not None:
        cached["status"] = "cached"
        cached["message"] = "✅ Result loaded from 24h cache"
        return cached

    # ── Execute scan ───────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        _vt_rate_limit_wait()
        result = _vt_fetch_analysis(url)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"VirusTotal took {elapsed_ms:.0f}ms (status={result.get('status')})")

        # Only cache completed results + record success
        if result.get("status") == "completed":
            set_cached(url, result)
            circuit_breaker.record_success(CB_VT)
        elif result.get("status") == "error":
            circuit_breaker.record_failure(CB_VT)

        return result

    except requests.exceptions.Timeout:
        circuit_breaker.record_failure(CB_VT)
        logger.warning(f"VT timeout for {url}")
        return _vt_error_response("VirusTotal API timeout")
    except requests.exceptions.RequestException as e:
        circuit_breaker.record_failure(CB_VT)
        logger.error(f"VirusTotal error: {e}")
        return _vt_error_response(f"Network error: {str(e)[:80]}")
    except Exception as e:
        circuit_breaker.record_failure(CB_VT)
        logger.exception("Unexpected VirusTotal failure")
        return _vt_error_response(f"Unexpected error: {str(e)[:80]}")


def _vt_fetch_analysis(url: str) -> Dict:
    """Core VT v3 workflow: GET → POST if 404 → GET again."""
    headers = {"x-apikey": VT_API_KEY, "Accept": "application/json"}
    url_id = _vt_url_id(url)

    # Step 1: Check if URL already analyzed
    get_resp = _vt_session.get(
        f"{VT_BASE_URL}/urls/{url_id}",
        headers=headers, timeout=15,
    )

    if get_resp.status_code == 200:
        return _vt_parse_report(url, get_resp.json())

    # Step 2: Not found → submit for scanning
    if get_resp.status_code == 404:
        logger.info(f"VT: submitting new URL for scan")
        _vt_rate_limit_wait()
        post_resp = _vt_session.post(
            f"{VT_BASE_URL}/urls",
            headers=headers, data={"url": url}, timeout=15,
        )

        if post_resp.status_code == 429:
            return _vt_error_response("VirusTotal rate limit exceeded (daily quota?)")
        if post_resp.status_code == 401:
            return _vt_error_response("Invalid VirusTotal API key")

        post_resp.raise_for_status()

        time.sleep(3)
        _vt_rate_limit_wait()
        final_resp = _vt_session.get(
            f"{VT_BASE_URL}/urls/{url_id}",
            headers=headers, timeout=15,
        )
        if final_resp.status_code == 200:
            return _vt_parse_report(url, final_resp.json())
        return _vt_error_response("Scan queued but not yet complete — try again in 30s")

    if get_resp.status_code == 429:
        return _vt_error_response("VirusTotal rate limit (try later)")
    if get_resp.status_code == 401:
        return _vt_error_response("Invalid VirusTotal API key")

    return _vt_error_response(f"VT HTTP {get_resp.status_code}")


def _vt_parse_report(url: str, data: Dict) -> Dict:
    """Parse VT v3 /urls/{id} response into our clean format."""
    try:
        attrs = data.get("data", {}).get("attributes", {})
        stats = attrs.get("last_analysis_stats", {}) or {}

        malicious   = int(stats.get("malicious", 0))
        suspicious  = int(stats.get("suspicious", 0))
        harmless    = int(stats.get("harmless", 0))
        undetected  = int(stats.get("undetected", 0))
        timeout_cnt = int(stats.get("timeout", 0))
        total = malicious + suspicious + harmless + undetected + timeout_cnt

        categories = set()
        results = attrs.get("last_analysis_results", {}) or {}
        for engine_data in results.values():
            if engine_data.get("category") in ("malicious", "suspicious"):
                res = engine_data.get("result")
                if res and res not in ("clean", "unrated"):
                    categories.add(res)

        is_malicious = malicious >= 2 or (malicious >= 1 and suspicious >= 2)

        scan_ts = attrs.get("last_analysis_date")
        scan_date = (datetime.fromtimestamp(scan_ts).strftime("%Y-%m-%d %H:%M:%S")
                     if scan_ts else "Unknown")

        if is_malicious:
            msg = f"🚨 {malicious}/{total} engines flagged this URL as malicious"
        elif malicious > 0 or suspicious > 0:
            msg = f"⚠️ {malicious} malicious + {suspicious} suspicious out of {total}"
        else:
            msg = f"✅ Clean — {total} engines scanned, none flagged"

        return {
            "status": "completed",
            "is_malicious": is_malicious,
            "malicious_count":  malicious,
            "suspicious_count": suspicious,
            "harmless_count":   harmless,
            "undetected_count": undetected,
            "total_engines":    total,
            "detection_ratio":  f"{malicious}/{total}" if total else "0/0",
            "threat_categories": sorted(categories)[:10],
            "scan_date": scan_date,
            "permalink": f"https://www.virustotal.com/gui/url/{_vt_url_id(url)}",
            "message": msg,
        }

    except Exception as e:
        logger.exception("Failed to parse VT response")
        return _vt_error_response(f"Parse error: {str(e)[:80]}")


# ══════════════════════════════════════════════════════════════════════
#  Response Helpers
# ══════════════════════════════════════════════════════════════════════
def _vt_skip_response(message: str, status: str = "skipped") -> Dict:
    return {
        "status": status, "is_malicious": False,
        "malicious_count": 0, "suspicious_count": 0,
        "harmless_count": 0, "undetected_count": 0,
        "total_engines": 0, "detection_ratio": "N/A",
        "threat_categories": [], "scan_date": "N/A",
        "permalink": "", "message": message,
    }


def _vt_error_response(message: str) -> Dict:
    return {
        "status": "error", "is_malicious": False,
        "malicious_count": 0, "suspicious_count": 0,
        "harmless_count": 0, "undetected_count": 0,
        "total_engines": 0, "detection_ratio": "N/A",
        "threat_categories": [], "scan_date": "N/A",
        "permalink": "", "message": f"❌ {message}",
    }


# ══════════════════════════════════════════════════════════════════════
#  Unified API (for use in analyze_url() pipeline — Phase 6)
# ══════════════════════════════════════════════════════════════════════
def run_threat_intelligence(url: str, ml_confidence: float) -> Dict:
    """One-call wrapper for the entire TI stack."""
    return {
        "safe_browsing": check_safe_browsing(url),
        "virustotal":    check_virustotal(url, ml_confidence),
    }


def get_circuit_status() -> Dict:
    """For admin panel monitoring."""
    try:
        return {
            "safe_browsing": circuit_breaker.status(CB_GSB),
            "virustotal":    circuit_breaker.status(CB_VT),
        }
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════
#  Sanity test
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 API INTEGRATION — PHASE 8 TEST")
    print("=" * 70)

    test_urls = [
        ("http://malware.testing.google.test/testing/malware/",
         0.95, "Google's official malware test URL"),
        ("https://www.google.com", 0.10, "Legit — should skip VT"),
        ("https://www.github.com", 0.85, "Legit but high conf — will call VT"),
    ]

    for url, ml_conf, note in test_urls:
        print(f"\n🔗 {url}")
        print(f"   ({note})  ML confidence: {ml_conf}")
        print("-" * 70)

        gsb = check_safe_browsing(url)
        print(f"  🔵 Safe Browsing [{gsb['status']}]: {gsb['message']}")

        vt = check_virustotal(url, ml_confidence=ml_conf)
        print(f"  🟣 VirusTotal    [{vt['status']}]: {vt['message']}")
        if vt["status"] == "completed":
            print(f"      Detection: {vt['detection_ratio']}")
            if vt["threat_categories"]:
                print(f"      Categories: {', '.join(vt['threat_categories'][:5])}")

    print("\n🔌 Circuit breaker status:")
    for svc, st in get_circuit_status().items():
        print(f"   {svc}: {st}")

    print("\n" + "=" * 70)
    print("✅ Test complete")
    print("=" * 70)