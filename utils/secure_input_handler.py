"""
secure_input_handler.py — PhishGuard AI Secure Input Validation
================================================================

PURPOSE:
    Validate, sanitize, and rate-limit all URL inputs before
    they enter the analysis pipeline. This is the security
    perimeter of the entire system.

DEFENDS AGAINST:
    • Malformed / non-URL inputs (XSS injection attempts)
    • Private/loopback IP scans (SSRF — Server-Side Request Forgery)
    • Excessively long URLs (DoS buffer attacks)
    • Banned dangerous extensions (.exe, .dll etc.)
    • Homograph-level Unicode deception in inputs
    • Rate limit abuse (per-IP sliding window, in-memory)

INTEGRATION:
    # In Flask app.py — add ONE line before analyze_url():
    from secure_input_handler import validate_and_sanitize, RateLimiter

    limiter = RateLimiter(max_requests=10, window_seconds=60)

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        ip = request.remote_addr
        ok, err = limiter.check(ip)
        if not ok:
            return jsonify({"error": err}), 429

        result = validate_and_sanitize(data.get("url"))
        if not result.is_valid:
            return jsonify({"error": result.error}), 400

        analysis = analyze_url(result.safe_url, ...)

    # In Streamlit:
    from secure_input_handler import validate_and_sanitize
    result = validate_and_sanitize(url_input)
    if not result.is_valid:
        st.error(result.error)
        st.stop()
"""

from __future__ import annotations

import os
import re
import time
import ipaddress
import unicodedata
import threading
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, quote

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

MAX_URL_LENGTH       = 2048       # chars — Chrome's practical limit
MIN_URL_LENGTH       = 4          # "a.io"
MAX_DOMAIN_LABELS    = 10         # subdomain depth
ALLOWED_SCHEMES      = {"http", "https"}

# Banned file extensions (executable / dangerous download paths)
BANNED_EXTENSIONS    = {
    ".exe", ".dll", ".bat", ".cmd", ".ps1", ".vbs", ".js",
    ".jar", ".apk", ".dmg", ".msi", ".scr", ".com",
}

# Private / loopback IP ranges that must not be scanned (SSRF prevention)
PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 private
]

# Regex: detect basic XSS / injection patterns in URL
_XSS_RE  = re.compile(r"(<script|javascript:|vbscript:|data:text)", re.I)
_NULL_RE = re.compile(r"%00|\\x00|\x00")  # null byte injection


# ═══════════════════════════════════════════════════════════════════════════
#  RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """
    Returned by validate_and_sanitize().

    Attributes:
        is_valid:   True if URL passed all checks
        safe_url:   Cleaned, normalized URL (use this for analysis)
        original:   Raw input string
        error:      Human-readable error message (if not valid)
        warnings:   Non-blocking observations
        checks:     Dict of check_name → passed (bool)
    """
    is_valid:  bool
    safe_url:  str
    original:  str
    error:     str       = ""
    warnings:  List[str] = field(default_factory=list)
    checks:    Dict[str, bool] = field(default_factory=dict)

    def __bool__(self):
        return self.is_valid


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def validate_and_sanitize(raw_url: str) -> ValidationResult:
    """
    Validate and sanitize a URL input.

    This is the PRIMARY entry point. Call it before analyze_url().

    Args:
        raw_url: Raw string from user input / API request

    Returns:
        ValidationResult — check .is_valid before proceeding

    Example:
        result = validate_and_sanitize("  HTTP://Example.COM/path  ")
        if result.is_valid:
            analyze_url(result.safe_url)
        else:
            show_error(result.error)
    """
    original = raw_url
    checks   = {}
    warnings = []

    # ── 1. Null / empty check ─────────────────────────────────────────
    if not raw_url or not raw_url.strip():
        return ValidationResult(False, "", original,
                                error="URL cannot be empty",
                                checks={"not_empty": False})
    checks["not_empty"] = True

    # ── 2. Strip whitespace + control characters ──────────────────────
    url = raw_url.strip()
    url = _strip_control_chars(url)

    # ── 3. Null-byte injection check ──────────────────────────────────
    if _NULL_RE.search(url):
        return ValidationResult(False, "", original,
                                error="Invalid URL: null byte detected",
                                checks={**checks, "no_null_byte": False})
    checks["no_null_byte"] = True

    # ── 4. XSS / injection check ─────────────────────────────────────
    if _XSS_RE.search(url):
        return ValidationResult(False, "", original,
                                error="Invalid URL: script injection detected",
                                checks={**checks, "no_xss": False})
    checks["no_xss"] = True

    # ── 5. Length check ───────────────────────────────────────────────
    if len(url) > MAX_URL_LENGTH:
        return ValidationResult(False, "", original,
                                error=f"URL too long ({len(url)} chars, max {MAX_URL_LENGTH})",
                                checks={**checks, "length_ok": False})
    if len(url) < MIN_URL_LENGTH:
        return ValidationResult(False, "", original,
                                error=f"URL too short ({len(url)} chars, min {MIN_URL_LENGTH})",
                                checks={**checks, "length_ok": False})
    checks["length_ok"] = True

    # ── 6. Add scheme if missing ──────────────────────────────────────
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
        url = "http://" + url
        warnings.append("Scheme missing — defaulted to http://")

    # ── 7. Parse URL ──────────────────────────────────────────────────
    try:
        parsed = urlparse(url)
    except Exception:
        return ValidationResult(False, "", original,
                                error="Malformed URL — cannot parse",
                                checks={**checks, "parseable": False})

    if not parsed.netloc:
        return ValidationResult(False, "", original,
                                error="URL has no hostname",
                                checks={**checks, "has_netloc": False})
    checks["parseable"] = True
    checks["has_netloc"] = True

    # ── 8. Scheme whitelist ───────────────────────────────────────────
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        return ValidationResult(False, "", original,
                                error=f"Scheme '{scheme}' not allowed. Use http or https.",
                                checks={**checks, "scheme_ok": False})
    checks["scheme_ok"] = True

    # ── 9. Extract and validate hostname ─────────────────────────────
    host = parsed.hostname or ""
    host = host.lower()

    if not host:
        return ValidationResult(False, "", original,
                                error="No valid hostname found",
                                checks={**checks, "has_host": False})
    checks["has_host"] = True

    # ── 10. SSRF prevention — block private IPs ───────────────────────
    ip_check = _is_private_ip(host)
    if ip_check["is_private"]:
        return ValidationResult(False, "", original,
                                error=f"Private/loopback IPs are not allowed ({ip_check['reason']})",
                                checks={**checks, "not_private_ip": False})
    checks["not_private_ip"] = True

    # ── 11. Domain label depth ────────────────────────────────────────
    labels = host.split(".")
    if len(labels) > MAX_DOMAIN_LABELS:
        return ValidationResult(False, "", original,
                                error=f"Too many subdomain levels ({len(labels)}, max {MAX_DOMAIN_LABELS})",
                                checks={**checks, "label_depth_ok": False})
    checks["label_depth_ok"] = True

    # ── 12. Banned extension check ────────────────────────────────────
    path_lower = parsed.path.lower()
    for ext in BANNED_EXTENSIONS:
        if path_lower.endswith(ext):
            warnings.append(f"URL path ends with banned extension: {ext}")
            # Warning only — still scan (the URL might be reported as phishing)
            break
    checks["ext_checked"] = True

    # ── 13. Unicode / IDN normalization ──────────────────────────────
    safe_url, idn_warning = _normalize_unicode(url, parsed)
    if idn_warning:
        warnings.append(idn_warning)
    checks["unicode_normalized"] = True

    # ── 14. Final cleanup ─────────────────────────────────────────────
    safe_url = _final_clean(safe_url)
    checks["sanitized"] = True

    logger.debug(f"URL validated: {safe_url[:80]}  warnings={warnings}")

    return ValidationResult(
        is_valid = True,
        safe_url = safe_url,
        original = original,
        warnings = warnings,
        checks   = checks,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  PRIVATE IP CHECK (SSRF prevention)
# ═══════════════════════════════════════════════════════════════════════════

def _is_private_ip(host: str) -> Dict:
    """Return {is_private: bool, reason: str}."""
    # Loopback strings
    if host in ("localhost", "localhost.localdomain", "local"):
        return {"is_private": True, "reason": "localhost"}

    # Try to parse as IP
    try:
        addr = ipaddress.ip_address(host)
        for network in PRIVATE_RANGES:
            if addr in network:
                return {"is_private": True, "reason": f"{addr} in {network}"}
        # Special categories
        if addr.is_loopback:
            return {"is_private": True, "reason": "loopback"}
        if addr.is_link_local:
            return {"is_private": True, "reason": "link-local"}
        if addr.is_multicast:
            return {"is_private": True, "reason": "multicast"}
    except ValueError:
        pass  # Not an IP address — it's a domain name

    return {"is_private": False, "reason": ""}


# ═══════════════════════════════════════════════════════════════════════════
#  UNICODE / IDN NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_unicode(url: str, parsed) -> Tuple[str, Optional[str]]:
    """
    Normalize unicode characters in domain (IDN).
    Returns (normalized_url, warning_message_or_None).
    """
    host = parsed.hostname or ""
    warning = None

    try:
        # Detect non-ASCII characters in host
        if any(ord(c) > 127 for c in host):
            try:
                # Attempt IDNA encoding (punycodes it)
                ascii_host = host.encode("idna").decode("ascii")
                warning = f"International domain detected — encoded as: {ascii_host}"
                # Rebuild URL with punycode host
                netloc = parsed.netloc.replace(host, ascii_host)
                url = urlunparse(parsed._replace(netloc=netloc))
            except (UnicodeError, UnicodeDecodeError):
                warning = "Non-ASCII characters in hostname could not be encoded"

        # NFC normalization of the whole URL
        url = unicodedata.normalize("NFC", url)
    except Exception:
        pass  # Normalization failure is non-fatal

    return url, warning


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER SANITIZERS
# ═══════════════════════════════════════════════════════════════════════════

def _strip_control_chars(text: str) -> str:
    """Remove ASCII control characters (except tab/newline used in multiline)."""
    return "".join(c for c in text if unicodedata.category(c) != "Cc" or c in "\t\n")


def _final_clean(url: str) -> str:
    """Final cleanup: remove trailing whitespace, fragment (#anchor)."""
    url = url.strip()
    # Remove URL fragment (not needed for analysis, leaks info)
    if "#" in url:
        url = url[:url.index("#")]
    return url


# ═══════════════════════════════════════════════════════════════════════════
#  RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    In-memory sliding-window rate limiter.

    Tracks requests per identifier (IP address, user ID, etc.)
    using a deque per identifier — O(1) check + O(1) insert.

    Thread-safe with RLock.

    Usage:
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        # In Flask route:
        ok, msg = limiter.check(request.remote_addr)
        if not ok:
            return jsonify({"error": msg}), 429

        # In Streamlit (use session state for ID):
        ok, msg = limiter.check(st.session_state.get("user_id", "anon"))
    """

    def __init__(
        self,
        max_requests:   int = 10,
        window_seconds: int = 60,
        burst_max:      int = 5,       # extra burst requests allowed
        burst_window:   int = 10,      # burst window in seconds
    ):
        self.max_requests   = max_requests
        self.window_seconds = window_seconds
        self.burst_max      = burst_max
        self.burst_window   = burst_window

        self._requests: Dict[str, deque] = defaultdict(lambda: deque())
        self._blocked:  Dict[str, float] = {}   # identifier → block_until timestamp
        self._lock      = threading.RLock()

    def check(self, identifier: str) -> Tuple[bool, str]:
        """
        Check if identifier is within rate limit.

        Returns:
            (True, "")           — allowed, proceed
            (False, error_msg)   — blocked, return 429
        """
        now = time.time()

        with self._lock:
            # ── Check hard block (for repeat offenders) ──────────────
            block_until = self._blocked.get(identifier, 0)
            if now < block_until:
                remaining = int(block_until - now)
                return False, f"Rate limit exceeded. Retry after {remaining}s."

            # ── Slide window ──────────────────────────────────────────
            q = self._requests[identifier]
            while q and (now - q[0]) > self.window_seconds:
                q.popleft()

            # ── Burst check ───────────────────────────────────────────
            recent_burst = sum(1 for ts in q if (now - ts) <= self.burst_window)
            if recent_burst >= self.burst_max:
                # Soft block for 30s
                self._blocked[identifier] = now + 30
                logger.warning(f"Rate limit burst triggered: {identifier}")
                return False, "Too many requests in a short burst. Please wait 30 seconds."

            # ── Window check ──────────────────────────────────────────
            if len(q) >= self.max_requests:
                oldest  = q[0]
                wait    = int(self.window_seconds - (now - oldest)) + 1
                logger.info(f"Rate limited: {identifier} ({len(q)} reqs in window)")
                return False, f"Rate limit: {self.max_requests} requests per {self.window_seconds}s. Wait {wait}s."

            # ── Record this request ───────────────────────────────────
            q.append(now)
            remaining_quota = self.max_requests - len(q)

            return True, f"{remaining_quota} requests remaining in this window."

    def get_status(self, identifier: str) -> Dict:
        """Get current rate limit status for an identifier."""
        now = time.time()
        with self._lock:
            q = self._requests.get(identifier, deque())
            recent = [ts for ts in q if (now - ts) <= self.window_seconds]
            blocked_until = self._blocked.get(identifier, 0)
            return {
                "identifier":     identifier,
                "requests_made":  len(recent),
                "requests_limit": self.max_requests,
                "window_seconds": self.window_seconds,
                "is_blocked":     now < blocked_until,
                "block_remaining": max(0, int(blocked_until - now)),
            }

    def reset(self, identifier: str) -> None:
        """Reset rate limit for an identifier (admin use)."""
        with self._lock:
            self._requests.pop(identifier, None)
            self._blocked.pop(identifier, None)

    def cleanup_stale(self) -> int:
        """Remove stale identifiers (call periodically)."""
        now = time.time()
        removed = 0
        with self._lock:
            stale = [k for k, q in self._requests.items()
                     if not q or (now - max(q)) > self.window_seconds * 2]
            for k in stale:
                del self._requests[k]
                self._blocked.pop(k, None)
                removed += 1
        return removed


# ═══════════════════════════════════════════════════════════════════════════
#  BATCH URL VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

def validate_batch(urls: List[str], max_batch: int = 50) -> Dict:
    """
    Validate a list of URLs (batch mode).

    Returns:
        {
            "valid":   [safe_url, ...],
            "invalid": [{"original": str, "error": str}, ...],
            "warnings": {url: [str, ...]}
        }
    """
    if len(urls) > max_batch:
        return {
            "valid": [], "invalid": [],
            "error": f"Batch too large: {len(urls)} URLs (max {max_batch})"
        }

    valid_urls   = []
    invalid_urls = []
    url_warnings = {}

    for raw in urls:
        result = validate_and_sanitize(raw)
        if result.is_valid:
            valid_urls.append(result.safe_url)
            if result.warnings:
                url_warnings[result.safe_url] = result.warnings
        else:
            invalid_urls.append({"original": raw, "error": result.error})

    return {
        "valid":    valid_urls,
        "invalid":  invalid_urls,
        "warnings": url_warnings,
        "stats": {
            "total":   len(urls),
            "passed":  len(valid_urls),
            "failed":  len(invalid_urls),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT HELPER
# ═══════════════════════════════════════════════════════════════════════════

def render_validation_error(result: ValidationResult) -> None:
    """Display validation errors in Streamlit."""
    try:
        import streamlit as st
    except ImportError:
        return

    if not result.is_valid:
        st.error(f"❌ {result.error}")
        with st.expander("Security checks"):
            for check, passed in result.checks.items():
                icon = "✅" if passed else "❌"
                st.write(f"{icon} {check.replace('_', ' ')}")
    if result.warnings:
        for w in result.warnings:
            st.warning(f"⚠️ {w}")


# ═══════════════════════════════════════════════════════════════════════════
#  SELF TEST
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 65)
    print("🧪 SECURE INPUT HANDLER — SELF TEST")
    print("=" * 65)

    test_cases = [
        ("https://www.google.com",                         True,  "Valid HTTPS"),
        ("http://example.com/path?q=1",                    True,  "Valid HTTP with query"),
        ("example.com",                                     True,  "No scheme (auto-added)"),
        ("",                                                False, "Empty input"),
        ("not-a-url",                                       False, "No TLD / no netloc"),
        ("http://127.0.0.1/admin",                          False, "Loopback IP (SSRF)"),
        ("http://192.168.1.100/login",                      False, "Private IP (SSRF)"),
        ("http://10.0.0.1/db",                              False, "Private IP class A"),
        ("javascript:alert(1)",                             False, "XSS injection"),
        ("<script>alert(1)</script>",                       False, "XSS in URL"),
        ("http://example.com/" + "A"*3000,                 False, "URL too long"),
        ("ftp://ftp.example.com/file.zip",                  False, "Banned scheme"),
        ("http://malware.testing.google.test/malware/",     True,  "Valid test URL"),
        ("  https://EXAMPLE.COM/  ",                        True,  "Whitespace + caps"),
        ("http://xn--e1afmapc.com",                         True,  "Punycode IDN"),
    ]

    passed = failed = 0
    for raw, expected_valid, desc in test_cases:
        result = validate_and_sanitize(raw)
        ok = (result.is_valid == expected_valid)
        status = "✅" if ok else "❌"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status} [{desc}]")
        if not ok:
            print(f"      Expected valid={expected_valid}, got valid={result.is_valid}")
            print(f"      Error: {result.error}")
        elif result.is_valid:
            print(f"      Safe URL: {result.safe_url[:70]}")
        if result.warnings:
            print(f"      Warnings: {result.warnings}")

    print(f"\n  Results: {passed} passed, {failed} failed")

    # Rate limiter test
    print("\n🚦 Rate limiter test:")
    rl = RateLimiter(max_requests=3, window_seconds=10, burst_max=2, burst_window=5)
    test_ip = "1.2.3.4"
    for i in range(5):
        ok, msg = rl.check(test_ip)
        print(f"   Request #{i+1}: {'✅ ALLOWED' if ok else '❌ BLOCKED'} — {msg}")

    print("\n" + "=" * 65)
    print("✅ Module 5 test complete")
    print("=" * 65)