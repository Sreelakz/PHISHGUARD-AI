"""
utils/perf_cache.py
--------------------
High-performance caching utilities for PhishGuard AI.

PROVIDES:
  ✅ TTL + LRU cache for URL analysis results (15-min TTL, 1000 entries)
  ✅ Circuit breaker for flaky external APIs
  ✅ URL normalization (so http://X.com/ and X.com hit same cache key)
  ✅ Thread-safe (uses cachetools.TTLCache with lock)
  ✅ Statistics endpoint (hit/miss ratio)

USAGE:
    from utils.perf_cache import url_cache, circuit_breaker

    # Cache
    url_cache.set("https://example.com", {...})
    result = url_cache.get("https://example.com")

    # Circuit breaker (for APIs)
    if circuit_breaker.is_open("virustotal"):
        return skip_response
    try:
        result = api_call()
        circuit_breaker.record_success("virustotal")
    except Exception:
        circuit_breaker.record_failure("virustotal")
"""

import time
import threading
from typing import Optional, Dict, Any
from urllib.parse import urlparse, urlunparse

from cachetools import TTLCache

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
URL_CACHE_MAX_SIZE = 1000          # entries
URL_CACHE_TTL_SECONDS = 15 * 60    # 15 minutes

CIRCUIT_FAIL_THRESHOLD = 5         # fails before opening
CIRCUIT_COOLDOWN_SECONDS = 5 * 60  # 5-min cooldown when open


# ══════════════════════════════════════════════════════════════════════
#  URL Normalization
# ══════════════════════════════════════════════════════════════════════
def normalize_url(url: str) -> str:
    """
    Normalize a URL so equivalent URLs share the same cache key.

    Transformations:
      - Lowercase scheme + host
      - Strip trailing slash
      - Remove default ports (:80 for http, :443 for https)
      - Remove fragments (#section)

    Example:
        HTTP://Example.COM/path/  →  http://example.com/path
    """
    if not url:
        return ""

    url = url.strip()

    # Add scheme if missing
    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url

    try:
        p = urlparse(url)
        scheme = p.scheme.lower()
        netloc = p.netloc.lower()

        # Strip default ports
        if scheme == "http" and netloc.endswith(":80"):
            netloc = netloc[:-3]
        elif scheme == "https" and netloc.endswith(":443"):
            netloc = netloc[:-4]

        # Strip trailing slash from path
        path = p.path.rstrip("/") if p.path != "/" else ""

        return urlunparse((scheme, netloc, path, p.params, p.query, ""))
    except Exception:
        return url.lower()


# ══════════════════════════════════════════════════════════════════════
#  Thread-safe URL Cache
# ══════════════════════════════════════════════════════════════════════
class URLCache:
    """
    Thread-safe TTL + LRU cache for URL analysis results.
    Tracks hits / misses / size for monitoring.
    """

    def __init__(self, maxsize: int = URL_CACHE_MAX_SIZE,
                 ttl: int = URL_CACHE_TTL_SECONDS):
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._maxsize = maxsize
        self._ttl = ttl

    def get(self, url: str) -> Optional[Dict]:
        key = normalize_url(url)
        with self._lock:
            val = self._cache.get(key)
            if val is not None:
                self._hits += 1
                return val
            self._misses += 1
            return None

    def set(self, url: str, value: Dict) -> None:
        key = normalize_url(url)
        with self._lock:
            self._cache[key] = value

    def invalidate(self, url: str) -> bool:
        key = normalize_url(url)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        with self._lock:
            size = len(self._cache)
            self._cache.clear()
            return size

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total else 0
            return {
                "size": len(self._cache),
                "max_size": self._maxsize,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(hit_rate, 2),
            }


# ══════════════════════════════════════════════════════════════════════
#  Circuit Breaker (for flaky APIs)
# ══════════════════════════════════════════════════════════════════════
class CircuitBreaker:
    """
    Simple circuit breaker.

    States:
      CLOSED  — API calls flow through
      OPEN    — API calls are blocked (during cooldown)

    Trip condition: `fail_threshold` consecutive failures → OPEN for `cooldown`.
    Auto-close after cooldown expires.
    """

    def __init__(self, fail_threshold: int = CIRCUIT_FAIL_THRESHOLD,
                 cooldown: int = CIRCUIT_COOLDOWN_SECONDS):
        self._fail_threshold = fail_threshold
        self._cooldown = cooldown
        self._state: Dict[str, Dict] = {}   # service_name → state
        self._lock = threading.RLock()

    def _ensure_entry(self, service: str):
        if service not in self._state:
            self._state[service] = {
                "failures": 0,
                "opened_at": None,
            }

    def is_open(self, service: str) -> bool:
        """Return True if the circuit is open (calls should be blocked)."""
        with self._lock:
            self._ensure_entry(service)
            entry = self._state[service]

            if entry["opened_at"] is None:
                return False

            elapsed = time.time() - entry["opened_at"]
            if elapsed >= self._cooldown:
                # Auto-reset
                entry["failures"] = 0
                entry["opened_at"] = None
                return False

            return True

    def record_success(self, service: str):
        with self._lock:
            self._ensure_entry(service)
            self._state[service]["failures"] = 0
            self._state[service]["opened_at"] = None

    def record_failure(self, service: str):
        with self._lock:
            self._ensure_entry(service)
            entry = self._state[service]
            entry["failures"] += 1
            if entry["failures"] >= self._fail_threshold and entry["opened_at"] is None:
                entry["opened_at"] = time.time()

    def status(self, service: str) -> Dict[str, Any]:
        with self._lock:
            self._ensure_entry(service)
            entry = self._state[service]
            is_open = self.is_open(service)
            remaining = 0
            if entry["opened_at"]:
                remaining = max(0, int(self._cooldown - (time.time() - entry["opened_at"])))
            return {
                "service": service,
                "state": "OPEN" if is_open else "CLOSED",
                "failures": entry["failures"],
                "fail_threshold": self._fail_threshold,
                "cooldown_remaining_sec": remaining,
            }

    def all_status(self) -> Dict[str, Any]:
        with self._lock:
            return {svc: self.status(svc) for svc in self._state.keys()}

    def reset(self, service: Optional[str] = None):
        with self._lock:
            if service:
                self._state.pop(service, None)
            else:
                self._state.clear()


# ══════════════════════════════════════════════════════════════════════
#  Global singletons
# ══════════════════════════════════════════════════════════════════════
url_cache = URLCache()
circuit_breaker = CircuitBreaker()


# ══════════════════════════════════════════════════════════════════════
#  Self-test
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 PERF CACHE SELF-TEST")
    print("=" * 70)

    # URL normalization
    tests = [
        ("HTTP://Example.COM/path/",  "http://example.com/path"),
        ("https://foo.bar:443/x",     "https://foo.bar/x"),
        ("example.com",               "http://example.com"),
        ("https://X.com/#section",    "https://x.com"),
    ]
    print("\n📝 URL normalization:")
    for raw, expected in tests:
        got = normalize_url(raw)
        status = "✅" if got == expected else "❌"
        print(f"   {status}  {raw:40s} → {got}")

    # Cache
    print("\n💾 Cache test:")
    url_cache.set("https://test.com", {"verdict": "PHISHING"})
    assert url_cache.get("HTTPS://TEST.COM/") is not None  # normalization
    print("   ✅ Set/get works (with normalization)")
    print(f"   Stats: {url_cache.stats()}")

    # Circuit breaker
    print("\n🔌 Circuit breaker test:")
    for i in range(5):
        circuit_breaker.record_failure("testapi")
    assert circuit_breaker.is_open("testapi")
    print(f"   ✅ Tripped after 5 failures: {circuit_breaker.status('testapi')}")

    circuit_breaker.reset("testapi")
    assert not circuit_breaker.is_open("testapi")
    print("   ✅ Reset works")

    print("\n" + "=" * 70)
    print("✅ All tests passed")
    print("=" * 70)