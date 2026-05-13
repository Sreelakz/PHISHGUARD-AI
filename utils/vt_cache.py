"""
utils/vt_cache.py
------------------
Persistent JSON cache for VirusTotal API results.

Why cache?
  • VT free tier = 500 req/day, 4 req/min
  • Same URL often rescanned during demos/testing
  • 24h TTL → fresh enough for phishing (threats evolve daily)
"""

import os
import json
import time
import hashlib
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
CACHE_DIR = os.path.join(_PROJECT_ROOT, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "vt_cache.json")
CACHE_TTL_SECONDS = 24 * 60 * 60   # 24 hours

os.makedirs(CACHE_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────
def _url_key(url: str) -> str:
    """Hash URL to a stable cache key (privacy + filesystem safety)."""
    return hashlib.sha256(url.strip().lower().encode()).hexdigest()[:32]


def _load_cache() -> Dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Cache file corrupted, resetting: {e}")
        return {}


def _save_cache(cache: Dict) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError as e:
        logger.error(f"Failed to write cache: {e}")


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────
def get_cached(url: str) -> Optional[Dict]:
    """Return cached VT result if still fresh, else None."""
    cache = _load_cache()
    key = _url_key(url)
    entry = cache.get(key)
    if not entry:
        return None

    age = time.time() - entry.get("timestamp", 0)
    if age > CACHE_TTL_SECONDS:
        logger.info(f"Cache expired for {url[:50]} (age: {age/3600:.1f}h)")
        return None

    logger.info(f"✅ Cache HIT for {url[:50]} (age: {age/60:.1f}min)")
    return entry["result"]


def set_cached(url: str, result: Dict) -> None:
    """Store a VT result in the cache."""
    cache = _load_cache()
    cache[_url_key(url)] = {
        "url": url,
        "timestamp": time.time(),
        "result": result,
    }
    _save_cache(cache)


def clear_cache() -> int:
    """Wipe cache (for debugging). Returns number of entries removed."""
    cache = _load_cache()
    count = len(cache)
    _save_cache({})
    return count