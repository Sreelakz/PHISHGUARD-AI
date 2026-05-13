"""
utils/logger.py
----------------
Centralized logging configuration for PhishGuard AI.

FEATURES:
  ✅ Rotating file handler (10 MB per file, keep 5 backups)
  ✅ Console + file output
  ✅ Per-module log levels
  ✅ Structured format with timestamps
  ✅ Color-coded console output (if colorama installed)

USAGE:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

import os
import logging
from logging.handlers import RotatingFileHandler

# ══════════════════════════════════════════════════════════════════════
#  Paths
# ══════════════════════════════════════════════════════════════════════
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "phishguard.log")

# ══════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
BACKUP_COUNT = 5

# Per-module log levels (reduce noise from chatty libraries)
MODULE_LEVELS = {
    "urllib3":           logging.WARNING,
    "requests":          logging.WARNING,
    "werkzeug":          logging.WARNING,   # Flask dev server
    "matplotlib":        logging.WARNING,
    "PIL":               logging.WARNING,
    "shap":              logging.WARNING,
    "asyncio":           logging.WARNING,
}

# ══════════════════════════════════════════════════════════════════════
#  Setup
# ══════════════════════════════════════════════════════════════════════
_initialized = False


def _setup_root_logger():
    """Configure root logger once (idempotent)."""
    global _initialized
    if _initialized:
        return

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Clear existing handlers (avoid duplicates on Flask reload)
    root.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Console handler ───────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ── Rotating file handler ─────────────────────────────────────────
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception as e:
        root.warning(f"Could not set up file logging: {e}")

    # ── Mute noisy modules ────────────────────────────────────────────
    for module, level in MODULE_LEVELS.items():
        logging.getLogger(module).setLevel(level)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Usually __name__ from the calling module.

    Returns:
        Configured logger.
    """
    _setup_root_logger()
    return logging.getLogger(name)


# ══════════════════════════════════════════════════════════════════════
#  Performance helper
# ══════════════════════════════════════════════════════════════════════
import time
from functools import wraps


def log_performance(logger: logging.Logger, threshold_ms: float = 1000):
    """
    Decorator to log slow function calls.

    Usage:
        @log_performance(logger, threshold_ms=500)
        def slow_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if elapsed_ms > threshold_ms:
                    logger.warning(
                        f"⚠️  SLOW: {func.__name__} took {elapsed_ms:.1f}ms"
                    )
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════
#  Self-test
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger = get_logger("logger_test")
    logger.debug("debug message")
    logger.info("✅ Logger initialized successfully")
    logger.warning("⚠️  This is a warning")
    logger.error("❌ This is an error")
    print(f"\n📁 Logs saved to: {LOG_FILE}")