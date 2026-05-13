"""
feedback_handler.py — PhishGuard AI Feedback Learning System
=============================================================

PURPOSE:
    Collects user corrections when the model predicts wrong,
    stores them persistently, and uses that data to retrain
    the model — making it smarter over time.

PIPELINE POSITION:
    UI (Streamlit) → feedback_handler.save_feedback()
                   → feedback_handler.retrain_from_feedback()
                   → ml_model.py (existing)

STORAGE:
    • SQLite DB (primary)  — feedback/feedback.db
    • CSV mirror (export)  — feedback/feedback_export.csv

COMPATIBILITY:
    • Works with existing ml_model.py PhishingMLModel class
    • Works with existing FeatureExtractor output format
    • Drop-in with Streamlit UI (no changes needed to app.py)
"""

from __future__ import annotations

import os
import sys
import csv
import json
import sqlite3
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Optional, List, Tuple

# ── Path setup ────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _HERE  # adjust if placed inside backend/
FEEDBACK_DIR = os.path.join(_PROJECT_ROOT, "feedback")
FEEDBACK_DB  = os.path.join(FEEDBACK_DIR, "feedback.db")
FEEDBACK_CSV = os.path.join(FEEDBACK_DIR, "feedback_export.csv")
os.makedirs(FEEDBACK_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  DATABASE SETUP
#  Single table: feedback
#    id, url, features_json, predicted_label, actual_label,
#    confidence, timestamp, used_in_retrain
# ═══════════════════════════════════════════════════════════════════════════

@contextmanager
def _db():
    """Thread-safe SQLite connection context manager."""
    conn = sqlite3.connect(FEEDBACK_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


def _init_db() -> None:
    """Create feedback table + indexes on first run (idempotent)."""
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                url              TEXT    NOT NULL,
                features_json    TEXT,
                predicted_label  TEXT    NOT NULL,
                actual_label     TEXT    NOT NULL,
                confidence       REAL,
                feedback_type    TEXT    DEFAULT 'incorrect',
                source           TEXT    DEFAULT 'user',
                timestamp        TEXT    DEFAULT (datetime('now')),
                used_in_retrain  INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_label   ON feedback(actual_label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_retrain ON feedback(used_in_retrain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fb_ts      ON feedback(timestamp DESC)")
        conn.commit()


# Run init on import
_init_db()


# ═══════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION: save_feedback()
# ═══════════════════════════════════════════════════════════════════════════

def save_feedback(
    url:             str,
    features:        Dict,
    predicted_label: str,
    actual_label:    str,
    confidence:      float = 0.0,
    source:          str   = "user",
) -> Dict:
    """
    Save a feedback record when a user marks a prediction as incorrect.

    Args:
        url:             The URL that was analyzed
        features:        Feature dict from FeatureExtractor (or subset)
        predicted_label: What the model predicted ("PHISHING" / "LEGITIMATE")
        actual_label:    What the user says is correct
        confidence:      Model's confidence score (0.0–1.0)
        source:          "user" | "expert" | "auto"

    Returns:
        {"success": True, "id": int, "message": str}
        {"success": False, "error": str}

    Example:
        result = save_feedback(
            url="http://bad-site.com",
            features={"url_length": 45, "has_ip_address": 1, ...},
            predicted_label="LEGITIMATE",
            actual_label="PHISHING",
            confidence=0.43,
        )
    """
    # ── Validation ────────────────────────────────────────────────────
    if not url or not url.strip():
        return {"success": False, "error": "URL is required"}

    valid_labels = {"PHISHING", "LEGITIMATE"}
    predicted_label = (predicted_label or "").strip().upper()
    actual_label    = (actual_label    or "").strip().upper()

    if predicted_label not in valid_labels:
        return {"success": False, "error": f"Invalid predicted_label: {predicted_label}"}
    if actual_label not in valid_labels:
        return {"success": False, "error": f"Invalid actual_label: {actual_label}"}

    # ── Determine feedback type ───────────────────────────────────────
    feedback_type = "correct" if predicted_label == actual_label else "incorrect"

    # ── Serialise features (keep only JSON-safe types) ────────────────
    try:
        features_json = json.dumps(_sanitize_features(features))
    except Exception as e:
        features_json = json.dumps({})
        logger.warning(f"Feature serialization warning: {e}")

    # ── Insert into DB ────────────────────────────────────────────────
    try:
        with _db() as conn:
            cur = conn.execute(
                """INSERT INTO feedback
                   (url, features_json, predicted_label, actual_label,
                    confidence, feedback_type, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (url.strip(), features_json, predicted_label, actual_label,
                 float(confidence), feedback_type, source)
            )
            conn.commit()
            record_id = cur.lastrowid

        logger.info(f"Feedback saved [id={record_id}] {predicted_label}→{actual_label}: {url[:60]}")

        # ── Mirror to CSV ─────────────────────────────────────────────
        _append_to_csv({
            "id": record_id, "url": url, "predicted_label": predicted_label,
            "actual_label": actual_label, "confidence": confidence,
            "feedback_type": feedback_type, "source": source,
            "timestamp": datetime.now().isoformat(),
        })

        return {
            "success":       True,
            "id":            record_id,
            "feedback_type": feedback_type,
            "message":       (
                "✅ Feedback saved — thank you! The model will learn from this."
                if feedback_type == "incorrect"
                else "✅ Confirmed correct prediction."
            ),
        }

    except sqlite3.Error as e:
        logger.error(f"DB error saving feedback: {e}")
        return {"success": False, "error": f"Database error: {str(e)[:120]}"}


# ═══════════════════════════════════════════════════════════════════════════
#  RETRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def retrain_from_feedback(
    min_samples:    int  = 20,
    only_incorrect: bool = True,
    mark_used:      bool = True,
    verbose:        bool = True,
) -> Dict:
    """
    Retrain the ML model by incorporating feedback corrections.

    STRATEGY:
        1. Load existing training features + labels from feedback DB
        2. Load the current model's training data (via ml_model)
        3. Combine: original data + corrected feedback samples
           (feedback samples are upweighted 3x to emphasise corrections)
        4. Retrain on combined dataset
        5. Save new model (overwrites existing)
        6. Mark used feedback rows so they aren't reused

    Args:
        min_samples:    Minimum feedback records needed to trigger retraining
        only_incorrect: If True, only use misclassified corrections
        mark_used:      Mark used feedback rows in DB
        verbose:        Print progress

    Returns:
        {"success": True, "samples_used": int, "metrics": {...}}
        {"success": False, "reason": str}

    Compatibility:
        Requires ml_model.py with PhishingMLModel class that exposes:
            model.train(X, y)   — trains the model
            model.save()        — saves to disk
            model.get_training_data() — optional: returns (X, y) for current model
    """
    _log = print if verbose else logger.info

    # ── Load feedback records ─────────────────────────────────────────
    records = get_feedback_records(
        only_incorrect=only_incorrect,
        only_unused=True
    )

    if len(records) < min_samples:
        reason = (f"Not enough feedback yet ({len(records)}/{min_samples} samples). "
                  f"Collect at least {min_samples - len(records)} more corrections.")
        _log(f"⚠️  Retraining skipped: {reason}")
        return {"success": False, "reason": reason, "current_count": len(records)}

    _log(f"\n{'='*60}")
    _log(f"🔁 FEEDBACK RETRAINING — {len(records)} samples")
    _log(f"{'='*60}")

    # ── Parse features + labels ───────────────────────────────────────
    feedback_X: List[Dict] = []
    feedback_y: List[int]  = []

    for rec in records:
        try:
            feats = json.loads(rec["features_json"] or "{}")
            if not feats:
                continue
            feedback_X.append(feats)
            # Encode: PHISHING=1, LEGITIMATE=0
            feedback_y.append(1 if rec["actual_label"] == "PHISHING" else 0)
        except Exception as e:
            logger.warning(f"Skipping malformed record id={rec['id']}: {e}")

    if len(feedback_X) < min_samples:
        return {"success": False,
                "reason": f"Only {len(feedback_X)} usable records after parsing."}

    # ── Load existing model + training pipeline ───────────────────────
    try:
        # Import here to avoid circular imports at module load
        try:
            from backend.ml_model import get_model
        except ModuleNotFoundError:
            from ml_model import get_model

        ml = get_model()

        if not ml.is_trained():
            return {"success": False,
                    "reason": "No trained model found. Train the base model first."}

        _log(f"✅ Base model loaded")

    except Exception as e:
        return {"success": False, "reason": f"Failed to load ml_model: {str(e)[:120]}"}

    # ── Build combined training set ───────────────────────────────────
    try:
        try:
            from utils.preprocessing import preprocess_pipeline
        except ModuleNotFoundError:
            from preprocessing import preprocess_pipeline  # type: ignore

        # Load base dataset (small sample for speed)
        base_csv = os.path.join(_PROJECT_ROOT, "data", "dataset.csv")
        if os.path.exists(base_csv):
            _log("📂 Loading base dataset...")
            X_base, y_base, _ = preprocess_pipeline(base_csv, sample_size=5000, fetch_html=False)
            X_combined = list(X_base.to_dict("records")) + feedback_X * 3  # 3x upsample corrections
            y_combined = list(y_base) + feedback_y * 3
            _log(f"   Base: {len(y_base)} | Feedback (3x): {len(feedback_y)*3} | Total: {len(y_combined)}")
        else:
            # No base dataset — retrain on feedback only
            _log("⚠️  Base dataset not found. Retraining on feedback only.")
            X_combined = feedback_X * 5   # 5x upsample for stability
            y_combined = feedback_y * 5

    except Exception as e:
        logger.warning(f"Could not load base dataset ({e}). Using feedback only.")
        X_combined = feedback_X * 5
        y_combined = feedback_y * 5

    # ── Retrain ───────────────────────────────────────────────────────
    _log("\n🏋️  Retraining model...")
    try:
        import pandas as pd
        X_df = pd.DataFrame(X_combined)
        metrics = ml.train(X_df, y_combined, test_size=0.2, verbose=verbose)
        ml.save()
        _log(f"\n✅ Model retrained and saved!")
        _log(f"   Accuracy: {metrics.get('accuracy', 0)*100:.2f}%")
        _log(f"   F1-Score: {metrics.get('f1_score',  0)*100:.2f}%")
    except Exception as e:
        return {"success": False, "reason": f"Retraining failed: {str(e)[:200]}"}

    # ── Mark feedback as used ─────────────────────────────────────────
    if mark_used:
        used_ids = [rec["id"] for rec in records]
        _mark_feedback_used(used_ids)
        _log(f"✅ Marked {len(used_ids)} feedback records as used")

    return {
        "success":        True,
        "samples_used":   len(feedback_X),
        "total_combined": len(X_combined),
        "metrics":        metrics,
        "message":        f"Model retrained with {len(feedback_X)} feedback samples. "
                          f"New accuracy: {metrics.get('accuracy', 0)*100:.1f}%",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  QUERY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_feedback_records(
    only_incorrect: bool = True,
    only_unused:    bool = False,
    limit:          int  = 10000,
) -> List[Dict]:
    """Retrieve feedback records from DB."""
    with _db() as conn:
        q      = "SELECT * FROM feedback WHERE 1=1"
        params = []
        if only_incorrect:
            q += " AND feedback_type='incorrect'"
        if only_unused:
            q += " AND used_in_retrain=0"
        q += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cur = conn.execute(q, params)
        return [dict(r) for r in cur.fetchall()]


def get_feedback_stats() -> Dict:
    """Summary statistics for the admin panel / Streamlit dashboard."""
    with _db() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM feedback");                              total    = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM feedback WHERE feedback_type='incorrect'"); wrong = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM feedback WHERE used_in_retrain=0 AND feedback_type='incorrect'"); pending = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM feedback WHERE used_in_retrain=1");         used   = c.fetchone()[0]
        c.execute("SELECT actual_label, COUNT(*) as cnt FROM feedback GROUP BY actual_label")
        by_label = {r[0]: r[1] for r in c.fetchall()}

    return {
        "total_feedback":     total,
        "incorrect_count":    wrong,
        "pending_retrain":    pending,
        "used_in_retrain":    used,
        "by_actual_label":    by_label,
        "retrain_ready":      pending >= 20,
        "retrain_threshold":  20,
    }


def export_feedback_csv(path: Optional[str] = None) -> str:
    """Export all feedback to CSV. Returns the output path."""
    out_path = path or FEEDBACK_CSV
    records  = get_feedback_records(only_incorrect=False, only_unused=False)

    if not records:
        return out_path

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    return out_path


# ═══════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _sanitize_features(features: Dict) -> Dict:
    """Keep only JSON-serializable scalar values from features dict."""
    safe = {}
    for k, v in features.items():
        if isinstance(k, str) and isinstance(v, (int, float, bool, str, type(None))):
            safe[k] = v
    return safe


def _append_to_csv(record: Dict) -> None:
    """Append a single record to the CSV mirror (creates headers if new)."""
    try:
        write_header = not os.path.exists(FEEDBACK_CSV)
        with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(record)
    except Exception as e:
        logger.warning(f"CSV mirror write failed: {e}")


def _mark_feedback_used(ids: List[int]) -> None:
    """Mark records as used_in_retrain=1."""
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    with _db() as conn:
        conn.execute(
            f"UPDATE feedback SET used_in_retrain=1 WHERE id IN ({placeholders})",
            ids
        )
        conn.commit()


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT INTEGRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def render_feedback_widget(
    url:             str,
    features:        Dict,
    predicted_label: str,
    confidence:      float,
) -> Optional[Dict]:
    """
    Call this from Streamlit after showing a result.
    Renders the feedback buttons and handles the save.

    Usage in streamlit_app.py:
        from feedback_handler import render_feedback_widget
        feedback = render_feedback_widget(url, features, "PHISHING", 0.87)

    Returns the save_feedback result dict if submitted, else None.
    """
    try:
        import streamlit as st
    except ImportError:
        return None

    st.divider()
    st.markdown("#### 📝 Was this prediction correct?")

    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        correct_btn = st.button("✅ Correct", key=f"fb_correct_{hash(url)}")
    with col2:
        wrong_btn   = st.button("❌ Wrong",   key=f"fb_wrong_{hash(url)}")

    if correct_btn:
        result = save_feedback(url, features, predicted_label, predicted_label, confidence)
        st.success(result["message"])
        return result

    if wrong_btn:
        opposite = "LEGITIMATE" if predicted_label == "PHISHING" else "PHISHING"
        result = save_feedback(url, features, predicted_label, opposite, confidence)
        st.warning(result["message"])

        # Show retrain suggestion
        stats = get_feedback_stats()
        if stats["retrain_ready"]:
            st.info(f"💡 {stats['pending_retrain']} corrections collected — "
                    f"consider retraining via the Admin panel.")
        return result

    return None


def render_retrain_panel() -> None:
    """
    Admin panel widget for triggering retraining from Streamlit.

    Usage in admin page:
        from feedback_handler import render_retrain_panel
        render_retrain_panel()
    """
    try:
        import streamlit as st
    except ImportError:
        return

    st.subheader("🔁 Feedback-Based Retraining")

    stats = get_feedback_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Feedback",    stats["total_feedback"])
    col2.metric("Corrections",       stats["incorrect_count"])
    col3.metric("Pending Retrain",   stats["pending_retrain"])

    if not stats["retrain_ready"]:
        remaining = stats["retrain_threshold"] - stats["pending_retrain"]
        st.info(f"Collect {remaining} more corrections to enable retraining.")
        return

    st.success(f"✅ {stats['pending_retrain']} corrections ready for retraining!")

    if st.button("🚀 Retrain Model Now", type="primary"):
        with st.spinner("Retraining... this may take a minute."):
            result = retrain_from_feedback(min_samples=20, verbose=False)

        if result["success"]:
            m = result["metrics"]
            st.success(f"✅ Model retrained!")
            st.json({
                "Samples Used":   result["samples_used"],
                "Accuracy":       f"{m.get('accuracy',0)*100:.2f}%",
                "F1-Score":       f"{m.get('f1_score',0)*100:.2f}%",
            })
        else:
            st.error(f"❌ Retraining failed: {result['reason']}")

    # Export button
    st.divider()
    if st.button("📥 Export Feedback CSV"):
        path = export_feedback_csv()
        with open(path, "rb") as f:
            st.download_button(
                label="⬇️ Download feedback_export.csv",
                data=f.read(),
                file_name="feedback_export.csv",
                mime="text/csv",
            )


# ═══════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import pprint

    print("=" * 60)
    print("🧪 FEEDBACK HANDLER — SELF TEST")
    print("=" * 60)

    dummy_features = {
        "url_length": 78, "num_dots": 4, "has_ip_address": 1,
        "uses_https": 0, "has_suspicious_keywords": 1,
        "num_dashes": 2, "entropy": 4.7, "num_subdomains": 3,
    }

    # Test: incorrect prediction (LEGITIMATE was predicted but it's PHISHING)
    r1 = save_feedback(
        url="http://192.168.1.1/login/verify?bank=paypal",
        features=dummy_features,
        predicted_label="LEGITIMATE",
        actual_label="PHISHING",
        confidence=0.43,
    )
    print(f"\n📥 Save feedback (wrong prediction): {r1}")

    # Test: correct prediction
    r2 = save_feedback(
        url="https://www.google.com",
        features={"url_length": 22, "uses_https": 1, "has_ip_address": 0},
        predicted_label="LEGITIMATE",
        actual_label="LEGITIMATE",
        confidence=0.95,
    )
    print(f"📥 Save feedback (correct prediction): {r2}")

    print(f"\n📊 Feedback Stats:")
    pprint.pprint(get_feedback_stats())

    print(f"\n📋 Records (incorrect only):")
    recs = get_feedback_records(only_incorrect=True)
    for r in recs[:3]:
        print(f"   [{r['id']}] {r['url'][:50]:50s} | {r['predicted_label']} → {r['actual_label']}")

    csv_path = export_feedback_csv()
    print(f"\n📁 CSV exported to: {csv_path}")

    print("\n" + "=" * 60)
    print("✅ Module 1 test complete")
    print("=" * 60)