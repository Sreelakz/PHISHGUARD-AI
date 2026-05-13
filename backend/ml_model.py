"""
backend/ml_model.py
--------------------
Production-grade phishing detection ML model — PHASE 8 OPTIMIZED.

ARCHITECTURE:
  Stage 1 — Rule engine (rules_engine.py) checks for critical overrides
  Stage 2 — ML ensemble (XGBoost + RandomForest) on real dataset features
  Stage 3 — Confidence blending between rules and ML

PHASE 8 OPTIMIZATIONS:
  ✅ Feature importances cached on first call (computed once, reused forever)
  ✅ Structured logging via utils.logger
  ✅ Lazy feature-vector conversion (numpy array pre-built once per call)
  ✅ Faster predict_proba via direct ndarray input (skips validation overhead)

BACKWARD COMPATIBLE: Same class, same methods, same return format.
"""

# ══════════════════════════════════════════════════════════════════════════
#  Smart imports
# ══════════════════════════════════════════════════════════════════════════
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from backend.rules_engine import check_critical_rules, check_high_rules
except ModuleNotFoundError:
    from rules_engine import check_critical_rules, check_high_rules

try:
    from utils.logger import get_logger
except ModuleNotFoundError:
    import logging
    def get_logger(name): return logging.getLogger(name)

# ══════════════════════════════════════════════════════════════════════════
#  Standard imports
# ══════════════════════════════════════════════════════════════════════════
import json
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════
#  Paths
# ══════════════════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "phishing_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


# ══════════════════════════════════════════════════════════════════════════
#  Main class
# ══════════════════════════════════════════════════════════════════════════
class PhishingMLModel:
    """Hybrid rule-based + ML phishing detector."""

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.metrics: Dict = {}
        # Phase 8: cached feature importances
        self._importances_cache: Optional[Dict[str, float]] = None
        self._importances_raw: Optional[np.ndarray] = None
        os.makedirs(MODELS_DIR, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    #  TRAINING
    # ────────────────────────────────────────────────────────────────────
    def train(self, X: pd.DataFrame, y: np.ndarray,
              test_size: float = 0.2, verbose: bool = True) -> Dict:
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        if verbose:
            print("=" * 70)
            print(f"🤖 TRAINING PHISHING DETECTION MODEL")
            print("=" * 70)
            print(f"📊 Training set: {X_train.shape}")
            print(f"📊 Testing set:  {X_test.shape}")
            print(f"🏷️  Class balance: {np.bincount(y_train)}")
            print(f"🧠 Using: {'XGBoost + RF Ensemble' if XGBOOST_AVAILABLE else 'RandomForest only'}")

        self.pipeline = self._build_pipeline()

        if verbose:
            print("\n⏳ Training in progress (may take 1-3 minutes)...")
        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
            "f1_score":  round(f1_score(y_test, y_pred), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "n_train": int(len(X_train)),
            "n_test":  int(len(X_test)),
            "n_features": X.shape[1],
            "model_type": "XGBoost+RF Ensemble" if XGBOOST_AVAILABLE else "RandomForest",
            "feature_names": self.feature_names,
        }

        # Invalidate importance cache after retraining
        self._importances_cache = None
        self._importances_raw = None

        if verbose:
            self._print_report(y_test, y_pred)

        return self.metrics

    def _build_pipeline(self) -> Pipeline:
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=2,
            n_jobs=-1, random_state=42, class_weight="balanced",
        )

        if XGBOOST_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", n_jobs=-1,
                random_state=42, verbosity=0,
            )
            estimator = VotingClassifier(
                estimators=[("rf", rf), ("xgb", xgb)],
                voting="soft", weights=[1, 2], n_jobs=-1,
            )
        else:
            estimator = rf

        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ])

    def _print_report(self, y_test, y_pred):
        m = self.metrics
        print("\n" + "=" * 70)
        print("📈 MODEL PERFORMANCE")
        print("=" * 70)
        print(f"  Accuracy:  {m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1-Score:  {m['f1_score']:.4f}")
        print(f"  ROC-AUC:   {m['roc_auc']:.4f}")
        print("\n📊 Confusion Matrix:")
        cm = m["confusion_matrix"]
        print(f"              Predicted")
        print(f"              Legit  Phish")
        print(f"    Legit  →  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"    Phish  →  {cm[1][0]:5d}  {cm[1][1]:5d}")
        print("\n📋 Detailed Report:")
        print(classification_report(y_test, y_pred,
              target_names=["Legitimate", "Phishing"]))
        print("=" * 70)

    # ────────────────────────────────────────────────────────────────────
    #  PERSISTENCE
    # ────────────────────────────────────────────────────────────────────
    def save(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("No model to save — call train() first.")
        joblib.dump(self.pipeline, MODEL_PATH)
        with open(FEATURES_PATH, "w") as f:
            json.dump(self.feature_names, f, indent=2)
        with open(METRICS_PATH, "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Model saved to {MODEL_PATH}")
        print(f"\n💾 Model saved to {MODEL_PATH}")
        print(f"💾 Metrics saved to {METRICS_PATH}")

    def load(self) -> bool:
        try:
            self.pipeline = joblib.load(MODEL_PATH)
            with open(FEATURES_PATH) as f:
                self.feature_names = json.load(f)
            with open(METRICS_PATH) as f:
                self.metrics = json.load(f)

            # Phase 8: pre-compute importances on load (one-time cost)
            self._precompute_importances()

            logger.info(f"Model loaded from {MODEL_PATH} ({len(self.feature_names)} features)")
            return True
        except FileNotFoundError:
            logger.warning(f"No trained model found at {MODEL_PATH}")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def is_trained(self) -> bool:
        return self.pipeline is not None

    # ────────────────────────────────────────────────────────────────────
    #  PREDICTION (Hybrid: Rules → ML → Blend)
    # ────────────────────────────────────────────────────────────────────
    def predict(self, features: dict) -> Tuple[str, float, Optional[str]]:
        # Stage 1: Critical rule override
        critical = check_critical_rules(features)
        if critical is not None:
            return ("PHISHING", critical.confidence, critical.description)

        # Stage 2: ML prediction
        if not self.is_trained():
            logger.warning("Model not trained — falling back to rule-only mode")
            return self._rule_only_predict(features)

        try:
            vec = self._features_to_vector(features)
            proba = self.pipeline.predict_proba(vec.reshape(1, -1))[0]
            ml_phish_prob = float(proba[1])
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._rule_only_predict(features)

        # Stage 3: High-rule boosting
        high_rules = check_high_rules(features)
        if high_rules:
            rule_conf = max(r.confidence for r in high_rules)
            if ml_phish_prob > 0.3:
                phish_prob = 0.6 * rule_conf + 0.4 * ml_phish_prob
            else:
                phish_prob = rule_conf * 0.85
            reason = high_rules[0].description
        else:
            phish_prob = ml_phish_prob
            reason = None

        label = "PHISHING" if phish_prob > 0.5 else "LEGITIMATE"
        confidence = round(phish_prob if label == "PHISHING" else 1 - phish_prob, 3)
        return (label, confidence, reason)

    def predict_proba(self, features: dict) -> Dict[str, float]:
        if not self.is_trained():
            critical = check_critical_rules(features)
            if critical:
                return {"legitimate": 1 - critical.confidence,
                        "phishing": critical.confidence}
            return {"legitimate": 0.5, "phishing": 0.5}

        try:
            vec = self._features_to_vector(features)
            proba = self.pipeline.predict_proba(vec.reshape(1, -1))[0]
            return {"legitimate": round(float(proba[0]), 4),
                    "phishing":   round(float(proba[1]), 4)}
        except Exception as e:
            logger.error(f"predict_proba failed: {e}")
            return {"legitimate": 0.5, "phishing": 0.5}

    def _rule_only_predict(self, features: dict) -> Tuple[str, float, Optional[str]]:
        critical = check_critical_rules(features)
        if critical:
            return ("PHISHING", critical.confidence, critical.description)
        high = check_high_rules(features)
        if high:
            return ("PHISHING", high[0].confidence, high[0].description)
        return ("LEGITIMATE", 0.6, None)

    # ────────────────────────────────────────────────────────────────────
    #  FEATURE IMPORTANCE (PHASE 8: CACHED)
    # ────────────────────────────────────────────────────────────────────
    def _precompute_importances(self):
        """Compute raw importances once at load time (expensive for VotingClassifier)."""
        if not self.is_trained():
            return

        try:
            clf = self.pipeline.named_steps["clf"]

            if hasattr(clf, "estimators_"):
                importances_list = []
                for est in clf.estimators_:
                    if hasattr(est, "feature_importances_"):
                        importances_list.append(est.feature_importances_)
                if importances_list:
                    self._importances_raw = np.mean(importances_list, axis=0)
            elif hasattr(clf, "feature_importances_"):
                self._importances_raw = clf.feature_importances_

        except Exception as e:
            logger.error(f"Importance pre-computation failed: {e}")
            self._importances_raw = None

    def get_feature_importances(self, top_n: int = 10) -> Dict[str, float]:
        """
        Return top N most important features. Cached after first call.
        """
        if not self.is_trained():
            return {}

        # Compute raw importances if not yet cached
        if self._importances_raw is None:
            self._precompute_importances()

        if self._importances_raw is None:
            return {}

        # Cache key = top_n (different N → different result)
        cache_key = f"top_{top_n}"
        if self._importances_cache and cache_key in self._importances_cache:
            return self._importances_cache[cache_key]

        try:
            paired = sorted(
                zip(self.feature_names, self._importances_raw),
                key=lambda x: x[1], reverse=True
            )
            result = {name: round(float(imp), 4) for name, imp in paired[:top_n]}

            if self._importances_cache is None:
                self._importances_cache = {}
            self._importances_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            return {}

    # ────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ────────────────────────────────────────────────────────────────────
    def _features_to_vector(self, features: dict) -> np.ndarray:
        """Convert feature dict → ordered numpy vector for ML input."""
        return np.array([features.get(name, 0) for name in self.feature_names],
                        dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════
#  Singleton pattern
# ══════════════════════════════════════════════════════════════════════════
_model_instance: Optional[PhishingMLModel] = None


def get_model() -> PhishingMLModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = PhishingMLModel()
        _model_instance.load()
    return _model_instance


# ══════════════════════════════════════════════════════════════════════════
#  Sanity test
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time

    print("=" * 70)
    print("🧪 ML MODEL — PHASE 8 TEST")
    print("=" * 70)

    model = PhishingMLModel()
    if model.load():
        print("\n✅ Loaded existing model from disk")
        print(f"   Feature count: {len(model.feature_names)}")
        print(f"   Accuracy: {model.metrics.get('accuracy', 'N/A')}")

        test_features = {
            "url_length": 50, "num_dots": 3, "has_ip_address": 1,
            "uses_https": 0, "has_suspicious_keywords": 1,
        }

        # Benchmark predict
        t0 = time.perf_counter()
        label, conf, reason = model.predict(test_features)
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"\n🔎 Prediction took {dt_ms:.2f} ms")
        print(f"   Label: {label}  | Conf: {conf}  | Reason: {reason}")

        # Benchmark importance (cached)
        print("\n⚡ Feature importance timing:")
        t0 = time.perf_counter()
        _ = model.get_feature_importances(10)
        dt1 = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        _ = model.get_feature_importances(10)
        dt2 = (time.perf_counter() - t0) * 1000
        print(f"   1st call: {dt1:.3f} ms")
        print(f"   2nd call: {dt2:.3f} ms  ← should be ~0 (cached)")

        print(f"\n🔝 Top 5 features:")
        for name, imp in list(model.get_feature_importances(5).items()):
            print(f"   {name:30s} {imp:.4f}")
    else:
        print("\n⚠️  No trained model found. Run: python -m backend.train --sample 10000")

MLModel = PhishingMLModel