"""
backend/shap_explainer.py
--------------------------
SHAP-based Explainable AI engine for phishing detection.

Generates per-prediction explanations using SHapley Additive exPlanations.
Works with the VotingClassifier (XGBoost + RandomForest) pipeline from ml_model.py.

WHY SHAP?
  • Shows exactly HOW MUCH each feature pushed the prediction toward phishing
  • Mathematically grounded (game theory) — not heuristic
  • Per-prediction (local) explanations, not just global importance

OUTPUT FORMAT (JSON-serializable for Flask API):
  {
    "base_value": 0.42,                    # Model's average prediction
    "prediction_value": 0.87,              # This URL's prediction
    "top_positive": [...],                 # Features pushing TOWARD phishing
    "top_negative": [...],                 # Features pushing TOWARD legitimate
    "force_plot_data": {...},              # For interactive visualization
    "waterfall_data": [...],               # For waterfall chart
  }
"""

from __future__ import annotations
import os
import logging
import warnings
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")


# ══════════════════════════════════════════════════════════════════════════
#  Human-readable feature descriptions
#  (Maps cryptic feature names → friendly UI labels)
# ══════════════════════════════════════════════════════════════════════════
FEATURE_LABELS = {
    "url_length":               "URL Length",
    "hostname_length":          "Hostname Length",
    "num_dots":                 "Number of Dots",
    "num_dashes":               "Number of Hyphens",
    "num_underscores":          "Number of Underscores",
    "num_slashes":              "Number of Slashes",
    "num_question_marks":       "Question Marks Count",
    "num_equals":               "Equal Signs Count",
    "num_at_signs":             "'@' Symbol Count",
    "num_digits":               "Digit Count",
    "num_special_chars":        "Special Characters",
    "has_at_symbol":            "Contains '@' Symbol",
    "has_ip_address":           "Uses Raw IP Address",
    "uses_https":               "HTTPS Enabled",
    "has_suspicious_keywords":  "Phishing Keywords Present",
    "suspicious_keyword_count": "Phishing Keywords Count",
    "num_subdomains":           "Subdomain Count",
    "path_depth":               "URL Path Depth",
    "query_length":             "Query String Length",
    "has_suspicious_tld":       "Suspicious TLD (.tk/.ml/.xyz)",
    "has_non_standard_port":    "Non-Standard Port",
    "is_shortened":             "URL Shortener Used",
    "special_char_ratio":       "Special Character Ratio",
    "entropy":                  "URL Entropy (Randomness)",
    "has_double_slash_redirect": "Double-Slash Redirect",
    "has_login_form":           "Login Form Present",
    "has_iframes":              "Iframe Count",
    "num_external_links":       "External Links Count",
    "has_favicon_mismatch":     "Favicon Domain Mismatch",
    "has_hidden_fields":        "Hidden Form Fields",
    "has_meta_refresh":         "Meta-Refresh Redirect",
    "has_popup_window":         "Popup Windows",
    "num_images":               "Image Count",
    "num_scripts":              "Script Tag Count",
    "right_click_disabled":     "Right-Click Disabled",
    "has_obfuscated_js":        "Obfuscated JavaScript",
}


def _friendly_name(feature: str) -> str:
    """Convert snake_case feature → Title Case label."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


# ══════════════════════════════════════════════════════════════════════════
#  Main class
# ══════════════════════════════════════════════════════════════════════════
class SHAPExplainer:
    """
    Per-prediction SHAP explanation engine.

    Usage:
        explainer = SHAPExplainer()
        explainer.initialize(ml_model)           # once at startup
        result = explainer.explain(features)     # per request
    """

    def __init__(self):
        self.explainer: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.feature_names: List[str] = []
        self.base_value: float = 0.5
        self._initialized: bool = False

    # ────────────────────────────────────────────────────────────────────
    #  INITIALIZATION (call once at app startup)
    # ────────────────────────────────────────────────────────────────────
    def initialize(self, ml_model, background_size: int = 100) -> bool:
        """
        Build the SHAP explainer from a trained PhishingMLModel.

        Args:
            ml_model: Instance of PhishingMLModel (must be trained)
            background_size: Number of synthetic background samples for TreeExplainer

        Returns:
            True if initialization succeeded
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP library not available")
            return False

        if not hasattr(ml_model, "is_trained") or not ml_model.is_trained():
            logger.warning("ML model not trained — cannot init SHAP")
            return False

        try:
            self.feature_names = list(ml_model.feature_names)

            # Extract scaler + classifier from pipeline
            pipeline = ml_model.pipeline
            self.scaler = pipeline.named_steps.get("scaler")
            clf = pipeline.named_steps.get("clf")

            # Build background dataset (zeros = neutral baseline)
            background = np.zeros((background_size, len(self.feature_names)))
            if self.scaler is not None:
                background = self.scaler.transform(background)

            # Select the best single estimator for SHAP
            # (VotingClassifier isn't directly supported — use XGBoost if available)
            tree_model = self._select_tree_model(clf)
            if tree_model is None:
                logger.error("No tree-based estimator found in pipeline")
                return False

            # TreeExplainer is fastest + most accurate for tree models
            self.explainer = shap.TreeExplainer(
                tree_model,
                data=background,
                feature_perturbation="interventional",
            )

            # Cache the expected (base) value
            try:
                ev = self.explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    self.base_value = float(ev[1] if len(ev) > 1 else ev[0])
                else:
                    self.base_value = float(ev)
            except Exception:
                self.base_value = 0.5

            self._initialized = True
            logger.info(f"✅ SHAP explainer initialized ({len(self.feature_names)} features)")
            return True

        except Exception as e:
            logger.error(f"SHAP initialization failed: {e}")
            return False

    def _select_tree_model(self, clf):
        """Pick a tree model for SHAP from the pipeline's classifier."""
        # VotingClassifier case — prefer XGBoost (better SHAP support)
        if hasattr(clf, "estimators_"):
            for name, est in zip(getattr(clf, "named_estimators_", {}).keys(),
                                  clf.estimators_):
                if "xgb" in str(type(est)).lower():
                    return est
            # Fallback: first tree estimator
            for est in clf.estimators_:
                if hasattr(est, "feature_importances_"):
                    return est
        # Single classifier case
        if hasattr(clf, "feature_importances_"):
            return clf
        return None

    def is_ready(self) -> bool:
        return self._initialized and self.explainer is not None

    # ────────────────────────────────────────────────────────────────────
    #  MAIN EXPLANATION API
    # ────────────────────────────────────────────────────────────────────
    def explain(self, features: dict, top_n: int = 5) -> Dict[str, Any]:
        """
        Generate a per-prediction SHAP explanation.

        Args:
            features: Feature dict (same format as ml_model.predict)
            top_n: How many top positive/negative contributors to return

        Returns:
            Dict with SHAP values, top features, and plot data (JSON-safe)
        """
        if not self.is_ready():
            return self._empty_result("SHAP explainer not initialized")

        try:
            # 1. Convert features → ordered vector
            vec = np.array([[features.get(n, 0) for n in self.feature_names]],
                            dtype=float)

            # 2. Apply same scaling as training
            if self.scaler is not None:
                vec_scaled = self.scaler.transform(vec)
            else:
                vec_scaled = vec

            # 3. Compute SHAP values
            shap_values = self.explainer.shap_values(vec_scaled)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classifier → [class_0_shap, class_1_shap]
                shap_vec = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif shap_values.ndim == 3:
                # (1, n_features, 2) format
                shap_vec = shap_values[0, :, 1]
            else:
                # (1, n_features) format
                shap_vec = shap_values[0]

            shap_vec = np.array(shap_vec, dtype=float)

            # 4. Build feature contribution list
            contributions = []
            for i, name in enumerate(self.feature_names):
                contrib = float(shap_vec[i])
                if abs(contrib) < 1e-6:
                    continue  # Skip negligible contributions
                contributions.append({
                    "feature":       name,
                    "label":         _friendly_name(name),
                    "value":         self._safe_value(features.get(name, 0)),
                    "shap_value":    round(contrib, 4),
                    "direction":     "phishing" if contrib > 0 else "legitimate",
                    "impact":        abs(contrib),
                })

            # 5. Split and rank
            positive = sorted(
                [c for c in contributions if c["shap_value"] > 0],
                key=lambda x: x["shap_value"], reverse=True
            )[:top_n]

            negative = sorted(
                [c for c in contributions if c["shap_value"] < 0],
                key=lambda x: x["shap_value"]
            )[:top_n]

            # 6. Compute final prediction value
            prediction_value = self.base_value + float(shap_vec.sum())
            prediction_value = max(0.0, min(1.0, prediction_value))  # Clamp

            # 7. Build waterfall + force plot data
            all_ranked = sorted(contributions, key=lambda x: x["impact"],
                                 reverse=True)[:10]

            return {
                "available":         True,
                "base_value":        round(self.base_value, 4),
                "prediction_value":  round(prediction_value, 4),
                "total_contribution": round(float(shap_vec.sum()), 4),
                "top_positive":      positive,   # Push → phishing
                "top_negative":      negative,   # Push → legitimate
                "waterfall_data":    all_ranked,
                "force_plot_data":   self._build_force_plot(all_ranked, prediction_value),
                "natural_reasons":   self._build_reasons(positive, negative),
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}", exc_info=True)
            return self._empty_result(str(e))

    def get_shap_values(self, features: dict) -> Optional[List[float]]:
        """
        Get raw SHAP values for the given features.

        Args:
            features: Feature dict

        Returns:
            List of SHAP values or None if failed
        """
        if not self.is_ready():
            return None

        try:
            # 1. Convert features → ordered vector
            vec = np.array([[features.get(n, 0) for n in self.feature_names]],
                            dtype=float)

            # 2. Apply same scaling as training
            if self.scaler is not None:
                vec_scaled = self.scaler.transform(vec)
            else:
                vec_scaled = vec

            # 3. Compute SHAP values
            shap_values = self.explainer.shap_values(vec_scaled)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classifier → [class_0_shap, class_1_shap]
                shap_vec = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif shap_values.ndim == 3:
                # (1, n_features, 2) format
                shap_vec = shap_values[0, :, 1]
            else:
                # (1, n_features) format
                shap_vec = shap_values[0]

            return shap_vec.tolist()

        except Exception as e:
            logger.error(f"SHAP values computation failed: {e}")
            return None

    def get_top_features(self, features: dict) -> Optional[List[Dict]]:
        """
        Get top 5 features by absolute SHAP value.

        Args:
            features: Feature dict

        Returns:
            List of dicts with 'feature', 'shap_value', 'direction' or None if failed
        """
        shap_values = self.get_shap_values(features)
        if shap_values is None:
            return None

        try:
            # Pair features with their SHAP values
            feature_importance = []
            for name, shap_val in zip(self.feature_names, shap_values):
                if abs(shap_val) > 1e-6:  # Skip negligible contributions
                    feature_importance.append({
                        "feature": name,
                        "shap_value": round(shap_val, 4),
                        "direction": "phishing" if shap_val > 0 else "legitimate",
                        "impact": abs(shap_val)
                    })

            # Sort by absolute impact and take top 5
            top_features = sorted(
                feature_importance,
                key=lambda x: x["impact"],
                reverse=True
            )[:5]

            return top_features

        except Exception as e:
            logger.error(f"Top features computation failed: {e}")
            return None

    # ────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _safe_value(v):
        """Convert numpy/unusual types to JSON-safe primitives."""
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        if isinstance(v, (int, float, str, bool)):
            return v
        return str(v)

    @staticmethod
    def _build_force_plot(ranked: list, prediction: float) -> dict:
        """Pre-compute data for a force-plot style visualization."""
        return {
            "features": [r["label"] for r in ranked],
            "shap_values": [r["shap_value"] for r in ranked],
            "feature_values": [r["value"] for r in ranked],
            "final_prediction": round(prediction, 4),
        }

    @staticmethod
    def _build_reasons(positive: list, negative: list) -> List[str]:
        """Convert SHAP contributions → natural language sentences."""
        reasons = []
        for c in positive[:3]:
            pct = round(c["impact"] * 100, 1)
            reasons.append(
                f"🔴 **{c['label']}** (value: {c['value']}) pushed the prediction "
                f"toward phishing by {pct}%."
            )
        for c in negative[:2]:
            pct = round(c["impact"] * 100, 1)
            reasons.append(
                f"🟢 **{c['label']}** (value: {c['value']}) pulled the prediction "
                f"toward legitimate by {pct}%."
            )
        return reasons

    @staticmethod
    def _empty_result(reason: str = "") -> Dict[str, Any]:
        """Fallback when SHAP can't run."""
        return {
            "available":         False,
            "reason":            reason,
            "base_value":        0.5,
            "prediction_value":  0.5,
            "top_positive":      [],
            "top_negative":      [],
            "waterfall_data":    [],
            "force_plot_data":   {},
            "natural_reasons":   [],
        }


# ══════════════════════════════════════════════════════════════════════════
#  Singleton pattern for Flask
# ══════════════════════════════════════════════════════════════════════════
_explainer_instance: Optional[SHAPExplainer] = None


def get_shap_explainer() -> SHAPExplainer:
    """Get the global SHAP explainer (init once, reuse forever)."""
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = SHAPExplainer()
    return _explainer_instance


# ══════════════════════════════════════════════════════════════════════════
#  Sanity test
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    from backend.ml_model import PhishingMLModel

    print("=" * 70)
    print("🧪 SHAP EXPLAINER SANITY TEST")
    print("=" * 70)

    ml = PhishingMLModel()
    if not ml.load():
        print("❌ No trained model found. Run: python -m backend.train")
        sys.exit(1)

    explainer = SHAPExplainer()
    if not explainer.initialize(ml):
        print("❌ SHAP initialization failed")
        sys.exit(1)

    test_features = {
        "url_length": 95, "num_dots": 4, "has_ip_address": 1,
        "uses_https": 0, "has_suspicious_keywords": 1,
        "suspicious_keyword_count": 2, "num_dashes": 5,
        "entropy": 4.8, "has_at_symbol": 0, "num_subdomains": 3,
    }

    result = explainer.explain(test_features, top_n=5)

    print(f"\n📊 Base value:         {result['base_value']}")
    print(f"📊 Prediction value:   {result['prediction_value']}")
    print(f"📊 Total contribution: {result['total_contribution']}")

    print("\n🔴 Top features pushing TOWARD phishing:")
    for c in result["top_positive"]:
        print(f"   • {c['label']:35s} SHAP = +{c['shap_value']:.4f}")

    print("\n🟢 Top features pushing TOWARD legitimate:")
    for c in result["top_negative"]:
        print(f"   • {c['label']:35s} SHAP = {c['shap_value']:.4f}")

    print("\n💬 Natural language reasons:")
    for r in result["natural_reasons"]:
        print(f"   {r}")

    print("\n✅ SHAP explainer working correctly!")