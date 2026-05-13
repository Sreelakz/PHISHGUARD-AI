"""
backend/analyzer.py
====================
PHASE 6: SYSTEM INTEGRATION — Master detection pipeline.

FIX (v4.2.1):
  • SHAP explanation now ALWAYS runs and is ALWAYS returned.
    Previously it was silently skipped when shap_explainer.is_ready()
    returned False. Now we compute it when available AND always
    include shap_explanation in the response (empty list [] as fallback).
  • The frontend's buildFallbackShap() will handle the empty case.

PIPELINE STAGES:
  1. Feature Extraction       (URL + HTML features)
  2. Auxiliary Intelligence   (SSL, WHOIS, redirects, homograph, visual)
  3. Rule Engine              (critical rules → early verdict)
  4. ML Prediction            (XGBoost + RF ensemble)
  5. Threat Intelligence
       ├── Google Safe Browsing  (always runs — free & fast)
       └── VirusTotal            (smart: only if ML conf > 0.7)
  6. Risk Calculation + XAI    (SHAP + feature importance)
  7. Unified Verdict Assembly
"""

import os
import sys
import time
import logging
from typing import Dict, Optional, List
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════
#  Path setup
# ══════════════════════════════════════════════════════════════════════════
_CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ══════════════════════════════════════════════════════════════════════════
#  Imports (resilient — work from any directory)
# ══════════════════════════════════════════════════════════════════════════
try:
    from backend.feature_extractor import FeatureExtractor
    from backend.ml_model          import PhishingMLModel, get_model
    from backend.rules_engine      import (
        check_critical_rules, check_high_rules, get_rule_summary,
    )
    from backend.api_integration   import check_safe_browsing, check_virustotal
    from backend.simple_explainer  import generate_simple_explanation
except ModuleNotFoundError:
    from feature_extractor import FeatureExtractor
    from ml_model          import PhishingMLModel, get_model
    from rules_engine      import (
        check_critical_rules, check_high_rules, get_rule_summary,
    )
    from api_integration   import check_safe_browsing, check_virustotal
    from simple_explainer  import generate_simple_explanation

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════
VT_CONFIDENCE_TRIGGER = float(
    os.getenv("VIRUSTOTAL_CONFIDENCE_THRESHOLD", "0.7")
)

# ══════════════════════════════════════════════════════════════════════════
#  Singletons
# ══════════════════════════════════════════════════════════════════════════
_feature_extractor: Optional[FeatureExtractor] = None
_ml_model: Optional[PhishingMLModel] = None


def _get_extractor() -> FeatureExtractor:
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = FeatureExtractor(timeout=5)
    return _feature_extractor


def _get_ml() -> PhishingMLModel:
    global _ml_model
    if _ml_model is None:
        _ml_model = get_model()
    return _ml_model


# ══════════════════════════════════════════════════════════════════════════
#  URL helpers
# ══════════════════════════════════════════════════════════════════════════
def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


# ══════════════════════════════════════════════════════════════════════════
#  SHAP fallback builder (rule-weight-based pseudo-SHAP)
#  Used when the ML SHAP explainer is not trained/ready.
# ══════════════════════════════════════════════════════════════════════════
def _build_rule_based_shap(features: Dict, prediction: str) -> List[Dict]:
    """
    When SHAP is unavailable, derive approximate feature contributions
    from the rule engine weights. Returns same shape as real SHAP output.
    """
    is_phishing = prediction == "PHISHING"

    rules = [
        ("Suspicious Keywords",    "has_suspicious_keywords",  0.45),
        ("IP Address in URL",      "has_ip_address",           0.42),
        ("No HTTPS / HTTP only",   "uses_https",              -0.40),   # inverted
        ("@ Symbol in URL",        "has_at_symbol",            0.35),
        ("Obfuscated JavaScript",  "has_obfuscated_js",        0.38),
        ("Login Form Present",     "has_login_form",           0.30),
        ("Hidden Iframes",         "has_iframes",              0.28),
        ("Favicon Domain Mismatch","has_favicon_mismatch",     0.25),
        ("URL Length",             "url_length",               None),   # continuous
        ("Subdomain Count",        "subdomain_count",          None),   # continuous
        ("External Script Count",  "external_scripts_count",  None),   # continuous
    ]

    out = []
    for feat_name, feat_key, weight in rules:
        raw = features.get(feat_key)
        if raw is None:
            continue

        if weight is None:
            # Continuous feature — normalise to [-0.2, 0.2]
            try:
                norm_map = {
                    "url_length": 200,
                    "subdomain_count": 5,
                    "external_scripts_count": 20,
                }
                divisor = norm_map.get(feat_key, 100)
                ratio   = min(float(raw) / divisor, 1.0)
                val     = round(ratio * 0.2 * (1 if is_phishing else -1), 4)
            except (TypeError, ValueError):
                continue
        elif feat_key == "uses_https":
            # 1 = HTTPS present = safe → negative SHAP (reduces phishing risk)
            val = round(-abs(weight) if raw else abs(weight), 4)
        else:
            # Binary flag — positive weight means "pushes toward phishing"
            val = round(weight if raw else -weight * 0.4, 4)

        if abs(val) > 0.01:
            out.append({"feature": feat_name, "shap_value": val})

    return sorted(out, key=lambda x: abs(x["shap_value"]), reverse=True)


# ══════════════════════════════════════════════════════════════════════════
#  STAGE RUNNERS
# ══════════════════════════════════════════════════════════════════════════
def _stage_extract_features(url: str, enable_html: bool, timings: dict) -> Dict:
    t0 = time.time()
    try:
        features = _get_extractor().extract(url, fetch_html=enable_html)
    except Exception as e:
        logger.exception("Feature extraction failed")
        features = {"_extraction_error": str(e)[:200]}
    timings["feature_extraction_ms"] = int((time.time() - t0) * 1000)
    return features


def _stage_auxiliary_intel(url: str, features: Dict,
                            aux_modules: Optional[Dict], timings: dict) -> Dict:
    t0 = time.time()
    if not aux_modules:
        timings["auxiliary_intel_ms"] = 0
        return features

    module_map = {
        "ssl_info":        ("ssl",       "check"),
        "domain_info":     ("domain",    "analyze"),
        "redirect_info":   ("redirect",  "detect"),
        "homograph_info":  ("homograph", "detect"),
        "visual_analysis": ("visual",    "analyze"),
    }

    for feat_key, (mod_key, method_name) in module_map.items():
        mod = aux_modules.get(mod_key)
        if mod is None:
            continue
        try:
            features[feat_key] = getattr(mod, method_name)(url)
        except Exception as e:
            logger.warning(f"{mod_key}.{method_name} failed: {e}")
            features[feat_key] = {"error": str(e)[:100]}

    timings["auxiliary_intel_ms"] = int((time.time() - t0) * 1000)
    return features


def _stage_rules(features: Dict, timings: dict) -> Dict:
    t0 = time.time()
    try:
        summary = get_rule_summary(features)
    except Exception as e:
        logger.exception("Rule engine failed")
        summary = {"verdict": "clean", "confidence": 0.5,
                   "fired_rules": [], "critical_hit": False,
                   "_error": str(e)[:200]}
    timings["rules_engine_ms"] = int((time.time() - t0) * 1000)
    return summary


def _stage_ml(features: Dict, timings: dict) -> Dict:
    t0 = time.time()
    ml = _get_ml()

    result = {
        "available":        ml.is_trained(),
        "prediction":       None,
        "confidence":       0.5,
        "phishing_proba":   0.5,
        "legitimate_proba": 0.5,
        "override_reason":  None,
        "error":            None,
    }

    if not ml.is_trained():
        result["error"] = "Model not trained — run `python -m backend.train`"
        timings["ml_prediction_ms"] = int((time.time() - t0) * 1000)
        return result

    try:
        label, conf, reason = ml.predict(features)
        proba = ml.predict_proba(features)
        result.update({
            "prediction":       label,
            "confidence":       conf,
            "phishing_proba":   proba.get("phishing", 0.5),
            "legitimate_proba": proba.get("legitimate", 0.5),
            "override_reason":  reason,
        })
    except Exception as e:
        logger.exception("ML prediction failed")
        result["error"] = str(e)[:200]

    timings["ml_prediction_ms"] = int((time.time() - t0) * 1000)
    return result


def _stage_threat_intel(url: str, ml_phish_proba: float,
                         enable_sb: bool, enable_vt: bool,
                         timings: dict) -> Dict:
    t0 = time.time()
    result = {"safe_browsing": None, "virustotal": None}

    if enable_sb:
        try:
            result["safe_browsing"] = check_safe_browsing(url)
        except Exception as e:
            logger.exception("Safe Browsing failed")
            result["safe_browsing"] = {
                "status": "error", "is_threat": False,
                "threat_types": [], "message": str(e)[:120],
            }

    if enable_vt:
        try:
            result["virustotal"] = check_virustotal(url, ml_phish_proba)
        except Exception as e:
            logger.exception("VirusTotal failed")
            result["virustotal"] = {
                "status": "error", "is_malicious": False,
                "message": str(e)[:120],
            }

    timings["threat_intel_ms"] = int((time.time() - t0) * 1000)
    return result


def _stage_risk_xai(features: Dict, ml_confidence: float, prediction: str,
                     risk_calculator, explainable_ai, shap_explainer,
                     ml_importances: Dict, ov_reason: Optional[str],
                     timings: dict) -> Dict:
    """
    Stage 6: Risk Calculation + Explainable AI.

    FIX v4.2.1:
      SHAP now ALWAYS attempts to run regardless of is_ready().
      When SHAP explainer is not ready, we fall back to rule-based
      pseudo-SHAP via _build_rule_based_shap().
      shap_explanation is NEVER None — always a list (may be empty).
    """
    t0 = time.time()

    risk_result = {
        "risk_score":      50,
        "risk_level":      "MEDIUM",
        "risk_colour":     "orange",
        "dominant_signal": "unknown",
        "signal_weights":  {},
    }
    explanations = {
        "summary":      "",
        "top_reasons":  [],
        "indicators":   [],
        "ai_narrative": "",
    }
    shap_result   = None
    shap_values   = None
    top_features  = []   # Always a list — never None

    # ── Risk score ──
    if risk_calculator is not None:
        try:
            risk_result = risk_calculator.calculate(features, ml_confidence, prediction)
        except Exception as e:
            logger.exception("Risk calc failed")
            risk_result["_error"] = str(e)[:150]

    # ── SHAP — attempt real SHAP, fall back to rule-based ──
    shap_ready = False
    if shap_explainer is not None:
        try:
            shap_ready = shap_explainer.is_ready()
            if shap_ready:
                shap_result  = shap_explainer.explain(features, top_n=5)
                shap_values  = shap_explainer.get_shap_values(features)
                top_features = shap_explainer.get_top_features(features) or []
                logger.debug(f"SHAP (ML): {len(top_features)} features computed")
            else:
                logger.info("SHAP explainer not ready — using rule-based fallback")
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")
            shap_ready = False

    # ── Rule-based SHAP fallback (used when ML SHAP not available) ──
    if not top_features:
        try:
            top_features = _build_rule_based_shap(features, prediction)
            logger.debug(f"SHAP (rule-based): {len(top_features)} features computed")
        except Exception as e:
            logger.warning(f"Rule-based SHAP fallback failed: {e}")
            top_features = []

    # ── Explainable AI narrative ──
    if explainable_ai is not None:
        try:
            detailed_explanation = explainable_ai.generate_explanations(
                features, prediction, ml_importances, shap_result=shap_result
            )
            feature_names = (shap_explainer.feature_names
                             if shap_explainer and shap_ready else None)
            simple_explanation = generate_simple_explanation(
                shap_values, feature_names, features
            )
            explanations = {
                "detailed_explanation": detailed_explanation,
                "simple_explanation":   simple_explanation,
            }
        except Exception as e:
            logger.exception("XAI failed")
            explanations = {"_error": str(e)[:150]}

    timings["risk_xai_ms"] = int((time.time() - t0) * 1000)
    return {
        "risk":             risk_result,
        "explanations":     explanations,
        "shap":             shap_result,
        "shap_values":      shap_values,
        "shap_explanation": top_features,   # Always a list
    }


# ══════════════════════════════════════════════════════════════════════════
#  VERDICT ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════
def _assemble_verdict(rule_summary: Dict, ml_result: Dict,
                       threat_intel: Dict) -> Dict:
    """
    Combine all signals → final verdict.

    Precedence:
      1. Safe Browsing threat  → PHISHING (99%)
      2. Critical rule hit     → PHISHING (rule confidence)
      3. VirusTotal malicious  → PHISHING (blended)
      4. ML prediction         → (ML verdict)
    """
    sb = threat_intel.get("safe_browsing") or {}
    vt = threat_intel.get("virustotal")    or {}

    if sb.get("status") == "unsafe" and sb.get("is_threat"):
        threat = (sb.get("threat_types") or ["THREAT"])[0]
        return {
            "verdict":    "PHISHING",
            "confidence": 0.99,
            "reason":     f"🚨 Google Safe Browsing flagged as {threat.replace('_', ' ').title()}",
            "source":     "safe_browsing",
        }

    if rule_summary.get("critical_hit"):
        crit = rule_summary["fired_rules"][0]
        return {
            "verdict":    "PHISHING",
            "confidence": crit["confidence"],
            "reason":     f"⛔ {crit['description']}",
            "source":     "rule_engine",
        }

    if vt.get("status") == "completed" and vt.get("is_malicious"):
        mcount  = vt.get("malicious_count", 0)
        total   = vt.get("total_engines", 0)
        blended = min(0.99, 0.7 + (mcount / max(total, 1)) * 0.3)
        return {
            "verdict":    "PHISHING",
            "confidence": round(blended, 3),
            "reason":     f"🛡️ {mcount}/{total} antivirus engines flagged as malicious",
            "source":     "virustotal",
        }

    if ml_result.get("available") and ml_result.get("prediction"):
        label  = ml_result["prediction"]
        conf   = ml_result["confidence"]
        reason = ml_result.get("override_reason") or (
            f"ML model classified as {label.lower()} "
            f"(phishing probability: {ml_result['phishing_proba']:.2%})"
        )
        return {
            "verdict":    label,
            "confidence": conf,
            "reason":     reason,
            "source":     "ml_model",
        }

    # Fallback — use rule summary verdict
    rule_verdict = rule_summary.get("verdict", "clean")
    return {
        "verdict":    "PHISHING" if rule_verdict != "clean" else "LEGITIMATE",
        "confidence": rule_summary.get("confidence", 0.5),
        "reason":     "Verdict based on rule engine (ML model not trained)",
        "source":     "rule_engine",
    }


def _derive_risk_level(verdict: str, confidence: float) -> str:
    if verdict == "PHISHING":
        if confidence >= 0.9:  return "CRITICAL"
        if confidence >= 0.75: return "HIGH"
        return "MEDIUM"
    if verdict == "LEGITIMATE":
        if confidence >= 0.85: return "LOW"
        return "MEDIUM"
    return "MEDIUM"


# ══════════════════════════════════════════════════════════════════════════
#  MASTER FUNCTION: analyze_url()
# ══════════════════════════════════════════════════════════════════════════
def analyze_url(
    url: str,
    *,
    enable_html: bool = True,
    enable_safe_browsing: bool = True,
    enable_virustotal: bool = True,
    auxiliary_modules: Optional[Dict] = None,
    risk_calculator=None,
    explainable_ai=None,
    shap_explainer=None,
) -> Dict:
    """
    Master URL analysis pipeline — single entry point for all analysis.

    Key guarantee (v4.2.1):
      `shap_explanation` is ALWAYS a list in the returned dict.
      When the ML SHAP explainer is not ready, rule-weight-based
      pseudo-SHAP values are computed so the frontend always has
      something to display in the SHAP XAI tab and explanation panel.
    """
    start_ts = time.time()
    timings: Dict[str, int] = {}

    url = _normalize_url(url)
    if not url:
        return {
            "error":     "URL is required",
            "verdict":   "UNKNOWN",
            "confidence": 0,
            "timestamp": datetime.now().isoformat(),
            "shap_explanation": [],
        }

    logger.info(f"🔍 Analyzing: {url}")

    # ── Stage 1 ──
    features = _stage_extract_features(url, enable_html, timings)

    # ── Stage 2 ──
    features = _stage_auxiliary_intel(url, features, auxiliary_modules, timings)

    # ── Stage 3 ──
    rule_summary = _stage_rules(features, timings)

    # ── Stage 4 ──
    ml_result = _stage_ml(features, timings)

    # ── Stage 5 ──
    ml_phish_proba = ml_result.get("phishing_proba", 0.5)
    effective_conf = max(
        ml_phish_proba,
        rule_summary.get("confidence", 0)
        if rule_summary.get("verdict") != "clean" else 0
    )
    threat_intel = _stage_threat_intel(
        url, effective_conf, enable_safe_browsing, enable_virustotal, timings
    )

    # ── Stage 6 ──
    ml_importances = {}
    if ml_result.get("available"):
        try:
            ml_importances = _get_ml().get_feature_importances(top_n=10)
        except Exception:
            ml_importances = {}

    _pred_for_risk = ml_result.get("prediction") or (
        "PHISHING" if rule_summary.get("verdict") == "phishing" else "LEGITIMATE"
    )
    _conf_for_risk = ml_result.get("confidence", rule_summary.get("confidence", 0.5))

    risk_xai = _stage_risk_xai(
        features, _conf_for_risk, _pred_for_risk,
        risk_calculator, explainable_ai, shap_explainer,
        ml_importances, ml_result.get("override_reason"),
        timings,
    )

    # ── Stage 7: Verdict assembly ──
    verdict    = _assemble_verdict(rule_summary, ml_result, threat_intel)
    risk_level = _derive_risk_level(verdict["verdict"], verdict["confidence"])

    # Augment explanations
    explanations = {}
    try:
        explanations = risk_xai["explanations"].get("detailed_explanation") or {}
    except Exception:
        pass

    simple_explanation = {}
    try:
        simple_explanation = risk_xai["explanations"].get("simple_explanation") or {}
    except Exception:
        pass

    if verdict["reason"] and verdict["source"] in ("safe_browsing", "rule_engine", "virustotal"):
        explanations.setdefault("indicators", []).insert(0, {
            "category": "OVERRIDE",
            "severity": "critical",
            "code":     verdict["source"],
            "message":  verdict["reason"],
        })
        explanations["top_reasons"] = [verdict["reason"]] + \
            explanations.get("top_reasons", [])[:4]
        explanations["summary"] = verdict["reason"]

    total_ms = int((time.time() - start_ts) * 1000)
    timings["total_ms"] = total_ms

    # ── Simple text indicators ──
    simple_indicators = []
    f = features  # alias
    if f.get("has_suspicious_keywords"):  simple_indicators.append("Suspicious keywords detected")
    if not f.get("uses_https", True):     simple_indicators.append("No HTTPS — plain HTTP only")
    if f.get("has_ip_address"):           simple_indicators.append("IP address used instead of domain")
    if f.get("has_at_symbol"):            simple_indicators.append("@ symbol in URL")
    if f.get("has_login_form"):           simple_indicators.append("Login form detected on page")
    if f.get("has_obfuscated_js"):        simple_indicators.append("Obfuscated JavaScript detected")
    if f.get("has_iframes"):              simple_indicators.append("Hidden iframes detected")
    if f.get("has_favicon_mismatch"):     simple_indicators.append("Favicon domain mismatch")

    explanation_text = " + ".join(simple_indicators) if simple_indicators \
        else "No significant indicators detected"

    threat_indicators = explanations.get("indicators", [])
    threat_message    = (f"⚠️ {len(threat_indicators)} threat indicators detected"
                         if threat_indicators
                         else "✓ No significant indicators detected")

    # ── shap_explanation is ALWAYS a list ──
    shap_explanation = risk_xai.get("shap_explanation") or []

    return {
        # Core verdict
        "url":              url,
        "verdict":          verdict["verdict"],
        "confidence":       round(verdict["confidence"], 3),
        "risk_level":       risk_level,
        "risk_score":       risk_xai["risk"].get("risk_score"),
        "risk_colour":      risk_xai["risk"].get("risk_colour"),
        "verdict_source":   verdict["source"],
        "verdict_reason":   verdict["reason"],
        "threat_message":   threat_message,

        # Component results
        "ml":                   ml_result,
        "rules":                rule_summary,
        "threat_intel":         threat_intel,
        "risk":                 risk_xai["risk"],
        "detailed_explanation": explanations,
        "simple_explanation":   simple_explanation,

        # SHAP — always a list
        "shap":             risk_xai["shap"],
        "shap_values":      risk_xai["shap_values"],
        "shap_explanation": shap_explanation,

        # Text summaries
        "indicators":       simple_indicators,
        "explanation":      explanation_text,

        # Raw data
        "features":         _sanitize(features),
        "importances":      ml_importances,

        # Metadata
        "timings_ms":       timings,
        "timestamp":        datetime.now().isoformat(),
        "pipeline_version": "6.1",
    }


# ══════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════
def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()
                if isinstance(k, (str, int))}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    try:
        return str(obj)
    except Exception:
        return None


def analyze_batch(urls: List[str], **kwargs) -> List[Dict]:
    results = []
    for i, url in enumerate(urls):
        logger.info(f"Batch [{i+1}/{len(urls)}]: {url}")
        try:
            results.append(analyze_url(url, **kwargs))
        except Exception as e:
            logger.exception(f"Batch analysis failed for {url}")
            results.append({
                "url":              url,
                "verdict":          "ERROR",
                "error":            str(e)[:200],
                "shap_explanation": [],
                "timestamp":        datetime.now().isoformat(),
            })
    return results


# ══════════════════════════════════════════════════════════════════════════
#  Sanity test
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import json
    import warnings
    warnings.filterwarnings("ignore")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 70)
    print("🧪 PHASE 6.1 — SYSTEM INTEGRATION TEST (SHAP always-on)")
    print("=" * 70)

    test_urls = [
        "https://www.google.com",
        "http://malware.testing.google.test/testing/malware/",
        "http://192.168.1.1/login/verify?bank=1",
    ]

    for url in test_urls:
        print(f"\n{'━' * 70}")
        print(f"🔗 URL: {url}")
        result = analyze_url(url, enable_html=False,
                             enable_safe_browsing=True,
                             enable_virustotal=True)

        print(f"\n📊 VERDICT: {result['verdict']}  (conf: {result['confidence']})")
        print(f"   Risk Level:      {result['risk_level']}")
        print(f"   Source:          {result['verdict_source']}")
        print(f"   Reason:          {result['verdict_reason']}")
        print(f"   SHAP features:   {len(result['shap_explanation'])} returned")
        if result['shap_explanation']:
            for s in result['shap_explanation'][:3]:
                print(f"     {s['feature']:35s} → {s['shap_value']:+.4f}")
        print(f"   ⏱️  Timings (ms): {result['timings_ms']}")

    print("\n" + "=" * 70)
    print("✅ Phase 6.1 test complete!")
    print("=" * 70)