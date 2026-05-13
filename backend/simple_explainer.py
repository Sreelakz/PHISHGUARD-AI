"""
simple_explainer.py — PhishGuard AI Human-Readable Explanations
================================================================

PURPOSE:
    Convert raw SHAP values and feature importance scores into
    plain-English sentences that non-technical users can understand.
    Also provides emoji-annotated one-liners for the Streamlit UI.

KEY FUNCTION:
    generate_simple_explanation(shap_values, feature_names, features)
    → "This URL is suspicious due to: high URL length (78 chars),
       presence of 'login' keyword, and missing HTTPS."

EXTRA HELPERS:
    explain_verdict()          — one-sentence verdict banner
    explain_feature()          — single feature explanation
    build_explanation_card()   — full structured card for UI
    render_explanation_card()  — Streamlit widget

DESIGN PHILOSOPHY:
    Every explanation has THREE layers:
      1. TECHNICAL  — feature name + value (for analysts)
      2. PLAIN      — what it means in English (for users)
      3. RISK       — why it matters for phishing (for executives)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE KNOWLEDGE BASE
#  For every known feature: a plain-English template and risk note.
# ═══════════════════════════════════════════════════════════════════════════

_FEATURE_TEMPLATES: Dict[str, Dict] = {
    # URL structure
    "url_length": {
        "label":   "URL Length",
        "unit":    "chars",
        "high_risk_threshold": 54,
        "plain_high": "unusually long URL ({value} chars)",
        "plain_low":  "normal URL length ({value} chars)",
        "risk_note":  "phishing URLs are often very long to obscure the real domain",
        "icon": "📏",
    },
    "num_dots": {
        "label":   "Number of Dots",
        "unit":    "dots",
        "high_risk_threshold": 3,
        "plain_high": "excessive dots in URL ({value})",
        "plain_low":  "normal number of dots ({value})",
        "risk_note":  "extra dots indicate multiple subdomains — a common disguise tactic",
        "icon": "⚫",
    },
    "num_dashes": {
        "label":   "Number of Hyphens",
        "unit":    "hyphens",
        "high_risk_threshold": 3,
        "plain_high": "many hyphens in URL ({value})",
        "plain_low":  "few hyphens ({value})",
        "risk_note":  "hyphens mimic legitimate brand names (e.g. paypal-secure.com)",
        "icon": "➖",
    },
    "entropy": {
        "label":   "URL Entropy",
        "unit":    "",
        "high_risk_threshold": 4.0,
        "plain_high": "high randomness/entropy score ({value:.2f})",
        "plain_low":  "normal entropy score ({value:.2f})",
        "risk_note":  "high entropy suggests random character sequences typical of auto-generated phishing URLs",
        "icon": "🎲",
    },
    "num_subdomains": {
        "label":   "Subdomain Count",
        "unit":    "levels",
        "high_risk_threshold": 2,
        "plain_high": "many subdomain levels ({value})",
        "plain_low":  "normal subdomain depth ({value})",
        "risk_note":  "deep subdomains are used to fake brand names (e.g. secure.paypal.fake.com)",
        "icon": "🌐",
    },
    # Binary flags
    "has_ip_address": {
        "label":   "IP Address in URL",
        "unit":    "",
        "binary":  True,
        "plain_1": "raw IP address used instead of domain name",
        "plain_0": "uses proper domain name",
        "risk_note": "legitimate sites never use raw IPs — this hides the real destination",
        "icon": "🔢",
        "weight": 3.0,
    },
    "has_at_symbol": {
        "label":   "@ Symbol in URL",
        "unit":    "",
        "binary":  True,
        "plain_1": "@ symbol present in URL",
        "plain_0": "no @ symbol",
        "risk_note": "browsers ignore everything before @, masking the true destination",
        "icon": "📧",
        "weight": 3.0,
    },
    "has_suspicious_keywords": {
        "label":   "Suspicious Keywords",
        "unit":    "",
        "binary":  True,
        "plain_1": "contains phishing keywords ('login', 'verify', 'secure', 'update')",
        "plain_0": "no phishing keywords detected",
        "risk_note": "phishing pages mimic login/verification pages of trusted brands",
        "icon": "🔑",
        "weight": 2.0,
    },
    "has_suspicious_tld": {
        "label":   "Suspicious Domain Extension",
        "unit":    "",
        "binary":  True,
        "plain_1": "uses a high-risk domain extension (.tk, .ml, .xyz, .pw)",
        "plain_0": "uses a standard domain extension",
        "risk_note": "these TLDs are free/cheap and disproportionately used by phishing sites",
        "icon": "🌍",
        "weight": 2.0,
    },
    "uses_https": {
        "label":   "HTTPS Encryption",
        "unit":    "",
        "binary":  True,
        "invert":  True,      # 0 is bad, 1 is good
        "plain_1": "uses secure HTTPS encryption",
        "plain_0": "missing HTTPS — plain HTTP only",
        "risk_note": "legitimate sites always use HTTPS; absence is a phishing red flag",
        "icon": "🔒",
        "weight": 2.5,
    },
    "has_double_slash_redirect": {
        "label":   "Double-Slash Redirect",
        "unit":    "",
        "binary":  True,
        "plain_1": "double-slash (//) redirect pattern detected",
        "plain_0": "no redirect obfuscation",
        "risk_note": "attackers use // to redirect victims through trusted domains",
        "icon": "↪️",
        "weight": 1.5,
    },
    "has_non_standard_port": {
        "label":   "Non-Standard Port",
        "unit":    "",
        "binary":  True,
        "plain_1": "non-standard port in URL",
        "plain_0": "standard port",
        "risk_note": "unusual ports bypass corporate firewalls and look suspicious",
        "icon": "🚪",
        "weight": 1.5,
    },
    # HTML features
    "has_login_form": {
        "label":   "Login Form Present",
        "unit":    "",
        "binary":  True,
        "plain_1": "page contains a password/login form",
        "plain_0": "no login form found",
        "risk_note": "credential harvesting via fake login forms is phishing's primary goal",
        "icon": "📝",
        "weight": 2.5,
    },
    "has_iframes": {
        "label":   "Hidden iFrames",
        "unit":    "",
        "binary":  True,
        "plain_1": "iframes detected on page",
        "plain_0": "no iframes",
        "risk_note": "iframes load hidden content to exfiltrate data without user awareness",
        "icon": "🖼️",
        "weight": 1.5,
    },
    "has_obfuscated_js": {
        "label":   "Obfuscated JavaScript",
        "unit":    "",
        "binary":  True,
        "plain_1": "obfuscated JavaScript code detected",
        "plain_0": "no obfuscated scripts",
        "risk_note": "obfuscation hides malicious code from security scanners",
        "icon": "💻",
        "weight": 2.0,
    },
    "has_favicon_mismatch": {
        "label":   "Favicon Mismatch",
        "unit":    "",
        "binary":  True,
        "plain_1": "favicon loaded from a different domain",
        "plain_0": "favicon matches domain",
        "risk_note": "attackers load favicons from legitimate brands to fake their identity",
        "icon": "🎭",
        "weight": 2.0,
    },
    "has_meta_refresh": {
        "label":   "Auto-Redirect (meta refresh)",
        "unit":    "",
        "binary":  True,
        "plain_1": "page auto-redirects visitors (meta refresh)",
        "plain_0": "no auto-redirect",
        "risk_note": "silent redirects move victims to the actual phishing page",
        "icon": "🔄",
        "weight": 1.5,
    },
    "right_click_disabled": {
        "label":   "Right-Click Disabled",
        "unit":    "",
        "binary":  True,
        "plain_1": "right-click context menu disabled",
        "plain_0": "normal browser interaction",
        "risk_note": "disabling right-click prevents users from inspecting the page source",
        "icon": "🖱️",
        "weight": 1.0,
    },
}

# Severity mapping (icon + label) based on SHAP magnitude
_SEVERITY_MAP = [
    (0.3,  "CRITICAL", "🔴"),
    (0.15, "HIGH",     "🟠"),
    (0.08, "MEDIUM",   "🟡"),
    (0.0,  "LOW",      "🟢"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def generate_simple_explanation(
    shap_values:   Optional[List[float]],
    feature_names: Optional[List[str]],
    features:      Optional[Dict] = None,
    top_n:         int = 3,
    prediction:    str = "PHISHING",
) -> Dict:
    """
    Convert SHAP values + feature names into plain-English explanations.

    Args:
        shap_values:   List of SHAP values (one per feature)
        feature_names: List of feature names aligned with shap_values
        features:      Optional raw feature dict (adds actual values to text)
        top_n:         Number of top features to highlight (default 3)
        prediction:    "PHISHING" or "LEGITIMATE" (affects framing)

    Returns:
        {
            "one_liner":   str,      # e.g. "Suspicious due to: IP address, login keyword"
            "full_text":   str,      # paragraph-length explanation
            "top_features": [        # top N with details
                {
                    "feature":   str,
                    "value":     any,
                    "shap":      float,
                    "severity":  str,
                    "icon":      str,
                    "plain":     str,    # plain-English description
                    "risk_note": str,    # why it matters
                }
            ],
            "confidence_statement": str,
        }

    Usage:
        explanation = generate_simple_explanation(
            shap_values=result["shap"]["top_features"],
            feature_names=[f["feature"] for f in result["shap"]["top_features"]],
            features=result["features"],
            prediction=result["verdict"],
        )
        print(explanation["one_liner"])
    """
    features = features or {}

    # ── Handle dict-of-dicts input (from shap_explainer.explain()) ───
    if shap_values and isinstance(shap_values[0], dict):
        # Input is a list of {"feature": str, "shap_value": float, ...}
        items = shap_values
        sv    = [item.get("shap_value", 0) for item in items]
        fn    = [item.get("feature", f"feat_{i}") for i, item in enumerate(items)]
        shap_values   = sv
        feature_names = fn

    # ── Fallback: use feature dict directly ──────────────────────────
    if not shap_values or not feature_names:
        return _explain_from_features_only(features, prediction, top_n)

    # ── Build ranked feature list ─────────────────────────────────────
    ranked = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    top_features = []
    for fname, sval in ranked:
        fval     = features.get(fname)
        tmpl     = _FEATURE_TEMPLATES.get(fname, {})
        plain    = _explain_feature(fname, fval, sval, tmpl)
        sev, icon = _get_severity(sval)
        top_features.append({
            "feature":   fname,
            "value":     fval,
            "shap":      round(sval, 4),
            "severity":  sev,
            "icon":      icon,
            "plain":     plain,
            "risk_note": tmpl.get("risk_note", ""),
            "label":     tmpl.get("label", fname.replace("_"," ").title()),
        })

    # ── Build one-liner ───────────────────────────────────────────────
    risk_parts  = [f["plain"] for f in top_features if f["shap"] > 0]
    clean_parts = [f["plain"] for f in top_features if f["shap"] < 0]

    if prediction == "PHISHING":
        if risk_parts:
            one_liner = f"⚠️ Suspicious due to: {_oxford_join(risk_parts[:3])}."
        else:
            one_liner = "⚠️ ML model detected suspicious patterns."
    else:
        if clean_parts:
            one_liner = f"✅ Appears legitimate: {_oxford_join(clean_parts[:3])}."
        else:
            one_liner = "✅ No major phishing signals detected."

    # ── Build full paragraph ──────────────────────────────────────────
    full_text = _build_full_text(top_features, prediction, features)

    # ── Confidence statement ──────────────────────────────────────────
    max_shap = max((abs(f["shap"]) for f in top_features), default=0)
    confidence_statement = _confidence_statement(max_shap, prediction)

    return {
        "one_liner":             one_liner,
        "full_text":             full_text,
        "top_features":          top_features,
        "confidence_statement":  confidence_statement,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  ADDITIONAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def explain_verdict(
    verdict:    str,
    confidence: float,
    risk_level: str,
    reason:     str = "",
) -> str:
    """
    Single sentence verdict explanation for the top of the UI.

    Example output:
        "🚨 PHISHING DETECTED with 93% confidence (CRITICAL risk).
         Google Safe Browsing flagged this URL."
    """
    emoji = "🚨" if verdict == "PHISHING" else "✅"
    conf_pct = f"{confidence*100:.0f}%"
    rl    = risk_level.upper()

    base = f"{emoji} **{verdict}** detected with {conf_pct} confidence ({rl} risk)."
    if reason:
        base += f" {reason}"
    return base


def explain_feature(feature_name: str, value: Any) -> str:
    """
    Plain-English explanation for a single feature value.

    Example:
        explain_feature("has_ip_address", 1)
        → "🔢 Raw IP address used instead of domain name"
    """
    tmpl = _FEATURE_TEMPLATES.get(feature_name, {})
    if not tmpl:
        return f"{feature_name.replace('_',' ')}: {value}"
    return f"{tmpl.get('icon','•')} {_explain_feature(feature_name, value, 0, tmpl)}"


def build_explanation_card(result: Dict) -> Dict:
    """
    Build a complete, structured explanation card from an analyze_url() result.

    Returns a dict suitable for the Streamlit card UI or JSON API.

    Usage:
        card = build_explanation_card(result)
        print(card["one_liner"])
        for reason in card["reasons"]:
            print(f"  {reason['icon']} {reason['plain']}")
    """
    shap_data   = result.get("shap") or {}
    features    = result.get("features") or {}
    prediction  = result.get("verdict", result.get("prediction", "UNKNOWN"))
    confidence  = result.get("confidence", 0)
    risk_level  = result.get("risk_level", "MEDIUM")
    reason      = result.get("verdict_reason", "")

    # Extract SHAP top features
    top_shap = shap_data.get("top_features", [])

    # Generate simple explanation
    if top_shap:
        simple = generate_simple_explanation(
            shap_values   = top_shap,
            feature_names = [f.get("feature","") for f in top_shap],
            features      = features,
            prediction    = prediction,
            top_n         = 5,
        )
    else:
        simple = _explain_from_features_only(features, prediction, top_n=3)

    return {
        "verdict":              prediction,
        "confidence":           confidence,
        "risk_level":           risk_level,
        "one_liner":            simple["one_liner"],
        "full_text":            simple["full_text"],
        "confidence_statement": simple["confidence_statement"],
        "verdict_banner":       explain_verdict(prediction, confidence, risk_level, reason),
        "reasons":              simple["top_features"],
        "feature_explanations": {
            k: explain_feature(k, v)
            for k, v in features.items()
            if k in _FEATURE_TEMPLATES and isinstance(v, (int, float))
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT WIDGET
# ═══════════════════════════════════════════════════════════════════════════

def render_explanation_card(result: Dict) -> None:
    """
    Render a human-readable explanation card in Streamlit.

    Usage:
        from simple_explainer import render_explanation_card
        render_explanation_card(analysis_result)
    """
    try:
        import streamlit as st
    except ImportError:
        return

    card = build_explanation_card(result)
    is_phishing = card["verdict"] == "PHISHING"

    # ── Banner ────────────────────────────────────────────────────────
    if is_phishing:
        st.error(card["verdict_banner"])
    else:
        st.success(card["verdict_banner"])

    # ── One-liner ─────────────────────────────────────────────────────
    st.markdown(f"**{card['one_liner']}**")

    # ── Full text ─────────────────────────────────────────────────────
    with st.expander("📖 Full Explanation", expanded=True):
        st.write(card["full_text"])
        st.caption(card["confidence_statement"])

    # ── Top reasons ───────────────────────────────────────────────────
    if card["reasons"]:
        st.markdown("#### 🔍 Key Risk Factors")
        for reason in card["reasons"][:5]:
            sev = reason["severity"].lower()
            color = {"critical":"red","high":"orange","medium":"yellow","low":"blue"}.get(sev,"gray")
            st.markdown(
                f":{color}[{reason['icon']} **{reason['label']}**] — "
                f"{reason['plain']}"
            )
            if reason.get("risk_note"):
                st.caption(f"  ↳ {reason['risk_note']}")

    # ── Feature legend ────────────────────────────────────────────────
    with st.expander("🔬 All Feature Explanations"):
        for feat, expl in card["feature_explanations"].items():
            st.write(expl)


# ═══════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _explain_feature(name: str, value: Any, shap_val: float, tmpl: Dict) -> str:
    """Generate a plain-English string for a single feature."""
    if not tmpl:
        return f"{name.replace('_',' ')}: {value}"

    if tmpl.get("binary"):
        inverted = tmpl.get("invert", False)
        v = int(value) if value is not None else 0
        # For inverted (uses_https): 1=good, 0=bad
        key = "plain_0" if (inverted and v == 1) else \
              ("plain_1" if (inverted and v == 0) else \
               ("plain_1" if v == 1 else "plain_0"))
        return tmpl.get(key, f"{name}: {value}")

    # Numeric
    threshold = tmpl.get("high_risk_threshold", 99999)
    is_high   = (value is not None and float(value) >= threshold)
    template  = tmpl.get("plain_high" if is_high else "plain_low", "{value}")
    try:
        return template.format(value=value)
    except Exception:
        return f"{name}: {value}"


def _explain_from_features_only(
    features:   Dict,
    prediction: str,
    top_n:      int = 3,
) -> Dict:
    """Fallback: generate explanation from raw features when SHAP unavailable."""
    risk_signals  = []
    clean_signals = []

    for fname, tmpl in _FEATURE_TEMPLATES.items():
        value = features.get(fname)
        if value is None:
            continue

        weight = tmpl.get("weight", 1.0)

        if tmpl.get("binary"):
            inverted = tmpl.get("invert", False)
            v = int(value)
            is_risk = (v == 0 if inverted else v == 1)
            if is_risk:
                risk_signals.append((weight, fname, value, tmpl))
            else:
                clean_signals.append((weight, fname, value, tmpl))
        else:
            threshold = tmpl.get("high_risk_threshold", 99999)
            if float(value) >= threshold:
                risk_signals.append((weight, fname, value, tmpl))

    risk_signals.sort(reverse=True)
    top = risk_signals[:top_n]

    top_features = []
    for w, fname, val, tmpl in top:
        plain = _explain_feature(fname, val, w, tmpl)
        top_features.append({
            "feature":   fname,
            "value":     val,
            "shap":      w * 0.1,
            "severity":  "high" if w >= 2.0 else "medium",
            "icon":      tmpl.get("icon", "⚠️"),
            "plain":     plain,
            "risk_note": tmpl.get("risk_note", ""),
            "label":     tmpl.get("label", fname.replace("_"," ").title()),
        })

    risk_parts = [f["plain"] for f in top_features]

    if prediction == "PHISHING" and risk_parts:
        one_liner = f"⚠️ Suspicious due to: {_oxford_join(risk_parts)}."
    elif prediction == "PHISHING":
        one_liner = "⚠️ ML model detected suspicious URL patterns."
    else:
        one_liner = "✅ No significant phishing indicators detected."

    return {
        "one_liner":            one_liner,
        "full_text":            _build_full_text(top_features, prediction, features),
        "top_features":         top_features,
        "confidence_statement": "Analysis based on rule-based feature scoring.",
    }


def _build_full_text(top_features: List[Dict], prediction: str, features: Dict) -> str:
    """Build a readable paragraph explanation."""
    uses_https = features.get("uses_https", 1)
    has_ip     = features.get("has_ip_address", 0)
    url_len    = features.get("url_length", 0)

    if prediction == "PHISHING":
        intro = "Our AI flagged this URL as a potential phishing site. "
    else:
        intro = "Our AI classified this URL as legitimate. "

    details = []
    for f in top_features[:3]:
        if f["shap"] > 0 or prediction == "PHISHING":
            details.append(f["plain"])

    if details:
        body = f"The main concern(s) are: {'; '.join(details)}. "
    else:
        body = "No major risk factors were detected. "

    if has_ip:
        body += "Using a raw IP address is a classic phishing technique. "
    if not uses_https and url_len > 30:
        body += "The combination of HTTP (no encryption) and a long URL is particularly suspicious. "

    if prediction == "PHISHING":
        advice = "Do not enter any personal information on this page."
    else:
        advice = "Standard security practices still apply — verify the URL before entering credentials."

    return intro + body + advice


def _get_severity(shap_val: float) -> Tuple[str, str]:
    """Map SHAP magnitude to severity label + icon."""
    abs_val = abs(shap_val)
    for threshold, label, icon in _SEVERITY_MAP:
        if abs_val >= threshold:
            return label, icon
    return "LOW", "🟢"


def _confidence_statement(max_shap: float, prediction: str) -> str:
    """Generate a confidence sentence based on the strongest SHAP value."""
    if max_shap >= 0.3:
        return f"The AI is highly confident in this {'PHISHING' if prediction=='PHISHING' else 'LEGITIMATE'} classification."
    elif max_shap >= 0.15:
        return "The AI is moderately confident. Manual review is recommended."
    else:
        return "The AI has low confidence. This case is borderline — treat with caution."


def _oxford_join(items: List[str]) -> str:
    """Join list with Oxford comma: a, b, and c."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# ═══════════════════════════════════════════════════════════════════════════
#  SELF TEST
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import pprint

    print("=" * 65)
    print("🧪 SIMPLE EXPLAINER — SELF TEST")
    print("=" * 65)

    # Test 1: SHAP-based explanation
    mock_shap_features = [
        {"feature": "has_ip_address",          "shap_value": 0.42},
        {"feature": "uses_https",               "shap_value": -0.31},  # negative = risky when missing
        {"feature": "has_suspicious_keywords",  "shap_value": 0.28},
        {"feature": "url_length",               "shap_value": 0.19},
        {"feature": "num_subdomains",           "shap_value": 0.12},
    ]

    mock_features = {
        "has_ip_address":         1,
        "uses_https":             0,
        "has_suspicious_keywords":1,
        "url_length":             87,
        "num_subdomains":         3,
        "num_dashes":             4,
    }

    result = generate_simple_explanation(
        shap_values   = mock_shap_features,
        feature_names = [f["feature"] for f in mock_shap_features],
        features      = mock_features,
        prediction    = "PHISHING",
        top_n         = 3,
    )

    print(f"\n🔍 One-liner:\n   {result['one_liner']}")
    print(f"\n📝 Full text:\n   {result['full_text']}")
    print(f"\n💬 Confidence:\n   {result['confidence_statement']}")
    print(f"\n🔝 Top Features:")
    for f in result["top_features"]:
        print(f"   {f['icon']} [{f['severity']:8s}] {f['label']:30s} | SHAP={f['shap']:+.3f} | {f['plain']}")

    # Test 2: Feature-only (no SHAP)
    print("\n" + "-"*55)
    print("📦 Test 2: Feature-only explanation (no SHAP)")
    result2 = generate_simple_explanation(
        shap_values   = None,
        feature_names = None,
        features      = mock_features,
        prediction    = "PHISHING",
    )
    print(f"   One-liner: {result2['one_liner']}")

    # Test 3: Single feature explainer
    print("\n" + "-"*55)
    print("🔬 Test 3: Single feature explanations")
    for feat, val in mock_features.items():
        print(f"   {explain_feature(feat, val)}")

    # Test 4: Verdict explanation
    print("\n" + "-"*55)
    print("🎯 Test 4: Verdict banner")
    print(f"   {explain_verdict('PHISHING', 0.93, 'CRITICAL', 'ML model detected IP-based URL.')}")
    print(f"   {explain_verdict('LEGITIMATE', 0.88, 'LOW', '')}")

    print("\n" + "=" * 65)
    print("✅ Module 6 test complete")
    print("=" * 65)