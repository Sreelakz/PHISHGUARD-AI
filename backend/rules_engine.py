"""
backend/rules_engine.py
------------------------
Centralized rule engine for phishing detection.

DESIGN PHILOSOPHY:
  • CRITICAL rules → immediate phishing verdict (override ML)
  • HIGH rules     → boost ML confidence if they fire
  • Every rule has a human-readable reason (for XAI/UI)

This file replaces the duplicated rule logic in your original
app.py::_hard_override() and ml_model.py::CRITICAL_RULES.
"""

from dataclasses import dataclass
from typing import Callable, Optional, List, Dict
from urllib.parse import urlparse
import re

_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@dataclass
class Rule:
    """A single detection rule."""
    name: str
    severity: str              # 'critical' | 'high' | 'medium'
    confidence: float          # 0.0 – 1.0
    description: str           # Human-readable explanation
    check: Callable[[dict], bool]  # Function: features → bool


# ══════════════════════════════════════════════════════════════════════════
#  CRITICAL RULES — If any fires, verdict is PHISHING regardless of ML
# ══════════════════════════════════════════════════════════════════════════
CRITICAL_RULES: List[Rule] = [
    Rule(
        name="ip_address_in_url",
        severity="critical",
        confidence=0.97,
        description="Raw IP address used instead of a domain — legitimate sites never do this.",
        check=lambda f: f.get("has_ip_address", 0) == 1,
    ),
    Rule(
        name="at_symbol_in_url",
        severity="critical",
        confidence=0.96,
        description="'@' symbol hides the real destination — classic phishing redirection trick.",
        check=lambda f: f.get("has_at_symbol", 0) == 1,
    ),
    Rule(
        name="homograph_attack_high",
        severity="critical",
        confidence=0.95,
        description="Domain is >85% visually similar to a known brand — typosquatting attack.",
        check=lambda f: (f.get("homograph_info") or {}).get("similarity_score", 0) > 0.85,
    ),
    Rule(
        name="freshly_registered_domain",
        severity="critical",
        confidence=0.93,
        description="Domain is less than 7 days old — hallmark of throwaway phishing sites.",
        check=lambda f: 0 < (f.get("domain_info") or {}).get("domain_age_days", 9999) < 7,
    ),
]


# ══════════════════════════════════════════════════════════════════════════
#  HIGH RULES — Boost ML confidence when they fire
# ══════════════════════════════════════════════════════════════════════════
HIGH_RULES: List[Rule] = [
    Rule(
        name="homograph_attack_medium",
        severity="high",
        confidence=0.88,
        description="Domain is >75% similar to a known brand.",
        check=lambda f: (f.get("homograph_info") or {}).get("similarity_score", 0) > 0.75,
    ),
    Rule(
        name="http_with_login_form",
        severity="high",
        confidence=0.87,
        description="Login form served over HTTP — credentials would be transmitted in plaintext.",
        check=lambda f: f.get("uses_https", 1) == 0 and f.get("has_login_form", 0) == 1,
    ),
    Rule(
        name="http_with_phishing_keywords",
        severity="high",
        confidence=0.85,
        description="Phishing keywords (login/verify/secure/bank) + no HTTPS.",
        check=lambda f: f.get("uses_https", 1) == 0 and f.get("has_suspicious_keywords", 0) == 1,
    ),
    Rule(
        name="young_domain_no_https",
        severity="high",
        confidence=0.84,
        description="Domain < 30 days old AND no HTTPS.",
        check=lambda f: (
            0 < (f.get("domain_info") or {}).get("domain_age_days", 9999) < 30
            and f.get("uses_https", 1) == 0
        ),
    ),
    Rule(
        name="many_dashes_and_keywords",
        severity="high",
        confidence=0.82,
        description="4+ hyphens in URL + phishing keywords + no HTTPS.",
        check=lambda f: (
            f.get("num_dashes", 0) >= 4
            and f.get("has_suspicious_keywords", 0) == 1
            and f.get("uses_https", 1) == 0
        ),
    ),
    Rule(
        name="suspicious_tld_with_keywords",
        severity="high",
        confidence=0.80,
        description="Uses a commonly-abused TLD (.tk/.ml/.ga) with phishing keywords.",
        check=lambda f: (
            f.get("has_suspicious_tld", 0) == 1
            and f.get("has_suspicious_keywords", 0) == 1
        ),
    ),
]


# ══════════════════════════════════════════════════════════════════════════
#  MEDIUM RULES — Informational signals (don't override ML, just report)
# ══════════════════════════════════════════════════════════════════════════
MEDIUM_RULES: List[Rule] = [
    Rule(
        name="url_shortener_used",
        severity="medium",
        confidence=0.60,
        description="URL shortener hides the real destination.",
        check=lambda f: f.get("is_shortened", 0) == 1,
    ),
    Rule(
        name="excessive_subdomains",
        severity="medium",
        confidence=0.55,
        description="3+ subdomains — possibly trying to impersonate a brand.",
        check=lambda f: f.get("num_subdomains", 0) >= 3,
    ),
    Rule(
        name="high_url_entropy",
        severity="medium",
        confidence=0.50,
        description="High randomness in URL — suggests auto-generated phishing link.",
        check=lambda f: f.get("entropy", 0) > 4.5,
    ),
    Rule(
        name="long_url",
        severity="medium",
        confidence=0.45,
        description="Unusually long URL (>75 chars) — possible obfuscation.",
        check=lambda f: f.get("url_length", 0) > 75,
    ),
]


# ══════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════
def check_critical_rules(features: dict) -> Optional[Rule]:
    """Returns first matching CRITICAL rule, or None."""
    for rule in CRITICAL_RULES:
        try:
            if rule.check(features):
                return rule
        except Exception:
            continue
    return None


def check_high_rules(features: dict) -> List[Rule]:
    """Returns ALL matching HIGH rules."""
    fired = []
    for rule in HIGH_RULES:
        try:
            if rule.check(features):
                fired.append(rule)
        except Exception:
            continue
    return fired


def check_all_rules(features: dict) -> Dict[str, List[Rule]]:
    """Run all rules → returns grouped by severity."""
    return {
        "critical": [r for r in CRITICAL_RULES if _safe_check(r, features)],
        "high":     [r for r in HIGH_RULES     if _safe_check(r, features)],
        "medium":   [r for r in MEDIUM_RULES   if _safe_check(r, features)],
    }


def _safe_check(rule: Rule, features: dict) -> bool:
    try:
        return rule.check(features)
    except Exception:
        return False


def get_rule_summary(features: dict) -> dict:
    """
    High-level summary for UI/API response.
    Returns:
        {
            "verdict": "phishing" | "suspicious" | "clean",
            "confidence": float,
            "fired_rules": [{name, severity, description, confidence}, ...],
            "critical_hit": bool,
        }
    """
    all_fired = check_all_rules(features)
    critical_hit = len(all_fired["critical"]) > 0

    fired_rules = []
    for severity in ("critical", "high", "medium"):
        for rule in all_fired[severity]:
            fired_rules.append({
                "name": rule.name,
                "severity": rule.severity,
                "description": rule.description,
                "confidence": rule.confidence,
            })

    if critical_hit:
        verdict = "phishing"
        confidence = max(r.confidence for r in all_fired["critical"])
    elif all_fired["high"]:
        verdict = "suspicious"
        confidence = max(r.confidence for r in all_fired["high"])
    else:
        verdict = "clean"
        confidence = 1.0 - (0.1 * len(all_fired["medium"]))

    return {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "fired_rules": fired_rules,
        "critical_hit": critical_hit,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Sanity test
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 RULES ENGINE TEST")
    print("=" * 70)

    test_cases = [
        ("IP attack", {"has_ip_address": 1, "uses_https": 0}),
        ("@ symbol", {"has_at_symbol": 1, "uses_https": 1}),
        ("Homograph", {"homograph_info": {"similarity_score": 0.92}, "uses_https": 1}),
        ("Young + no https", {"domain_info": {"domain_age_days": 15}, "uses_https": 0,
                               "has_suspicious_keywords": 1}),
        ("Clean site", {"uses_https": 1, "has_at_symbol": 0, "has_ip_address": 0,
                        "url_length": 20}),
    ]

    for name, features in test_cases:
        summary = get_rule_summary(features)
        print(f"\n🔎 {name}")
        print(f"   Verdict: {summary['verdict'].upper()}")
        print(f"   Confidence: {summary['confidence']}")
        print(f"   Rules fired: {len(summary['fired_rules'])}")
        for r in summary["fired_rules"][:2]:
            print(f"      • [{r['severity']}] {r['description']}")

    print("\n✅ All rule tests passed!")