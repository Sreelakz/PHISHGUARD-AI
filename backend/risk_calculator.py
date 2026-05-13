"""
risk_calculator.py — Weighted multi-factor risk score (0–100).

Combines signals from all detection layers with tuned weights so that the
final score is more expressive than a raw ML confidence value.
"""

from __future__ import annotations
from typing import Any, Dict


# Weight configuration — all weights sum to ~100 when maximally triggered
WEIGHTS: Dict[str, float] = {
    # URL structure
    "url_length_risk":        5.0,
    "has_at_symbol":          8.0,
    "has_ip_address":         8.0,
    "has_suspicious_keywords":6.0,
    "excessive_dashes":       4.0,
    "suspicious_tld":         7.0,
    "has_double_slash":       3.0,
    "high_entropy":           4.0,
    # Domain
    "domain_age":             12.0,
    "domain_unresolvable":    10.0,
    # SSL
    "no_https":               10.0,
    "invalid_cert":           8.0,
    # HTML / behaviour
    "login_form":             6.0,
    "many_iframes":           5.0,
    "favicon_mismatch":       5.0,
    "obfuscated_js":          6.0,
    "redirect_chain":         5.0,
    # Visual / homograph
    "homograph":              10.0,
    "visual_brand_impersonation": 8.0,
    # ML model boost
    "ml_confidence_boost":    5.0,
}


class RiskCalculator:
    def calculate(
        self,
        features: Dict[str, Any],
        ml_confidence: float,
        prediction: str,
    ) -> Dict[str, Any]:
        """
        Returns:
        {
          "risk_score": 0-100,
          "risk_level": "Critical" | "High" | "Medium" | "Low" | "Safe",
          "risk_colour": hex string,
          "signal_weights": {signal: contribution},
          "dominant_signal": str,
        }
        """
        contributions: Dict[str, float] = {}

        d = features.get("domain_info") or {}
        s = features.get("ssl_info") or {}
        r = features.get("redirect_info") or {}
        h = features.get("homograph_info") or {}
        v = features.get("visual_analysis") or {}

        # ── URL ────────────────────────────────────────────────────────────
        url_len = features.get("url_length", 0)
        contributions["url_length_risk"] = WEIGHTS["url_length_risk"] * min(1.0, (url_len - 54) / 100) if url_len > 54 else 0

        contributions["has_at_symbol"] = WEIGHTS["has_at_symbol"] * features.get("has_at_symbol", 0)
        contributions["has_ip_address"] = WEIGHTS["has_ip_address"] * features.get("has_ip_address", 0)
        contributions["has_suspicious_keywords"] = WEIGHTS["has_suspicious_keywords"] * features.get("has_suspicious_keywords", 0)
        contributions["excessive_dashes"] = WEIGHTS["excessive_dashes"] * min(1.0, features.get("num_dashes", 0) / 5)
        contributions["suspicious_tld"] = WEIGHTS["suspicious_tld"] * features.get("has_suspicious_tld", 0)
        contributions["has_double_slash"] = WEIGHTS["has_double_slash"] * features.get("has_double_slash_redirect", 0)
        entropy = features.get("entropy", 0)
        contributions["high_entropy"] = WEIGHTS["high_entropy"] * min(1.0, max(0, entropy - 3.5) / 2.0)

        # ── Domain ─────────────────────────────────────────────────────────
        age = d.get("domain_age_days", 9999)
        domain_age_factor = max(0, 1 - age / 365) if age < 365 else 0
        contributions["domain_age"] = WEIGHTS["domain_age"] * domain_age_factor
        contributions["domain_unresolvable"] = WEIGHTS["domain_unresolvable"] * int(not d.get("is_registered", True))

        # ── SSL ────────────────────────────────────────────────────────────
        contributions["no_https"] = WEIGHTS["no_https"] * int(not s.get("uses_https", False))
        uses_https_invalid = s.get("uses_https", False) and not s.get("certificate_valid", False)
        contributions["invalid_cert"] = WEIGHTS["invalid_cert"] * int(uses_https_invalid)

        # ── HTML ───────────────────────────────────────────────────────────
        contributions["login_form"] = WEIGHTS["login_form"] * features.get("has_login_form", 0)
        iframes = features.get("has_iframes", 0)
        contributions["many_iframes"] = WEIGHTS["many_iframes"] * min(1.0, iframes / 5)
        contributions["favicon_mismatch"] = WEIGHTS["favicon_mismatch"] * features.get("has_favicon_mismatch", 0)
        contributions["obfuscated_js"] = WEIGHTS["obfuscated_js"] * features.get("has_obfuscated_js", 0)
        rcount = r.get("redirect_count", 0)
        contributions["redirect_chain"] = WEIGHTS["redirect_chain"] * min(1.0, rcount / 4)

        # ── Visual / Homograph ─────────────────────────────────────────────
        contributions["homograph"] = WEIGHTS["homograph"] * h.get("similarity_score", 0)
        v_risk = v.get("visual_risk_score", 0) / 100
        contributions["visual_brand_impersonation"] = WEIGHTS["visual_brand_impersonation"] * v_risk

        # ── ML boost ───────────────────────────────────────────────────────
        if prediction == "PHISHING":
            ml_boost = WEIGHTS["ml_confidence_boost"] * min(1.0, (ml_confidence - 0.5) / 0.5)
            contributions["ml_confidence_boost"] = max(0, ml_boost)
        else:
            contributions["ml_confidence_boost"] = 0

        # ── Aggregate ──────────────────────────────────────────────────────
        raw_score = sum(contributions.values())
        risk_score = round(min(100.0, max(0.0, raw_score)), 1)

        level, colour = self._classify(risk_score)

        dominant = max(contributions, key=contributions.get) if contributions else "—"

        return {
            "risk_score":      risk_score,
            "risk_level":      level,
            "risk_colour":     colour,
            "signal_weights":  {k: round(v, 2) for k, v in contributions.items() if v > 0},
            "dominant_signal": dominant,
        }

    @staticmethod
    def _classify(score: float):
        if score >= 75:
            return "Critical", "#e53e3e"
        elif score >= 55:
            return "High",     "#dd6b20"
        elif score >= 35:
            return "Medium",   "#d69e2e"
        elif score >= 15:
            return "Low",      "#3182ce"
        else:
            return "Safe",     "#38a169"