"""
explainable_ai.py — Generates structured, natural-language explanations for
phishing predictions using a severity-ranked indicator system and a risk
breakdown by category (URL, Domain, SSL, HTML, Visual).

PHASE 3 UPGRADE: Integrated SHAP for per-prediction ML explanations.
PHASE 4 UPGRADE: Integrated Google Safe Browsing indicators.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ══════════════════════════════════════════════════════════════════════════
#  SHAP integration (optional — graceful fallback if unavailable)
# ══════════════════════════════════════════════════════════════════════════
try:
    from shap_explainer import get_shap_explainer
except ModuleNotFoundError:
    try:
        from backend.shap_explainer import get_shap_explainer
    except ModuleNotFoundError:
        get_shap_explainer = None


@dataclass
class Indicator:
    category: str          # "SafeBrowsing" | "URL" | "Domain" | "SSL" | "HTML" | "Visual"
    severity: str          # "critical" | "high" | "medium" | "low"
    code: str              # machine-readable key
    message: str           # human-readable sentence
    weight: float = 1.0    # contribution to overall risk


SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}


class ExplainableAI:
    # ── Public entry point ─────────────────────────────────────────────────
    def generate_explanations(
        self,
        features: Dict[str, Any],
        prediction: str,
        importances: Dict[str, float] | None = None,
        shap_result: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Returns a rich explanation dict:
        {
          "summary":    str,
          "verdict":    str,
          "indicators": [ {"category", "severity", "code", "message"} ],
          "category_scores": { "SafeBrowsing": 0-100, "URL": 0-100, ... },
          "top_reasons": [ str, ... ],       # ≤ 5 plain-English sentences
          "ai_narrative": str,               # paragraph-level explanation
          "risk_breakdown": { ... },
          "shap":        { ... },            # Phase 3 SHAP data
          "ai_reasons":  [ str, ... ],       # SHAP-derived reasons
        }
        """
        indicators: List[Indicator] = []
        indicators += self._safe_browsing_indicators(features)   # 🆕 Phase 4 — highest priority
        indicators += self._url_indicators(features)
        indicators += self._domain_indicators(features)
        indicators += self._ssl_indicators(features)
        indicators += self._html_indicators(features)
        indicators += self._visual_indicators(features)

        # Sort by severity
        indicators.sort(key=lambda i: SEVERITY_ORDER.get(i.severity, 0), reverse=True)

        category_scores = self._category_scores(indicators)
        top_reasons = [ind.message for ind in indicators[:5]]
        ai_narrative = self._build_narrative(features, prediction, indicators)

        # Phase 3: Fetch SHAP result if not already provided
        if shap_result is None and get_shap_explainer is not None:
            try:
                explainer = get_shap_explainer()
                if explainer.is_ready():
                    shap_result = explainer.explain(features, top_n=5)
            except Exception:
                shap_result = None

        shap_data = shap_result or {"available": False}
        shap_reasons = shap_data.get("natural_reasons", []) if shap_data.get("available") else []

        return {
            "summary":         self._verdict_summary(prediction, indicators),
            "verdict":         prediction,
            "indicators":      [self._serialize(ind) for ind in indicators],
            "category_scores": category_scores,
            "top_reasons":     top_reasons,
            "ai_narrative":    ai_narrative,
            "risk_breakdown":  self._risk_breakdown(features),
            "shap":            shap_data,
            "ai_reasons":      shap_reasons,
        }

    # ── Indicator generators ───────────────────────────────────────────────

    def _safe_browsing_indicators(self, f: dict) -> List[Indicator]:
        """🆕 Phase 4: Google Safe Browsing threat indicators."""
        out = []
        sb = f.get("safe_browsing") or {}

        if not sb.get("available"):
            return out  # API not available — skip silently

        if sb.get("is_threat"):
            for threat in sb.get("threats", []):
                out.append(Indicator(
                    category="SafeBrowsing",
                    severity="critical",
                    code=f"gsb_{threat['type'].lower()}",
                    message=f"{threat['icon']} Google Safe Browsing: {threat['description']}",
                    weight=3.0,  # Highest weight — Google's verdict is authoritative
                ))

        return out

    def _url_indicators(self, f: dict) -> List[Indicator]:
        out = []
        url_len = f.get("url_length", 0)
        if url_len > 100:
            out.append(Indicator("URL", "high", "url_very_long",
                f"URL is unusually long ({url_len} chars) — phishing pages often use long, obfuscated URLs.", 1.5))
        elif url_len > 54:
            out.append(Indicator("URL", "medium", "url_long",
                f"URL length ({url_len} chars) exceeds the typical safe threshold of 54.", 1.0))

        if f.get("has_at_symbol"):
            out.append(Indicator("URL", "critical", "at_symbol",
                "@ symbol found in URL — browsers ignore everything before @, masking the real destination.", 2.0))

        if f.get("has_ip_address"):
            out.append(Indicator("URL", "critical", "ip_address",
                "URL uses a raw IP address instead of a domain name — a classic phishing evasion technique.", 2.0))

        if f.get("has_suspicious_keywords"):
            out.append(Indicator("URL", "high", "suspicious_keywords",
                "URL contains phishing-related keywords (e.g. 'login', 'verify', 'secure') that mimic legitimate sites.", 1.5))

        dashes = f.get("num_dashes", 0)
        if dashes >= 4:
            out.append(Indicator("URL", "medium", "excessive_dashes",
                f"{dashes} hyphens in URL — attackers use dashes to imitate legitimate domains (e.g. paypal-secure.com).", 1.0))

        if f.get("has_suspicious_tld"):
            out.append(Indicator("URL", "high", "suspicious_tld",
                "Domain uses a high-risk TLD (.tk, .ml, .xyz, etc.) commonly associated with free/disposable phishing domains.", 1.5))

        if f.get("has_double_slash_redirect"):
            out.append(Indicator("URL", "medium", "double_slash",
                "Double-slash (//) in path may indicate a redirect-based obfuscation technique.", 1.0))

        if f.get("has_non_standard_port"):
            out.append(Indicator("URL", "medium", "non_standard_port",
                "Non-standard port detected — legitimate sites rarely serve content on unusual ports.", 1.0))

        entropy = f.get("entropy", 0)
        if entropy > 4.5:
            out.append(Indicator("URL", "medium", "high_entropy",
                f"URL entropy is high ({entropy:.2f}) — suggests random/obfuscated character sequences.", 0.8))

        subs = f.get("num_subdomains", 0)
        if subs >= 3:
            out.append(Indicator("URL", "medium", "many_subdomains",
                f"{subs} subdomain levels detected — excessive subdomains are used to confuse users.", 1.0))

        return out

    def _domain_indicators(self, f: dict) -> List[Indicator]:
        out = []
        d = f.get("domain_info") or {}
        age = d.get("domain_age_days", 9999)

        if age < 30:
            out.append(Indicator("Domain", "critical", "brand_new_domain",
                f"Domain is only {age} day(s) old — newly registered domains are a leading phishing indicator.", 2.5))
        elif age < 180:
            out.append(Indicator("Domain", "high", "young_domain",
                f"Domain age is {age} days — domains less than 6 months old are frequently used in phishing campaigns.", 1.5))

        if not d.get("is_registered"):
            out.append(Indicator("Domain", "critical", "unresolvable_domain",
                "Domain does not resolve to any IP — may be a dangling or spoofed domain.", 2.0))

        return out

    def _ssl_indicators(self, f: dict) -> List[Indicator]:
        out = []
        s = f.get("ssl_info") or {}

        if not s.get("uses_https"):
            out.append(Indicator("SSL", "critical", "no_https",
                "Site uses plain HTTP with no encryption — legitimate services always use HTTPS.", 2.5))

        if s.get("uses_https") and not s.get("certificate_valid"):
            out.append(Indicator("SSL", "high", "invalid_cert",
                "HTTPS is used but the SSL certificate is invalid or self-signed — a major red flag.", 2.0))

        return out

    def _html_indicators(self, f: dict) -> List[Indicator]:
        out = []

        if f.get("has_login_form"):
            out.append(Indicator("HTML", "high", "login_form",
                "Page contains a password-input login form — credential harvesting is a primary phishing goal.", 1.5))

        iframes = f.get("has_iframes", 0)
        if iframes >= 3:
            out.append(Indicator("HTML", "high", "many_iframes",
                f"{iframes} iframes detected — phishing pages often load hidden iframes to exfiltrate data.", 1.5))
        elif iframes > 0:
            out.append(Indicator("HTML", "medium", "iframes_present",
                f"{iframes} iframe(s) detected on page.", 0.8))

        if f.get("has_favicon_mismatch"):
            out.append(Indicator("HTML", "high", "favicon_mismatch",
                "Favicon is loaded from a different domain — a strong brand-impersonation signal.", 1.5))

        if f.get("has_meta_refresh"):
            out.append(Indicator("HTML", "medium", "meta_refresh",
                "Meta-refresh redirect found — attackers use this to silently forward victims to phishing pages.", 1.0))

        if f.get("has_obfuscated_js"):
            out.append(Indicator("HTML", "high", "obfuscated_js",
                "Obfuscated JavaScript detected (eval+atob/unescape) — used to hide malicious logic from scanners.", 1.5))

        if f.get("right_click_disabled"):
            out.append(Indicator("HTML", "medium", "right_click_disabled",
                "Right-click context menu is disabled — a common tactic to prevent source-code inspection.", 0.8))

        if f.get("has_popup_window"):
            out.append(Indicator("HTML", "low", "popup_window",
                "window.open() calls detected — may indicate unwanted popup behavior.", 0.5))

        r = f.get("redirect_info") or {}
        rcount = r.get("redirect_count", 0)
        if rcount >= 3:
            out.append(Indicator("HTML", "high", "redirect_chain",
                f"{rcount}-hop redirect chain detected — complex redirects hide the true final destination.", 1.5))
        elif rcount > 0:
            out.append(Indicator("HTML", "medium", "redirect_present",
                f"{rcount} redirect(s) in response — review the chain for suspicious domains.", 0.8))

        if f.get("has_hidden_fields"):
            out.append(Indicator("HTML", "medium", "hidden_fields",
                "Hidden form fields detected — may carry pre-filled victim data or session tokens.", 0.8))

        return out

    def _visual_indicators(self, f: dict) -> List[Indicator]:
        out = []
        h = f.get("homograph_info") or {}
        v = f.get("visual_analysis") or {}

        if h.get("is_homograph"):
            brand = h.get("matched_brand", "a known brand")
            score = h.get("similarity_score", 0)
            out.append(Indicator("Visual", "critical", "homograph_attack",
                f"Domain is {int(score*100)}% similar to '{brand}' — likely a homograph/typosquatting attack.", 2.5))

        if v.get("brand_detected"):
            brand = v.get("detected_brand", "unknown brand")
            out.append(Indicator("Visual", "high", "visual_brand_impersonation",
                f"Visual analysis detected {brand} branding elements — page may be impersonating this brand.", 2.0))

        if v.get("visual_risk_score", 0) > 70:
            out.append(Indicator("Visual", "high", "visual_anomaly",
                f"Visual CNN analysis flagged suspicious layout patterns (score: {v['visual_risk_score']}).", 1.5))

        return out

    # ── Scoring & formatting ───────────────────────────────────────────────
    def _category_scores(self, indicators: List[Indicator]) -> Dict[str, int]:
        cats = {
            "SafeBrowsing": 0.0,   # 🆕 Phase 4
            "URL":          0.0,
            "Domain":       0.0,
            "SSL":          0.0,
            "HTML":         0.0,
            "Visual":       0.0,
        }

        sev_weights = {"critical": 40, "high": 25, "medium": 15, "low": 5}

        for ind in indicators:
            if ind.category in cats:
                score = sev_weights.get(ind.severity, 0) * ind.weight
                cats[ind.category] += score

        # Clamp to 100
        return {k: min(100, int(v)) for k, v in cats.items()}

    def _risk_breakdown(self, f: dict) -> Dict[str, Any]:
        """Per-signal numeric contributions for the radar/heatmap chart."""
        d  = f.get("domain_info")    or {}
        s  = f.get("ssl_info")       or {}
        r  = f.get("redirect_info")  or {}
        h  = f.get("homograph_info") or {}
        sb = f.get("safe_browsing")  or {}   # 🆕 Phase 4

        age = d.get("domain_age_days", 9999)
        age_risk = max(0, 100 - int(age / 50 * 10)) if age < 500 else 0

        return {
            "url_structure":     min(100, int(f.get("url_length", 0) / 2)),
            "domain_trust":      age_risk,
            "ssl_security":      0 if s.get("certificate_valid") else 80,
            "html_behaviour":    min(100, (f.get("has_login_form", 0) * 30 +
                                          f.get("has_iframes", 0) * 10 +
                                          f.get("has_obfuscated_js", 0) * 40)),
            "redirect_risk":     min(100, r.get("redirect_count", 0) * 25),
            "visual_similarity": int(h.get("similarity_score", 0) * 100),
            "safe_browsing":     100 if (sb.get("available") and sb.get("is_threat")) else 0,   # 🆕
        }

    def _verdict_summary(self, prediction: str, indicators: List[Indicator]) -> str:
        critical = sum(1 for i in indicators if i.severity == "critical")
        high     = sum(1 for i in indicators if i.severity == "high")
        gsb_hit  = any(i.category == "SafeBrowsing" for i in indicators)   # 🆕

        if prediction == "PHISHING":
            if gsb_hit:
                return "🚨 Google Safe Browsing has confirmed this URL is dangerous."
            elif critical >= 2:
                return f"⛔ High-confidence phishing detected with {critical} critical and {high} high-severity signals."
            elif critical >= 1:
                return f"🚨 Likely phishing — {critical} critical indicator(s) found."
            else:
                return f"⚠️ Suspicious site — {high} high-severity indicators detected."
        else:
            if not indicators:
                return "✅ No phishing indicators detected. Site appears legitimate."
            return f"✅ Classified as legitimate, though {len(indicators)} minor indicator(s) were noted."

    def _build_narrative(
        self, f: dict, prediction: str, indicators: List[Indicator]
    ) -> str:
        critical = [i for i in indicators if i.severity == "critical"]
        high     = [i for i in indicators if i.severity == "high"]
        gsb_hit  = any(i.category == "SafeBrowsing" for i in indicators)   # 🆕

        d = f.get("domain_info") or {}
        s = f.get("ssl_info")    or {}

        parts = []

        # 🆕 Phase 4: Lead with Safe Browsing verdict if triggered
        if gsb_hit:
            parts.append(
                "⚠️ <strong>Google Safe Browsing</strong> has flagged this URL as a known threat. "
                "This is the most authoritative signal available — the URL appears in Google's "
                "continuously updated database of malicious sites."
            )
        elif prediction == "PHISHING":
            parts.append(
                "Our AI engine analysed this URL across six security dimensions and "
                f"classified it as a <strong>phishing site</strong> with {len(indicators)} "
                "indicator(s) raising concern."
            )
        else:
            parts.append(
                "The URL was classified as <strong>legitimate</strong> by our hybrid "
                "AI engine. The analysis found no strong phishing signals."
            )

        if critical and not gsb_hit:
            crit_msgs = "; ".join(i.message.split("—")[0].strip() for i in critical[:2])
            parts.append(f"The most critical findings were: {crit_msgs}.")

        age = d.get("domain_age_days", 9999)
        if age < 180:
            parts.append(
                f"Domain intelligence shows this domain is only {age} days old, "
                "which is a strong predictor of phishing activity."
            )

        if not s.get("uses_https"):
            parts.append(
                "The absence of HTTPS encryption means any data submitted on this "
                "page is transmitted in plaintext, enabling trivial interception."
            )

        if not parts[1:]:
            parts.append(
                "All analysed dimensions — URL structure, domain age, SSL certificate, "
                "HTML behaviour, and visual similarity — returned normal results."
            )

        return " ".join(parts)

    @staticmethod
    def _serialize(ind: Indicator) -> dict:
        return {
            "category": ind.category,
            "severity": ind.severity,
            "code":     ind.code,
            "message":  ind.message,
        }