"""
visual_analyzer.py — Simulates CNN-based visual phishing detection.

In a production deployment this module would:
  1. Capture a screenshot using Playwright/Selenium.
  2. Pass it through a fine-tuned ResNet-50 / EfficientNet model trained on
     the VisualPhish / PhishIntention datasets.
  3. Return brand-detection results and a visual anomaly score.

For the current demo we apply a deterministic heuristic on URL + domain
features to produce a realistic visual risk score and brand-detection signal.
Replace `_simulate_cnn_inference` with a real model call when ready.
"""

from __future__ import annotations
from urllib.parse import urlparse
from difflib import SequenceMatcher

KNOWN_BRANDS = {
    "google":    ["google", "gmail", "goog1e", "g00gle"],
    "paypal":    ["paypal", "paypa1", "paypai"],
    "amazon":    ["amazon", "arnazon", "amaz0n", "amazn"],
    "apple":     ["apple", "app1e", "appleid"],
    "microsoft": ["microsoft", "micros0ft", "microsofft", "outlook", "office365"],
    "facebook":  ["facebook", "faceb00k", "face-book"],
    "netflix":   ["netflix", "netf1ix", "net-flix"],
    "instagram": ["instagram", "1nstagram"],
    "twitter":   ["twitter", "tw1tter", "twltter"],
    "ebay":      ["ebay", "ebay-com", "e-bay"],
    "linkedin":  ["linkedin", "linkedln"],
    "dropbox":   ["dropbox", "dr0pbox"],
    "chase":     ["chase", "chasebank"],
    "wellsfargo":["wellsfargo", "wells-fargo"],
    "bankofamerica": ["bankofamerica", "bank0famerica"],
}


class VisualAnalyzer:
    def analyze(self, url: str) -> dict:
        result = {
            "visual_risk_score":   0,
            "brand_detected":      False,
            "detected_brand":      None,
            "similarity_score":    0.0,
            "cnn_confidence":      0.0,
            "screenshot_taken":    False,   # True once Playwright integration added
            "matched_keywords":    [],
        }

        try:
            parsed = urlparse(url)
            netloc = (parsed.netloc or "").replace("www.", "").lower()
            domain_part = netloc.split(".")[0] if netloc else ""
            path = (parsed.path or "").lower()

            brand, score, matches = self._detect_brand(domain_part, path)

            if brand:
                result["brand_detected"]   = True
                result["detected_brand"]   = brand
                result["similarity_score"] = score
                result["matched_keywords"] = matches

            result["visual_risk_score"] = self._compute_visual_risk(
                url, brand, score, parsed
            )
            result["cnn_confidence"] = round(result["visual_risk_score"] / 100, 2)

        except Exception:
            pass

        return result

    # ── Brand detection ────────────────────────────────────────────────────
    def _detect_brand(self, domain_part: str, path: str) -> tuple[str | None, float, list]:
        best_brand: str | None = None
        best_score = 0.0
        best_matches: list = []

        for brand, variants in KNOWN_BRANDS.items():
            # Direct alias match
            if domain_part == brand:
                return None, 0.0, []          # it IS the real brand — not suspicious

            for variant in variants:
                seq_score = SequenceMatcher(None, domain_part, variant).ratio()
                if seq_score > best_score and seq_score > 0.75 and domain_part != brand:
                    best_score = seq_score
                    best_brand = brand
                    best_matches = [variant]

            # Brand name buried in path (e.g. evil.com/paypal-login)
            if brand in path and domain_part != brand:
                path_score = 0.70
                if path_score > best_score:
                    best_score = path_score
                    best_brand = brand
                    best_matches = [f"path contains '{brand}'"]

        return best_brand, round(best_score, 2), best_matches

    # ── Risk scoring ───────────────────────────────────────────────────────
    def _compute_visual_risk(
        self, url: str, brand: str | None, similarity: float, parsed
    ) -> int:
        score = 0

        if brand and similarity > 0.8:
            score += 60
        elif brand and similarity > 0.75:
            score += 40

        # Non-HTTPS impersonating a known brand is worse
        if brand and parsed.scheme != "https":
            score += 20

        # Lots of URL length / dashes suggests visual spoofing
        if len(url) > 80:
            score += 10
        if url.count("-") >= 3:
            score += 10

        return min(100, score)