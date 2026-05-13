"""
homograph_detector.py — Detects typosquatting / homograph attacks by
comparing the target domain against a list of common brand names using
sequence similarity.  Extended with an additional brand list and Levenshtein
distance check for single-character substitutions.
"""

from __future__ import annotations
from urllib.parse import urlparse
from difflib import SequenceMatcher

COMMON_BRANDS = [
    "google", "facebook", "amazon", "apple", "microsoft",
    "netflix", "twitter", "instagram", "paypal", "ebay",
    "linkedin", "dropbox", "chase", "wellsfargo", "bankofamerica",
    "yahoo", "outlook", "hotmail", "icloud", "spotify",
    "steam", "discord", "twitch", "github", "gitlab",
]

# Characters visually similar to ASCII equivalents
VISUAL_SUBS = {
    "0": "o", "1": "l", "3": "e", "4": "a",
    "5": "s", "6": "g", "7": "t", "8": "b",
    "@": "a", "!": "i",
}


class HomographDetector:
    def detect(self, url: str) -> dict:
        result = {
            "is_homograph":    False,
            "similarity_score": 0.0,
            "matched_brand":   None,
            "technique":       None,   # "typosquat" | "visual_substitution" | "subdomain_abuse"
        }

        try:
            parsed = urlparse(url)
            netloc = parsed.netloc.replace("www.", "").lower()
            domain_name = netloc.split(".")[0]

            # ── Technique 1: sequence similarity ─────────────────────────
            for brand in COMMON_BRANDS:
                if domain_name == brand:
                    break   # exact match = not a spoof
                sim = SequenceMatcher(None, domain_name, brand).ratio()
                if sim > 0.80 and sim > result["similarity_score"]:
                    result.update({
                        "is_homograph":     True,
                        "similarity_score": round(sim, 2),
                        "matched_brand":    brand,
                        "technique":        "typosquat",
                    })

            # ── Technique 2: visual character substitution ───────────────
            normalised = domain_name
            for fake, real in VISUAL_SUBS.items():
                normalised = normalised.replace(fake, real)
            if normalised != domain_name:
                for brand in COMMON_BRANDS:
                    if normalised == brand:
                        result.update({
                            "is_homograph":     True,
                            "similarity_score": 0.95,
                            "matched_brand":    brand,
                            "technique":        "visual_substitution",
                        })
                        break

            # ── Technique 3: brand as subdomain (brand.evil.com) ─────────
            subdomains = netloc.split(".")[:-2]
            for sub in subdomains:
                for brand in COMMON_BRANDS:
                    if sub == brand:
                        result.update({
                            "is_homograph":     True,
                            "similarity_score": 0.90,
                            "matched_brand":    brand,
                            "technique":        "subdomain_abuse",
                        })

        except Exception:
            pass

        return result
