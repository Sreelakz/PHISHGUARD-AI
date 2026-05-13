"""
redirect_detector.py — Follows HTTP redirects and analyses the chain for
suspicious patterns (cross-domain hops, HTTP→HTTPS downgrade, etc.).
"""

from __future__ import annotations
import requests
from urllib.parse import urlparse


class RedirectDetector:
    def detect(self, url: str) -> dict:
        result = {
            "has_redirects":          False,
            "redirect_count":         0,
            "redirect_chain":         [],
            "cross_domain_redirects": 0,
            "final_url":              url,
            "suspicious_redirect":    False,
        }

        try:
            session = requests.Session()
            resp = session.get(
                url,
                timeout=5,
                headers={"User-Agent": "Mozilla/5.0"},
                allow_redirects=True,
            )

            result["redirect_count"] = len(resp.history)
            result["has_redirects"]  = result["redirect_count"] > 0
            result["final_url"]      = resp.url

            if resp.history:
                chain = [r.url for r in resp.history] + [resp.url]
                result["redirect_chain"] = chain

                # Cross-domain count
                base = _base_domain(urlparse(url).netloc)
                cross = sum(
                    1 for u in chain[1:]
                    if _base_domain(urlparse(u).netloc) != base
                )
                result["cross_domain_redirects"] = cross

                # Suspicious: cross-domain redirects OR HTTP final destination
                result["suspicious_redirect"] = (
                    cross > 0 or not resp.url.startswith("https")
                )

        except Exception:
            pass

        return result


def _base_domain(netloc: str) -> str:
    parts = netloc.replace("www.", "").split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else netloc