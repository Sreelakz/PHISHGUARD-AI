"""
domain_intelligence.py — Analyses domain registration metadata.

In production, replace the simulated WHOIS block with a real call to:
  • python-whois  (pip install python-whois)
  • whoisapi.com  (REST API)
  • ipinfo.io     (IP geolocation / ASN)

The simulation uses deterministic heuristics on the domain string so that
the same URL always returns the same result during testing.
"""

from __future__ import annotations
import hashlib
import socket
from datetime import datetime, timedelta
from urllib.parse import urlparse

HIGH_RISK_TLDS = {".tk", ".ml", ".ga", ".cf", ".gq", ".xyz",
                  ".top", ".click", ".link", ".online", ".site"}

FREE_DNS_PROVIDERS = {"freenom", "afraid", "changeip", "dyn", "noip"}


class DomainIntelligence:
    def analyze(self, url: str) -> dict:
        result = {
            "domain":           None,
            "domain_age_days":  0,
            "is_registered":    False,
            "expiration_date":  None,
            "registrar":        None,
            "country":          None,
            "risk_tld":         False,
            "ip_address":       None,
            "reverse_dns":      None,
            "asn":              None,
        }

        try:
            parsed = urlparse(url)
            netloc = parsed.netloc or ""
            domain = netloc.replace("www.", "").split(":")[0]
            result["domain"] = domain

            # ── DNS resolution ─────────────────────────────────────────────
            try:
                ip = socket.gethostbyname(domain)
                result["is_registered"] = True
                result["ip_address"] = ip

                try:
                    result["reverse_dns"] = socket.gethostbyaddr(ip)[0]
                except Exception:
                    pass
            except socket.gaierror:
                result["is_registered"] = False
                return result

            # ── TLD risk flag ──────────────────────────────────────────────
            tld = "." + domain.rsplit(".", 1)[-1].lower() if "." in domain else ""
            result["risk_tld"] = tld in HIGH_RISK_TLDS

            # ── Deterministic WHOIS simulation ────────────────────────────
            # Use a hash of the domain so repeated calls give consistent results.
            seed = int(hashlib.md5(domain.encode()).hexdigest(), 16)

            # Suspicious patterns → shorter age
            is_suspicious = (
                result["risk_tld"]
                or any(kw in domain for kw in ["login", "secure", "verify", "bank"])
                or any(prov in domain for prov in FREE_DNS_PROVIDERS)
            )

            if is_suspicious:
                age_days = (seed % 90) + 1            # 1–90 days
                exp_days = (seed % 180) + 30          # 30–210 days
                registrar = "NameCheap" if seed % 2 else "GoDaddy"
            else:
                age_days = (seed % 4500) + 365        # 1–13 years
                exp_days = (seed % 1000) + 200        # 200–1200 days
                registrars = ["MarkMonitor", "CSC Corporate", "Network Solutions",
                               "Tucows", "Wild West Domains"]
                registrar = registrars[seed % len(registrars)]

            result["domain_age_days"] = age_days
            result["registrar"]       = registrar
            result["expiration_date"] = (
                datetime.now() + timedelta(days=exp_days)
            ).strftime("%Y-%m-%dT%H:%M:%S")

            # Simulated country based on TLD
            tld_country = {
                ".uk": "GB", ".de": "DE", ".fr": "FR", ".cn": "CN",
                ".ru": "RU", ".in": "IN", ".br": "BR",
            }
            result["country"] = tld_country.get(tld, "US")

        except Exception:
            pass

        return result