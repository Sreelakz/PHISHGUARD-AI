"""
ssl_checker.py — Verifies SSL/TLS certificate details for a given URL.
Adds certificate grade estimation and days-until-expiry calculation.
"""

from __future__ import annotations
import ssl
import socket
from datetime import datetime
from urllib.parse import urlparse


class SSLChecker:
    def check(self, url: str) -> dict:
        result = {
            "uses_https":        False,
            "certificate_valid": False,
            "issuer":            None,
            "subject":           None,
            "expiration_date":   None,
            "days_until_expiry": None,
            "is_expired":        False,
            "tls_version":       None,
            "grade":             "F",        # A, B, C, F
        }

        try:
            parsed = urlparse(url)
            hostname = (parsed.netloc or "").split(":")[0]

            if parsed.scheme != "https":
                return result

            result["uses_https"] = True

            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=5) as raw_sock:
                with context.wrap_socket(raw_sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    result["tls_version"] = ssock.version()
                    result["certificate_valid"] = True

                    if cert:
                        # Issuer
                        if "issuer" in cert:
                            issuer = dict(x[0] for x in cert["issuer"])
                            result["issuer"] = issuer.get("organizationName", "Unknown")

                        # Subject (CN)
                        if "subject" in cert:
                            subj = dict(x[0] for x in cert["subject"])
                            result["subject"] = subj.get("commonName")

                        # Expiry
                        if "notAfter" in cert:
                            exp_str = cert["notAfter"]          # e.g. "Dec 31 23:59:59 2024 GMT"
                            result["expiration_date"] = exp_str
                            try:
                                exp_dt = datetime.strptime(exp_str, "%b %d %H:%M:%S %Y %Z")
                                days_left = (exp_dt - datetime.utcnow()).days
                                result["days_until_expiry"] = days_left
                                result["is_expired"] = days_left < 0
                            except Exception:
                                pass

                    result["grade"] = self._grade(result)

        except ssl.SSLError:
            result["uses_https"]        = True
            result["certificate_valid"] = False
        except Exception:
            pass

        return result

    @staticmethod
    def _grade(r: dict) -> str:
        if not r["certificate_valid"]:
            return "F"
        days = r.get("days_until_expiry")
        tls  = r.get("tls_version", "")
        if tls in ("TLSv1.3",) and (days is None or days > 30):
            return "A"
        if tls in ("TLSv1.2",) and (days is None or days > 14):
            return "B"
        if days is not None and days < 14:
            return "C"
        return "B"
