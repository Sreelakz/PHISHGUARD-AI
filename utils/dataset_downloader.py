"""
utils/dataset_downloader.py
----------------------------
Fetches a sample phishing dataset. Use this if you can't access Kaggle.

Primary source: PhishTank (phishing URLs)
Legitimate source: Top Alexa domains (from a public GitHub mirror)
"""

import os
import pandas as pd
import requests
from io import StringIO

# Public datasets that don't require auth
PHISHING_URL = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt"
LEGIT_URL = "https://raw.githubusercontent.com/zapret-info/z-i/master/dump.csv"  # fallback


def download_sample_dataset(output_path: str = "data/dataset.csv", max_per_class: int = 5000):
    """
    Build a balanced phishing/legitimate dataset from public sources.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("📥 Downloading phishing URLs from PhishTank mirror...")
    try:
        resp = requests.get(PHISHING_URL, timeout=30)
        resp.raise_for_status()
        phishing_urls = [line.strip() for line in resp.text.splitlines()
                         if line.strip() and line.startswith("http")]
        phishing_urls = phishing_urls[:max_per_class]
        print(f"   ✅ Got {len(phishing_urls)} phishing URLs")
    except Exception as e:
        print(f"   ❌ Failed to download phishing: {e}")
        return None

    print("📥 Generating legitimate URLs from top domains list...")
    # Top 10,000 sites (hardcoded small sample for reliability)
    legit_urls = _generate_legit_urls(max_per_class)
    print(f"   ✅ Got {len(legit_urls)} legitimate URLs")

    # Build DataFrame
    df = pd.DataFrame({
        "url": phishing_urls + legit_urls,
        "label": [1] * len(phishing_urls) + [0] * len(legit_urls)
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset saved to {output_path} ({len(df)} rows)")
    return df


def _generate_legit_urls(n: int = 5000):
    """Fetch top domains from a public Alexa mirror."""
    try:
        url = "https://raw.githubusercontent.com/zakird/crux-top-lists/main/data/global/202309.csv.gz"
        df = pd.read_csv(url, compression="gzip", nrows=n)
        return ["https://" + domain for domain in df.iloc[:, 0].tolist()]
    except Exception:
        # Fallback hardcoded list
        base = ["google.com", "youtube.com", "facebook.com", "wikipedia.org",
                "amazon.com", "twitter.com", "instagram.com", "linkedin.com",
                "github.com", "stackoverflow.com", "reddit.com", "netflix.com",
                "microsoft.com", "apple.com", "yahoo.com", "bing.com",
                "cnn.com", "bbc.com", "nytimes.com", "medium.com"]
        return [f"https://www.{d}/page/{i}" for i in range(n // len(base) + 1) for d in base][:n]


if __name__ == "__main__":
    download_sample_dataset()