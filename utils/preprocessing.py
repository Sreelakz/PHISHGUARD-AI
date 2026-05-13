"""
utils/preprocessing.py
-----------------------
Loads and cleans the phishing dataset, then extracts features for ML training.
"""

import os
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.feature_extractor import FeatureExtractor


LABEL_MAPPINGS = {
    "phishing": 1, "bad": 1, "malicious": 1, "malware": 1, "1": 1, 1: 1,
    "legitimate": 0, "good": 0, "benign": 0, "safe": 0, "0": 0, 0: 0,
}


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load CSV with auto-detection of url/label columns.
    Handles common dataset formats from Kaggle, PhishTank, UCI.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. "
                                f"Run `python utils/dataset_downloader.py` first.")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Auto-detect URL column
    url_col = None
    for candidate in ["url", "urls", "link", "website", "domain"]:
        if candidate in df.columns:
            url_col = candidate
            break
    if not url_col:
        raise ValueError(f"No URL column found. Got columns: {list(df.columns)}")

    # Auto-detect label column
    label_col = None
    for candidate in ["label", "result", "type", "class", "status", "is_phishing"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if not label_col:
        raise ValueError(f"No label column found. Got columns: {list(df.columns)}")

    df = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label"})

    # Normalize labels
    if df["label"].dtype == object:
        df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["label"] = df["label"].map(LABEL_MAPPINGS)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove nulls, duplicates, invalid URLs."""
    initial = len(df)
    df = df.dropna(subset=["url", "label"])
    df = df[df["url"].str.len() > 4]
    df = df.drop_duplicates(subset=["url"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    print(f"🧹 Cleaned: {initial} → {len(df)} rows "
          f"({df['label'].sum()} phishing, {(df['label']==0).sum()} legitimate)")
    return df.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame, fetch_html: bool = False,
                         sample_size: int = None) -> tuple:
    """
    Extract features from URLs → returns (X, y) ready for ML.

    Args:
        fetch_html: If True, fetches HTML (SLOW, ~3s/URL). Keep False for training.
        sample_size: Downsample for quick experiments (e.g., 5000).
    """
    if sample_size and len(df) > sample_size:
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 2), random_state=42)
        ).reset_index(drop=True)
        print(f"⚡ Sampled down to {len(df)} rows for speed")

    print(f"⚙️  Extracting features from {len(df)} URLs "
          f"({'with' if fetch_html else 'without'} HTML)...")

    fe = FeatureExtractor()
    X = fe.extract_batch(df["url"].tolist(), fetch_html=fetch_html, verbose=True)
    y = df["label"].values

    print(f"✅ Feature matrix ready: {X.shape}")
    return X, y


def preprocess_pipeline(csv_path: str = "data/dataset.csv",
                        sample_size: int = 10000,
                        fetch_html: bool = False):
    """End-to-end pipeline: load → clean → extract → return (X, y)."""
    print("=" * 70)
    print("🚀 PREPROCESSING PIPELINE")
    print("=" * 70)
    df = load_dataset(csv_path)
    df = clean_dataset(df)
    X, y = build_feature_matrix(df, fetch_html=fetch_html, sample_size=sample_size)
    print("=" * 70)
    return X, y, df


if __name__ == "__main__":
    X, y, df = preprocess_pipeline(sample_size=2000)
    print("\n📊 Sample features:")
    print(X.head(3).T)
    print(f"\n📈 Label distribution: {pd.Series(y).value_counts().to_dict()}")