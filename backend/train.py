"""
backend/train.py
-----------------
Standalone training script.

Usage (from project root):
    python -m backend.train                    # Train with defaults (10K sample)
    python -m backend.train --sample 5000      # Train on 5K sample (fast)
    python -m backend.train --full             # Train on full dataset (slow)
    python -m backend.train --html             # Include HTML features (very slow)
"""

# ══════════════════════════════════════════════════════════════════════════
#  Smart path setup — works from any directory
# ══════════════════════════════════════════════════════════════════════════
import os
import sys
import argparse

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from utils.preprocessing import preprocess_pipeline
    from backend.ml_model import PhishingMLModel
except ModuleNotFoundError:
    from preprocessing import preprocess_pipeline  # type: ignore
    from ml_model import PhishingMLModel


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Train phishing detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m backend.train                         # Quick train on 10K sample
    python -m backend.train --sample 5000           # Super fast (5K sample)
    python -m backend.train --full                  # Train on full dataset
    python -m backend.train --dataset data/my.csv   # Custom dataset path
        """,
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(_PROJECT_ROOT, "data", "dataset.csv"),
        help="Path to dataset CSV (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--sample", type=int, default=10000,
        help="Sample size for training (default: 10000)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use full dataset (overrides --sample, may take long)",
    )
    parser.add_argument(
        "--html", action="store_true",
        help="Fetch HTML features (VERY slow — ~3-5s per URL)",
    )
    args = parser.parse_args()

    sample_size = None if args.full else args.sample

    print("=" * 70)
    print("🚀 PHISHGUARD AI — MODEL TRAINING")
    print("=" * 70)
    print(f"  Project root: {_PROJECT_ROOT}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Sample size:  {'FULL' if args.full else sample_size}")
    print(f"  Fetch HTML:   {args.html}")
    print("=" * 70)

    # Verify dataset exists
    if not os.path.exists(args.dataset):
        print(f"\n❌ ERROR: Dataset not found at {args.dataset}")
        print("   Please download the dataset first:")
        print("   → python -m utils.dataset_downloader")
        print("   OR place your dataset.csv in the data/ folder")
        sys.exit(1)

    # Step 1: Load + preprocess
    try:
        X, y, df = preprocess_pipeline(
            csv_path=args.dataset,
            sample_size=sample_size,
            fetch_html=args.html,
        )
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        sys.exit(1)

    # Step 2: Train
    model = PhishingMLModel()
    try:
        metrics = model.train(X, y, test_size=0.2, verbose=True)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

    # Step 3: Save
    try:
        model.save()
    except Exception as e:
        print(f"\n❌ Saving model failed: {e}")
        sys.exit(1)

    # Step 4: Quick sanity check
    print("\n" + "=" * 70)
    print("🔍 POST-TRAIN SANITY CHECK")
    print("=" * 70)
    print(f"\n🔝 Top 10 Most Important Features:")
    importances = model.get_feature_importances(10)
    if importances:
        max_imp = max(importances.values())
        for i, (name, imp) in enumerate(importances.items(), 1):
            bar_length = int((imp / max_imp) * 40)
            bar = "█" * bar_length
            print(f"   {i:2d}. {name:30s} {imp:.4f}  {bar}")
    else:
        print("   (No feature importances available)")

    # Quick live prediction test
    print("\n🧪 Live Prediction Test:")
    test_cases = [
        {
            "name": "Suspicious IP URL",
            "features": {
                "url_length": 45, "num_dots": 3, "has_ip_address": 1,
                "uses_https": 0, "has_suspicious_keywords": 1,
                "num_dashes": 1, "entropy": 4.2,
            },
        },
        {
            "name": "Legitimate-looking URL",
            "features": {
                "url_length": 22, "num_dots": 2, "has_ip_address": 0,
                "uses_https": 1, "has_suspicious_keywords": 0,
                "num_dashes": 0, "entropy": 3.5,
            },
        },
    ]

    for case in test_cases:
        label, conf, reason = model.predict(case["features"])
        emoji = "🚨" if label == "PHISHING" else "✅"
        print(f"   {emoji} {case['name']:30s} → {label} ({conf*100:.1f}%)")
        if reason:
            print(f"      └─ Reason: {reason}")

    # Final summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"   📦 Model:    {os.path.join('models', 'phishing_model.pkl')}")
    print(f"   📋 Features: {os.path.join('models', 'feature_names.json')}")
    print(f"   📊 Metrics:  {os.path.join('models', 'metrics.json')}")
    print(f"\n   🎯 Final Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   🎯 Final F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"\n   Next step: Build Phase 3 (SHAP explainability)")
    print("=" * 70)


if __name__ == "__main__":
    main()