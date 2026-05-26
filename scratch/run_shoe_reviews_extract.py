"""
Run extract_dynamic_features on shoe_reviews.xlsx (debug MIN_FREQ output).
Usage: python scratch/run_shoe_reviews_extract.py [path/to/shoe_reviews.xlsx]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from feature_analysis import extract_dynamic_features

DEFAULT_PATHS = [
    ROOT / "shoe_reviews.xlsx",
    ROOT / "data" / "shoe_reviews.xlsx",
    Path.home() / "Desktop" / "shoe_reviews.xlsx",
    Path.home() / "Desktop" / "amazon_sentiment_analysis" / "shoe_reviews.xlsx",
]


def load_reviews(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    col = None
    for c in df.columns:
        if str(c).lower() in ("review", "reviews", "text", "reviewtext", "review_text"):
            col = c
            break
    if col is None:
        col = df.select_dtypes(include=["object"]).columns[0]
    out = df.rename(columns={col: "review"})
    if "predicted_sentiment" not in out.columns:
        out["predicted_sentiment"] = "neutral"
    return out


def main():
    paths = [Path(sys.argv[1])] if len(sys.argv) > 1 else DEFAULT_PATHS
    data = None
    used = None
    for p in paths:
        if p.exists():
            data = load_reviews(p)
            used = p
            break
    if data is None:
        print("shoe_reviews.xlsx not found. Tried:")
        for p in paths:
            print(f"  {p}")
        sys.exit(1)

    print(f"Loaded {len(data)} reviews from {used}\n")
    print("=" * 60)
    feats = extract_dynamic_features(data, product_name="Running Shoes", top_n=20)
    print("=" * 60)
    print(f"\nFinal feature groups ({len(feats)}): {list(feats.keys())}")


if __name__ == "__main__":
    main()
