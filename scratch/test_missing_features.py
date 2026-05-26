"""
Sample test for Missing Feature Detector pipeline.
Run from repo root:
  python scratch/test_missing_features.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from missing_feature_detector import detect_missing_features  # noqa: E402

MOCK_REVIEWS = [
    {
        "id": "r1",
        "text": "Great sound but I wish it had noise cancelling. Really need that for flights.",
        "rating": 3,
        "date": "2025-01-10",
    },
    {
        "id": "r2",
        "text": "Battery is fine. If only they included a USB-C cable in the box — please add one desperately.",
        "rating": 2,
        "date": "2025-01-11",
    },
    {
        "id": "r3",
        "text": "Comfortable fit. Would be better if there was a pocket for the case. Missing that feature badly.",
        "rating": 3,
        "date": "2025-01-12",
    },
    {
        "id": "r4",
        "text": "Love everything! Best purchase ever.",
        "rating": 5,
        "date": "2025-01-13",
    },
    {
        "id": "r5",
        "text": "It doesn't have wireless charging. Needs a Qi pad built in. Should include that at this price.",
        "rating": 2,
        "date": "2025-01-14",
    },
    {
        "id": "r6",
        "text": "Wish it had noise cancelling like the Pro model. The Pro has ANC and this doesn't have it.",
        "rating": 2,
        "date": "2025-01-15",
    },
]

API_SAMPLE = {
    "product_id": "TEST-HEADPHONES",
    "reviews": MOCK_REVIEWS,
}


if __name__ == "__main__":
    print("=== Missing Feature Detector — mock run ===\n")
    results = detect_missing_features(MOCK_REVIEWS)
    print(json.dumps(results, indent=2))
    print(f"\nFound {len(results)} clustered missing-feature(s).")
    print("\n--- POST /api/missing-features body (sample) ---")
    print(json.dumps(API_SAMPLE, indent=2))
