"""
Verify desire-expression fixes for missing feature detector.
Run: python scratch/test_missing_desire_fix.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from missing_feature_detector import (  # noqa: E402
    detect_missing_features,
    extract_desire_sentences,
    extract_missing_feature,
    filter_candidate_reviews,
)

REVIEW_1 = {
    "id": "t1",
    "text": "it would be good if it has a good leather",
    "rating": 3,
    "sentiment_score": 0.45,
}
REVIEW_2 = {
    "id": "t2",
    "text": "it is a perfect product if the colour was good",
    "rating": 4,
    "sentiment_score": 0.55,
}


def main():
    reviews = [REVIEW_1, REVIEW_2]
    print("=" * 60)
    print("STAGE 1 — filter_candidate_reviews")
    print("=" * 60)
    passed = filter_candidate_reviews(reviews)
    for r in passed:
        print(f"  kept id={r['id']} rating={r.get('rating')} score={r.get('_sentiment_score')}")

    print("\n" + "=" * 60)
    print("STAGE 2 — desire sentences + raw features per review")
    print("=" * 60)
    for rev in passed:
        text = rev["text"]
        print(f"\nReview {rev['id']}: {text!r}")
        sents = extract_desire_sentences(text)
        print(f"  desire sentences ({len(sents)}): {sents}")
        for s in sents:
            feats = extract_missing_feature(s)
            print(f"    -> raw features: {feats}")

    print("\n" + "=" * 60)
    print("STAGE 3 — full pipeline (debug=True)")
    print("=" * 60)
    result = detect_missing_features(reviews, debug=True)

    print("\n" + "=" * 60)
    print("ASSERTIONS")
    print("=" * 60)
    features = {item["feature"].lower() for item in result}
    joined = " ".join(features)
    ok1 = "leather" in joined
    ok2 = "colour" in joined or "color" in joined
    print(f"  review_1 -> leather: {'PASS' if ok1 else 'FAIL'} (features: {features})")
    print(f"  review_2 -> colour:  {'PASS' if ok2 else 'FAIL'} (features: {features})")
    if not (ok1 and ok2):
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
