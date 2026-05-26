"""
Fake review detection for ReviewIntellect.
Uses only review text, rating, date, and cross-review patterns (no reviewer_id).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Lexicons ───────────────────────────────────────────────────────────────────

NEGATIVE_WORDS = [
    "broke", "broken", "terrible", "awful", "defective",
    "failed", "fail", "worst", "horrible", "waste", "poor",
    "useless", "apart", "cracked", "separated", "collapsed",
    "damaged", "disgusting", "nightmare", "catastrophic",
    "falling", "peeling", "ripped", "shredded",
]

POSITIVE_WORDS = [
    "love", "perfect", "excellent", "amazing", "outstanding",
    "fantastic", "superb", "brilliant", "flawless", "incredible",
]

GENERIC_PHRASES = [
    "highly recommend", "great product", "love it",
    "best ever", "exceeded expectations", "worth every penny",
    "five stars", "amazing product", "perfect product",
    "excellent product", "good quality", "love this product",
    "very happy", "will buy again", "no complaints",
    # Substring-friendly patterns (matched with `p in text_lower`)
    "best purchase",
    "amazing quality",
    "great quality",
    "love this",
    "absolutely love",
    "must buy",
    "totally recommend",
    "100 percent recommend",
    "zero complaints",
    "nothing bad",
    "no issues at all",
    "works perfectly",
    "does exactly",
    "as described",
]

SPECIFIC_PARTS = [
    "sole", "insole", "cushioning", "heel", "toe", "laces",
    "grip", "stitching", "material", "leather", "mesh",
    "arch", "upper", "lining", "outsole", "strap", "zipper",
    "buckle", "collar", "tread", "rubber", "padding", "width",
]

_OPINION_WORDS = set(NEGATIVE_WORDS) | set(POSITIVE_WORDS)
_WORD_RE = {w: re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in _OPINION_WORDS}
_PART_RE = {p: re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE) for p in SPECIFIC_PARTS}

OUTPUT_COLUMNS = [
    "mismatch_score",
    "burst_day",
    "max_similarity",
    "is_duplicate",
    "generic_score",
    "specificity_score",
    "is_generic",
    "is_too_short",
    "fake_score",
    "fake_label",
    "signals_fired",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map input columns to Review, Rating, Date (case-insensitive)."""
    mapping = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in ("review", "reviews", "text", "reviewtext"):
            mapping[col] = "Review"
        elif key in ("rating", "ratings", "stars", "score"):
            mapping[col] = "Rating"
        elif key in ("date", "dates", "review_date"):
            mapping[col] = "Date"
    out = df.rename(columns=mapping)
    for required in ("Review", "Rating", "Date"):
        if required not in out.columns:
            out[required] = np.nan
    return out


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).strip()


def _safe_rating(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _count_word_matches(text: str, words: List[str]) -> int:
    lower = text.lower()
    return sum(1 for w in words if _WORD_RE[w].search(lower))


def _count_phrase_matches(text: str, phrases: List[str]) -> int:
    lower = text.lower()
    return sum(1 for p in phrases if p in lower)


def _signal_mismatch(text: str, rating: Optional[float]) -> Tuple[float, bool]:
    """Sentiment-rating contradiction score and whether rule fired."""
    if not text or rating is None:
        return 0.0, False

    neg_count = _count_word_matches(text, NEGATIVE_WORDS)
    pos_count = _count_word_matches(text, POSITIVE_WORDS)
    total_opinion = neg_count + pos_count
    if total_opinion == 0:
        return 0.0, False

    contradicting = 0
    fired = False
    if rating >= 4 and neg_count >= 2:
        contradicting = neg_count
        fired = True
    elif rating <= 2 and pos_count >= 3:
        contradicting = pos_count
        fired = True

    score = contradicting / total_opinion
    return round(min(1.0, score), 3), fired


def _assign_burst_day(work: pd.DataFrame) -> pd.Series:
    """
    Burst flags from the FULL dataframe (all rows must already be present).
    Uses value_counts after append so new rows are included in date totals.
    """
    safe_dates = work["Date"].apply(
        lambda x: _safe_str(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else ""
    )
    date_counts = safe_dates.value_counts().to_dict()
    burst = pd.Series(False, index=work.index)

    for date, count in date_counts.items():
        if not date:
            continue
        mask = safe_dates == date
        if count >= 3:
            burst.loc[mask] = True
        elif count == 2:
            ratings = work.loc[mask, "Rating"].apply(_safe_rating).tolist()
            if (
                len(ratings) == 2
                and ratings[0] is not None
                and ratings[1] is not None
                and ratings[0] == ratings[1]
            ):
                burst.loc[mask] = True

    return burst


def _compute_similarity_scores(texts: List[str]) -> List[float]:
    """Max cosine similarity to any other review (0 if single/empty)."""
    n = len(texts)
    if n <= 1:
        return [0.0] * n

    cleaned = [t if t else "empty" for t in texts]
    try:
        vec = TfidfVectorizer(lowercase=True, stop_words="english")
        matrix = vec.fit_transform(cleaned)
        sim = cosine_similarity(matrix)
        np.fill_diagonal(sim, -1.0)
        max_sim = sim.max(axis=1)
        return [round(float(max(0.0, s)), 3) for s in max_sim]
    except Exception:
        return [0.0] * n


def _signal_generic(text: str, rating: Optional[float]) -> Tuple[int, int, bool]:
    g = _count_phrase_matches(text, GENERIC_PHRASES)
    s = sum(1 for p in SPECIFIC_PARTS if _PART_RE[p].search(text))
    is_gen = bool(
        g >= 1
        and s == 0
        and rating is not None
        and float(rating) == 5.0
    )
    return g, s, is_gen


def _signal_too_short(text: str, rating: Optional[float]) -> bool:
    if not text or rating is None:
        return False
    words = text.split()
    return len(words) < 6 and rating in (1.0, 5.0)


def compute_fake_score(row: Dict[str, Any]) -> int:
    score = 0
    if row.get("mismatch_score", 0) > 0.3:
        score += 30
    if row.get("burst_day"):
        score += 20
    if row.get("max_similarity", 0) > 0.5:
        score += 25
    if row.get("is_generic"):
        score += 15
    if row.get("is_too_short"):
        score += 10
    return min(score, 100)


def fake_label_from_score(score: int) -> str:
    if score <= 29:
        return "Likely genuine"
    if score <= 59:
        return "Suspicious"
    return "Likely fake"


def _signals_fired_list(row: Dict[str, Any]) -> List[str]:
    fired = []
    if row.get("mismatch_score", 0) > 0.3:
        fired.append("mismatch")
    if row.get("burst_day"):
        fired.append("burst")
    if row.get("is_duplicate"):
        fired.append("duplicate")
    if row.get("is_generic"):
        fired.append("generic")
    if row.get("is_too_short"):
        fired.append("too_short")
    return fired


def build_full_review_dataframe(
    existing_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Merge stored reviews with incoming rows, then dedupe.
    Burst and similarity require this full dataset — never pass a single row alone.
    """
    cols = ["Review", "Rating", "Date"]
    existing_df = pd.DataFrame(existing_rows) if existing_rows else pd.DataFrame(columns=cols)
    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame(columns=cols)
    if existing_df.empty:
        return _normalize_columns(new_df)
    if new_df.empty:
        return _normalize_columns(existing_df)
    full = pd.concat([existing_df, new_df], ignore_index=True)
    full = _normalize_columns(full)
    full = full.drop_duplicates(subset=["Review", "Rating", "Date"], keep="last")
    return full.reset_index(drop=True)


def detect_fake_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect potentially fake reviews.

    Input columns: Review (str), Rating (float 1-5), Date (str YYYY-MM-DD)
    Adds signal columns and fake_score / fake_label.

    IMPORTANT: df must contain the COMPLETE review set (existing + any new rows).
    Burst detection and TF-IDF similarity need cross-review context.
    """
    work = _normalize_columns(df.copy())
    n = len(work)
    work["mismatch_score"] = 0.0
    work["burst_day"] = False
    work["max_similarity"] = 0.0
    work["is_duplicate"] = False
    work["generic_score"] = 0
    work["specificity_score"] = 0
    work["is_generic"] = False
    work["is_too_short"] = False
    work["fake_score"] = 0
    work["fake_label"] = "Likely genuine"
    work["signals_fired"] = [[] for _ in range(n)]

    if n == 0:
        return work

    work["burst_day"] = _assign_burst_day(work)

    texts = [_safe_str(work.iloc[i]["Review"]) for i in range(n)]
    max_sims = _compute_similarity_scores(texts)

    for i in range(n):
        text = texts[i]
        rating = _safe_rating(work.iloc[i]["Rating"])
        if not text:
            continue

        ms, _ = _signal_mismatch(text, rating)
        g, sp, is_gen = _signal_generic(text, rating)
        too_short = _signal_too_short(text, rating)
        max_sim = max_sims[i]

        row_data = {
            "mismatch_score": ms,
            "burst_day": bool(work.iloc[i]["burst_day"]),
            "max_similarity": max_sim,
            "is_duplicate": max_sim > 0.5,
            "generic_score": g,
            "specificity_score": sp,
            "is_generic": is_gen,
            "is_too_short": too_short,
        }
        row_data["fake_score"] = compute_fake_score(row_data)
        row_data["fake_label"] = fake_label_from_score(row_data["fake_score"])
        row_data["signals_fired"] = _signals_fired_list(row_data)

        for k, v in row_data.items():
            if k == "signals_fired":
                work.at[work.index[i], k] = v
            else:
                work.iloc[i, work.columns.get_loc(k)] = v

    return work


def build_detection_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate summary for API response."""
    if df.empty or "fake_label" not in df.columns:
        return {
            "total": 0,
            "likely_fake": 0,
            "suspicious": 0,
            "likely_genuine": 0,
            "fake_percentage": 0.0,
            "top_signals": [],
        }

    labels = df["fake_label"].fillna("Likely genuine")
    total = len(df)
    likely_fake = int((labels == "Likely fake").sum())
    suspicious = int((labels == "Suspicious").sum())
    likely_genuine = int((labels == "Likely genuine").sum())
    flagged = int((labels != "Likely genuine").sum())
    flagged_pct = round(100.0 * flagged / max(total, 1), 1)

    signal_counts: Counter = Counter()
    if "signals_fired" in df.columns:
        for sigs in df["signals_fired"]:
            if isinstance(sigs, list):
                signal_counts.update(sigs)
    top_signals = [s for s, _ in signal_counts.most_common()]

    return {
        "total": total,
        "likely_fake": likely_fake,
        "suspicious": suspicious,
        "likely_genuine": likely_genuine,
        "flagged": flagged,
        "fake_percentage": flagged_pct,
        "top_signals": top_signals,
    }


def detect_fake_reviews_from_records(
    reviews: List[Dict[str, Any]],
    existing_reviews: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    API helper: list of {text, rating, date} -> enriched rows + summary.
    Merges existing_reviews (from DB) with incoming so burst/similarity see full data.
    """
    def _to_row(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Review": _safe_str(r.get("text", r.get("Review", ""))),
            "Rating": r.get("rating", r.get("Rating")),
            "Date": _safe_str(r.get("date", r.get("Date", ""))),
        }

    incoming_rows = [_to_row(r) for r in reviews]
    existing_rows = [_to_row(r) for r in (existing_reviews or [])]
    df = build_full_review_dataframe(existing_rows, incoming_rows)
    result = detect_fake_reviews(df)
    summary = build_detection_summary(result)

    def _row_to_api(row: pd.Series) -> Dict[str, Any]:
        return {
            "text": _safe_str(row.get("Review", "")),
            "rating": _safe_rating(row.get("Rating")),
            "date": _safe_str(row.get("Date", "")),
            "mismatch_score": float(row.get("mismatch_score", 0)),
            "burst_day": bool(row.get("burst_day", False)),
            "max_similarity": float(row.get("max_similarity", 0)),
            "is_duplicate": bool(row.get("is_duplicate", False)),
            "generic_score": int(row.get("generic_score", 0)),
            "specificity_score": int(row.get("specificity_score", 0)),
            "is_generic": bool(row.get("is_generic", False)),
            "is_too_short": bool(row.get("is_too_short", False)),
            "fake_score": int(row.get("fake_score", 0)),
            "fake_label": str(row.get("fake_label", "Likely genuine")),
            "signals_fired": list(row.get("signals_fired", [])),
        }

    result_by_key: Dict[Tuple[str, Optional[float], str], Dict[str, Any]] = {}
    for _, row in result.iterrows():
        key = (
            _safe_str(row.get("Review", "")),
            _safe_rating(row.get("Rating")),
            _safe_str(row.get("Date", "")),
        )
        result_by_key[key] = _row_to_api(row)

    out_reviews = []
    for inc in incoming_rows:
        key = (inc["Review"], _safe_rating(inc["Rating"]), inc["Date"])
        if key in result_by_key:
            out_reviews.append(result_by_key[key])

    summary = build_detection_summary(result)
    return out_reviews, summary


# ── Self-test ──────────────────────────────────────────────────────────────────

def _load_shoe_reviews_for_test() -> Optional[pd.DataFrame]:
    """Load shoe_reviews.xlsx if present (Review, Rating, Date)."""
    from pathlib import Path

    candidates = [
        Path(__file__).resolve().parent.parent / "shoe_reviews.xlsx",
        Path(__file__).resolve().parent.parent / "data" / "shoe_reviews.xlsx",
        Path.home() / "Desktop" / "shoe_reviews.xlsx",
        Path.home() / "Desktop" / "amazon_sentiment_analysis" / "shoe_reviews.xlsx",
    ]
    for path in candidates:
        if not path.exists():
            continue
        raw = pd.read_excel(path)
        col_map = {}
        for c in raw.columns:
            k = str(c).strip().lower()
            if k in ("review", "reviews", "text"):
                col_map[c] = "Review"
            elif k in ("rating", "ratings", "stars"):
                col_map[c] = "Rating"
            elif k in ("date", "dates"):
                col_map[c] = "Date"
        df = raw.rename(columns=col_map)
        for req in ("Review", "Rating", "Date"):
            if req not in df.columns:
                df[req] = np.nan
        return df[["Review", "Rating", "Date"]].dropna(subset=["Review"])
    return None


if __name__ == "__main__":
    TEST_REVIEW = {
        "Review": (
            "Absolutely love this product. Amazing quality. "
            "Highly recommend to everyone. Best purchase ever."
        ),
        "Rating": 5.0,
        "Date": "2024-08-24",
    }

    shoe_df = _load_shoe_reviews_for_test()
    if shoe_df is not None:
        print(f"Loaded shoe_reviews ({len(shoe_df)} rows)\n")
        full_df = pd.concat([shoe_df, pd.DataFrame([TEST_REVIEW])], ignore_index=True)
        out = detect_fake_reviews(full_df)
        target = out[
            out["Review"].astype(str).str.contains("Absolutely love", case=False, na=False)
        ]
        if target.empty:
            print("ERROR: test review row not found")
        else:
            row = target.iloc[-1]
            print("=== Verification row (appended fake review) ===")
            for field in OUTPUT_COLUMNS:
                print(f"  {field}: {row.get(field)}")
            checks = [
                ("burst_day", bool(row["burst_day"]) is True),
                ("is_generic", bool(row["is_generic"]) is True),
                ("generic_score >= 2", int(row["generic_score"]) >= 2),
                ("specificity_score == 0", int(row["specificity_score"]) == 0),
                ("fake_score >= 35", int(row["fake_score"]) >= 35),
                ('fake_label == "Suspicious"', str(row["fake_label"]) == "Suspicious"),
            ]
            print("\nChecks:")
            for name, ok in checks:
                print(f"  {'PASS' if ok else 'FAIL'}: {name}")
        print()
    else:
        print("shoe_reviews.xlsx not found — using synthetic burst dataset\n")
        burst_day = "2024-08-24"
        synthetic = pd.DataFrame(
            [
                {"Review": f"Generic great product five stars review {i}", "Rating": 5.0, "Date": burst_day}
                for i in range(3)
            ]
            + [TEST_REVIEW]
        )
        out = detect_fake_reviews(synthetic)
        row = out.iloc[-1]
        print("=== Verification row (synthetic 3 + test) ===")
        for field in OUTPUT_COLUMNS:
            print(f"  {field}: {row.get(field)}")
        print()

    samples = pd.DataFrame([
        {
            "Review": "Love it perfect product amazing five stars best ever",
            "Rating": 5.0,
            "Date": "2024-01-10",
        },
        {
            "Review": "Terrible broke apart defective waste horrible nightmare",
            "Rating": 5.0,
            "Date": "2024-01-10",
        },
        {
            "Review": "Love it perfect product amazing five stars best ever",
            "Rating": 5.0,
            "Date": "2024-01-10",
        },
        {
            "Review": "Excellent amazing outstanding flawless incredible perfect",
            "Rating": 1.0,
            "Date": "2024-02-01",
        },
        {
            "Review": "Great sole cushioning",
            "Rating": 5.0,
            "Date": "2024-03-01",
        },
    ])

    print("=== Fake Review Detector — sample run ===\n")
    out = detect_fake_reviews(samples)
    cols = ["Review", "Rating", "Date"] + [c for c in OUTPUT_COLUMNS if c in out.columns]
    for _, row in out[cols].iterrows():
        print("-" * 60)
        print(f"Review: {str(row['Review'])[:70]}...")
        print(f"Rating: {row['Rating']}  Date: {row['Date']}")
        print(f"  mismatch_score:   {row['mismatch_score']}")
        print(f"  burst_day:        {row['burst_day']}")
        print(f"  max_similarity:   {row['max_similarity']}")
        print(f"  is_duplicate:     {row['is_duplicate']}")
        print(f"  generic_score:    {row['generic_score']}")
        print(f"  specificity_score:{row['specificity_score']}")
        print(f"  is_generic:       {row['is_generic']}")
        print(f"  is_too_short:     {row['is_too_short']}")
        print(f"  fake_score:       {row['fake_score']}")
        print(f"  fake_label:       {row['fake_label']}")
        print(f"  signals_fired:    {row.get('signals_fired', [])}")

    print("\n=== Summary ===")
    print(build_detection_summary(out))
