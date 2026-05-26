"""
Missing Feature Detector — desire-expression mining.
Uses spaCy when available; regex + sklearn fallbacks otherwise (no NLTK).
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ── Lazy-loaded models ─────────────────────────────────────────────────────────
_nlp = None  # False = tried and failed; None = not loaded yet
_embedder = None
_embedder_failed = False

GENERIC_SKIP = {
    "it", "this", "that", "one", "ones", "something", "anything", "everything",
    "thing", "things", "stuff", "product", "item", "model", "version", "feature",
    "option", "bit", "lot", "kind", "sort", "way", "part", "area", "place",
    "they", "them", "their", "some", "more", "less", "much", "many",
    "the", "a", "an", "and", "or", "but", "for", "with", "to", "of", "in", "on",
    "only", "had", "would", "included", "added", "supported", "also", "just",
}

URGENCY_WORDS = {"really", "desperately", "please", "badly", "urgently", "seriously", "definitely"}

DESIRE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"wish\s+it\s+had",
        r"wish\s+(?:they|it|this)\s+(?:had|would)",
        r"if\s+only",
        r"would\s+be\s+better\s+if",
        r"would\s+be\s+(?:good|nice|great|better|perfect)\s+if",
        r"if\s+(?:the|it|they|this)\s+.{2,30}?\s+(?:was|were|had|is)\s+(?:better|good|different|more|less)",
        r"(?:good|nice|great|perfect)\s+(?:product|shoe|item)\s+(?:but|if|except)",
        r"(?:just|only)\s+(?:wish|hope|want).{0,30}?(?:had|have|include)",
        r"(?:could|should)\s+(?:really\s+)?use\s+(?:a|an|some|more)",
        r"(?:a\s+bit|little)\s+more\s+.{3,30}?\s+would",
        r"(?:improve|fix|change)\s+the\s+.{3,30}",
        r"needs?\s+a(?:n)?\s+",
        r"\bmissing\b",
        r"doesn'?t\s+have",
        r"do\s+not\s+have",
        r"should\s+include",
        r"please\s+add",
        r"would\s+love\s+(?:to\s+see|if)",
        r"hope\s+they\s+add",
        r"lacks?\s+(?:a\s+)?",
        r"without\s+(?:a\s+)?",
        r"no\s+(?:built[- ]in|integrated)\s+",
    ]
]

CONDITIONAL_MARKERS = re.compile(r"\b(if|wish|would|could|should)\b", re.IGNORECASE)

# Regex capture groups for feature phrases (fallback when spaCy unavailable)
FEATURE_CAPTURE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"wish\s+it\s+had\s+(?:a\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"wish\s+(?:they|it|this)\s+(?:had|would)\s+(?:a\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"if\s+only\s+(?:they|it|the(?:y)?)?\s*(?:had|included|added|supported)\s+(?:a\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"would\s+be\s+better\s+if\s+(?:it\s+)?(?:had\s+)?(?:a\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"needs?\s+a(?:n)?\s+([a-z][a-z0-9\s-]{2,40})",
        r"please\s+add\s+(?:a\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"should\s+include\s+(?:a\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"\bmissing\s+(?:a\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"doesn'?t\s+have\s+(?:a\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"do\s+not\s+have\s+(?:a\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"lacks?\s+(?:a\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"hope\s+they\s+add\s+(?:a\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"would\s+be\s+(?:good|nice|great|better|perfect)\s+if\s+(?:it\s+)?(?:has|have|had)?\s*(?:a\s+)?(?:good\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"if\s+(?:the|it|this|they)\s+([a-z][a-z0-9\s-]{2,30}?)\s+(?:was|were|had|is)\s+(?:better|good|different|more|less)",
        r"(?:good|nice|great|perfect)\s+(?:product|shoe|item)\s+if\s+(?:the\s+)?([a-z][a-z0-9]+)",
        r"(?:good|nice|great|perfect)\s+(?:product|shoe|item)\s+but\s+(?:the\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"(?:just|only)\s+(?:wish|hope|want).{0,30}?(?:had|have|include)\s+(?:a\s+)?([a-z][a-z0-9\s-]{2,40})",
        r"(?:could|should)\s+(?:really\s+)?use\s+(?:a|an|some|more)\s+([a-z][a-z0-9\s-]{2,40})",
        r"(?:a\s+bit|little)\s+more\s+([a-z][a-z0-9\s-]{3,40})\s+would",
        r"(?:improve|fix|change)\s+the\s+([a-z][a-z0-9\s-]{3,40})",
    ]
]

TOP_N = 10


@dataclass
class DesireHit:
    review_id: Any
    review_text: str
    sentence: str
    feature_phrase: str
    urgency_hits: int = 0


def spacy_available() -> bool:
    """Return True if spaCy model loaded successfully."""
    return _get_nlp() is not None


def _get_nlp():
    global _nlp
    if _nlp is False:
        return None
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
        logger.info("Missing Feature Detector: using spaCy en_core_web_sm")
        return _nlp
    except Exception as exc:
        logger.warning("spaCy unavailable (%s); using regex fallback", exc)
        _nlp = False
        return None


def _load_embedder():
    global _embedder, _embedder_failed
    if _embedder_failed:
        return None
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Missing Feature Detector: using sentence-transformers")
        return _embedder
    except Exception as exc:
        logger.warning("sentence-transformers unavailable (%s); using TF-IDF clustering", exc)
        _embedder_failed = True
        return None


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text))
    return [p.strip() for p in parts if p and len(p.strip()) > 5]


def _sanitize_phrase(raw: str) -> Optional[str]:
    if not raw:
        return None
    raw = re.sub(r"[^a-z0-9\s-]", " ", raw.lower())
    # Trim trailing clause residue (e.g. "colour was good" -> "colour")
    raw = re.split(
        r"\s+(?:was|were|had|has|have|is|are|would|could|should)\b",
        raw,
        maxsplit=1,
    )[0].strip()
    words = []
    for w in raw.split():
        if w in GENERIC_SKIP:
            break
        if len(w) >= 2:
            words.append(w)
        if len(words) >= 4:
            break
    phrase = " ".join(words).strip()
    if len(phrase) < 3 or phrase in GENERIC_SKIP:
        return None
    return phrase


def estimate_sentiment_score(
    rating: Optional[int] = None,
    sentiment_score: Optional[float] = None,
    sentiment_label: Optional[str] = None,
) -> float:
    if sentiment_score is not None:
        return float(sentiment_score)
    if sentiment_label:
        lbl = sentiment_label.lower()
        if lbl == "negative":
            return 0.15
        if lbl == "mixed":
            return 0.35
        if lbl == "neutral":
            return 0.5
        if lbl == "positive":
            return 0.85
    if rating is not None:
        return {1: 0.1, 2: 0.22, 3: 0.42, 4: 0.72, 5: 0.9}.get(int(rating), 0.5)
    return 0.5


def _safe_rating(rating: Any) -> Optional[int]:
    if rating is None:
        return None
    try:
        return int(rating)
    except (TypeError, ValueError):
        try:
            return int(float(rating))
        except (TypeError, ValueError):
            return None


def filter_candidate_reviews(
    reviews: List[Dict[str, Any]],
    rating_max: int = 4,
    sentiment_max: float = 0.6,
) -> List[Dict[str, Any]]:
    out = []
    for r in reviews:
        rating = _safe_rating(r.get("rating"))
        score = estimate_sentiment_score(
            rating=rating,
            sentiment_score=r.get("sentiment_score"),
            sentiment_label=r.get("sentiment"),
        )
        r = {**r, "_sentiment_score": score, "rating": rating}
        rating_ok = rating is None or rating <= rating_max
        sentiment_ok = score < sentiment_max
        if rating_ok or sentiment_ok:
            out.append(r)
    return out


def extract_desire_sentences(text: str) -> List[str]:
    if not text or not str(text).strip():
        return []
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(str(text))
        return [
            sent.text.strip()
            for sent in doc.sents
            if any(p.search(sent.text) for p in DESIRE_PATTERNS)
        ]
    return [
        s for s in _split_sentences(text)
        if any(p.search(s) for p in DESIRE_PATTERNS)
    ]


def _extract_missing_feature_regex(sentence: str) -> List[str]:
    phrases = []
    for pat in FEATURE_CAPTURE_PATTERNS:
        for m in pat.finditer(sentence):
            cleaned = _sanitize_phrase(m.group(1))
            if cleaned:
                phrases.append(cleaned)
    if not phrases and any(p.search(sentence) for p in DESIRE_PATTERNS):
        for n in re.findall(r"\b([a-z]{4,})\b", sentence.lower()):
            if n not in GENERIC_SKIP:
                phrases.append(n)
                break
    seen = set()
    out = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _clean_spacy_chunk(chunk) -> Optional[str]:
    words = []
    for tok in chunk:
        if tok.is_pronoun or tok.pos_ == "PRON":
            continue
        if tok.text.lower() in GENERIC_SKIP:
            continue
        if tok.is_stop and tok.pos_ not in ("NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"):
            continue
        if tok.is_alpha or tok.text == "-":
            words.append(tok.text.lower())
    phrase = " ".join(words).strip()
    if not phrase or phrase in GENERIC_SKIP:
        return None
    # Allow single feature nouns (e.g. "colour", "leather")
    if len(phrase) < 3 and len(words) == 1 and len(words[0]) < 4:
        return None
    return phrase


def _desire_trigger_end(sentence: str) -> int:
    """Character index after the last desire-pattern match (for post-trigger extraction)."""
    ends = [m.end() for p in DESIRE_PATTERNS for m in p.finditer(sentence)]
    return max(ends) if ends else -1


def _is_conditional_sentence(sentence: str) -> bool:
    return bool(CONDITIONAL_MARKERS.search(sentence))


def _chunk_has_subj_or_obj(chunk) -> bool:
    root = chunk.root
    if root.dep_ in ("nsubj", "dobj", "attr", "pobj"):
        return True
    return any(t.dep_ in ("nsubj", "dobj", "attr", "pobj") for t in chunk)


def extract_missing_feature(sentence: str) -> List[str]:
    """
    1. Noun chunks after desire trigger span
    2. If empty + conditional sentence: nsubj/dobj noun chunks
    3. Regex fallback when spaCy unavailable or still empty
    """
    if not sentence.strip():
        return []
    nlp = _get_nlp()
    if nlp is None:
        return _extract_missing_feature_regex(sentence)

    doc = nlp(sentence)
    phrases: List[str] = []
    trigger_end = _desire_trigger_end(sentence)

    # Pass 1: noun chunks at or after the desire trigger
    for chunk in doc.noun_chunks:
        if trigger_end >= 0 and chunk.start_char < trigger_end - 2:
            continue
        cleaned = _clean_spacy_chunk(chunk)
        if cleaned and cleaned not in GENERIC_SKIP:
            phrases.append(cleaned)

    # Pass 2: conditional subject/object nouns (e.g. "if the colour was good")
    if not phrases and _is_conditional_sentence(sentence):
        for chunk in doc.noun_chunks:
            if not _chunk_has_subj_or_obj(chunk):
                continue
            cleaned = _clean_spacy_chunk(chunk)
            if not cleaned or cleaned in GENERIC_SKIP:
                continue
            # Prefer the feature being discussed, not generic "product"
            if cleaned == "product" and len(list(doc.noun_chunks)) > 1:
                continue
            phrases.append(cleaned)

    # Pass 3: head nouns after trigger
    if not phrases and trigger_end >= 0:
        for tok in doc:
            if tok.idx is not None and tok.idx < trigger_end:
                continue
            if tok.pos_ in ("NOUN", "PROPN") and tok.text.lower() not in GENERIC_SKIP and not tok.is_stop:
                phrases.append(tok.lemma_.lower())

    if not phrases:
        phrases = _extract_missing_feature_regex(sentence)

    seen = set()
    unique = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _urgency_count(text: str) -> int:
    lower = text.lower()
    return sum(1 for w in URGENCY_WORDS if re.search(rf"\b{re.escape(w)}\b", lower))


def _singleton_clusters(n: int) -> List[List[int]]:
    return [[i] for i in range(n)]


def _tfidf_matrix(phrases: List[str]):
    """Build TF-IDF matrix; fall back to character n-grams if word vocab is empty."""
    try:
        vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-z][a-z]+\b")
        return vec.fit_transform(phrases)
    except ValueError:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
        return vec.fit_transform(phrases)


def _cluster_with_tfidf(
    feature_phrases: List[str],
    similarity_threshold: float,
) -> List[List[int]]:
    n = len(feature_phrases)
    if n <= 1:
        return _singleton_clusters(n)

    try:
        if n == 2:
            mat = _tfidf_matrix(feature_phrases)
            sim = cosine_similarity(mat)[0, 1]
            return [[0, 1]] if sim >= similarity_threshold else [[0], [1]]

        mat = _tfidf_matrix(feature_phrases)
        distance_threshold = 1.0 - similarity_threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(mat.toarray())
        clusters: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[int(label)].append(idx)
        return list(clusters.values())
    except Exception as exc:
        logger.warning("TF-IDF clustering failed (%s); using singleton clusters", exc)
        return _singleton_clusters(n)


def cluster_features(
    feature_phrases: List[str],
    similarity_threshold: float = 0.35,
) -> List[List[int]]:
    n = len(feature_phrases)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    embedder = _load_embedder()
    if embedder is not None:
        try:
            if n == 2:
                emb = embedder.encode(feature_phrases, convert_to_numpy=True)
                sim = float(cosine_similarity(emb)[0, 1])
                return [[0, 1]] if sim >= similarity_threshold else [[0], [1]]

            embeddings = embedder.encode(feature_phrases, convert_to_numpy=True)
            distance_threshold = 1.0 - similarity_threshold
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)
            clusters: Dict[int, List[int]] = defaultdict(list)
            for idx, label in enumerate(labels):
                clusters[int(label)].append(idx)
            return list(clusters.values())
        except Exception as exc:
            logger.warning("Embedding clustering failed (%s); falling back to TF-IDF", exc)

    return _cluster_with_tfidf(feature_phrases, similarity_threshold)


def score_clusters(
    clusters: List[List[DesireHit]],
    _all_hits: List[DesireHit],
    _phrase_list: List[str],
) -> List[Dict[str, Any]]:
    scored = []
    for cluster_hits in clusters:
        if not cluster_hits:
            continue
        count = len(cluster_hits)
        urgency_boost = sum(h.urgency_hits for h in cluster_hits)
        score = float(count) + 0.5 * urgency_boost
        feature = Counter(h.feature_phrase for h in cluster_hits).most_common(1)[0][0]
        if not feature or feature.strip() in GENERIC_SKIP:
            continue
        examples = []
        seen_ex = set()
        for h in cluster_hits:
            snippet = h.sentence if len(h.sentence) <= 280 else h.sentence[:277] + "..."
            if snippet not in seen_ex:
                seen_ex.add(snippet)
                examples.append(snippet)
        scored.append({
            "feature": feature,
            "count": count,
            "score": round(score, 2),
            "examples": examples,
        })
    scored.sort(key=lambda x: (x["score"], x["count"]), reverse=True)
    return scored[:TOP_N]


def detect_missing_features(
    reviews: List[Dict[str, Any]],
    similarity_threshold: float = 0.35,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    try:
        candidates = filter_candidate_reviews(reviews)
        if debug:
            print(f"\n[filter] {len(candidates)}/{len(reviews)} reviews passed "
                  f"(rating<={4} or sentiment<0.6)")

        hits: List[DesireHit] = []

        for rev in candidates:
            text = str(rev.get("text") or "").strip()
            if not text:
                continue
            rid = rev.get("id", rev.get("review_id", ""))
            if debug:
                print(f"\n[review {rid}] {text!r}")

            for sentence in extract_desire_sentences(text):
                phrases = extract_missing_feature(sentence)
                if debug:
                    print(f"  [desire sentence] {sentence!r}")
                    print(f"  [raw features]    {phrases}")
                urg = _urgency_count(sentence)
                for phrase in phrases:
                    if not phrase or phrase in GENERIC_SKIP:
                        continue
                    hits.append(
                        DesireHit(
                            review_id=rid,
                            review_text=text,
                            sentence=sentence,
                            feature_phrase=phrase,
                            urgency_hits=urg,
                        )
                    )

        if not hits:
            if debug:
                print("\n[cluster] no hits — skipping")
            return []

        unique_phrases = list(dict.fromkeys(h.feature_phrase for h in hits))
        if debug:
            print(f"\n[pre-cluster] unique phrases: {unique_phrases}")
        try:
            cluster_idx_groups = cluster_features(unique_phrases, similarity_threshold)
        except Exception as exc:
            logger.warning("Clustering failed (%s); one cluster per phrase", exc)
            cluster_idx_groups = _singleton_clusters(len(unique_phrases))

        phrase_to_unique_idx = {p: i for i, p in enumerate(unique_phrases)}
        clusters_hits: List[List[DesireHit]] = []
        for cluster_idxs in cluster_idx_groups:
            cluster_hits = [
                h for h in hits if phrase_to_unique_idx.get(h.feature_phrase) in cluster_idxs
            ]
            if cluster_hits:
                clusters_hits.append(cluster_hits)

        result = score_clusters(clusters_hits, hits, unique_phrases)
        if debug:
            print(f"[final] {result}")
        return result
    except Exception as exc:
        logger.exception("detect_missing_features failed: %s", exc)
        return []
