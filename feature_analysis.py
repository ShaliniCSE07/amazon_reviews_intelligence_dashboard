from collections import Counter, defaultdict
import math
import os
import re

import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import RegexpParser
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = PorterStemmer()

# ── NLTK data (quiet download if missing) ────────────────────────────────────
_NLTK_PACKAGES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("corpora/stopwords", "stopwords"),
]
for _path, _pkg in _NLTK_PACKAGES:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

_NP_GRAMMAR = r"NP: {<JJ.*>*<NN.*>+}"
_NP_PARSER = RegexpParser(_NP_GRAMMAR)

# Output caps (user-facing feature count)
MAX_ASPECT_CANDIDATES = 50   # unique aspects considered before ranking
MAX_OUTPUT_FEATURES = 50     # hard cap on returned feature groups
DEFAULT_TOP_N = 8
MIN_FREQ_ABSOLUTE = 2

# =========================
# FILTER LISTS
# =========================

FEATURE_BLACKLIST = {
    # Time / logistics / meta
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "time", "times", "hour", "hours", "minute", "minutes", "today", "yesterday",
    "box", "boxes", "package", "packages", "order", "orders", "shipping",
    "delivery", "support", "service", "customer", "return", "returns",
    "product", "products", "item", "items", "thing", "things", "stuff",
    "amazon", "seller", "purchase", "buy", "bought", "money", "price",
    "review", "reviews", "star", "stars", "rating", "ratings",
    "people", "person", "someone", "anyone", "everyone", "way", "ways",
    "lot", "lots", "bit", "kind", "sort", "number", "part", "parts",
    # Product name / category
    "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
    "trainer", "trainers", "footwear", "pair", "pairs",
    # Context / environment (confirmed in shoe review output)
    "gym", "work", "weather", "terrain", "ground", "surfaces", "surface",
    "concrete", "tiles", "gravel", "puddles", "outdoor", "outdoors",
    "distance", "distances", "running", "run", "runs", "walking", "walk",
    "wear", "wearing", "workouts", "workout", "hikes", "hike", "hiking",
    "sports", "sport",
    # Body / user experience (not shoe parts)
    "feet", "foot", "knee", "heels", "blisters", "blister",
    "socks", "sock", "pain", "orthotics", "orthotic",
    "barefoot", "physio", "doctor", "podiatrist",
    # Vague / abstract nouns
    "quality", "design", "construction", "options", "option",
    "value", "protection", "condition", "process", "feature", "features",
    "safety", "period", "batch", "brand", "brands", "cost",
    "packaging", "separation", "materials", "material",
    "absorption", "shock", "touch", "side", "care", "wash",
    "instructions", "instruction", "photos", "photo", "wardrobe",
    "compliments", "compliment", "everything", "nothing",
    "downside", "nightmare", "knotting", "undone", "thicker",
    # Things / environment nouns
    "stones", "stone", "lint", "hair", "strips", "strip",
    "pattern", "counter", "smell", "smells", "wet", "blue",
    # Mis-tagged as NNP/NN (spaCy) — block even if they slip past POS filter
    "comfortable", "decent", "fine", "lightweight", "nice", "fell", "took",
    "outdoor", "described", "roomy", "collects", "scuffs", "pinches", "fits",
    "supportive", "average", "okay", "looks", "descent",
    # Meta / misc noise
    "daily", "weekly", "monthly", "yearly", "morning", "evening", "night",
    "ago", "later", "soon", "while", "during", "since", "until",
    "bad", "worse", "worst", "good", "better", "best", "small", "smaller",
    "large", "larger", "big", "bigger", "little", "tiny", "huge",
    "feels", "felt", "feel", "seems", "seemed", "looked", "sound", "sounds",
}

STOPWORDS = {
    "the", "is", "and", "a", "an", "to", "for", "it", "this", "was", "are", "of",
    "in", "on", "with", "very", "i", "you", "they", "that", "but", "so", "my", "not",
    "without", "after", "from", "too", "use", "used", "using", "also",
    "get", "got", "like", "just", "now", "then", "than", "when", "while",
    "still", "even", "one", "two", "really", "much", "many",
    "excellent", "great", "good", "amazing", "wonderful", "perfect",
    "poor", "bad", "terrible", "horrible", "disappointed", "disappointing",
    "recommend", "recommended", "highly", "quite",
    "also", "would", "could", "should", "ever", "never", "back",
}

_NLTK_STOP = set(stopwords.words("english"))
_ALL_STOP = _NLTK_STOP | STOPWORDS | FEATURE_BLACKLIST
_BLACKLIST_STEMS = {stemmer.stem(w) for w in FEATURE_BLACKLIST}

# Domain → product-relevant aspect nouns (whitelist)
DOMAIN_WHITELISTS = {
    "apparel": {
        "fit", "size", "sizing", "comfort", "sole", "soles", "insole", "insoles",
        "material", "fabric", "leather", "mesh", "stitching", "seam", "seams",
        "grip", "traction", "durability", "cushioning", "cushion", "arch", "heel",
        "toe", "width", "length", "breathability", "lace", "laces",
        "strap", "straps", "padding", "weight", "look", "style", "color",
        "waterproof", "warmth", "flexibility", "stretch", "waist", "collar",
        "sleeve", "pocket", "zipper", "button", "lining", "sole", "upper",
    },
    "electronics": {
        "battery", "screen", "display", "camera", "speaker", "sound", "audio",
        "performance", "speed", "processor", "memory", "storage", "charger",
        "charging", "connectivity", "bluetooth", "wifi", "port", "ports",
        "keyboard", "touchpad", "resolution", "brightness", "heat", "cooling",
        "build", "design", "weight", "durability", "software", "interface",
    },
    "books": {
        "plot", "story", "character", "characters", "writing", "style",
        "ending", "chapter", "chapters", "pace", "dialogue", "theme",
        "cover", "pages", "binding", "illustration", "illustrations",
        "genre", "author", "narrative", "prose", "twist",
    },
    "general": {
        "quality", "durability", "design", "build", "material", "comfort",
        "performance", "value", "fit", "size", "weight", "look", "feel",
        "functionality", "ease", "installation", "setup", "usability",
    },
}

# Map product-name keywords → domain key
_DOMAIN_HINTS = {
    "apparel": (
        "shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "sandal", "sandals",
        "apparel", "clothing", "dress", "shirt", "pants", "jeans", "jacket", "coat",
        "sock", "socks", "trainer", "trainers", "footwear", "heel", "heels", "slipper",
        "hoodie", "sweater", "shorts", "legging", "leggings", "underwear", "bra",
    ),
    "electronics": (
        "phone", "smartphone", "laptop", "tablet", "camera", "headphone", "headphones",
        "earphone", "earbuds", "monitor", "tv", "television", "speaker", "charger",
        "keyboard", "mouse", "gpu", "cpu", "computer", "pc", "macbook", "iphone",
        "android", "watch", "smartwatch", "console", "gaming",
    ),
    "books": (
        "book", "books", "novel", "ebook", "e-book", "paperback", "hardcover",
        "textbook", "audiobook", "kindle",
    ),
}

# Lexicon for opinion-bearing sentences (co-occurrence / PMI)
_POSITIVE_OPINION = {
    "love", "loved", "like", "liked", "great", "excellent", "amazing", "perfect",
    "good", "best", "comfortable", "happy", "satisfied", "recommend", "awesome",
    "wonderful", "fantastic", "impressed", "pleased", "nice", "beautiful",
}
_NEGATIVE_OPINION = {
    "hate", "hated", "dislike", "disliked", "bad", "terrible", "awful", "horrible",
    "poor", "worst", "uncomfortable", "disappointed", "disappointing", "regret",
    "broken", "defective", "fail", "failed", "useless", "cheap", "flimsy", "pain",
}
_OPINION_WORDS = _POSITIVE_OPINION | _NEGATIVE_OPINION


# =========================
# DOMAIN & FILTERING
# =========================

def detect_product_domain(product_name: str) -> str:
    """Infer domain from product name / category string."""
    if not product_name:
        return "general"
    tokens = re.findall(r"[a-z0-9]+", product_name.lower())
    for domain, hints in _DOMAIN_HINTS.items():
        if any(t in hints or any(h in t for h in hints) for t in tokens):
            return domain
    return "general"


def _product_tokens(product_name: str) -> set:
    if not product_name:
        return set()
    tokens = set()
    for word in re.findall(r"[a-z0-9]+", product_name.lower()):
        tokens.add(word)
        tokens.add(stemmer.stem(word))
        if word.endswith("y"):
            tokens.add(word[:-1] + "ies")
        tokens.add(word + "s")
        tokens.add(word + "es")
    return tokens


def is_blacklisted(term: str) -> bool:
    t = term.lower().strip()
    if not t or t in _ALL_STOP or t in FEATURE_BLACKLIST:
        return True
    # Reject very short tokens and pure sentiment/size words
    if len(t) < 3:
        return True
    parts = t.split()
    if any(p in FEATURE_BLACKLIST or p in _ALL_STOP for p in parts):
        return True
    if any(p in _OPINION_WORDS for p in parts):
        return True
    if any(stemmer.stem(p) in _BLACKLIST_STEMS for p in parts):
        return True
    return False


def passes_domain_filter(term: str, domain: str, strict_whitelist: bool = False) -> bool:
    """
    Whitelist: keep term if it matches a known domain aspect (or contains one).
    When strict_whitelist is True (apparel), only whitelisted aspects pass.
    """
    t = term.lower().strip()
    whitelist = DOMAIN_WHITELISTS.get(domain, DOMAIN_WHITELISTS["general"])
    expanded = whitelist | DOMAIN_WHITELISTS["general"]

    for aspect in expanded:
        if t == aspect or aspect in t.split() or t in aspect:
            return True
        if stemmer.stem(t) == stemmer.stem(aspect):
            return True

    if strict_whitelist and domain == "apparel":
        return False
    return not strict_whitelist


def filter_aspect_term(term: str, domain: str, product_tokens: set) -> bool:
    """Combined blacklist + domain whitelist gate."""
    if is_blacklisted(term):
        return False
    if term.lower() in product_tokens:
        return False
    if any(pt in term.lower().split() for pt in product_tokens if len(pt) > 3):
        return False
    strict = domain == "apparel"
    return passes_domain_filter(term, domain, strict_whitelist=strict)


# =========================
# OPINION SENTENCES & NP CHUNKING
# =========================

def is_opinion_bearing_sentence(sentence: str) -> bool:
    """True if sentence has sentiment/opinion signals (lexicon or comparative JJ)."""
    lower = sentence.lower()
    tokens = re.findall(r"[a-z']+", lower)
    if any(w in _OPINION_WORDS for w in tokens):
        return True
    try:
        tagged = pos_tag(word_tokenize(sentence))
        if any(tag in ("JJR", "JJS", "RB") for _, tag in tagged):
            # JJR/JJS often modify aspects; RB catches "very comfortable"
            if any(tag.startswith("JJ") for _, tag in tagged):
                return True
    except Exception:
        pass
    return False


def _token_is_clean_aspect_word(word: str) -> bool:
    """True if word is not blacklisted/stop and long enough."""
    w = word.lower()
    return (
        w.isalpha()
        and len(w) >= 3
        and w not in FEATURE_BLACKLIST
        and w not in _ALL_STOP
    )


def _leaf_nouns_from_np(subtree) -> list:
    """Only true common nouns (NN/NNS). NNP/NNPS excluded — tagger mislabels adjectives/verbs."""
    nouns = []
    for word, tag in subtree.leaves():
        if tag not in ("NN", "NNS"):  # never NNP, NNPS, JJ, VBG, RB
            continue
        w = word.lower()
        if _token_is_clean_aspect_word(w):
            nouns.append(w)
    return nouns


def extract_aspects_from_sentence(sentence: str) -> list:
    """
    Noun-phrase chunking + NN/NNS from opinion-bearing sentences only.
    Returns canonical single-token or short bigram aspect strings.
    """
    if not is_opinion_bearing_sentence(sentence):
        return []

    aspects = []
    try:
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        tree = _NP_PARSER.parse(tagged)

        for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
            np_nouns = _leaf_nouns_from_np(subtree)
            if not np_nouns:
                continue
            # Head noun (last NN/NNS in phrase)
            head = np_nouns[-1]
            if _token_is_clean_aspect_word(head):
                aspects.append(head)
            if len(np_nouns) >= 2:
                bigram = " ".join(np_nouns[-2:])
                both_clean = all(
                    w not in FEATURE_BLACKLIST and w not in _ALL_STOP
                    for w in bigram.split()
                )
                if len(bigram) <= 24 and both_clean:
                    aspects.append(bigram)

        # Standalone NN/NNS in opinion sentences (kept by design)
        for word, tag in tagged:
            if tag in ("NN", "NNS") and len(word) > 2:
                w = word.lower()
                if _token_is_clean_aspect_word(w):
                    aspects.append(w)
    except Exception:
        pass

    return list(dict.fromkeys(aspects))


# =========================
# PMI + TF-IDF SCORING
# =========================

def _pmi_scores(
    aspect_counts: Counter,
    aspect_sentiment_cooc: defaultdict,
    total_sentences: int,
) -> dict:
    """
    PMI-style score: how strongly an aspect co-occurs with opinion words
    vs chance. Higher = more distinctive opinion-linked feature.
    """
    total_opinion_hits = sum(
        sum(co.values()) for co in aspect_sentiment_cooc.values()
    )
    if total_opinion_hits == 0:
        total_opinion_hits = 1

    scores = {}
    for aspect, freq in aspect_counts.items():
        p_a = (freq + 1) / (total_sentences + 1)
        co = aspect_sentiment_cooc.get(aspect, {})
        co_total = sum(co.values())
        p_co = (co_total + 1) / (total_opinion_hits + 1)
        # log PMI approximation
        pmi = math.log((p_co + 1e-9) / (p_a + 1e-9))
        scores[aspect] = pmi * math.log1p(freq)
    return scores


def _tfidf_boost(review_texts: list, candidates: list) -> dict:
    """TF-IDF over corpus for candidate terms (unigrams matching aspects)."""
    if not review_texts or not candidates:
        return {c: 0.0 for c in candidates}
    try:
        vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",
            max_features=5000,
        )
        matrix = vec.fit_transform(review_texts)
        names = vec.get_feature_names_out()
        mean_scores = matrix.mean(axis=0).A1
        name_to_score = dict(zip(names, mean_scores))
        out = {}
        for cand in candidates:
            parts = cand.split()
            vals = [name_to_score.get(p, 0.0) for p in parts]
            out[cand] = sum(vals) / max(1, len(vals))
        return out
    except Exception:
        return {c: 0.0 for c in candidates}


def rank_aspects(
    aspect_counts: Counter,
    aspect_sentiment_cooc: defaultdict,
    total_sentences: int,
    review_texts: list,
) -> list:
    """Combine PMI (opinion co-occurrence) and TF-IDF (distinctiveness)."""
    candidates = list(aspect_counts.keys())
    pmi = _pmi_scores(aspect_counts, aspect_sentiment_cooc, total_sentences)
    tfidf = _tfidf_boost(review_texts, candidates)

    max_pmi = max(pmi.values()) if pmi else 1.0
    max_tfidf = max(tfidf.values()) if tfidf else 1.0

    combined = []
    for aspect in candidates:
        p_norm = pmi.get(aspect, 0) / (max_pmi + 1e-9)
        t_norm = tfidf.get(aspect, 0) / (max_tfidf + 1e-9)
        score = 0.55 * p_norm + 0.45 * t_norm
        combined.append((aspect, score))
    return sorted(combined, key=lambda x: x[1], reverse=True)


# =========================
# FEATURE EXTRACTION (main)
# =========================

def extract_dynamic_features(data, product_name="", top_n=None, domain=None):
    """
    Domain-aware aspect extraction from reviews.

    Pipeline:
      1. Opinion-bearing sentences only
      2. NP chunking + NN/NNS POS tags
      3. Blacklist + domain whitelist filtering
      4. Rank by PMI (sentiment co-occurrence) + TF-IDF

    Returns at most MAX_OUTPUT_FEATURES (50) feature groups; default display top_n=8.
    """
    if top_n is None:
        top_n = DEFAULT_TOP_N
    top_n = min(int(top_n), MAX_OUTPUT_FEATURES)

    domain = domain or detect_product_domain(product_name)
    product_tokens = _product_tokens(product_name)
    aspect_counts = Counter()
    aspect_sentiment_cooc = defaultdict(lambda: Counter())
    review_texts = []
    total_opinion_sentences = 0

    for _, row in data.iterrows():
        text = str(row.get("review", ""))
        if not text.strip():
            continue
        review_texts.append(text.lower())

        for sentence in sent_tokenize(text):
            if not is_opinion_bearing_sentence(sentence):
                continue
            total_opinion_sentences += 1

            sent_tokens = set(re.findall(r"[a-z']+", sentence.lower()))
            opinion_hits = [w for w in sent_tokens if w in _OPINION_WORDS]

            for aspect in extract_aspects_from_sentence(sentence):
                if not filter_aspect_term(aspect, domain, product_tokens):
                    continue
                aspect_counts[aspect] += 1
                for ow in opinion_hits:
                    aspect_sentiment_cooc[aspect][ow] += 1

    MIN_FREQ = 2
    aspect_counts = Counter({
        k: v for k, v in aspect_counts.items() if v >= MIN_FREQ
    })

    if os.environ.get("FEATURE_DEBUG"):
        print(f"After MIN_FREQ filter: {len(aspect_counts)} aspects")
        for term, count in aspect_counts.most_common():
            print(f"  {count}x {term}")

    if not aspect_counts:
        return {}

    # Keep only the top N candidates by raw frequency before PMI/TF-IDF
    if len(aspect_counts) > MAX_ASPECT_CANDIDATES:
        aspect_counts = Counter(dict(aspect_counts.most_common(MAX_ASPECT_CANDIDATES)))
        aspect_sentiment_cooc = defaultdict(
            lambda: Counter(),
            {k: aspect_sentiment_cooc[k] for k in aspect_counts},
        )

    ranked = rank_aspects(
        aspect_counts,
        aspect_sentiment_cooc,
        max(1, total_opinion_sentences),
        review_texts,
    )

    dynamic_features = {}
    seen_stems = {}

    for aspect, _score in ranked:
        # Normalize to head token for grouping synonyms
        head = aspect.split()[-1] if " " in aspect else aspect
        stem = stemmer.stem(head)

        if stem in seen_stems:
            parent = seen_stems[stem]
            if aspect not in dynamic_features[parent]:
                dynamic_features[parent].append(aspect)
            continue

        seen_stems[stem] = head
        keywords = {head, aspect}
        if head.endswith("y"):
            keywords.add(head[:-1] + "ies")
        else:
            keywords.add(head + "s")

        # Domain whitelist aliases (e.g. sole → soles already in list)
        wl = DOMAIN_WHITELISTS.get(domain, set()) | DOMAIN_WHITELISTS["general"]
        for w in wl:
            if stemmer.stem(w) == stem:
                keywords.add(w)

        dynamic_features[head] = sorted(keywords)

        if len(dynamic_features) >= top_n:
            break

    return dict(list(dynamic_features.items())[:MAX_OUTPUT_FEATURES])


def extract_feature_insights_api(
    reviews: list,
    product_name: str = "",
    category: str = "",
    top_n: int = 20,
) -> list:
    """
    Build API/DB feature insight rows using extract_dynamic_features (filtered pipeline).
    Each review dict needs: text, sentiment (or predicted_sentiment), optional rating.
    """
    if not reviews:
        return []

    rows = []
    for r in reviews:
        rows.append({
            "review": str(r.get("text", "")),
            "predicted_sentiment": r.get("sentiment") or r.get("predicted_sentiment") or "Neutral",
            "rating": r.get("rating"),
        })
    df = pd.DataFrame(rows)
    label = product_name or category or "Product"
    domain = detect_product_domain(label)

    dynamic = extract_dynamic_features(
        df,
        product_name=label,
        top_n=min(top_n, MAX_OUTPUT_FEATURES),
        domain=domain,
    )
    if not dynamic:
        return []

    insights = []
    for head, keywords in dynamic.items():
        mentions = 0
        sentiment_sum = 0.0
        pos_quotes = []
        neg_quotes = []

        for r in reviews:
            text = str(r.get("text", ""))
            if not text.strip():
                continue
            lower = text.lower()
            if not any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in keywords):
                continue

            mentions += 1
            sent_lbl = r.get("sentiment") or r.get("predicted_sentiment") or "Neutral"
            if sent_lbl == "Positive":
                sentiment_sum += 1.0
            elif sent_lbl == "Negative":
                sentiment_sum -= 1.0

            for sent in sent_tokenize(text):
                sl = sent.lower()
                if not any(re.search(rf"\b{re.escape(kw)}\b", sl) for kw in keywords):
                    continue
                clean = sent.strip()
                if sent_lbl == "Positive" and len(pos_quotes) < 3:
                    pos_quotes.append(clean)
                elif sent_lbl == "Negative" and len(neg_quotes) < 3:
                    neg_quotes.append(clean)

        avg = sentiment_sum / max(mentions, 1)
        ui_score = round(((avg + 1.0) / 2.0) * 100.0, 1)

        insights.append({
            "feature_name": head.replace("_", " ").title(),
            "mentions": mentions,
            "sentiment_score": ui_score,
            "quotes_pos": list(dict.fromkeys(pos_quotes))[:3],
            "quotes_neg": list(dict.fromkeys(neg_quotes))[:3],
        })

    return sorted(insights, key=lambda x: x["mentions"], reverse=True)


def filter_review_keywords(keywords: list) -> list:
    """Drop blacklisted/noise tokens from per-review keyword lists (word cloud)."""
    out = []
    for w in keywords:
        if not w or is_blacklisted(str(w)):
            continue
        out.append(w)
    return out


def feature_sentiment_breakdown(data, features=None):
    """
    Analyzes sentiment for each feature.
    If features is None, it returns an empty analysis.
    """
    if features is None:
        return {}

    feature_results = {}

    for feature, keywords in features.items():

        total_mentions = 0
        positive_count = 0
        negative_count = 0

        for _, row in data.iterrows():

            review = str(row["review"]).lower()
            sentiment = row["predicted_sentiment"]

            if any(word in review for word in keywords):

                total_mentions += 1

                if sentiment == "positive":
                    positive_count += 1
                elif sentiment == "negative":
                    negative_count += 1

        if total_mentions > 0:

            if positive_count > negative_count:
                summary = f"Most users are satisfied with {feature}."

            elif negative_count > positive_count:
                summary = f"Many users reported issues in {feature}."

            else:
                summary = f"Users have mixed opinions on {feature}."

            feature_results[feature] = {
                "mentions": total_mentions,
                "positive": positive_count,
                "negative": negative_count,
                "summary": summary
            }

    return feature_results


def extract_keywords(reviews, top_n=10, product_name=""):
    """
    Extracts top keywords from a list of reviews (legacy; uses shared stoplists).
    """
    words = []
    local_stopwords = set(_ALL_STOP)
    if product_name:
        local_stopwords |= _product_tokens(product_name)

    for text in reviews:
        tokens = re.findall(r"\b[a-z]+\b", str(text).lower())
        for w in tokens:
            if w not in local_stopwords and len(w) > 3 and not is_blacklisted(w):
                words.append(w)

    return Counter(words).most_common(top_n)


def extract_sentiment_keywords(data, sentiment=None, top_n=10, product_name=""):
    """
    Extracts top keywords for a specific sentiment.
    Filters out product name and generic stopwords.
    """
    words = []
    local_stopwords = set(_ALL_STOP)
    if product_name:
        local_stopwords |= _product_tokens(product_name)

    for _, row in data.iterrows():
        if sentiment is None or row["predicted_sentiment"] == sentiment:
            tokens = re.findall(r"\b[a-z]+\b", str(row["review"]).lower())
            for w in tokens:
                if w not in local_stopwords and len(w) > 3 and not is_blacklisted(w):
                    words.append(w)
    return Counter(words).most_common(top_n)
