import os
import re
import math
import json
import collections
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

# NLP imports — NLTK only (spaCy removed for Python 3.12 compatibility)
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, RegexpParser
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# ── Ensure all required NLTK data is present ─────────────────────────────────
_NLTK_PACKAGES = [
    ('sentiment/vader_lexicon.zip', 'vader_lexicon'),
    ('tokenizers/punkt',            'punkt'),
    ('tokenizers/punkt_tab',        'punkt_tab'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
    ('corpora/stopwords',           'stopwords'),
]
for path, pkg in _NLTK_PACKAGES:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOP_WORDS = set(stopwords.words('english'))

# Simple noun-phrase grammar: optional adjectives followed by one or more nouns
_NP_GRAMMAR = r"""
  NP: {<JJ.*>*<NN.*>+}
"""
_NP_PARSER = RegexpParser(_NP_GRAMMAR)


def _sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK punkt tokenizer."""
    try:
        return sent_tokenize(text)
    except Exception:
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


def _noun_phrases(text: str) -> List[str]:
    """
    Extract noun phrases from text using NLTK POS tagging + regex chunking.
    Returns lowercased, stop-word-filtered phrases.
    """
    phrases = []
    for sent in _sentences(text):
        try:
            tokens = word_tokenize(sent)
            tagged = pos_tag(tokens)
            tree = _NP_PARSER.parse(tagged)
            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                words = [
                    w.lower() for w, tag in subtree.leaves()
                    if w.isalpha() and w.lower() not in _STOP_WORDS
                    and w.lower() not in ('i','we','he','she','it','they','you',
                                          'my','your','our','their','this','that')
                ]
                if words:
                    phrases.append(' '.join(words))
        except Exception:
            pass
    return phrases


def _nouns_adjectives(text: str) -> List[str]:
    """Return individual nouns and adjectives from text (lowercased, no stop-words)."""
    tokens = []
    try:
        tagged = pos_tag(word_tokenize(text))
        tokens = [
            w.lower() for w, tag in tagged
            if tag.startswith(('NN', 'JJ', 'NNP')) and w.isalpha()
            and w.lower() not in _STOP_WORDS
        ]
    except Exception:
        pass
    return tokens


# ─────────────────────────────────────────────────────────────────────────────

class NLPPipeline:
    def __init__(self):
        self.vader = None
        self._sentiment_pipe = None
        self._emotion_pipe = None
        self.use_transformers = True
        self._load_vader()

    def _load_vader(self):
        try:
            self.vader = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"Warning: Could not load VADER: {e}")
            self.vader = None

    def _load_sentiment_model(self):
        if self._sentiment_pipe is not None:
            return self._sentiment_pipe
        if not self.use_transformers:
            return None
        try:
            from transformers import pipeline
            print("Loading RoBERTa Sentiment Model...")
            self._sentiment_pipe = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1,
                max_length=512,
                truncation=True
            )
            return self._sentiment_pipe
        except Exception as e:
            print(f"Transformers sentiment model failed to load, using VADER fallback: {e}")
            self.use_transformers = False
            return None

    def _load_emotion_model(self):
        if self._emotion_pipe is not None:
            return self._emotion_pipe
        if not self.use_transformers:
            return None
        try:
            from transformers import pipeline
            print("Loading DistilRoBERTa Emotion Model...")
            self._emotion_pipe = pipeline(
                "sentiment-analysis",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=-1,
                max_length=512,
                truncation=True,
                return_all_scores=True
            )
            return self._emotion_pipe
        except Exception as e:
            print(f"Transformers emotion model failed to load, using Lexicon fallback: {e}")
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyzes the text sentiment and returns (label, confidence).
        Supported labels: Positive, Negative, Neutral, Mixed.
        """
        if not text or not text.strip():
            return "Neutral", 1.0

        sentences = _sentences(text)
        sentiment_pipe = self._load_sentiment_model()

        if sentiment_pipe:
            try:
                res = sentiment_pipe(text[:1500])[0]
                label_map = {
                    "positive": "Positive",
                    "neutral":  "Neutral",
                    "negative": "Negative"
                }
                full_label = label_map.get(res["label"].lower(), "Neutral")
                full_conf  = res["score"]

                if len(sentences) >= 2:
                    pos_count = neg_count = 0
                    sent_res = sentiment_pipe(sentences[:6])
                    for r in sent_res:
                        lbl   = r["label"].lower()
                        score = r["score"]
                        if lbl == "positive" and score > 0.65:
                            pos_count += 1
                        elif lbl == "negative" and score > 0.65:
                            neg_count += 1
                    if pos_count >= 1 and neg_count >= 1:
                        return "Mixed", min(0.95, (pos_count + neg_count) / len(sentences))

                return full_label, full_conf
            except Exception as e:
                print(f"Transformer sentiment error: {e}, falling back to VADER")

        # VADER fallback
        if self.vader:
            scores   = self.vader.polarity_scores(text)
            compound = scores["compound"]

            if len(sentences) >= 2:
                pos_sent = neg_sent = 0
                for sent in sentences[:8]:
                    s = self.vader.polarity_scores(sent)
                    if s["compound"] > 0.35:
                        pos_sent += 1
                    elif s["compound"] < -0.35:
                        neg_sent += 1
                if pos_sent >= 1 and neg_sent >= 1:
                    return "Mixed", 0.85

            if compound >= 0.05:
                return "Positive", min(1.0, 0.5 + abs(compound) * 0.5)
            elif compound <= -0.05:
                return "Negative", min(1.0, 0.5 + abs(compound) * 0.5)
            else:
                return "Neutral", 1.0 - abs(compound)

        return "Neutral", 0.5

    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """
        Detects emotions: frustration, delight, disappointment, surprise.
        """
        emotions: Dict[str, float] = {
            "frustration": 0.0, "delight": 0.0,
            "disappointment": 0.0, "surprise": 0.0
        }
        if not text or not text.strip():
            return emotions

        emotion_pipe = self._load_emotion_model()
        if emotion_pipe:
            try:
                res    = emotion_pipe(text[:1500])[0]
                scores = {item["label"].lower(): item["score"] for item in res}
                emotions["frustration"]   = round(scores.get("anger", 0.0) + scores.get("disgust", 0.0), 3)
                emotions["delight"]       = round(scores.get("joy", 0.0), 3)
                emotions["disappointment"]= round(scores.get("sadness", 0.0) + scores.get("fear", 0.0) * 0.5, 3)
                emotions["surprise"]      = round(scores.get("surprise", 0.0), 3)
                total = sum(emotions.values())
                if total > 0:
                    for k in emotions:
                        emotions[k] = round(emotions[k] / max(1.0, total), 3)
                return emotions
            except Exception as e:
                print(f"Transformer emotion error: {e}, falling back to Lexicon")

        # Lexicon fallback
        text_lower = text.lower()
        lexicon = {
            "delight":       ["love","loved","delight","delighted","happy","great","excellent",
                              "awesome","perfect","amazing","wonderful","glad","satisfied","pleased"],
            "frustration":   ["annoyed","frustrated","angry","terrible","waste","worst","hate",
                              "garbage","poor","pain","bother","suck","sucks","awful","useless"],
            "disappointment":["disappointed","unhappy","regret","sad","broke","fail","failed",
                              "unfortunately","defect","defective","broken","hopes","letdown"],
            "surprise":      ["surprise","surprised","unexpected","shocked","amazed",
                              "astonished","suddenly","miracle"]
        }
        for emo, words in lexicon.items():
            count = sum(1 for w in words if re.search(r'\b' + w + r'\b', text_lower))
            emotions[emo] = min(1.0, count * 0.25)

        sentiment_label, _ = self.analyze_sentiment(text)
        if sentiment_label == "Positive":
            emotions["delight"]       = max(emotions["delight"], 0.4)
        elif sentiment_label == "Negative":
            emotions["frustration"]   = max(emotions["frustration"], 0.4)
            emotions["disappointment"]= max(emotions["disappointment"], 0.3)

        total = sum(emotions.values())
        if total > 0:
            for k in emotions:
                emotions[k] = round(emotions[k] / total, 3)
        else:
            emotions = {"frustration": 0.1, "delight": 0.1,
                        "disappointment": 0.1, "surprise": 0.1}
        return emotions

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extracts keywords using TF-IDF over NLTK-extracted noun phrases & adjectives.
        """
        if not text or len(text.split()) < 3:
            return []
        try:
            candidates = list(set(_noun_phrases(text) + _nouns_adjectives(text)))
            if not candidates:
                return []

            vectorizer  = TfidfVectorizer(ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            cand_scores: Dict[str, float] = {}
            for cand in candidates:
                if cand in feature_names:
                    idx = np.where(feature_names == cand)[0][0]
                    cand_scores[cand] = scores[idx]
                else:
                    score_sum = count = 0
                    for word in cand.split():
                        if word in feature_names:
                            idx = np.where(feature_names == word)[0][0]
                            score_sum += scores[idx]
                            count += 1
                    cand_scores[cand] = score_sum / max(1, count)

            sorted_cand = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
            return [k for k, v in sorted_cand[:top_n]]
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            words = [w.lower() for w in re.findall(r'\b\w{4,}\b', text)
                     if w.lower() not in _STOP_WORDS]
            return list(set(words))[:top_n]

    def score_review_quality(self, text: str) -> float:
        """
        Scores review quality (0–100) based on length, vocabulary richness,
        and specificity (ratio of nouns/adjectives).
        """
        words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
        word_count = len(words)
        if word_count == 0:
            return 0.0

        length_score = min(100.0, (word_count / 150.0) * 100.0)
        unique_words = len(set(words))
        ttr = (unique_words / word_count) * 100.0

        try:
            tagged = pos_tag(word_tokenize(text))
            specific_count = sum(1 for _, tag in tagged
                                 if tag.startswith(('NN', 'JJ', 'NNP', 'CD')))
            specificity = (specific_count / len(tagged)) * 100.0 if tagged else 0.0
        except Exception:
            specificity = 30.0

        score = 0.4 * length_score + 0.3 * ttr + 0.3 * specificity
        return round(min(100.0, max(0.0, score)), 1)

    def detect_fake_review(self, text: str, rating: Optional[int],
                           other_reviews: List[str] = None) -> Tuple[bool, List[str]]:
        """
        Flags potentially fake/spam reviews based on various heuristics.
        """
        reasons: List[str] = []

        if not text or not text.strip():
            return False, []

        # 1. Shouting check
        chars = [c for c in text if c.isalpha()]
        if len(chars) > 20:
            caps_ratio = sum(1 for c in chars if c.isupper()) / len(chars)
            if caps_ratio > 0.4:
                reasons.append("Excessive capitalization (shouting)")

        # 2. Sentiment–Rating mismatch
        if rating is not None:
            sentiment, confidence = self.analyze_sentiment(text)
            if rating in [1, 2] and sentiment == "Positive" and confidence > 0.85:
                reasons.append("Sentiment-Rating mismatch (positive text, low rating)")
            elif rating in [4, 5] and sentiment == "Negative" and confidence > 0.85:
                reasons.append("Sentiment-Rating mismatch (negative text, high rating)")

        # 3. Repetition check
        words = text.lower().split()
        if len(words) > 10:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            if trigrams:
                most_common = collections.Counter(trigrams).most_common(1)[0]
                if most_common[1] / len(trigrams) > 0.2:
                    reasons.append("High phrase repetition detected")

        # 4. Duplicate check
        if other_reviews:
            for r in other_reviews:
                if r != text and len(r) > 10 and len(text) > 10:
                    if r.lower() == text.lower() or (len(text) > 50 and text.lower() in r.lower()):
                        reasons.append("Duplicate review text found in batch")
                        break

        # 5. Low lexical diversity
        if len(words) > 50:
            if len(set(words)) / len(words) < 0.25:
                reasons.append("Extremely low vocabulary diversity (possible templated spam)")

        return bool(reasons), reasons

    def generate_extractive_summary(self, reviews: List[str],
                                    num_sentences: int = 4) -> str:
        """
        Generates a concise summary using TextRank-style TF-IDF centrality.
        """
        if not reviews:
            return "No reviews available to summarize."

        all_sentences: List[str] = []
        for r in reviews:
            for sent in _sentences(r):
                clean = sent.strip().replace("\n", " ")
                word_count = len(clean.split())
                if 6 <= word_count <= 30:
                    if not any(p in clean.lower() for p in
                               ["buy it","perfect","good product","very nice","love it"]):
                        all_sentences.append(clean)

        unique_sentences = list(dict.fromkeys(all_sentences))

        if len(unique_sentences) <= num_sentences:
            return " ".join(unique_sentences) if unique_sentences else "No summary sentences extracted."

        try:
            vectorizer   = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(unique_sentences)
            sim_matrix   = cosine_similarity(tfidf_matrix)
            scores       = np.sum(sim_matrix, axis=1)
            top_indices  = np.argsort(scores)[::-1][:num_sentences]
            selected     = [unique_sentences[i] for i in sorted(top_indices)]
            return " ".join(selected)
        except Exception as e:
            print(f"Summarization error: {e}")
            return " ".join(unique_sentences[:num_sentences])

    def extract_aspects_and_cluster(self, reviews: List[str],
                                    sentiments: List[str]) -> List[Dict[str, Any]]:
        """
        Extracts product aspects using NLTK noun-phrase chunking,
        clusters them and aggregates sentiment per feature.
        """
        if not reviews:
            return []

        synonyms = {
            "battery":     ["battery","charge","charging","charger","power","backup","life"],
            "screen":      ["screen","display","monitor","panel","glass","oled","amoled","lcd"],
            "camera":      ["camera","photo","video","lens","sensor","picture","megapixels","zoom"],
            "sound":       ["sound","audio","speaker","speakers","volume","music","bass","treble"],
            "price":       ["price","cost","value","money","affordable","cheap","expensive"],
            "design":      ["design","build","quality","material","feel","look","weight","aesthetic"],
            "performance": ["speed","performance","processor","ram","cpu","gpu","fast","lag","hang","freeze"],
        }

        aspect_mentions: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)

        for rev, sent_lbl in zip(reviews, sentiments):
            sent_score = 1.0 if sent_lbl == "Positive" else (-1.0 if sent_lbl == "Negative" else 0.0)
            for sent_text in _sentences(rev):
                # Noun-phrase candidates
                for phrase in _noun_phrases(sent_text):
                    if 2 < len(phrase) <= 30 and len(phrase.split()) <= 2:
                        aspect_mentions[phrase].append((sent_text, sent_score))
                # Individual nouns as fallback
                for noun in _nouns_adjectives(sent_text):
                    if len(noun) > 2:
                        aspect_mentions[noun].append((sent_text, sent_score))

        if not aspect_mentions:
            return []

        raw_aspects = list(aspect_mentions.keys())
        aspect_to_cluster: Dict[str, str] = {}

        for aspect in raw_aspects:
            matched = False
            for cat, keywords in synonyms.items():
                if any(kw in aspect or aspect in kw for kw in keywords):
                    aspect_to_cluster[aspect] = cat
                    matched = True
                    break
            if not matched:
                aspect_to_cluster[aspect] = aspect

        # Agglomerative clustering for unmapped aspects
        unmapped = [a for a in raw_aspects if aspect_to_cluster[a] == a]
        if len(unmapped) >= 4:
            try:
                vec     = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                tfidf   = vec.fit_transform(unmapped)
                n_clust = max(2, len(unmapped) // 3)
                labels  = AgglomerativeClustering(
                    n_clusters=n_clust, metric='cosine', linkage='average'
                ).fit_predict(tfidf.toarray())

                cluster_groups: Dict[int, List[str]] = collections.defaultdict(list)
                for aspect, label in zip(unmapped, labels):
                    cluster_groups[label].append(aspect)

                for label, group in cluster_groups.items():
                    best_name = sorted(group, key=lambda a: len(aspect_mentions[a]), reverse=True)[0]
                    for aspect in group:
                        aspect_to_cluster[aspect] = best_name
            except Exception as e:
                print(f"Hierarchical clustering error: {e}")

        # Aggregate per cluster
        clusters: Dict[str, Dict[str, Any]] = collections.defaultdict(
            lambda: {"mentions": 0, "sentiment_sum": 0.0, "quotes": []}
        )
        for aspect, mentions_list in aspect_mentions.items():
            cluster_name = aspect_to_cluster[aspect].capitalize()
            clusters[cluster_name]["mentions"] += len(mentions_list)
            for quote, score in mentions_list:
                clusters[cluster_name]["sentiment_sum"] += score
                clusters[cluster_name]["quotes"].append((quote, score))

        feature_insights: List[Dict[str, Any]] = []
        min_mentions = 2 if len(reviews) > 5 else 1

        for name, data in clusters.items():
            if data["mentions"] < min_mentions:
                continue
            sentiment_score = (data["sentiment_sum"] / len(data["quotes"])) if data["quotes"] else 0.0
            ui_score = round(((sentiment_score + 1.0) / 2.0) * 100.0, 1)

            pos_quotes = list(dict.fromkeys([q for q, s in data["quotes"] if s > 0.3]))[:3]
            neg_quotes = list(dict.fromkeys([q for q, s in data["quotes"] if s < -0.3]))[:3]

            feature_insights.append({
                "feature_name":   name,
                "mentions":       data["mentions"],
                "sentiment_score": ui_score,
                "quotes_pos":     pos_quotes,
                "quotes_neg":     neg_quotes,
            })

        return sorted(feature_insights, key=lambda x: x["mentions"], reverse=True)
