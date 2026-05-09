from preprocess import clean_text
from collections import Counter
import re


# =========================
# FEATURE EXTRACTION
# =========================

product_features = {
    "quality": ["quality", "build", "durable"],
    "camera quality": ["camera", "photo", "picture"],
    "display": ["display", "screen"],
    "battery": ["battery", "charging", "charge"],
    "performance": ["performance", "speed", "fast", "slow"],
    "speaker": ["speaker", "sound", "audio"],
    "design": ["design", "look", "style"],
    "user experience": ["experience", "usage", "easy", "smooth"]
}


def feature_sentiment_breakdown(data):

    feature_results = {}

    for feature, keywords in product_features.items():

        total_mentions = 0
        positive_count = 0
        negative_count = 0

        for _, row in data.iterrows():

            review = clean_text(str(row["review"])).lower()
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


# =========================
# KEYWORD SYSTEM (IMPROVED)
# =========================

STOPWORDS = {
    # basic stopwords
    "the","is","and","a","an","to","for","it","this","was","are","of",
    "in","on","with","very","i","you","they","that","but","so","my","not",

    # noise words (IMPORTANT for reviews)
    "without","after","from","too","use","used","using","also",
    "get","got","like","just","now","then","than","when","while",
    "still","even","one","two","really","much","many","lot","lot of",
    "product","phone","item","amazon"
}


def extract_keywords(reviews, top_n=10):

    words = []

    for text in reviews:

        tokens = re.findall(r'\b[a-z]+\b', str(text).lower())

        for w in tokens:
            if w not in STOPWORDS and len(w) > 3:
                words.append(w)

    return Counter(words).most_common(top_n)


def extract_sentiment_keywords(data, sentiment=None, top_n=10):

    words = []

    for _, row in data.iterrows():

        if sentiment is None or row["predicted_sentiment"] == sentiment:

            tokens = re.findall(r'\b[a-z]+\b', str(row["review"]).lower())

            for w in tokens:
                if w not in STOPWORDS and len(w) > 3:
                    words.append(w)

    return Counter(words).most_common(top_n)