import pandas as pd
import nltk

# Ensure NLTK data is accessible
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from feature_analysis import extract_dynamic_features, feature_sentiment_breakdown

# Mock data for a Laptop
data = pd.DataFrame({
    "review": [
        "The keyboard is amazing and the screen is very bright.",
        "I hate the battery life, it drains too fast.",
        "The performance is laggy but the design is sleek.",
        "The keyboard feels cheap.",
        "Great screen quality and battery is okay.",
        "The laptop gets hot while gaming.",
        "Excellent build quality and keyboard backlighting."
    ],
    "predicted_sentiment": ["positive", "negative", "negative", "negative", "positive", "negative", "positive"]
})

print("\n--- Testing dynamic extraction for 'Laptop' ---")
features = extract_dynamic_features(data, "Laptop", top_n=5)
print("Extracted Features:", list(features.keys()))

results = feature_sentiment_breakdown(data, features)
for f, r in results.items():
    print(f"Feature: {f:12} | Mentions: {r['mentions']} | Positive: {r['positive']} | Negative: {r['negative']} | Summary: {r['summary']}")

# Mock data for a Book
book_data = pd.DataFrame({
    "review": [
        "The plot was very engaging and the characters were well-developed.",
        "I didn't like the ending, it felt rushed.",
        "The writing style is beautiful.",
        "The cover design is stunning.",
        "Too many typos in the pages.",
        "The characters are a bit boring but the plot is good."
    ],
    "predicted_sentiment": ["positive", "negative", "positive", "positive", "negative", "neutral"]
})

print("\n--- Testing dynamic extraction for 'Book' ---")
features = extract_dynamic_features(book_data, "Book", top_n=5)
print("Extracted Features:", list(features.keys()))

results = feature_sentiment_breakdown(book_data, features)
for f, r in results.items():
    print(f"Feature: {f:12} | Mentions: {r['mentions']} | Positive: {r['positive']} | Negative: {r['negative']} | Summary: {r['summary']}")
