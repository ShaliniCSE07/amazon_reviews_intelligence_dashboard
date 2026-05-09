import pandas as pd
import joblib
import os

from preprocess import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# LOAD DATA
# =========================
data = pd.read_json(
    "data/Cell_Phones_and_Accessories_5.json",
    lines=True
)

data = data.sample(10000, random_state=42)

data = data[["reviewText", "overall"]].dropna()


# =========================
# LABEL CREATION
# =========================
def convert_sentiment(score):
    return "positive" if score >= 4 else "negative"


data["sentiment"] = data["overall"].apply(convert_sentiment)
data["clean_review"] = data["reviewText"].apply(clean_text)


# =========================
# BALANCING DATA
# =========================
min_count = data["sentiment"].value_counts().min()

balanced_data = data.groupby("sentiment").sample(min_count, random_state=42)


# =========================
# FEATURES / LABELS
# =========================
X = balanced_data["clean_review"]
y = balanced_data["sentiment"]


# =========================
# VECTORIZE
# =========================
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 2)
)

X_vectorized = vectorizer.fit_transform(X)


# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# MODEL
# =========================
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(X_train, y_train)


# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# =========================
# SAFE SAVE (IMPORTANT FIX)
# =========================

# remove old corrupted files first
for file in ["model.pkl", "vectorizer.pkl"]:
    if os.path.exists(file):
        os.remove(file)

# save fresh files
joblib.dump(model, "model.pkl", compress=3)
joblib.dump(vectorizer, "vectorizer.pkl", compress=3)

print("\n✅ Model & Vectorizer saved safely!")