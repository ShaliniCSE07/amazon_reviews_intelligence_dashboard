import pandas as pd
import joblib

from preprocess import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Load dataset
data = pd.read_json(
    "data/Cell_Phones_and_Accessories_5.json",
    lines=True
)

# Sample for faster training
data = data.sample(
    10000,
    random_state=42
)

# Keep required columns only
data = data[
    ["reviewText", "overall"]
]

# Remove null values
data = data.dropna()


# Convert star ratings to sentiment
def convert_sentiment(score):

    if score >= 4:
        return "positive"
    else:
        return "negative"


data["sentiment"] = data[
    "overall"
].apply(convert_sentiment)


# Clean reviews
data["clean_review"] = data[
    "reviewText"
].apply(clean_text)


# Balance dataset
print("\nBefore balancing:")
print(
    data["sentiment"]
    .value_counts()
)

min_count = data[
    "sentiment"
].value_counts().min()

balanced_data = (
    data.groupby("sentiment")
    .sample(
        min_count,
        random_state=42
    )
)

print("\nAfter balancing:")
print(
    balanced_data["sentiment"]
    .value_counts()
)


# Features and labels
X = balanced_data[
    "clean_review"
]

y = balanced_data[
    "sentiment"
]


# Vectorization
vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 2)
)

X_vectorized = vectorizer.fit_transform(X)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)


# Predict
y_pred = model.predict(
    X_test
)


# Evaluation
print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# Save files
joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

print(
    "\nModel and vectorizer saved successfully."
)