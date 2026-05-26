import joblib
from preprocess import clean_text

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def predict_sentiment(text):
    cleaned_text = clean_text(text)
    X = vectorizer.transform([cleaned_text])
    prediction = model.predict(X)[0]
    return prediction