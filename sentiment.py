import joblib
from preprocess import clean_text

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_sentiment(review):
    cleaned = clean_text(review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]