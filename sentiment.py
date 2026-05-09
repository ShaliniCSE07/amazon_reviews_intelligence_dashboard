import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def predict_sentiment(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction