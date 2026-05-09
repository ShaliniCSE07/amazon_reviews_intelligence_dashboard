print("Step 1")

import joblib
print("Step 2")

from preprocess import clean_text
print("Step 3")

model = joblib.load('model.pkl')
print("Step 4")

vectorizer = joblib.load('vectorizer.pkl')
print("Step 5")

review = input("Enter your Amazon review: ")
print("Step 6")

cleaned_review = clean_text(review)
print("Step 7")

review_vector = vectorizer.transform([cleaned_review])
print("Step 8")

prediction = model.predict(review_vector)
print("Step 9")

print("Predicted Sentiment:", prediction[0])