# 📊 Amazon Review Intelligence Dashboard

A Streamlit-based Machine Learning application that analyzes Amazon product reviews to generate sentiment insights, feature-level breakdowns, keyword trends, and AI-style summaries.

---

# 🚀 Features

- Sentiment classification (Positive / Negative / Neutral)
- Feature-wise sentiment analysis (Battery, Camera, Display, etc.)
- Top keyword extraction (General, Positive, Negative)
- AI-style smart summary of reviews
- Rating distribution visualization (1–5 stars)
- Drill-down into feature-specific reviews
- Download full analyzed report as CSV

---

# 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

---

# 📂 Project Structure

- app.py → Streamlit frontend dashboard  
- train.py → Model training script  
- sentiment.py → Sentiment prediction logic  
- preprocess.py → Text cleaning utilities  
- feature_analysis.py → Feature extraction + insights  
- summary.py → AI-style summary generator  
- model.pkl → Trained ML model  
- vectorizer.pkl → TF-IDF vectorizer  
- data/ → Amazon review dataset (not uploaded due to size limits)

---

# ⚙️ How to Run

# 1. Clone the repository
```bash
git clone https://github.com/your-username/amazon_reviews_intelligence_dashboard.git
cd amazon_reviews_intelligence_dashboard
```
# 2.Install dependencies
```bash
pip install -r requirements.txt
```
# 3.Train the model (first time only)
```bash
python train.py
```
# 4.Run the Streamlit app
```bash
streamlit run app.py
```
