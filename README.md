# 📊 Amazon Review Intelligence Dashboard

A Streamlit-based Machine Learning application that analyzes Amazon product reviews to generate sentiment insights, feature-level breakdowns, keyword trends, and AI-style summaries.

---

# What it does

Aspect extraction — pulls real product features (sole, cushioning, grip) from reviews using NP chunking, POS tagging, and domain whitelists. Filters out noise words automatically.
Sentiment analysis — scores each extracted feature as positive, negative, or mixed using PMI + TF-IDF ranking.
Fake review detection — flags suspicious reviews using four signals: burst detection, duplicate clustering, generic language detection, and sentiment-rating mismatch.
Missing feature detector — scans negative reviews for desire expressions ("wish it had", "if only") to surface what customers want but the product lacks.
Priority matrix — maps every feature on a frequency vs complaint-rate quadrant so product teams know what to fix first.
---

# Tech stack
Backend — Python, FastAPI
NLP — NLTK (POS tagging, NP chunking), spaCy (en_core_web_sm), scikit-learn (TF-IDF, cosine similarity)
Embeddings — sentence-transformers with all-MiniLM-L6-v2 for duplicate review clustering
Frontend — React, Tailwind CSS
Data — pandas, openpyxl for reading Excel and CSV files with columns Review, Rating, Date

---

# ⚙️ How to Run

## Running ReviewIntellect locally

### Prerequisites
Make sure these are installed on your machine before starting:
- Python 3.9 or above — https://python.org
- Node.js 18 or above — https://nodejs.org
- Git — https://git-scm.com

---

### Step 1 — Clone the repository

Open Command Prompt or PowerShell and run:

git clone https://github.com/your-username/reviewintellect.git
cd reviewintellect

---

### Step 2 — Set up the backend

Navigate to the backend folder and create a virtual environment:

cd backend
python -m venv venv
venv\Scripts\activate

Install all dependencies:

pip install fastapi uvicorn nltk spacy scikit-learn sentence-transformers pandas openpyxl

Download the spaCy English model:

python -m spacy download en_core_web_sm

Download NLTK data (run this once inside Python):

python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords')"

Start the backend server:

uvicorn main:app --reload

Backend will be running at http://localhost:8000

---

### Step 3 — Set up the frontend

Open a new terminal window, navigate to the frontend folder:

cd frontend
npm install
npm run dev

Frontend will be running at http://localhost:5173

---

### Step 4 — Open the app

Go to http://localhost:5173 in your browser. Upload any Excel or CSV file with columns Review, Rating, Date to start analyzing.

---


