import streamlit as st
import pandas as pd
from collections import Counter

from summary import generate_feature_summary
from sentiment import predict_sentiment
from feature_analysis import (
    feature_sentiment_breakdown,
    extract_sentiment_keywords
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Amazon Review Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# 🌈 MODERN UI STYLE (NO FEATURE CHANGE)
# =========================
st.markdown("""
<style>

/* Background */
body {
    background-color: #0b1220;
}

/* Main container spacing */
.block-container {
    padding: 2rem 3rem;
}

/* Title */
.main-title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00c6ff, #7f00ff, #ff4ecd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}

/* Card style */
.card {
    background: #111827;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #1f2937;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* Summary box */
.summary-box {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: #e2e8f0;
    padding: 18px;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    line-height: 1.6;
}

/* ✨ Glow effect for expander headers */
div[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid #1f2937;
    background: #0f172a;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #111827;
    border-radius: 10px;
    border: 1px solid #1f2937;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #111827;
    border-radius: 12px;
    padding: 10px;
}

/* Buttons glow */
div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    transition: 0.3s ease-in-out;
    box-shadow: 0 0 10px rgba(0,114,255,0.4);
}

div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(0,198,255,0.8);
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER (NO CHANGE IN LOGIC)
# =========================
st.markdown("<div class='main-title'>Amazon Review Intelligence Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload reviews and unlock AI-powered insights</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])


# =========================
# HELPERS (UNCHANGED)
# =========================
def detect_review_column(data):
    for col in data.columns:
        if col.lower() in ["review","reviewtext","reviews","text","comment","content"]:
            return col
    return data.columns[0]


def detect_rating_column(data):
    for col in data.columns:
        if col.lower() in ["rating","stars","score"]:
            return col
    return None


def get_feature_reviews(data, keywords):
    filtered = []
    for _, row in data.iterrows():
        review = str(row["review"]).lower()
        if any(word in review for word in keywords):
            filtered.append(row)
    return pd.DataFrame(filtered)


# =========================
# AI SUMMARY (UNCHANGED)
# =========================
def generate_ai_style_summary(data):

    pos_reviews = data[data["predicted_sentiment"] == "positive"]["review"]
    neg_reviews = data[data["predicted_sentiment"] == "negative"]["review"]

    pos_words = ["battery","display","camera","quality","performance","design"]
    neg_words = ["heat","slow","lag","poor","issue","problem","drain"]

    pos_hits, neg_hits = [], []

    for r in pos_reviews:
        for w in pos_words:
            if w in r.lower():
                pos_hits.append(w)

    for r in neg_reviews:
        for w in neg_words:
            if w in r.lower():
                neg_hits.append(w)

    pos_top = list(set(pos_hits))[:3]
    neg_top = list(set(neg_hits))[:3]

    if pos_top and neg_top:
        return f"Users are highly satisfied with {', '.join(pos_top)} but consistently report issues with {', '.join(neg_top)}."
    elif pos_top:
        return f"Users are highly satisfied with {', '.join(pos_top)} overall."
    elif neg_top:
        return f"Users mostly report issues with {', '.join(neg_top)}."
    else:
        return "Users have mixed feedback across different features."


# =========================
# TOP INSIGHTS (UNCHANGED)
# =========================
def extract_top_insights(data):

    pos_words = ["battery","display","camera","quality","performance","design"]
    neg_words = ["heat","slow","lag","poor","issue","problem","drain"]

    pos_hits, neg_hits = [], []

    for _, row in data.iterrows():

        text = row["review"].lower()

        if row["predicted_sentiment"] == "positive":
            for w in pos_words:
                if w in text:
                    pos_hits.append(w)

        if row["predicted_sentiment"] == "negative":
            for w in neg_words:
                if w in text:
                    neg_hits.append(w)

    return Counter(pos_hits).most_common(3), Counter(neg_hits).most_common(3)


# =========================
# MAIN APP
# =========================
if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    review_column = detect_review_column(data)
    rating_column = detect_rating_column(data)

    data = data.rename(columns={review_column: "review"})
    data = data.dropna(subset=["review"])
    data["review"] = data["review"].astype(str)

    reviews = data["review"].tolist()

    # =========================
    # SENTIMENT ANALYSIS
    # =========================
    st.subheader("Analyzing Reviews")

    sentiments = []
    progress = st.progress(0)

    for i, r in enumerate(reviews):
        sentiments.append(predict_sentiment(r))
        progress.progress((i + 1) / len(reviews))

    data["predicted_sentiment"] = sentiments

    # =========================
    # PREVIEW
    # =========================
    st.divider()
    st.subheader("Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    # =========================
    # SENTIMENT OVERVIEW
    # =========================
    st.divider()
    st.subheader("Sentiment Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Positive", sum(data["predicted_sentiment"]=="positive"))
    col2.metric("Negative", sum(data["predicted_sentiment"]=="negative"))
    col3.metric("Neutral", sum(data["predicted_sentiment"]=="neutral"))

    st.bar_chart(data["predicted_sentiment"].value_counts())

    # =========================
    # RATING DISTRIBUTION
    # =========================
    if rating_column:

        st.divider()
        st.subheader("Rating Distribution")

        counts = data[rating_column].value_counts().sort_index(ascending=False)
        total = counts.sum()

        for star in [5,4,3,2,1]:

            count = counts.get(star, 0)
            pct = (count / total * 100) if total else 0
            bar = "█" * int(pct // 5)

            col1,col2,col3 = st.columns([1,6,2])

            with col1:
                st.markdown(f"**{star}★**")
            with col2:
                st.markdown(f"`{bar}`")
            with col3:
                st.markdown(f"{pct:.0f}% ({count})")

    # =========================
    # FEATURE INSIGHTS (DRILL-DOWN PRESERVED)
    # =========================
    st.divider()
    st.subheader("Feature Insights 🧩")

    feature_results = feature_sentiment_breakdown(data)

    for feature, result in feature_results.items():

        with st.expander(f"🔍 {feature.capitalize()} ({result['mentions']})"):

            col1, col2, col3 = st.columns(3)

            col1.metric("Positive", result["positive"])
            col2.metric("Negative", result["negative"])
            col3.metric("Total", result["mentions"])

            st.markdown(f"<div class='summary-box'>{result['summary']}</div>", unsafe_allow_html=True)

            st.write("### Related Reviews")

            keywords = feature.lower().split()
            feature_reviews = get_feature_reviews(data, keywords)

            if not feature_reviews.empty:

                st.dataframe(feature_reviews[["review","predicted_sentiment"]], use_container_width=True)

                st.bar_chart(feature_reviews["predicted_sentiment"].value_counts())

            else:
                st.warning("No matching reviews found.")

    # =========================
    # KEYWORDS
    # =========================
    st.divider()
    st.subheader("Keyword Insights 🔑")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("Top")
        for w,c in extract_sentiment_keywords(data, None):
            st.write(f"{w} ({c})")

    with col2:
        st.markdown("Positive")
        for w,c in extract_sentiment_keywords(data, "positive"):
            st.write(f"{w} ({c})")

    with col3:
        st.markdown("Negative")
        for w,c in extract_sentiment_keywords(data, "negative"):
            st.write(f"{w} ({c})")

    # =========================
    # AI SUMMARY
    # =========================
    st.divider()
    st.subheader("AI Insights 🧠")

    st.markdown(f"<div class='summary-box'>{generate_ai_style_summary(data)}</div>", unsafe_allow_html=True)

    # =========================
    # TOP ISSUES
    # =========================
    st.divider()
    st.subheader("Top Issues & Highlights 📌")

    top_pos, top_neg = extract_top_insights(data)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Highlights")
        for w,c in top_pos:
            st.success(f"{w} ({c})")

    with col2:
        st.markdown("### Issues")
        for w,c in top_neg:
            st.error(f"{w} ({c})")

    # =========================
    # DOWNLOAD
    # =========================
    st.divider()
    st.subheader("Download Report 📥")

    st.download_button(
        "Download CSV Report",
        data.to_csv(index=False).encode("utf-8"),
        "report.csv",
        "text/csv"
    )

    # =========================
    # DATA
    # =========================
    st.divider()
    st.subheader("Detailed Reviews")
    st.dataframe(data, use_container_width=True)