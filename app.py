import streamlit as st
import pandas as pd
from collections import Counter

from summary import generate_feature_summary
from sentiment import predict_sentiment
from feature_analysis import (
    feature_sentiment_breakdown,
    extract_sentiment_keywords,
    extract_dynamic_features
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
# STYLING
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.summary-box {
    background-color: #f3f4f6;
    color: #111827;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #d1d5db;
    line-height: 1.8;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("Amazon Review Intelligence Dashboard")
st.caption("Upload customer reviews and analyze product feedback")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
product_name = st.text_input("Product Name/Category (e.g., Laptop, Headphone, Book)", value="Smartphone")

# =========================
# HELPERS
# =========================
def detect_review_column(data):
    # Expanded list of common review column names
    target_cols = ["review", "reviewtext", "reviews", "text", "comment", "content", "body", "feedback", "message"]
    for col in data.columns:
        if col.lower() in target_cols:
            return col
    # Fallback: find column with longest average string length
    string_cols = data.select_dtypes(include=['object']).columns
    if not string_cols.empty:
        return data[string_cols].apply(lambda x: x.str.len().mean()).idxmax()
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
# AI SUMMARY
# =========================
def generate_ai_style_summary(data, product_name):
    """
    Generates a natural language summary based on top dynamic keywords.
    """
    pos_keywords = extract_sentiment_keywords(data, "positive", top_n=5, product_name=product_name)
    neg_keywords = extract_sentiment_keywords(data, "negative", top_n=5, product_name=product_name)

    pos_top = [w for w, c in pos_keywords]
    neg_top = [w for w, c in neg_keywords]

    if pos_top and neg_top:
        return f"For the **{product_name}**, users are highly satisfied with features like **{', '.join(pos_top[:3])}** but consistently report issues with **{', '.join(neg_top[:3])}**."
    elif pos_top:
        return f"For the **{product_name}**, users are highly satisfied with **{', '.join(pos_top[:3])}** overall."
    elif neg_top:
        return f"For the **{product_name}**, users mostly report issues with **{', '.join(neg_top[:3])}** regarding the **{product_name}**."
    else:
        return f"Users have mixed feedback across different aspects of the **{product_name}**."


# =========================
# TOP INSIGHTS
# =========================
def extract_top_insights(data, product_name):
    """
    Dynamically extracts top positive and negative keywords for highlights.
    """
    pos_top = extract_sentiment_keywords(data, "positive", top_n=3, product_name=product_name)
    neg_top = extract_sentiment_keywords(data, "negative", top_n=3, product_name=product_name)

    return pos_top, neg_top


# =========================
# MAIN APP
# =========================
if uploaded_file:

    # Load file
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    review_column = detect_review_column(data)
    rating_column = detect_rating_column(data)

    data = data.rename(columns={review_column: "review"})
    
    # Clean data: handle missing values but be less aggressive with duplicates
    data = data.dropna(subset=["review"])
    
    # Only drop rows where the review is empty or whitespace
    data = data[data["review"].str.strip() != ""]
    
    # Optional: Keep duplicates as they might be distinct reviews with same content
    # data = data.drop_duplicates(subset=["review"]) 
    
    data["review"] = data["review"].astype(str)
    
    # Reset index to ensure perfect alignment when assigning results
    data = data.reset_index(drop=True)

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

    with col1:
        st.metric("Positive", sum(data["predicted_sentiment"]=="positive"))
    with col2:
        st.metric("Negative", sum(data["predicted_sentiment"]=="negative"))
    with col3:
        st.metric("Neutral", sum(data["predicted_sentiment"]=="neutral"))

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
    # FEATURE INSIGHTS (DRILL-DOWN)
    # =========================
    st.divider()
    st.subheader(f"Feature Insights for {product_name} 🧩")
    
    # Dynamically extract features based on reviews and product name
    dynamic_features = extract_dynamic_features(data, product_name)
    feature_results = feature_sentiment_breakdown(data, features=dynamic_features)

    for feature, result in feature_results.items():

        with st.expander(f"🔍 {feature.capitalize()} ({result['mentions']} mentions)"):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Positive", result["positive"])

            with col2:
                st.metric("Negative", result["negative"])

            with col3:
                st.metric("Total", result["mentions"])

            st.info(result["summary"])

            # DRILL-DOWN SECTION
            st.write("### 📄 Related Reviews")

            keywords = feature.lower().split()

            feature_reviews = get_feature_reviews(data, keywords)

            if not feature_reviews.empty:

                st.dataframe(
                    feature_reviews[["review", "predicted_sentiment"]],
                    use_container_width=True
                )

                st.write("### 📊 Sentiment Breakdown")

                st.bar_chart(
                    feature_reviews["predicted_sentiment"].value_counts()
                )

            else:
                st.warning("No matching reviews found.")

    # =========================
    # KEYWORDS
    # =========================
    st.divider()
    st.subheader("Keyword Insights 🔑")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Top Keywords")
        for w,c in extract_sentiment_keywords(data, None, product_name=product_name):
            st.write(f"{w} ({c})")

    with col2:
        st.markdown("### Positive")
        for w,c in extract_sentiment_keywords(data, "positive", product_name=product_name):
            st.write(f"{w} ({c})")

    with col3:
        st.markdown("### Negative")
        for w,c in extract_sentiment_keywords(data, "negative", product_name=product_name):
            st.write(f"{w} ({c})")

    # =========================
    # AI SUMMARY
    # =========================
    st.divider()
    st.subheader(f"AI Insights for {product_name} 🧠")

    ai_summary = generate_ai_style_summary(data, product_name)

    st.markdown(f"""
    <div class="summary-box">
        {ai_summary}
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # TOP ISSUES & HIGHLIGHTS
    # =========================
    st.divider()
    st.subheader("Top Issues & Highlights 📌")

    top_pos, top_neg = extract_top_insights(data, product_name)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟢 Highlights")
        for w,c in top_pos:
            st.success(f"{w} ({c})")

    with col2:
        st.markdown("### 🔴 Issues")
        for w,c in top_neg:
            st.error(f"{w} ({c})")

    # =========================
    # DOWNLOAD REPORT
    # =========================
    st.divider()
    st.subheader("Download Report 📥")

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Full Report (CSV)",
        data=csv,
        file_name="amazon_review_report.csv",
        mime="text/csv"
    )

    # =========================
    # FINAL DATA
    # =========================
    st.divider()
    st.subheader("Detailed Reviews")
    st.dataframe(data, use_container_width=True)