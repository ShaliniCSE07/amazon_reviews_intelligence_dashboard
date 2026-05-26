import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
import pandas as pd
import io

# Project root (feature_analysis.py lives one level up)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Local imports
from database import engine, get_db
import models
import schemas
from scraper import scrape_amazon_reviews
from nlp_pipeline import NLPPipeline
from export_pdf import generate_pdf_report
from routers.missing_features_router import router as missing_features_router
from routers.fake_review_router import router as fake_review_router
from missing_feature_detector import detect_missing_features, estimate_sentiment_score
from feature_analysis import extract_feature_insights_api, filter_review_keywords

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Amazon Product Review Sentiment Analyser API")
app.include_router(missing_features_router)
app.include_router(fake_review_router)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NLP Pipeline
nlp_processor = NLPPipeline()

@app.get("/")
def read_root():
    return {"message": "Amazon Reviews Sentiment Analyser API is running."}

@app.get("/api/products", response_model=list[schemas.ProductResponse])
def get_products(db: Session = Depends(get_db)):
    products = db.query(models.Product).order_by(models.Product.created_at.desc()).all()
    res = []
    for p in products:
        count = db.query(models.Review).filter(models.Review.product_id == p.id).count()
        res.append(schemas.ProductResponse(
            id=p.id,
            title=p.title,
            category=p.category,
            overall_rating=p.overall_rating,
            summary=p.summary,
            url=p.url,
            reviews_count=count
        ))
    return res

def _process_and_save_analysis(db: Session, product_info: Dict[str, Any], raw_reviews: list[Dict[str, Any]]) -> schemas.AnalysisResponse:
    # 1. Save or Update Product
    db_product = db.query(models.Product).filter(models.Product.id == product_info["id"]).first()
    if db_product:
        # Clear old reviews and features to recalculate
        db.query(models.Review).filter(models.Review.product_id == db_product.id).delete()
        db.query(models.FeatureInsight).filter(models.FeatureInsight.product_id == db_product.id).delete()
        
        db_product.title = product_info["title"]
        db_product.category = product_info["category"]
        db_product.overall_rating = product_info["overall_rating"]
        db_product.url = product_info["url"]
    else:
        db_product = models.Product(
            id=product_info["id"],
            title=product_info["title"],
            category=product_info["category"],
            overall_rating=product_info["overall_rating"],
            url=product_info["url"]
        )
        db.add(db_product)
    
    db.commit()

    # 2. Extract texts and ratings for ML processing
    texts = [r["text"] for r in raw_reviews]
    
    # Run NLP analyses
    sentiments = []
    confidences = []
    emotions_list = []
    qualities = []
    fakes = []
    fake_reasons_list = []
    keywords_list = []

    for idx, r in enumerate(raw_reviews):
        # Sentiment
        lbl, conf = nlp_processor.analyze_sentiment(r["text"])
        sentiments.append(lbl)
        confidences.append(conf)
        
        # Emotions
        emo = nlp_processor.analyze_emotion(r["text"])
        emotions_list.append(emo)
        
        # Quality
        qual = nlp_processor.score_review_quality(r["text"])
        qualities.append(qual)
        
        # Fake review detection
        is_f, reasons = nlp_processor.detect_fake_review(r["text"], r.get("rating"), texts)
        fakes.append(is_f)
        fake_reasons_list.append(reasons)
        
        # Keywords (filtered — no days/weeks/bad/etc. in word cloud)
        kws = filter_review_keywords(nlp_processor.extract_keywords(r["text"]))
        keywords_list.append(kws)

    # 3. Feature aspects — use feature_analysis.py (not legacy nlp_pipeline chunker)
    review_rows = [
        {"text": t, "sentiment": s, "rating": raw_reviews[i].get("rating")}
        for i, (t, s) in enumerate(zip(texts, sentiments))
    ]
    feature_insights = extract_feature_insights_api(
        review_rows,
        product_name=product_info.get("title", ""),
        category=product_info.get("category", ""),
        top_n=20,
    )

    # 4. Summary Generation
    summary_text = nlp_processor.generate_extractive_summary(texts)
    db_product.summary = summary_text
    db.commit()

    # 5. Write reviews and feature insights to database
    db_reviews = []
    for idx, r in enumerate(raw_reviews):
        db_rev = models.Review(
            product_id=db_product.id,
            review_id=r["review_id"],
            text=r["text"],
            rating=r["rating"],
            sentiment=sentiments[idx],
            sentiment_confidence=confidences[idx],
            emotions=json.dumps(emotions_list[idx]),
            quality_score=qualities[idx],
            is_fake=fakes[idx],
            fake_reasons=json.dumps(fake_reasons_list[idx]),
            keywords=json.dumps(keywords_list[idx]),
            date=r["date"],
            author=r["author"],
            language=r["language"]
        )
        db.add(db_rev)
        db_reviews.append(db_rev)

    db_features = []
    for f in feature_insights:
        db_feat = models.FeatureInsight(
            product_id=db_product.id,
            feature_name=f["feature_name"],
            mentions=f["mentions"],
            sentiment_score=f["sentiment_score"],
            quotes_pos=json.dumps(f["quotes_pos"]),
            quotes_neg=json.dumps(f["quotes_neg"])
        )
        db.add(db_feat)
        db_features.append(db_feat)

    db.commit()

    # 6. Format Response
    return _build_analysis_payload(db_product, db_reviews, db_features)


def _reviews_for_missing_detector(reviews: list[models.Review]) -> list[dict]:
    return [
        {
            "id": r.review_id or r.id,
            "text": r.text,
            "rating": r.rating,
            "date": r.date,
            "sentiment": r.sentiment,
            "sentiment_score": estimate_sentiment_score(
                rating=r.rating,
                sentiment_label=r.sentiment,
            ),
        }
        for r in reviews
    ]


def _missing_features_payload(reviews: list[models.Review]) -> list[schemas.MissingFeatureItem]:
    try:
        raw = detect_missing_features(_reviews_for_missing_detector(reviews))
        return [
            schemas.MissingFeatureItem(
                feature=x["feature"],
                count=x["count"],
                score=x["score"],
                examples=x["examples"],
            )
            for x in raw
        ]
    except Exception as e:
        print(f"Missing feature detection skipped: {e}")
        return []


def _build_analysis_payload(product: models.Product, reviews: list[models.Review], features: list[models.FeatureInsight]) -> schemas.AnalysisResponse:
    # 1. Product payload
    prod_resp = schemas.ProductResponse(
        id=product.id,
        title=product.title,
        category=product.category,
        overall_rating=product.overall_rating,
        summary=product.summary,
        url=product.url,
        reviews_count=len(reviews)
    )

    # 2. Review payload list
    review_items = []
    sentiment_dist = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
    emotion_totals = {"frustration": 0.0, "delight": 0.0, "disappointment": 0.0, "surprise": 0.0}
    
    for r in reviews:
        sentiment_dist[r.sentiment] = sentiment_dist.get(r.sentiment, 0) + 1
        
        emos = json.loads(r.emotions) if r.emotions else {}
        for k, v in emos.items():
            emotion_totals[k] = emotion_totals.get(k, 0.0) + v
            
        review_items.append(schemas.ReviewItem(
            id=r.id,
            text=r.text,
            rating=r.rating,
            sentiment=r.sentiment,
            sentiment_confidence=r.sentiment_confidence,
            emotions=emos,
            quality_score=r.quality_score,
            is_fake=r.is_fake,
            fake_reasons=json.loads(r.fake_reasons) if r.fake_reasons else [],
            keywords=json.loads(r.keywords) if r.keywords else [],
            date=r.date,
            author=r.author,
            language=r.language
        ))

    # Average emotions
    total_reviews = len(reviews)
    emotion_dist = {}
    if total_reviews > 0:
        for k, v in emotion_totals.items():
            emotion_dist[k] = round((v / total_reviews) * 100, 1)
    else:
        emotion_dist = {"frustration": 0.0, "delight": 0.0, "disappointment": 0.0, "surprise": 0.0}

    # 3. Features payload list
    feature_items = []
    for f in features:
        feature_items.append(schemas.FeatureInsightItem(
            id=f.id,
            feature_name=f.feature_name,
            mentions=f.mentions,
            sentiment_score=f.sentiment_score,
            quotes_pos=json.loads(f.quotes_pos) if f.quotes_pos else [],
            quotes_neg=json.loads(f.quotes_neg) if f.quotes_neg else []
        ))

    # 4. Trend data aggregation (group by date)
    df_reviews = pd.DataFrame([{
        "date": r.date,
        "sentiment": r.sentiment,
        "score": 1.0 if r.sentiment == "Positive" else (-1.0 if r.sentiment == "Negative" else 0.0)
    } for r in reviews])
    
    trend_data = []
    if not df_reviews.empty and "date" in df_reviews.columns:
        # Group by date and calculate average score
        trend_grp = df_reviews.groupby("date").agg(
            avg_score=("score", "mean"),
            count=("sentiment", "count")
        ).reset_index().sort_values("date")
        
        for _, row in trend_grp.iterrows():
            if row["date"]:
                # Map avg_score [-1, 1] to a sentiment percentage [0, 100]
                percentage_score = round(((row["avg_score"] + 1.0) / 2.0) * 100, 1)
                trend_data.append({
                    "date": row["date"],
                    "sentiment_score": percentage_score,
                    "count": int(row["count"])
                })

    return schemas.AnalysisResponse(
        product=prod_resp,
        reviews=review_items,
        features=feature_items,
        sentiment_distribution=sentiment_dist,
        emotion_distribution=emotion_dist,
        trend_data=trend_data,
        missing_features=_missing_features_payload(reviews),
    )

@app.post("/api/analyze/asin", response_model=schemas.AnalysisResponse)
def analyze_asin(req: schemas.ASINRequest, db: Session = Depends(get_db)):
    try:
        product_info, raw_reviews = scrape_amazon_reviews(req.url_or_asin, req.max_reviews)
        if not raw_reviews:
            raise HTTPException(status_code=400, detail="Could not scrape or generate reviews for this URL/ASIN.")
        
        return _process_and_save_analysis(db, product_info, raw_reviews)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/paste", response_model=schemas.AnalysisResponse)
def analyze_paste(req: schemas.PasteReviewsRequest, db: Session = Depends(get_db)):
    try:
        # Construct product metadata
        import hashlib
        # Generate a unique product ID based on name hashing
        prod_id = "PST-" + hashlib.md5(req.product_name.encode('utf-8')).hexdigest()[:8].upper()
        
        ratings = req.ratings or [5] * len(req.reviews)
        if len(ratings) < len(req.reviews):
            ratings = ratings + [3] * (len(req.reviews) - len(ratings))

        import datetime
        dates = req.dates or []
        if len(dates) < len(req.reviews):
            base_date = datetime.date.today()
            # Generate spread of dates
            dates = [(base_date - datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(req.reviews))]

        raw_reviews = []
        for idx, text in enumerate(req.reviews):
            raw_reviews.append({
                "review_id": f"PR-{idx}",
                "text": text,
                "rating": ratings[idx],
                "author": f"User_{idx+1}",
                "date": dates[idx],
                "language": "en"
            })

        avg_rating = sum(ratings) / len(ratings) if ratings else 5.0
        product_info = {
            "id": prod_id,
            "title": req.product_name,
            "category": req.category or "General",
            "overall_rating": round(avg_rating, 1),
            "url": None,
            "reviews_count": len(raw_reviews)
        }

        return _process_and_save_analysis(db, product_info, raw_reviews)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/{product_id}", response_model=schemas.AnalysisResponse)
def get_analysis(product_id: str, db: Session = Depends(get_db)):
    product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Analysis for this product was not found.")
        
    reviews = db.query(models.Review).filter(models.Review.product_id == product_id).all()
    features = db.query(models.FeatureInsight).filter(models.FeatureInsight.product_id == product_id).all()
    
    return _build_analysis_payload(product, reviews, features)

@app.delete("/api/analysis/{product_id}")
def delete_analysis(product_id: str, db: Session = Depends(get_db)):
    product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found.")
    db.delete(product)
    db.commit()
    return {"message": f"Successfully deleted analysis session for product {product_id}."}

@app.get("/api/export/pdf/{product_id}")
def export_pdf(product_id: str, db: Session = Depends(get_db)):
    product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found.")
        
    reviews = db.query(models.Review).filter(models.Review.product_id == product_id).all()
    features = db.query(models.FeatureInsight).filter(models.FeatureInsight.product_id == product_id).all()
    
    payload = _build_analysis_payload(product, reviews, features)
    # Convert pydantic back to dict
    pdf_bytes = generate_pdf_report(payload.dict())
    
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=sentiment_report_{product_id}.pdf"}
    )

@app.get("/api/export/csv/{product_id}")
def export_csv(product_id: str, db: Session = Depends(get_db)):
    reviews = db.query(models.Review).filter(models.Review.product_id == product_id).all()
    if not reviews:
        raise HTTPException(status_code=404, detail="Reviews not found.")
        
    data = []
    for r in reviews:
        data.append({
            "review_id": r.review_id,
            "author": r.author,
            "date": r.date,
            "rating": r.rating,
            "text": r.text,
            "sentiment": r.sentiment,
            "sentiment_confidence": r.sentiment_confidence,
            "quality_score": r.quality_score,
            "is_fake": r.is_fake,
            "fake_reasons": r.fake_reasons,
            "keywords": r.keywords
        })
        
    df = pd.DataFrame(data)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    return StreamingResponse(
        io.BytesIO(stream.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=sentiment_data_{product_id}.csv"}
    )

@app.post("/api/compare")
def compare_products(req: list[str], db: Session = Depends(get_db)):
    if len(req) < 2:
        raise HTTPException(status_code=400, detail="Must supply at least two product IDs to compare.")
        
    comparison_data = []
    for pid in req:
        product = db.query(models.Product).filter(models.Product.id == pid).first()
        if not product:
            continue
            
        reviews = db.query(models.Review).filter(models.Review.product_id == pid).all()
        features = db.query(models.FeatureInsight).filter(models.FeatureInsight.product_id == pid).all()
        
        payload = _build_analysis_payload(product, reviews, features)
        comparison_data.append(payload)
        
    return comparison_data
