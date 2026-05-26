from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True, index=True) # ASIN or generated ID
    title = Column(String, nullable=False)
    category = Column(String, nullable=True)
    overall_rating = Column(Float, default=0.0)
    summary = Column(Text, nullable=True)
    url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    reviews = relationship("Review", back_populates="product", cascade="all, delete-orphan")
    features = relationship("FeatureInsight", back_populates="product", cascade="all, delete-orphan")

class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    product_id = Column(String, ForeignKey("products.id"), nullable=False)
    review_id = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    rating = Column(Integer, nullable=True)
    sentiment = Column(String, nullable=False) # Positive, Negative, Neutral, Mixed
    sentiment_confidence = Column(Float, default=0.0)
    emotions = Column(Text, nullable=True) # JSON string of emotion scores
    quality_score = Column(Float, default=0.0)
    is_fake = Column(Boolean, default=False)
    fake_reasons = Column(Text, nullable=True) # JSON string of reasons
    keywords = Column(Text, nullable=True) # JSON string of list of keywords
    date = Column(String, nullable=True) # YYYY-MM-DD
    author = Column(String, nullable=True)
    language = Column(String, default="en")

    product = relationship("Product", back_populates="reviews")

class FeatureInsight(Base):
    __tablename__ = "feature_insights"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    product_id = Column(String, ForeignKey("products.id"), nullable=False)
    feature_name = Column(String, nullable=False) # e.g. "battery life"
    mentions = Column(Integer, default=0)
    sentiment_score = Column(Float, default=0.0) # -1.0 to 1.0 or 0 to 100
    quotes_pos = Column(Text, nullable=True) # JSON string list of positive quotes
    quotes_neg = Column(Text, nullable=True) # JSON string list of negative quotes

    product = relationship("Product", back_populates="features")
