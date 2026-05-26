from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

class ASINRequest(BaseModel):
    url_or_asin: str = Field(..., description="Amazon product URL or ASIN code")
    max_reviews: Optional[int] = Field(50, description="Maximum reviews to fetch")

class PasteReviewsRequest(BaseModel):
    product_name: str = Field("Pasted Product", description="Name of the product")
    category: Optional[str] = Field("General", description="Product category")
    reviews: List[str] = Field(..., description="List of raw review texts")
    ratings: Optional[List[int]] = Field(None, description="Optional ratings corresponding to each review")
    dates: Optional[List[str]] = Field(None, description="Optional dates in YYYY-MM-DD format")

class ReviewItem(BaseModel):
    id: Optional[int] = None
    text: str
    rating: Optional[int] = None
    sentiment: str
    sentiment_confidence: float
    emotions: Dict[str, float]
    quality_score: float
    is_fake: bool
    fake_reasons: List[str]
    keywords: List[str]
    date: Optional[str] = None
    author: Optional[str] = None
    language: str

class FeatureInsightItem(BaseModel):
    id: Optional[int] = None
    feature_name: str
    mentions: int
    sentiment_score: float
    quotes_pos: List[str]
    quotes_neg: List[str]

class ProductResponse(BaseModel):
    id: str
    title: str
    category: Optional[str] = None
    overall_rating: float
    summary: Optional[str] = None
    url: Optional[str] = None
    reviews_count: int

class AnalysisResponse(BaseModel):
    product: ProductResponse
    reviews: List[ReviewItem]
    features: List[FeatureInsightItem]
    sentiment_distribution: Dict[str, int]
    emotion_distribution: Dict[str, float]
    trend_data: List[Dict[str, Any]]
    missing_features: List["MissingFeatureItem"] = []


# ── Missing Feature Detector ───────────────────────────────────────────────────

class MissingFeatureReviewInput(BaseModel):
    id: Union[str, int] = Field(..., description="Review identifier")
    text: str
    rating: Optional[int] = None
    date: Optional[str] = None
    sentiment_score: Optional[float] = Field(
        None, description="Polarity 0–1; values below 0.4 count as negative"
    )
    sentiment: Optional[str] = Field(
        None, description="Optional label: Positive, Negative, Neutral, Mixed"
    )


class MissingFeaturesRequest(BaseModel):
    product_id: str
    reviews: List[MissingFeatureReviewInput]


class MissingFeatureItem(BaseModel):
    feature: str
    count: int
    score: float
    examples: List[str]


class MissingFeaturesResponse(BaseModel):
    product_id: str
    missing_features: List[MissingFeatureItem]
    cached: bool = False


# ── Fake Review Detector ───────────────────────────────────────────────────────

class FakeDetectionReviewInput(BaseModel):
    text: str
    rating: float = Field(..., ge=1.0, le=5.0)
    date: str = Field(..., description="YYYY-MM-DD")


class FakeDetectionRequest(BaseModel):
    product_id: str
    reviews: List[FakeDetectionReviewInput]


class FakeDetectionReviewResult(BaseModel):
    text: str
    rating: Optional[float] = None
    date: Optional[str] = None
    mismatch_score: float = 0.0
    burst_day: bool = False
    max_similarity: float = 0.0
    is_duplicate: bool = False
    generic_score: int = 0
    specificity_score: int = 0
    is_generic: bool = False
    is_too_short: bool = False
    fake_score: int = 0
    fake_label: str = "Likely genuine"
    signals_fired: List[str] = []


class FakeDetectionSummary(BaseModel):
    total: int
    likely_fake: int
    suspicious: int
    likely_genuine: int
    flagged: int = 0
    fake_percentage: float
    top_signals: List[str] = []


class FakeDetectionResponse(BaseModel):
    product_id: str
    reviews: List[FakeDetectionReviewResult]
    summary: FakeDetectionSummary
