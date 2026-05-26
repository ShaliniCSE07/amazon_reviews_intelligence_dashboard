"""
POST /api/missing-features — Missing Feature Detector API with in-memory cache.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

import schemas
from missing_feature_detector import detect_missing_features

router = APIRouter(prefix="/api", tags=["missing-features"])

# product_id -> { "digest": str, "result": dict, "ts": float }
_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_SEC = 3600


def clear_missing_features_cache() -> None:
    _CACHE.clear()


def _reviews_digest(reviews: List[schemas.MissingFeatureReviewInput]) -> str:
    payload = [
        {
            "id": r.id,
            "text": r.text,
            "rating": r.rating,
            "date": r.date,
            "sentiment_score": r.sentiment_score,
            "sentiment": r.sentiment,
        }
        for r in reviews
    ]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_reviews(
    reviews: List[schemas.MissingFeatureReviewInput],
) -> List[Dict[str, Any]]:
    return [
        {
            "id": r.id,
            "text": r.text,
            "rating": r.rating,
            "date": r.date,
            "sentiment_score": r.sentiment_score,
            "sentiment": r.sentiment,
        }
        for r in reviews
    ]


def _get_cached(product_id: str, digest: str) -> Optional[schemas.MissingFeaturesResponse]:
    entry = _CACHE.get(product_id)
    if not entry:
        return None
    if time.time() - entry["ts"] > _CACHE_TTL_SEC:
        del _CACHE[product_id]
        return None
    if entry["digest"] != digest:
        return None
    try:
        data = dict(entry["result"])
        data["cached"] = True
        return schemas.MissingFeaturesResponse(**data)
    except Exception:
        del _CACHE[product_id]
        return None


def _set_cache(product_id: str, digest: str, response: schemas.MissingFeaturesResponse) -> None:
    _CACHE[product_id] = {
        "digest": digest,
        "result": response.model_dump(),
        "ts": time.time(),
    }


@router.post("/missing-features", response_model=schemas.MissingFeaturesResponse)
def missing_features(req: schemas.MissingFeaturesRequest) -> schemas.MissingFeaturesResponse:
    if not req.reviews:
        raise HTTPException(status_code=400, detail="reviews must not be empty")

    digest = _reviews_digest(req.reviews)
    cached = _get_cached(req.product_id, digest)
    if cached:
        return cached

    review_dicts = _normalize_reviews(req.reviews)
    # Never return HTTP 500 — detector returns [] on internal errors
    items = detect_missing_features(review_dicts) or []

    response = schemas.MissingFeaturesResponse(
        product_id=req.product_id,
        missing_features=[
            schemas.MissingFeatureItem(
                feature=x["feature"],
                count=x["count"],
                score=x["score"],
                examples=x["examples"],
            )
            for x in items
        ],
        cached=False,
    )
    _set_cache(req.product_id, digest, response)
    return response
