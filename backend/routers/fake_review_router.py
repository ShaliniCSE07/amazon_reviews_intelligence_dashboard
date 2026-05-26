"""
POST /api/fake-detection — Fake review detector API.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
from database import get_db
from fake_review_detector import detect_fake_reviews_from_records

router = APIRouter(prefix="/api", tags=["fake-detection"])


def _existing_reviews_from_db(product_id: str, db: Session) -> list[dict]:
    """Load stored reviews so burst/similarity run on the full dataset."""
    rows = (
        db.query(models.Review)
        .filter(models.Review.product_id == product_id)
        .all()
    )
    return [
        {
            "text": r.text,
            "rating": float(r.rating) if r.rating is not None else None,
            "date": r.date or "",
        }
        for r in rows
    ]


@router.post("/fake-detection", response_model=schemas.FakeDetectionResponse)
def fake_detection(
    request: schemas.FakeDetectionRequest,
    db: Session = Depends(get_db),
):
    if not request.reviews:
        raise HTTPException(status_code=400, detail="At least one review is required")

    incoming = [
        {"text": r.text, "rating": r.rating, "date": r.date}
        for r in request.reviews
    ]
    existing = _existing_reviews_from_db(request.product_id, db)

    try:
        enriched, summary = detect_fake_reviews_from_records(
            incoming,
            existing_reviews=existing,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fake detection failed: {exc}") from exc

    return schemas.FakeDetectionResponse(
        product_id=request.product_id,
        reviews=[schemas.FakeDetectionReviewResult(**row) for row in enriched],
        summary=schemas.FakeDetectionSummary(**summary),
    )
