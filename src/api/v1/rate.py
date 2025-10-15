"""Rating endpoint - core SSR functionality."""

import time
from typing import List

import numpy as np
import polars as po
from fastapi import APIRouter, HTTPException
from semantic_similarity_rating import ResponseRater

from src.models.schemas import RateRequest, RateResponse, RatingResult

router = APIRouter()


@router.post("/rate", response_model=RateResponse)
async def rate_responses(request: RateRequest):
    """
    Convert text responses to probability distributions using SSR methodology.

    This endpoint takes a list of text responses and converts them into probability
    mass functions (PMFs) over a Likert scale defined by reference sentences.

    Args:
        request: RateRequest containing responses, reference sentences, and parameters

    Returns:
        RateResponse with PMFs and metrics for each response

    Example:
        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/v1/rate",
            json={
                "responses": ["I really like this product"],
                "reference_sentences": [
                    "I definitely would not purchase",
                    "I probably would not purchase",
                    "I might or might not purchase",
                    "I probably would purchase",
                    "I definitely would purchase"
                ],
                "temperature": 1.0,
                "epsilon": 0.01
            }
        )
        ```
    """
    start_time = time.time()

    try:
        # Create reference DataFrame
        num_points = len(request.reference_sentences)
        df = po.DataFrame(
            {
                "id": ["default"] * num_points,
                "int_response": list(range(1, num_points + 1)),
                "sentence": request.reference_sentences,
            }
        )

        # Initialize ResponseRater in text mode
        rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

        # Get PMFs from responses
        pmfs = rater.get_response_pmfs(
            reference_set_id="default",
            llm_responses=request.responses,
            temperature=request.temperature,
            epsilon=request.epsilon,
        )

        # Format results
        ratings: List[RatingResult] = []
        for response_text, pmf in zip(request.responses, pmfs):
            # Calculate metrics
            expected_value = float(sum((i + 1) * p for i, p in enumerate(pmf)))
            most_likely = int(np.argmax(pmf) + 1)
            confidence = float(np.max(pmf))

            ratings.append(
                RatingResult(
                    response=response_text,
                    pmf=pmf.tolist(),
                    expected_value=round(expected_value, 2),
                    most_likely_rating=most_likely,
                    confidence=round(confidence, 4),
                )
            )

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        return RateResponse(
            success=True,
            ratings=ratings,
            metadata={
                "num_responses": len(request.responses),
                "num_reference_sentences": num_points,
                "processing_time_ms": processing_time_ms,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
