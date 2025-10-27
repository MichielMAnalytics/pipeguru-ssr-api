"""Rating endpoint - core SSR functionality."""

import time
from typing import List

import numpy as np
import polars as po
from fastapi import APIRouter, HTTPException
from semantic_similarity_rating import ResponseRater

from src.core.reference_statements import get_reference_sets
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
        num_points = len(request.reference_sentences)

        # Determine which reference sets to use
        if request.use_multiple_reference_sets:
            # Use 6 reference sets from paper for better stability (KS sim ~0.88 vs ~0.72)
            all_reference_sets = get_reference_sets("purchase_intent")
            reference_sentences_list = all_reference_sets
            num_sets = len(all_reference_sets)
        else:
            # Use user-provided reference sentences only
            reference_sentences_list = [request.reference_sentences]
            num_sets = 1

        # Collect PMFs from all reference sets
        all_pmfs_per_response = [[] for _ in request.responses]

        for ref_idx, reference_sentences in enumerate(reference_sentences_list):
            # Create reference DataFrame
            df = po.DataFrame(
                {
                    "id": [f"set_{ref_idx}"] * num_points,
                    "int_response": list(range(1, num_points + 1)),
                    "sentence": reference_sentences,
                }
            )

            # Initialize ResponseRater
            rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

            # Get PMFs for this reference set
            pmfs_for_set = rater.get_response_pmfs(
                reference_set_id=f"set_{ref_idx}",
                llm_responses=request.responses,
                temperature=request.temperature,
                epsilon=request.epsilon,
            )

            # Collect PMFs for each response
            for response_idx, pmf in enumerate(pmfs_for_set):
                all_pmfs_per_response[response_idx].append(pmf)

        # Average PMFs across all reference sets for each response
        pmfs = [np.mean(pmf_list, axis=0) for pmf_list in all_pmfs_per_response]

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
                "num_reference_sets": num_sets,
                "use_multiple_reference_sets": request.use_multiple_reference_sets,
                "processing_time_ms": processing_time_ms,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
