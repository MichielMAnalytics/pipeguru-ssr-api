"""Pydantic schemas for API requests and responses."""

from typing import List

from pydantic import BaseModel, Field


class RateRequest(BaseModel):
    """Request schema for /v1/rate endpoint."""

    responses: List[str] = Field(
        ...,
        description="List of text responses to convert to PMFs",
        min_length=1,
        example=[
            "I really like this product",
            "Not sure about this",
        ],
    )
    reference_sentences: List[str] = Field(
        ...,
        description="Reference sentences for each Likert scale point",
        min_length=2,
        example=[
            "I definitely would not purchase",
            "I probably would not purchase",
            "I might or might not purchase",
            "I probably would purchase",
            "I definitely would purchase",
        ],
    )
    temperature: float = Field(
        default=1.0,
        description="Temperature scaling parameter (0.0 = sharp, >1.0 = smooth)",
        ge=0.0,
        le=10.0,
    )
    epsilon: float = Field(
        default=0.01,
        description="Regularization parameter for smoothing",
        ge=0.0,
        le=1.0,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "responses": [
                    "I really like this product, would definitely buy",
                    "Not sure, seems overpriced",
                ],
                "reference_sentences": [
                    "I definitely would not purchase",
                    "I probably would not purchase",
                    "I might or might not purchase",
                    "I probably would purchase",
                    "I definitely would purchase",
                ],
                "temperature": 1.0,
                "epsilon": 0.01,
            }
        }
    }


class RatingResult(BaseModel):
    """Individual rating result for a single response."""

    response: str = Field(..., description="Original text response")
    pmf: List[float] = Field(..., description="Probability mass function over scale points")
    expected_value: float = Field(..., description="Expected value (mean of distribution)")
    most_likely_rating: int = Field(
        ..., description="Most likely rating (1-indexed scale point)"
    )
    confidence: float = Field(
        ..., description="Confidence score (max probability in PMF)", ge=0.0, le=1.0
    )


class RateResponse(BaseModel):
    """Response schema for /v1/rate endpoint."""

    success: bool = Field(..., description="Whether the request succeeded")
    ratings: List[RatingResult] = Field(..., description="List of rating results")
    metadata: dict = Field(
        ...,
        description="Additional metadata about the request",
        example={
            "num_responses": 2,
            "num_reference_sentences": 5,
            "processing_time_ms": 1234,
        },
    )


class HealthResponse(BaseModel):
    """Response schema for /v1/health endpoint."""

    status: str = Field(..., description="Health status", example="healthy")
    version: str = Field(..., description="API version", example="0.1.0")
