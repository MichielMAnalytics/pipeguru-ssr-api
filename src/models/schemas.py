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


# Ad Prediction Schemas

class PredictAdRequest(BaseModel):
    """Request schema for /v1/predict-ad endpoint."""

    ad_image_base64: str = Field(
        ...,
        description="Base64-encoded ad image",
        min_length=100,
    )
    num_personas: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of synthetic personas to generate",
    )
    segment: str = Field(
        default="general_consumer",
        description="Persona segment (general_consumer, millennial_women, gen_z)",
    )
    reference_sentences: List[str] = Field(
        default=[
            "I definitely would not purchase",
            "I probably would not purchase",
            "I might or might not purchase",
            "I probably would purchase",
            "I definitely would purchase",
        ],
        description="Reference sentences for Likert scale",
        min_length=2,
    )
    temperature: float = Field(
        default=1.0,
        description="SSR temperature parameter",
        ge=0.0,
        le=10.0,
    )
    epsilon: float = Field(
        default=0.01,
        description="SSR epsilon parameter",
        ge=0.0,
        le=1.0,
    )


class PersonaResult(BaseModel):
    """Individual persona evaluation result."""

    persona_id: int = Field(..., description="Persona identifier")
    persona_description: str = Field(..., description="Truncated persona description")
    llm_response: str = Field(..., description="Truncated LLM response")
    pmf: List[float] = Field(..., description="Probability mass function")
    expected_value: float = Field(..., description="Expected value (1-5 scale)")


class PredictAdResponse(BaseModel):
    """Response schema for /v1/predict-ad endpoint."""

    predicted_conversion_rate: float = Field(
        ...,
        description="Predicted conversion rate (0-1)",
        ge=0.0,
        le=1.0,
    )
    confidence: float = Field(
        ...,
        description="Confidence score based on persona agreement",
        ge=0.0,
        le=1.0,
    )
    pmf_aggregate: List[float] = Field(
        ...,
        description="Aggregated probability mass function across all personas",
    )
    expected_value: float = Field(
        ...,
        description="Average expected purchase intent (1-5 scale)",
    )
    persona_results: List[PersonaResult] = Field(
        ...,
        description="Individual results for each persona",
    )
    cost: dict = Field(
        ...,
        description="Cost breakdown",
        example={"llm_calls": 20, "estimated_cost_usd": 0.30},
    )
    metadata: dict = Field(
        ...,
        description="Additional metadata",
    )
