"""Persona generation endpoint for testing and debugging."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.persona_generator import PersonaGenerator

router = APIRouter()
generator = PersonaGenerator()


class GeneratePersonasRequest(BaseModel):
    """Request schema for persona generation."""

    num_personas: int = Field(
        default=5, ge=1, le=100, description="Number of personas to generate"
    )
    segment: str = Field(
        default="general_consumer",
        description="Persona segment (general_consumer, millennial_women, gen_z)",
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )


class PersonaResponse(BaseModel):
    """Response schema for persona generation."""

    personas: List[str] = Field(..., description="List of generated persona descriptions")
    segment: str = Field(..., description="Segment used for generation")
    count: int = Field(..., description="Number of personas generated")


class SegmentsResponse(BaseModel):
    """Response schema for available segments."""

    segments: List[str] = Field(..., description="List of available persona segments")


@router.post("/generate-personas", response_model=PersonaResponse)
async def generate_personas(request: GeneratePersonasRequest):
    """
    Generate synthetic customer personas for ad testing.

    This endpoint is primarily for testing and debugging. In production,
    personas are generated automatically as part of the ad prediction pipeline.

    Args:
        request: GeneratePersonasRequest with num_personas, segment, and optional seed

    Returns:
        PersonaResponse with generated personas

    Example:
        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/v1/generate-personas",
            json={
                "num_personas": 3,
                "segment": "millennial_women",
                "seed": 42
            }
        )
        ```
    """
    # Validate segment
    available_segments = generator.get_available_segments()
    if request.segment not in available_segments:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid segment: {request.segment}. "
            f"Available segments: {available_segments}",
        )

    # Create generator with seed if provided
    if request.seed is not None:
        gen = PersonaGenerator(seed=request.seed)
    else:
        gen = generator

    # Generate personas
    try:
        personas = gen.generate_personas(
            num_personas=request.num_personas, segment=request.segment
        )

        return PersonaResponse(
            personas=personas, segment=request.segment, count=len(personas)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating personas: {str(e)}"
        )


@router.get("/personas/segments", response_model=SegmentsResponse)
async def get_available_segments():
    """
    Get list of available persona segments.

    Returns:
        SegmentsResponse with list of segment names

    Example:
        ```python
        import requests

        response = requests.get("http://localhost:8000/v1/personas/segments")
        print(response.json())
        # {"segments": ["general_consumer", "millennial_women", "gen_z"]}
        ```
    """
    return SegmentsResponse(segments=generator.get_available_segments())
