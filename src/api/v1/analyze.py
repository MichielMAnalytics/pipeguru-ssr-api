"""Creative analysis endpoint - simplified unified API."""

import logging
import time
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.core.ad_predictor import AdPredictor
from src.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter()


def get_ad_predictor() -> AdPredictor:
    """Dependency that provides an AdPredictor instance."""
    return AdPredictor()


def get_llm_client() -> LLMClient:
    """Dependency that provides an LLMClient instance."""
    return LLMClient()


# Request/Response Schemas

class AnalyzeCreativeRequest(BaseModel):
    """Request to analyze a creative with specific personas."""

    creative_base64: str = Field(
        ...,
        min_length=100,
        description="Base64-encoded image of the ad creative",
    )

    personas: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of persona descriptions to evaluate against",
    )

    # SSR parameters (optional, use defaults)
    reference_sentences: List[str] = Field(
        default=[
            "I definitely would not purchase",
            "I probably would not purchase",
            "I might or might not purchase",
            "I probably would purchase",
            "I definitely would purchase",
        ],
        min_length=2,
        description="Likert scale reference sentences",
    )

    temperature: float = Field(default=1.0, ge=0.0, le=10.0)
    epsilon: float = Field(default=0.01, ge=0.0, le=1.0)
    use_multiple_reference_sets: bool = Field(
        default=True,
        description="Use 6 reference sets for stability (recommended)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "creative_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                "personas": [
                    "You are Sarah, a 35-year-old marketing manager earning $85k/year in Seattle. You value quality and sustainability, shop online weekly, and have moderate price sensitivity.",
                    "You are Mike, a 28-year-old software engineer earning $110k/year in San Francisco. You prioritize convenience, are tech-savvy, and have low price sensitivity.",
                ],
            }
        }
    }


class PersonaAnalysisResult(BaseModel):
    """Individual persona evaluation result."""

    persona_id: int = Field(..., description="1-indexed persona identifier")
    persona_description: str = Field(..., description="Full persona description")

    # Qualitative feedback
    qualitative_feedback: str = Field(
        ..., description="LLM's reasoning: 'I would/wouldn't buy because...'"
    )

    # Quantitative results
    quantitative_score: int = Field(
        ..., ge=1, le=5, description="Most likely rating on 1-5 scale"
    )
    expected_value: float = Field(
        ..., description="Expected value (weighted average of PMF)"
    )
    pmf: List[float] = Field(
        ..., description="Probability distribution over 1-5 scale"
    )
    rating_certainty: float = Field(
        ..., ge=0.0, le=1.0, description="Certainty of this rating (max probability in PMF) - measures decisiveness"
    )


class AggregateResults(BaseModel):
    """Aggregated results across all personas."""

    average_score: float = Field(
        ..., description="Average expected value across personas"
    )
    predicted_purchase_intent: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted purchase intent (% likely to purchase)",
    )
    pmf_aggregate: List[float] = Field(
        ..., description="Averaged PMF across all personas"
    )
    persona_agreement: float = Field(
        ..., ge=0.0, le=1.0, description="Agreement/consensus between personas - measures how much personas agree on rating"
    )


class AnalyzeCreativeResponse(BaseModel):
    """Complete analysis response."""

    persona_results: List[PersonaAnalysisResult] = Field(
        ..., description="Individual results for each persona"
    )

    aggregate: AggregateResults = Field(
        ..., description="Aggregated metrics across all personas"
    )

    metadata: dict = Field(
        ...,
        description="Processing metadata",
        json_schema_extra={
            "example": {
                "num_personas": 10,
                "cost_usd": 0.15,
                "processing_time_seconds": 23.5,
                "llm_model": "gemini-2.5-flash",
            }
        },
    )


# Endpoint

@router.post("/analyze-creative", response_model=AnalyzeCreativeResponse)
async def analyze_creative(
    request: AnalyzeCreativeRequest,
    predictor: Annotated[AdPredictor, Depends(get_ad_predictor)],
):
    """
    Analyze ad creative with specific personas.

    Returns both qualitative feedback and quantitative scores per persona,
    plus aggregated metrics.

    Args:
        request: AnalyzeCreativeRequest with creative and personas

    Returns:
        AnalyzeCreativeResponse with per-persona and aggregate results

    Example:
        ```python
        import requests
        import base64

        with open("ad.jpg", "rb") as f:
            creative_b64 = base64.b64encode(f.read()).decode()

        response = requests.post(
            "http://localhost:8000/v1/analyze-creative",
            json={
                "creative_base64": creative_b64,
                "personas": [
                    "You are Sarah, a 35-year-old marketing manager...",
                    "You are Mike, a 28-year-old engineer..."
                ]
            }
        )

        result = response.json()
        print(f"Average score: {result['aggregate']['average_score']}/5")
        print(f"Purchase intent: {result['aggregate']['predicted_purchase_intent']:.1%}")
        ```

    Cost: ~$0.0015 per persona (e.g., 10 personas = ~$0.015)
    Time: ~10-30 seconds depending on number of personas
    """
    start_time = time.time()

    try:
        # Basic validation
        if not request.personas:
            raise ValueError("At least one persona is required")

        if len(request.creative_base64) < 100:
            raise ValueError("Invalid creative image (base64 too short)")

        # Run analysis
        logger.info(f"Analyzing creative with {len(request.personas)} personas")

        result = await predictor.analyze_creative(
            creative_base64=request.creative_base64,
            personas=request.personas,
            reference_sentences=request.reference_sentences,
            temperature=request.temperature,
            epsilon=request.epsilon,
            use_multiple_reference_sets=request.use_multiple_reference_sets,
        )

        # Add processing time
        processing_time = time.time() - start_time
        result["metadata"]["processing_time_seconds"] = round(processing_time, 2)

        logger.info(
            f"Analysis complete: avg_score={result['aggregate']['average_score']:.2f}, "
            f"purchase_intent={result['aggregate']['predicted_purchase_intent']:.1%}, "
            f"time={processing_time:.1f}s"
        )

        return AnalyzeCreativeResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing creative: {str(e)}")


# Persona Generation Endpoint

class GeneratePersonasRequest(BaseModel):
    """Request to generate synthetic personas based on characteristics."""

    num_personas: int = Field(
        ...,
        ge=1,
        le=20,
        description="Number of personas to generate (1-20)",
    )

    characteristics: Optional[str] = Field(
        None,
        description="Optional characteristics/constraints (e.g., 'tech-savvy millennials' or '40+ year old professionals')",
        max_length=500,
    )

    age_range: Optional[str] = Field(
        None,
        description="Optional age range (e.g., '25-35', '40-50')",
        example="25-40",
    )

    income_range: Optional[str] = Field(
        None,
        description="Optional income range (e.g., '$50k-75k', '$100k+')",
        example="$75k-100k",
    )

    location: Optional[str] = Field(
        None,
        description="Optional location/geography (e.g., 'urban tech hubs', 'midwest suburbs')",
        example="San Francisco, Seattle, Austin",
    )

    diversity: bool = Field(
        default=True,
        description="Whether to ensure diverse personas (age, gender, income, etc.)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "num_personas": 5,
                "characteristics": "tech-savvy, early adopters, value convenience",
                "age_range": "25-40",
                "income_range": "$75k-120k",
                "location": "urban tech hubs",
                "diversity": True,
            }
        }
    }


class GeneratePersonasResponse(BaseModel):
    """Response with generated personas."""

    personas: List[str] = Field(
        ..., description="List of generated persona descriptions ready for analysis"
    )

    metadata: dict = Field(
        ...,
        description="Generation metadata",
        json_schema_extra={
            "example": {
                "num_generated": 5,
                "characteristics_used": "tech-savvy, early adopters",
                "generation_time_seconds": 3.2,
            }
        },
    )


@router.post("/personas/generate", response_model=GeneratePersonasResponse)
async def generate_personas(
    request: GeneratePersonasRequest,
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
):
    """
    Generate synthetic personas based on specified characteristics.

    This is a helper endpoint to quickly create personas for testing or
    when you want AI-generated personas instead of writing them manually.

    Args:
        request: GeneratePersonasRequest with number and characteristics

    Returns:
        GeneratePersonasResponse with generated persona descriptions

    Example:
        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/v1/personas/generate",
            json={
                "num_personas": 5,
                "characteristics": "tech-savvy millennials",
                "age_range": "25-35",
                "income_range": "$75k-100k"
            }
        )

        personas = response.json()["personas"]
        # Use these personas in analyze-creative
        ```

    Cost: ~$0.001 per generation
    Time: ~2-5 seconds
    """
    start_time = time.time()

    try:
        # Build prompt for Gemini
        prompt = f"""Generate {request.num_personas} diverse customer personas for market research.

REQUIREMENTS:
- Each persona should be 2-3 sentences
- Format: "You are [Name], a [age]-year-old [occupation]..."
- Include: age, occupation, income level, location, shopping behavior, values
- Make them feel realistic and varied
"""

        if request.characteristics:
            prompt += f"\nCHARACTERISTICS: {request.characteristics}"

        if request.age_range:
            prompt += f"\nAGE RANGE: {request.age_range}"

        if request.income_range:
            prompt += f"\nINCOME RANGE: {request.income_range}"

        if request.location:
            prompt += f"\nLOCATION: {request.location}"

        if request.diversity:
            prompt += "\n\nIMPORTANT: Ensure diversity in gender, age (within range), income (within range), occupations, and perspectives."

        prompt += f"""

Generate exactly {request.num_personas} personas, one per line, numbered 1-{request.num_personas}.
Each should start with "You are..." and be self-contained.

Example format:
1. You are Sarah, a 32-year-old marketing manager earning $85k/year in Seattle. You value sustainability, shop online weekly, and have moderate price sensitivity.
2. You are Mike, a 28-year-old software engineer earning $110k/year in San Francisco. You prioritize convenience, are tech-savvy, and have low price sensitivity.
"""

        logger.info(
            f"Generating {request.num_personas} personas with characteristics: {request.characteristics}"
        )

        # Generate with Gemini
        response_text = await llm_client.generate_text(prompt)

        # Parse response into individual personas
        lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]

        personas = []
        for line in lines:
            # Remove numbering (e.g., "1. " or "1) ")
            cleaned = line
            if len(line) > 3 and line[0].isdigit() and line[1] in [".", ")", ":"]:
                cleaned = line[2:].strip()
            elif len(line) > 4 and line[:2].isdigit() and line[2] in [".", ")", ":"]:
                cleaned = line[3:].strip()

            # Only include if it starts with "You are" and is substantial
            if cleaned.lower().startswith("you are") and len(cleaned) > 50:
                personas.append(cleaned)

        # Ensure we have the requested number
        if len(personas) < request.num_personas:
            logger.warning(
                f"Generated {len(personas)} personas but {request.num_personas} were requested"
            )
            # Could retry or raise error, for now just return what we got
            if len(personas) == 0:
                raise ValueError("Failed to generate any valid personas. Please try again.")

        # Trim to exact number if we got more
        personas = personas[: request.num_personas]

        processing_time = time.time() - start_time

        logger.info(f"Generated {len(personas)} personas in {processing_time:.2f}s")

        return GeneratePersonasResponse(
            personas=personas,
            metadata={
                "num_generated": len(personas),
                "characteristics_used": request.characteristics or "general",
                "age_range": request.age_range,
                "income_range": request.income_range,
                "location": request.location,
                "generation_time_seconds": round(processing_time, 2),
            },
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Persona generation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating personas: {str(e)}"
        )
