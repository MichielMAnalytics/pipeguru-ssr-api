"""Creative analysis endpoint - simplified unified API."""

import logging
import time
from typing import Annotated, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.core.ad_predictor import AdPredictor
from src.core.auth import validate_api_key
from src.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(validate_api_key)])


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
        description="Base64-encoded image or video of the ad creative",
    )

    personas: List[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of persona descriptions to evaluate against (max 1000 for scalability testing)",
    )

    mime_type: Optional[str] = Field(
        None,
        description="Optional MIME type of the creative (e.g., 'video/mp4', 'image/jpeg'). Auto-detected if not provided.",
    )

    # Brand familiarity parameters (optional)
    brand_context: Optional[str] = Field(
        None,
        description="Optional brand context/background information. Provides personas with brand knowledge based on familiarity level.",
        max_length=5000,
    )

    brand_familiarity_distribution: Optional[Union[str, dict]] = Field(
        None,
        description="Optional distribution of brand familiarity levels across personas. Either a preset name (string) or custom distribution (dict mapping level to percentage). Requires brand_context.",
        examples=[
            "uniform",
            "new_brand",
            "established_brand",
            {"1": 0.3, "2": 0.5, "3": 0.2}
        ],
    )

    brand_familiarity_seed: Optional[int] = Field(
        None,
        description="Optional random seed for reproducible familiarity assignment. Only used with brand_familiarity_distribution.",
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

    def model_post_init(self, __context):
        """Validate brand familiarity parameters."""
        if self.brand_familiarity_distribution is not None and not self.brand_context:
            raise ValueError("brand_familiarity_distribution requires brand_context to be provided")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "creative_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                    "personas": [
                        "You are Sarah, a 35-year-old marketing manager earning $85k/year in Seattle. You value quality and sustainability, shop online weekly, and have moderate price sensitivity.",
                        "You are Mike, a 28-year-old software engineer earning $110k/year in San Francisco. You prioritize convenience, are tech-savvy, and have low price sensitivity.",
                    ],
                    "mime_type": None,  # Auto-detect
                },
                {
                    "creative_base64": "AAAAIGZ0eXBpc29tAAACAGlzb21pc28y...",
                    "personas": [
                        "You are Sarah, a 35-year-old marketing manager earning $85k/year in Seattle.",
                    ],
                    "mime_type": "video/mp4",
                },
                {
                    "creative_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                    "personas": [
                        "You are Sarah, a 35-year-old marketing manager...",
                        "You are Mike, a 28-year-old software engineer...",
                    ],
                    "brand_context": "Upfront is a Dutch sports nutrition company founded in 2020...",
                    "brand_familiarity_distribution": "emerging_brand",  # Preset
                },
                {
                    "creative_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                    "personas": ["You are Sarah...", "You are Mike..."],
                    "brand_context": "Upfront is a Dutch sports nutrition company...",
                    "brand_familiarity_distribution": {1: 0.3, 2: 0.5, 3: 0.2},  # Custom
                    "brand_familiarity_seed": 42,  # For reproducibility
                },
            ]
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

    # Brand familiarity (optional)
    brand_familiarity_level: Optional[int] = Field(
        None, ge=1, le=5, description="Brand familiarity level assigned to this persona (1=Never heard, 5=Brand advocate)"
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
    qualitative_summary: str = Field(
        ..., description="AI-generated summary of common themes, strengths, and concerns across all persona feedback"
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
                "llm_calls": 10,
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
    Analyze ad creative (image or video) with specific personas.

    Returns both qualitative feedback and quantitative scores per persona,
    plus aggregated metrics.

    Supports:
    - Images: JPEG, PNG, GIF, WebP
    - Videos: MP4, WebM, MOV, AVI, MPEG, FLV, WMV, 3GPP

    Args:
        request: AnalyzeCreativeRequest with creative and personas

    Returns:
        AnalyzeCreativeResponse with per-persona and aggregate results

    Brand Familiarity (Optional):
        Test how ad performance varies based on consumers' prior brand knowledge.

        Parameters:
        - brand_context: Comprehensive brand information (max 5000 chars)
        - brand_familiarity_distribution: Preset (e.g., "emerging_brand") or custom dict
        - brand_familiarity_seed: None (deterministic) or int (random with seed)

        Familiarity Levels:
        1. Never heard - Zero knowledge, first-time exposure
        2. Vaguely aware - Seen once or twice, superficial recognition
        3. Familiar - Knows what they do, general positioning
        4. Very familiar - Engaged with brand, purchased before
        5. Brand advocate - Deeply loyal, extensive knowledge

        Preset Distributions:
        - uniform: 20% / 20% / 20% / 20% / 20%
        - new_brand: 70% / 20% / 8% / 2% / 0%
        - emerging_brand: 40% / 30% / 20% / 8% / 2%
        - established_brand: 10% / 20% / 40% / 20% / 10%
        - popular_brand: 5% / 15% / 30% / 35% / 15%
        - cult_brand: 50% / 20% / 10% / 10% / 10%

    Examples:
        ```python
        import requests
        import base64

        # Example 1: Basic analysis (no brand familiarity)
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

        # Example 2: With brand familiarity (emerging brand)
        brand_context = \"\"\"Upfront is a Dutch sports nutrition company founded in 2020.
        Mission: Establish new standard for sports nutrition with radical transparency.
        Key differentiators: All ingredients on front, no artificial additives.
        Products: Protein powders, bars, energy gels. Distribution: Online + retail.\"\"\"

        response = requests.post(
            "http://localhost:8000/v1/analyze-creative",
            json={
                "creative_base64": creative_b64,
                "personas": [
                    "You are Sarah, a 28-year-old fitness enthusiast...",
                    "You are Mike, a 35-year-old marathon runner...",
                    "You are Elena, a 31-year-old yoga instructor...",
                    "You are Jordan, a 26-year-old CrossFit athlete...",
                    "You are Alex, a 33-year-old triathlete..."
                ],
                "brand_context": brand_context,
                "brand_familiarity_distribution": "emerging_brand",
                # No seed = deterministic assignment
            }
        )

        result = response.json()
        # Check brand familiarity distribution
        for persona in result["persona_results"]:
            print(f"Persona {persona['persona_id']}: Level {persona['brand_familiarity_level']}")

        # Example 3: Custom distribution with random seed
        response = requests.post(
            "http://localhost:8000/v1/analyze-creative",
            json={
                "creative_base64": creative_b64,
                "personas": ["Persona 1", "Persona 2", "Persona 3"],
                "brand_context": brand_context,
                "brand_familiarity_distribution": {1: 0.5, 2: 0.3, 3: 0.2},
                "brand_familiarity_seed": 42  # Reproducible random assignment
            }
        )

        # Example 4: Video analysis
        with open("ad.mp4", "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        response = requests.post(
            "http://localhost:8000/v1/analyze-creative",
            json={
                "creative_base64": video_b64,
                "personas": ["You are Sarah..."],
                "mime_type": "video/mp4"  # Optional - auto-detects if not provided
            }
        )
        ```

    Note: For videos, consider keeping file size under 10MB for optimal performance.
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
            mime_type=request.mime_type,
            brand_context=request.brand_context,
            brand_familiarity_distribution=request.brand_familiarity_distribution,
            brand_familiarity_seed=request.brand_familiarity_seed,
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
        le=1000,
        description="Number of personas to generate (1-1000)",
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
    """
    start_time = time.time()

    try:
        logger.info(
            f"Generating {request.num_personas} personas with characteristics: {request.characteristics}"
        )

        # Generate personas in batches (max 50 per LLM call for reliability)
        batch_size = 50
        all_personas = []

        num_batches = (request.num_personas + batch_size - 1) // batch_size  # Ceiling division

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_count = min(batch_size, request.num_personas - batch_start)

            # Build prompt for this batch
            prompt = f"""Generate {batch_count} diverse customer personas for market research.

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

Generate exactly {batch_count} personas, one per line, numbered 1-{batch_count}.
Each should start with "You are..." and be self-contained.

Example format:
1. You are Sarah, a 32-year-old marketing manager earning $85k/year in Seattle. You value sustainability, shop online weekly, and have moderate price sensitivity.
2. You are Mike, a 28-year-old software engineer earning $110k/year in San Francisco. You prioritize convenience, are tech-savvy, and have low price sensitivity.
"""

            logger.info(f"Generating batch {batch_idx + 1}/{num_batches} ({batch_count} personas)")

            # Generate with Gemini
            response_text = await llm_client.generate_text(prompt)

            # Parse response into individual personas
            lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]

            batch_personas = []
            for line in lines:
                # Remove numbering (e.g., "1. " or "1) ")
                cleaned = line
                if len(line) > 3 and line[0].isdigit() and line[1] in [".", ")", ":"]:
                    cleaned = line[2:].strip()
                elif len(line) > 4 and line[:2].isdigit() and line[2] in [".", ")", ":"]:
                    cleaned = line[3:].strip()
                elif len(line) > 5 and line[:3].isdigit() and line[3] in [".", ")", ":"]:
                    cleaned = line[4:].strip()

                # Only include if it starts with "You are" and is substantial
                if cleaned.lower().startswith("you are") and len(cleaned) > 50:
                    batch_personas.append(cleaned)

            # Ensure we got enough personas for this batch
            if len(batch_personas) < batch_count:
                logger.warning(
                    f"Batch {batch_idx + 1}: Generated {len(batch_personas)} personas but {batch_count} were requested"
                )

            all_personas.extend(batch_personas[:batch_count])

        personas = all_personas

        # Final validation
        if len(personas) == 0:
            raise ValueError("Failed to generate any valid personas. Please try again.")

        logger.info(f"Successfully generated {len(personas)} personas across {num_batches} batches")

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
