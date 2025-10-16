"""Ad prediction endpoint."""

import logging
import time
import traceback
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.core.ad_predictor import AdPredictor
from src.models.schemas import PredictAdRequest, PredictAdResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def get_ad_predictor() -> AdPredictor:
    """
    Dependency that provides an AdPredictor instance.

    FastAPI will call this function once per request and cache the result.
    """
    return AdPredictor()


@router.post("/predict-ad", response_model=PredictAdResponse)
async def predict_ad(
    request: PredictAdRequest,
    predictor: Annotated[AdPredictor, Depends(get_ad_predictor)]
):
    """
    Predict ad performance using synthetic personas and SSR methodology.

    This endpoint:
    1. Generates N synthetic personas from the specified segment
    2. Evaluates the ad with each persona using GPT-4 Vision
    3. Converts LLM responses to probability distributions using SSR
    4. Aggregates results into a final prediction

    Args:
        request: PredictAdRequest with ad image, personas config, and SSR parameters

    Returns:
        PredictAdResponse with prediction, confidence, PMF, and individual results

    Example:
        ```python
        import requests
        import base64

        # Load and encode image
        with open("ad.jpg", "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        response = requests.post(
            "http://localhost:8000/v1/predict-ad",
            json={
                "ad_image_base64": image_b64,
                "num_personas": 10,
                "segment": "millennial_women"
            }
        )

        result = response.json()
        print(f"Predicted conversion: {result['predicted_conversion_rate']:.1%}")
        print(f"Confidence: {result['confidence']:.1%}")
        ```

    Cost: ~$0.01-0.02 per persona (e.g., 20 personas = ~$0.20-0.40)
    Time: ~10-30 seconds depending on number of personas
    """
    start_time = time.time()

    try:
        # Validate segment
        available_segments = predictor.persona_generator.get_available_segments()
        if request.segment not in available_segments:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid segment: {request.segment}. "
                f"Available: {available_segments}",
            )

        # Run prediction
        result = await predictor.predict_ad_performance(
            ad_image_base64=request.ad_image_base64,
            num_personas=request.num_personas,
            segment=request.segment,
            reference_sentences=request.reference_sentences,
            temperature=request.temperature,
            epsilon=request.epsilon,
            use_multiple_reference_sets=request.use_multiple_reference_sets,
        )

        # Add processing time to metadata
        processing_time = time.time() - start_time
        result["metadata"]["processing_time_seconds"] = round(processing_time, 2)

        return PredictAdResponse(**result)

    except ValueError as e:
        logger.error(f"ValueError in predict_ad: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in predict_ad: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error predicting ad performance: {str(e)}"
        )
