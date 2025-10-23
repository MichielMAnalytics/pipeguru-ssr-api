"""
Test script for /v1/analyze-creative endpoint.

This script loads a single ad image (1/11.png) and analyzes it with 3 personas,
then prints the raw API response.
"""

import asyncio
import base64
import json
from pathlib import Path

import httpx


# Configuration
API_BASE_URL = "http://localhost:8000/v1"
DATASET_DIR = Path(__file__).parent.parent / "data" / "ad-visual-optimization"
IMAGE_PATH = DATASET_DIR / "Images" / "Ad images" / "1" / "11.png"

# Test personas
PERSONAS = [
    "You are Sarah, a 35-year-old marketing manager earning $85k/year in Seattle. You value quality and sustainability, shop online weekly, and have moderate price sensitivity.",
    "You are Mike, a 28-year-old software engineer earning $110k/year in San Francisco. You prioritize convenience, are tech-savvy, and have low price sensitivity.",
]


def load_image_as_base64(image_path: Path) -> str:
    """Load image file and convert to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


async def analyze_creative(
    creative_base64: str,
    personas: list[str],
) -> dict:
    """
    Call the /v1/analyze-creative endpoint.

    Args:
        creative_base64: Base64-encoded ad image
        personas: List of persona descriptions

    Returns:
        API response dict
    """
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/analyze-creative",
            json={
                "creative_base64": creative_base64,
                "personas": personas,
            },
        )
        response.raise_for_status()
        return response.json()


async def main():
    """Main test workflow."""
    print("=" * 80)
    print("Testing /v1/analyze-creative endpoint")
    print("=" * 80)
    print()

    # Check if API is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL.replace('/v1', '')}/v1/health")
            response.raise_for_status()
            print("✓ API is running")
    except Exception as e:
        print(f"✗ Error: API is not running at {API_BASE_URL}")
        print(f"  Please start the API with: uvicorn src.main:app --reload")
        return

    print()

    # Load image
    if not IMAGE_PATH.exists():
        print(f"✗ Error: Image not found at {IMAGE_PATH}")
        return

    print(f"Loading image: {IMAGE_PATH}")
    creative_base64 = load_image_as_base64(IMAGE_PATH)
    print(f"✓ Image loaded ({len(creative_base64)} bytes)")
    print()

    # Show personas
    print(f"Testing with {len(PERSONAS)} personas:")
    for i, persona in enumerate(PERSONAS, 1):
        # Truncate for display
        persona_preview = persona[:80] + "..." if len(persona) > 80 else persona
        print(f"  {i}. {persona_preview}")
    print()

    # Call API
    print("Calling /v1/analyze-creative endpoint...")
    print("(This will take 10-30 seconds)")
    print()

    try:
        result = await analyze_creative(creative_base64, PERSONAS)

        # Print raw JSON response
        print("=" * 80)
        print("RAW API RESPONSE")
        print("=" * 80)
        print()
        print(json.dumps(result, indent=2))
        print()
        print("=" * 80)

        # Print summary
        print()
        print("SUMMARY")
        print("=" * 80)
        print()

        # Individual results
        print("Individual Persona Results:")
        print()
        for persona_result in result["persona_results"]:
            print(f"Persona {persona_result['persona_id']}:")
            print(f"  Score: {persona_result['quantitative_score']}/5")
            print(f"  Expected Value: {persona_result['expected_value']:.2f}")
            print(f"  Rating Certainty: {persona_result['rating_certainty']:.1%}")
            print(f"  Feedback: {persona_result['qualitative_feedback'][:100]}...")
            print()

        # Aggregate results
        print("Aggregate Results:")
        print(f"  Average Score: {result['aggregate']['average_score']:.2f}/5")
        print(f"  Predicted Purchase Intent: {result['aggregate']['predicted_purchase_intent']:.1%}")
        print(f"  Persona Agreement: {result['aggregate']['persona_agreement']:.1%}")
        print()

        # Metadata
        print("Metadata:")
        print(f"  Num Personas: {result['metadata']['num_personas']}")
        print(f"  Cost: ${result['metadata']['cost_usd']:.2f}")
        print(f"  LLM Calls: {result['metadata']['llm_calls']}")
        print(f"  LLM Model: {result['metadata']['llm_model']}")
        print()

    except Exception as e:
        print(f"✗ Error calling API: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
