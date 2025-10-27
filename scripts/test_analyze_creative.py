"""
Test script for /v1/analyze-creative endpoint.

This script loads a single ad image and analyzes it with a configurable number of personas,
then prints performance metrics and results.
"""

import asyncio
import base64
import json
import time
from pathlib import Path

import httpx


# Configuration
API_BASE_URL = "http://localhost:8000/v1"
DATASET_DIR = Path(__file__).parent.parent / "data" / "ad-visual-optimization"
IMAGE_PATH = DATASET_DIR / "Images" / "Ad images" / "1" / "11.png"

# Number of personas to test (adjust this to test scalability)
NUM_PERSONAS = 100  # Change this to test with more personas

# Base persona templates (will be expanded)
BASE_PERSONAS = [
    "You are {name}, a {age}-year-old marketing manager earning ${income}k/year in Seattle. You value quality and sustainability, shop online weekly, and have moderate price sensitivity.",
    "You are {name}, a {age}-year-old software engineer earning ${income}k/year in San Francisco. You prioritize convenience, are tech-savvy, and have low price sensitivity.",
    "You are {name}, a {age}-year-old small business owner earning ${income}k/year in Austin. You're budget-conscious, shop monthly, and carefully research before purchasing.",
    "You are {name}, a {age}-year-old teacher earning ${income}k/year in Portland. You value education and community, prefer local businesses, and have moderate price sensitivity.",
    "You are {name}, a {age}-year-old nurse earning ${income}k/year in Boston. You prioritize practical value and durability, shop bi-weekly, and are moderately price-sensitive.",
]

NAMES = ["Sarah", "Mike", "Elena", "Jordan", "Alex", "Taylor", "Morgan", "Casey", "Riley", "Quinn"]


def generate_personas(num_personas: int) -> list[str]:
    """Generate diverse personas by cycling through templates."""
    personas = []
    for i in range(num_personas):
        template = BASE_PERSONAS[i % len(BASE_PERSONAS)]
        name = NAMES[i % len(NAMES)]
        age = 25 + (i % 40)  # Ages 25-64
        income = 50 + (i % 100)  # Income $50k-$150k
        persona = template.format(name=name, age=age, income=income)
        personas.append(persona)
    return personas


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
    print("Testing /v1/analyze-creative endpoint - CONCURRENCY TEST")
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

    # Generate personas
    print(f"Generating {NUM_PERSONAS} test personas...")
    personas = generate_personas(NUM_PERSONAS)
    print(f"✓ Generated {len(personas)} personas")
    print()

    # Show sample personas
    print("Sample personas (first 3):")
    for i in range(min(3, len(personas))):
        persona_preview = personas[i][:100] + "..." if len(personas[i]) > 100 else personas[i]
        print(f"  {i+1}. {persona_preview}")
    print()

    # Call API with timing
    print(f"Calling /v1/analyze-creative endpoint with {NUM_PERSONAS} personas...")
    print("(This may take 1-3 minutes depending on concurrency settings)")
    print()

    start_time = time.time()

    try:
        result = await analyze_creative(creative_base64, personas)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Performance metrics
        print("=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)
        print()
        print(f"✓ Successfully analyzed {NUM_PERSONAS} personas")
        print(f"✓ Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"✓ Time per persona: {elapsed_time/NUM_PERSONAS:.3f} seconds")
        print(f"✓ Throughput: {NUM_PERSONAS/elapsed_time:.2f} personas/second")
        print()

        # Aggregate results
        print("=" * 80)
        print("AGGREGATE RESULTS")
        print("=" * 80)
        print()
        print(f"  Average Score: {result['aggregate']['average_score']:.2f}/5")
        print(f"  Predicted Purchase Intent: {result['aggregate']['predicted_purchase_intent']:.1%}")
        print(f"  Persona Agreement: {result['aggregate']['persona_agreement']:.1%}")
        print()

        # Sample individual results (first 3)
        print("=" * 80)
        print("SAMPLE PERSONA RESULTS (First 3)")
        print("=" * 80)
        print()
        for persona_result in result["persona_results"][:3]:
            print(f"Persona {persona_result['persona_id']}:")
            print(f"  Score: {persona_result['quantitative_score']}/5")
            print(f"  Expected Value: {persona_result['expected_value']:.2f}")
            print(f"  Rating Certainty: {persona_result['rating_certainty']:.1%}")
            print(f"  Feedback: {persona_result['qualitative_feedback'][:150]}...")
            print()

        # Metadata
        print("=" * 80)
        print("API METADATA")
        print("=" * 80)
        print()
        print(f"  Num Personas: {result['metadata']['num_personas']}")
        print(f"  LLM Calls: {result['metadata']['llm_calls']}")
        print(f"  LLM Model: {result['metadata']['llm_model']}")
        print()

        # Save full results to JSON file
        output_file = Path(__file__).parent.parent / f"test_results_{NUM_PERSONAS}_personas.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✓ Full results saved to: {output_file}")
        print()

    except Exception as e:
        print(f"✗ Error calling API: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
