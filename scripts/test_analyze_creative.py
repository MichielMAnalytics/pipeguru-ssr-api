"""
Test script for /v1/analyze-creative endpoint.

This script loads a single ad image and analyzes it with a configurable number of personas,
then prints performance metrics and results.
"""

import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = os.getenv("SSR_API_KEY")
DATASET_DIR = Path(__file__).parent.parent / "data" / "ad-visual-optimization"
IMAGE_PATH = DATASET_DIR / "Images" / "Ad images" / "1" / "11.png"

# Number of personas to test (adjust this to test scalability)
NUM_PERSONAS = 5  # Change this to test with more personas

# Brand familiarity testing (set to None to skip)
TEST_BRAND_FAMILIARITY = True

# Distribution type: 'uniform', 'emerging_brand', or 'custom'
DISTRIBUTION_TYPE = 'custom'  # Use 'custom' to test all 5 levels with 5 personas

# Upfront brand context for testing
UPFRONT_BRAND_CONTEXT = """Upfront is a Dutch sports nutrition company founded in January 2020 by three childhood friends: Mark de Boer, Harro Schwencke, and Nick Schijvens. The brand's mission is to establish a new standard for sports nutrition in the Netherlands, with transparency and honesty at its core.

Core Brand Identity:
- "Wat oprecht is wint" (What is genuine wins) - their primary brand slogan
- The company name "Upfront" reflects their practice of placing all ingredients and nutritional values on the front of packaging
- They aim to "build the next Unilever, but genuine"

Key Differentiators:
- Radical Transparency: All ingredients displayed prominently on the front of packaging
- Complete transparency extends beyond packaging - they publish their purchase prices on their website
- No artificial flavors, colors, or sweeteners
- Minimalist Design: Simple, functional packaging that stands out by being deliberately "boring"

Product Range:
- Sports nutrition: protein powders, bars, shakes, gels
- Endurance products: electrolytes, energy gels
- Basic nutrition: protein oats, peanut butter, granola

Distribution:
- Started with direct-to-consumer online sales
- Expanded to retail with products in 502 Albert Heijn stores in Netherlands and Belgium
- Also available at Etos drugstores
- Plans to open 500 Upfront supermarkets across the Netherlands

Target Audience:
- Health-conscious consumers, particularly younger demographics
- Athletes and fitness enthusiasts
- People seeking transparent, honest nutrition without marketing deception"""

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
    brand_context: str | None = None,
    brand_familiarity_distribution: dict | str | None = None,
    brand_familiarity_seed: int | None = None,
) -> dict:
    """
    Call the /v1/analyze-creative endpoint.

    Args:
        creative_base64: Base64-encoded ad image
        personas: List of persona descriptions
        brand_context: Optional brand context
        brand_familiarity_distribution: Optional distribution (preset or custom)
        brand_familiarity_seed: Optional seed for reproducibility

    Returns:
        API response dict
    """
    payload = {
        "creative_base64": creative_base64,
        "personas": personas,
    }

    # Add brand familiarity parameters if provided
    if brand_context:
        payload["brand_context"] = brand_context
    if brand_familiarity_distribution:
        payload["brand_familiarity_distribution"] = brand_familiarity_distribution
    if brand_familiarity_seed is not None:
        payload["brand_familiarity_seed"] = brand_familiarity_seed

    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{API_BASE_URL}/analyze-creative",
            json=payload,
            headers=headers,
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
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

    # Prepare brand familiarity parameters
    brand_params = {}
    if TEST_BRAND_FAMILIARITY:
        print("=" * 80)
        print("BRAND FAMILIARITY CONFIGURATION")
        print("=" * 80)
        print()
        print("✓ Brand familiarity testing ENABLED")
        print(f"  Brand: Upfront (Dutch sports nutrition)")

        if DISTRIBUTION_TYPE == 'custom':
            # Custom distribution to ensure all 5 levels with 5 personas (20% each)
            print(f"  Distribution: custom (all 5 levels represented)")
            print(f"    - 20% never heard (level 1)")
            print(f"    - 20% vaguely aware (level 2)")
            print(f"    - 20% familiar (level 3)")
            print(f"    - 20% very familiar (level 4)")
            print(f"    - 20% brand advocates (level 5)")
            distribution = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
        elif DISTRIBUTION_TYPE == 'uniform':
            print(f"  Distribution: 'uniform' preset")
            print(f"    - 20% never heard (level 1)")
            print(f"    - 20% vaguely aware (level 2)")
            print(f"    - 20% familiar (level 3)")
            print(f"    - 20% very familiar (level 4)")
            print(f"    - 20% brand advocates (level 5)")
            distribution = "uniform"
        else:  # emerging_brand
            print(f"  Distribution: 'emerging_brand' preset")
            print(f"    - 40% never heard (level 1)")
            print(f"    - 30% vaguely aware (level 2)")
            print(f"    - 20% familiar (level 3)")
            print(f"    - 8% very familiar (level 4)")
            print(f"    - 2% brand advocates (level 5)")
            distribution = "emerging_brand"

        print(f"  Mode: Deterministic (exact percentages, no seed)")
        print()
        brand_params = {
            "brand_context": UPFRONT_BRAND_CONTEXT,
            "brand_familiarity_distribution": distribution,
            # No seed = deterministic mode (exact percentages)
        }
    else:
        print("Brand familiarity testing: DISABLED")
        print()

    # Call API with timing
    print("=" * 80)
    print("API CALL")
    print("=" * 80)
    print()
    print(f"Calling /v1/analyze-creative endpoint with {NUM_PERSONAS} personas...")
    print("(This may take 1-3 minutes depending on concurrency settings)")
    print()

    start_time = time.time()

    try:
        result = await analyze_creative(
            creative_base64,
            personas,
            **brand_params,
        )

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

        # Sample individual results (show all for small tests, first 3 for large tests)
        print("=" * 80)
        num_to_show = min(3, NUM_PERSONAS)
        print(f"PERSONA RESULTS (Showing {num_to_show} of {NUM_PERSONAS})")
        print("=" * 80)
        print()
        for persona_result in result["persona_results"][:num_to_show]:
            print(f"Persona {persona_result['persona_id']}:")
            if "brand_familiarity_level" in persona_result:
                level = persona_result["brand_familiarity_level"]
                level_labels = {
                    1: "Never heard",
                    2: "Vaguely aware",
                    3: "Familiar",
                    4: "Very familiar",
                    5: "Brand advocate"
                }
                print(f"  Brand Familiarity: Level {level} - {level_labels[level]}")
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

        # Brand familiarity metadata
        if result['metadata'].get('brand_familiarity', {}).get('enabled'):
            print("  Brand Familiarity:")
            bf = result['metadata']['brand_familiarity']
            print(f"    Distribution: {bf['distribution_used']}")
            print(f"    Seed: {bf['seed']}")
            print(f"    Assigned Levels:")
            for level_key, level_data in sorted(bf['level_distribution'].items()):
                print(f"      {level_data['label']}: {level_data['count']} personas ({level_data['percentage']}%)")
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
