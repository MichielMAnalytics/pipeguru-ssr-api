"""Test script for POST /v1/predict-ad endpoint."""

import asyncio
import base64
from pathlib import Path

from dotenv import load_dotenv
from src.utils.image_utils import prepare_image_for_api

# Load environment
load_dotenv()


async def test_predict_endpoint():
    """Test the ad prediction endpoint with our sample ad."""
    print("=" * 70)
    print("TEST: POST /v1/predict-ad Endpoint")
    print("=" * 70)
    print()

    # Step 1: Prepare the ad image
    print("Step 1: Loading ad image...")
    image_path = Path("test_data/sample_ad.jpg")

    if not image_path.exists():
        print(f"✗ Image not found: {image_path}")
        print("Please run the integration tests first to download the sample ad.")
        return

    image_b64 = prepare_image_for_api(str(image_path))
    print(f"✓ Image loaded: {len(image_b64)} bytes (base64)")
    print()

    # Step 2: Import and call the predictor directly
    print("Step 2: Testing predictor directly (not via HTTP)...")
    print("Configuration:")
    print("  • Personas: 3 (keeping it cheap for testing)")
    print("  • Segment: millennial_women")
    print("  • Cost estimate: ~$0.03-0.06")
    print()

    from src.core.ad_predictor import AdPredictor

    predictor = AdPredictor()

    reference_sentences = [
        "I definitely would not purchase",
        "I probably would not purchase",
        "I might or might not purchase",
        "I probably would purchase",
        "I definitely would purchase",
    ]

    print("Step 3: Running prediction...")
    print("(This will take 10-20 seconds...)")
    print()

    result = await predictor.predict_ad_performance(
        ad_image_base64=image_b64,
        num_personas=3,
        segment="millennial_women",
        reference_sentences=reference_sentences,
        temperature=1.0,
        epsilon=0.01,
    )

    # Step 4: Display results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Predicted Conversion Rate: {result['predicted_conversion_rate']:.1%}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Expected Value: {result['expected_value']:.2f}/5")
    print()

    print("PMF Distribution:")
    for i, prob in enumerate(result['pmf_aggregate'], 1):
        bar = "█" * int(prob * 50)  # Visual bar
        print(f"  {i}: {prob:.1%} {bar}")
    print()

    print("Individual Persona Results:")
    for persona_result in result['persona_results']:
        print(f"\nPersona {persona_result['persona_id']}:")
        print(f"  Expected value: {persona_result['expected_value']:.2f}/5")
        print(f"  Response: {persona_result['llm_response'][:150]}...")
    print()

    print("Cost:")
    print(f"  LLM calls: {result['cost']['llm_calls']}")
    print(f"  Estimated cost: ${result['cost']['estimated_cost_usd']:.2f}")
    print()

    print("✅ Prediction endpoint working!")
    print()

    # Step 5: Validation checks
    print("=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)
    checks = {
        "conversion_rate_valid": 0.0 <= result['predicted_conversion_rate'] <= 1.0,
        "confidence_valid": 0.0 <= result['confidence'] <= 1.0,
        "pmf_sums_to_one": abs(sum(result['pmf_aggregate']) - 1.0) < 0.01,
        "expected_value_valid": 1.0 <= result['expected_value'] <= 5.0,
        "has_persona_results": len(result['persona_results']) == 3,
    }

    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}: {passed}")

    print()

    if all(checks.values()):
        print("✅ All validation checks passed!")
    else:
        print("❌ Some checks failed")


if __name__ == "__main__":
    asyncio.run(test_predict_endpoint())
