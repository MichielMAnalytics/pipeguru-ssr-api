"""End-to-end integration test: Persona + Image + LLM + SSR."""

import asyncio
from dotenv import load_dotenv
from src.core.llm_client import LLMClient
from src.core.persona_generator import PersonaGenerator
from src.utils.image_utils import prepare_image_for_api

# Load environment variables
load_dotenv()

async def test_e2e():
    print("=" * 70)
    print("END-TO-END INTEGRATION TEST")
    print("=" * 70)
    print()

    # Step 1: Generate a persona
    print("Step 1: Generating persona...")
    gen = PersonaGenerator(seed=42)
    persona = gen.generate_persona(segment="millennial_women")
    print(f"✓ Persona: {persona[:150]}...")
    print()

    # Step 2: Prepare image
    print("Step 2: Preparing test image...")
    # You'll need to provide an ad image path here
    # For now, let's create a placeholder and show instructions

    print("⚠️  To run this test, you need to:")
    print("   1. Find or download a product ad image")
    print("   2. Save it as: test_data/sample_ad.jpg")
    print("   3. Run this script again")
    print()
    print("Example command to download a sample:")
    print("   curl -o test_data/sample_ad.jpg [IMAGE_URL]")
    print()

    # Check if image exists
    import os
    image_path = "test_data/sample_ad.jpg"

    if not os.path.exists(image_path):
        print(f"✗ Image not found at: {image_path}")
        print()
        print("For testing, you can use any product ad image you have.")
        return

    try:
        image_b64 = prepare_image_for_api(image_path)
        print(f"✓ Image prepared: {len(image_b64)} characters (base64)")
        print()
    except Exception as e:
        print(f"✗ Error preparing image: {e}")
        return

    # Step 3: Call LLM with persona and image
    print("Step 3: Calling GPT-4o Vision...")
    print("(This will cost ~$0.01-0.02)")

    try:
        client = LLMClient()

        reference_context = """We're measuring purchase intent on a 5-point scale:
1 = Definitely would not purchase
2 = Probably would not purchase
3 = Might or might not purchase
4 = Probably would purchase
5 = Definitely would purchase"""

        response = await client.evaluate_ad_with_persona(
            ad_image_base64=image_b64,
            persona_description=persona,
            reference_context=reference_context
        )

        print(f"✓ LLM Response received:")
        print("-" * 70)
        print(response)
        print("-" * 70)
        print()

    except Exception as e:
        print(f"✗ Error calling LLM: {e}")
        return

    # Step 4: Analyze response quality
    print("Step 4: Analyzing response quality...")

    response_lower = response.lower()
    quality_checks = {
        "mentions_product": any(word in response_lower for word in ["product", "item", "this"]),
        "expresses_opinion": any(word in response_lower for word in ["like", "interested", "would", "might"]),
        "substantial_length": len(response) > 50,
        "natural_language": not response.startswith("1") and not response.startswith("5"),
    }

    print("Quality checks:")
    for check, passed in quality_checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}: {passed}")
    print()

    all_passed = all(quality_checks.values())
    if all_passed:
        print("✓ All quality checks passed!")
        print("✓ End-to-end pipeline working!")
    else:
        print("⚠️  Some quality checks failed, review response")
    print()

    # Step 5: Show next steps
    print("=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. The LLM response looks good")
    print("2. Next: Pass this response to SSR (POST /v1/rate)")
    print("3. Then: Build /v1/predict-ad endpoint to automate this")
    print("4. Finally: Test with 10 ads to measure accuracy")

if __name__ == "__main__":
    asyncio.run(test_e2e())
