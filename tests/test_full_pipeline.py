"""Complete pipeline test: Persona → LLM → SSR → PMF."""

import asyncio
from dotenv import load_dotenv
from src.core.llm_client import LLMClient
from src.core.persona_generator import PersonaGenerator
from src.utils.image_utils import prepare_image_for_api
import polars as po
from semantic_similarity_rating import ResponseRater

load_dotenv()

async def test_full_pipeline():
    print("=" * 70)
    print("FULL PIPELINE TEST: Persona → LLM → SSR → PMF")
    print("=" * 70)
    print()

    # Step 1: Generate persona
    print("Step 1: Generate persona...")
    gen = PersonaGenerator(seed=42)
    persona = gen.generate_persona(segment="millennial_women")
    print(f"✓ Persona generated")
    print()

    # Step 2: Prepare image
    print("Step 2: Prepare image...")
    image_path = "test_data/sample_ad.jpg"
    image_b64 = prepare_image_for_api(image_path)
    print(f"✓ Image prepared")
    print()

    # Step 3: Get LLM response
    print("Step 3: Get LLM evaluation...")
    print("(Cost: ~$0.01-0.02)")
    client = LLMClient()

    reference_context = """We're measuring purchase intent on a 5-point scale:
1 = Definitely would not purchase
2 = Probably would not purchase
3 = Might or might not purchase
4 = Probably would purchase
5 = Definitely would purchase"""

    llm_response = await client.evaluate_ad_with_persona(
        ad_image_base64=image_b64,
        persona_description=persona,
        reference_context=reference_context
    )

    print(f"✓ LLM response: {llm_response[:100]}...")
    print()

    # Step 4: Convert to PMF using SSR
    print("Step 4: Convert to PMF using SSR...")

    # Create reference sentences
    reference_sentences = [
        "I definitely would not purchase",
        "I probably would not purchase",
        "I might or might not purchase",
        "I probably would purchase",
        "I definitely would purchase"
    ]

    # Create Polars DataFrame
    df = po.DataFrame({
        'id': ['purchase_intent'] * 5,
        'int_response': [1, 2, 3, 4, 5],
        'sentence': reference_sentences
    })

    # Initialize ResponseRater
    rater = ResponseRater(df, model_name="all-MiniLM-L6-v2")

    # Get PMF
    pmfs = rater.get_response_pmfs(
        reference_set_id='purchase_intent',
        llm_responses=[llm_response],
        temperature=1.0,
        epsilon=0.01
    )

    pmf = pmfs[0]

    print("✓ PMF calculated:")
    print(f"  Distribution: {[round(p, 3) for p in pmf]}")
    print()

    # Step 5: Calculate metrics
    print("Step 5: Calculate metrics...")
    expected_value = sum((i + 1) * p for i, p in enumerate(pmf))
    most_likely = int(pmf.argmax() + 1)
    confidence = float(pmf.max())

    print(f"  Expected value: {expected_value:.2f}/5")
    print(f"  Most likely rating: {most_likely}/5")
    print(f"  Confidence: {confidence:.1%}")
    print()

    # Step 6: Validate PMF
    print("Step 6: Validate PMF...")
    pmf_sum = sum(pmf)
    checks = {
        "sums_to_one": abs(pmf_sum - 1.0) < 0.001,
        "all_positive": all(p >= 0 for p in pmf),
        "makes_sense": 1.0 <= expected_value <= 5.0
    }

    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}: {passed}")
    print()

    if all(checks.values()):
        print("✅ FULL PIPELINE WORKING!")
        print()
        print("Summary:")
        print(f"  • Persona viewed ad → gave natural response")
        print(f"  • SSR converted to probability distribution")
        print(f"  • Predicted purchase intent: {expected_value:.2f}/5")
        print()
        print("🎯 Ready to build /v1/predict-ad endpoint!")
    else:
        print("❌ Some checks failed")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
