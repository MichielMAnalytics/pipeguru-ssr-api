"""
Evaluate ad performance predictions on the Art and Advertising Visual Optimization dataset.

This script:
1. Loads ads from the dataset (images + performance metrics)
2. For each ad, generates predictions using generic personas
3. Saves results for accuracy calculation

Phase 2 Validation Flow (Per Ad):
1. Load Ad (image + actual metrics from CSV)
2. Generate Generic Personas (20 per ad, segment="general_consumer")
3. Evaluate with Vision LLM (20 evaluations via GPT-4 Vision)
4. SSR Processing (20 PMFs)
5. Aggregate (average PMFs → predicted conversion rate)
6. Compare (save for later accuracy calculation)
"""

import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import httpx
import pandas as pd
from tqdm import tqdm


# Configuration
API_BASE_URL = "http://localhost:8000/v1"
DATASET_DIR = Path(__file__).parent.parent / "data" / "ad-visual-optimization"
CSV_PATH = DATASET_DIR / "art_design_ad_dataset.csv"
IMAGES_DIR = DATASET_DIR / "Images" / "Ad images"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_PATH = RESULTS_DIR / "phase2_predictions.json"

# Prediction parameters
NUM_PERSONAS = 20  # Number of personas per ad (cost: ~$0.40 per ad)
SEGMENT = "general_consumer"  # Using generic personas
SAMPLE_SIZE = 10  # Total number of ads to test
STRATIFIED = True  # Stratify by conversion rate (5 high, 5 low)


def load_image_as_base64(image_path: Path) -> str:
    """Load image file and convert to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_all_image_paths() -> List[Path]:
    """Get all image paths from the dataset."""
    return sorted(IMAGES_DIR.rglob("*.png"))


def load_dataset() -> pd.DataFrame:
    """Load the CSV with performance metrics."""
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV with {len(df)} rows")
    return df


def select_test_ads(
    all_images: List[Path],
    df: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
    stratified: bool = STRATIFIED
) -> List[Dict[str, Any]]:
    """
    Select a subset of ads for testing.

    If stratified=True, selects 50% high conversion and 50% low conversion ads.
    Otherwise, selects random sample.

    Args:
        all_images: List of all image paths
        df: DataFrame with performance metrics
        sample_size: Number of ads to select
        stratified: Whether to stratify by conversion rate

    Returns:
        List of dicts with image_path, image_index, and actual_metrics
    """
    # Since we have 301 images and 1000 CSV rows, we'll map images to the first 301 rows
    # This is a simplification - in production we'd need proper mapping
    image_count = min(len(all_images), len(df))

    if stratified and sample_size >= 2:
        # Sort by conversion rate
        df_subset = df.iloc[:image_count].copy()
        df_subset['image_idx'] = range(image_count)
        df_sorted = df_subset.sort_values('conversion_rate')

        # Select bottom 50% (low conversion) and top 50% (high conversion)
        n_low = sample_size // 2
        n_high = sample_size - n_low

        low_indices = df_sorted.head(n_low)['image_idx'].tolist()
        high_indices = df_sorted.tail(n_high)['image_idx'].tolist()
        selected_indices = low_indices + high_indices

        print(f"Selected {n_low} low-conversion ads and {n_high} high-conversion ads")
    else:
        # Random sample
        import random
        selected_indices = random.sample(range(min(len(all_images), image_count)), sample_size)
        print(f"Selected {sample_size} random ads")

    # Build test set
    test_ads = []
    for idx in selected_indices:
        image_path = all_images[idx]
        actual_metrics = df.iloc[idx].to_dict()

        test_ads.append({
            "image_path": str(image_path),
            "image_index": idx,
            "actual_conversion_rate": actual_metrics["conversion_rate"],
            "actual_ctr": actual_metrics["CTR"],
            "actual_engagement_ratio": actual_metrics["engagement_ratio"],
            "actual_metrics_full": actual_metrics
        })

    return test_ads


async def predict_ad_performance(
    ad_image_base64: str,
    num_personas: int,
    segment: str
) -> Dict[str, Any]:
    """
    Call the API to predict ad performance.

    Args:
        ad_image_base64: Base64-encoded ad image
        num_personas: Number of personas to generate
        segment: Persona segment (e.g., "general_consumer")

    Returns:
        API response dict with prediction results
    """
    async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
        response = await client.post(
            f"{API_BASE_URL}/predict-ad",
            json={
                "ad_image_base64": ad_image_base64,
                "num_personas": num_personas,
                "segment": segment
            }
        )
        response.raise_for_status()
        return response.json()


async def evaluate_single_ad(ad: Dict[str, Any], progress_bar: tqdm, ad_num: int, total_ads: int) -> Dict[str, Any]:
    """
    Evaluate a single ad and return prediction results.

    Args:
        ad: Dict with image_path and actual metrics
        progress_bar: tqdm progress bar to update
        ad_num: Current ad number (1-indexed)
        total_ads: Total number of ads

    Returns:
        Dict with prediction results and actual metrics
    """
    try:
        # Load image
        image_path = Path(ad["image_path"])
        # Show folder/filename for better context (e.g., "9/13.png")
        relative_path = f"{image_path.parent.name}/{image_path.name}"
        progress_bar.write(f"\n{'='*60}")
        progress_bar.write(f"Ad {ad_num}/{total_ads}: {relative_path}")
        progress_bar.write(f"Actual conversion: {ad['actual_conversion_rate']:.4f}")
        progress_bar.write(f"Loading image and generating {NUM_PERSONAS} personas...")

        ad_image_base64 = load_image_as_base64(image_path)

        # Get prediction
        prediction = await predict_ad_performance(
            ad_image_base64=ad_image_base64,
            num_personas=NUM_PERSONAS,
            segment=SEGMENT
        )

        # Combine with actual metrics
        result = {
            "image_path": ad["image_path"],
            "image_index": ad["image_index"],
            "actual_conversion_rate": ad["actual_conversion_rate"],
            "actual_ctr": ad["actual_ctr"],
            "actual_engagement_ratio": ad["actual_engagement_ratio"],
            "predicted_conversion_rate": prediction["predicted_conversion_rate"],
            "confidence": prediction["confidence"],
            "pmf_aggregate": prediction["pmf_aggregate"],
            "cost_usd": prediction["cost"]["estimated_cost_usd"],
            "processing_time": prediction["metadata"]["processing_time_seconds"],
            "num_personas": prediction["metadata"]["num_personas"],
        }

        progress_bar.write(f"✓ Predicted conversion: {prediction['predicted_conversion_rate']:.4f}")
        progress_bar.write(f"  Confidence: {prediction['confidence']:.1%} | Time: {prediction['metadata']['processing_time_seconds']:.1f}s | Cost: ${prediction['cost']['estimated_cost_usd']:.2f}")

        progress_bar.update(1)
        progress_bar.set_postfix({
            "actual": f"{ad['actual_conversion_rate']:.3f}",
            "predicted": f"{prediction['predicted_conversion_rate']:.3f}"
        })

        return result

    except Exception as e:
        progress_bar.write(f"\n{'='*60}")
        progress_bar.write(f"✗ Error evaluating ad {ad_num}/{total_ads}: {Path(ad['image_path']).name}")
        progress_bar.write(f"  {str(e)}")
        return {
            "image_path": ad["image_path"],
            "image_index": ad["image_index"],
            "error": str(e)
        }


async def evaluate_all_ads(test_ads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate all test ads.

    Args:
        test_ads: List of ads to evaluate

    Returns:
        List of prediction results
    """
    results = []
    total_ads = len(test_ads)

    with tqdm(total=total_ads, desc="Evaluating ads") as pbar:
        # Process ads sequentially to avoid rate limits
        # In production, could use asyncio.gather with semaphore for concurrency
        for i, ad in enumerate(test_ads, 1):
            result = await evaluate_single_ad(ad, pbar, ad_num=i, total_ads=total_ads)
            results.append(result)

            # Save intermediate results after each ad
            save_results(results, intermediate=True)

    return results


def save_results(results: List[Dict[str, Any]], intermediate: bool = False):
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)

    output_path = RESULTS_PATH
    if intermediate:
        output_path = RESULTS_DIR / "phase2_predictions_partial.json"

    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "num_ads": len(results),
                "num_personas_per_ad": NUM_PERSONAS,
                "segment": SEGMENT,
                "total_cost_usd": sum(r.get("cost_usd", 0) for r in results if "error" not in r)
            },
            "results": results
        }, f, indent=2)

    if not intermediate:
        print(f"\nResults saved to: {output_path}")


async def main():
    """Main evaluation workflow."""
    print("=" * 80)
    print("Phase 2 Validation: Ad Performance Prediction Evaluation")
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
        sys.exit(1)

    print()

    # Load dataset
    print("Loading dataset...")
    all_images = get_all_image_paths()
    df = load_dataset()
    print(f"Found {len(all_images)} images")
    print()

    # Select test ads
    print(f"Selecting {SAMPLE_SIZE} test ads...")
    test_ads = select_test_ads(all_images, df, SAMPLE_SIZE, STRATIFIED)
    print()

    # Show sample
    print("Sample ads selected:")
    for i, ad in enumerate(test_ads[:3]):
        print(f"  {i+1}. {Path(ad['image_path']).name} "
              f"(conversion: {ad['actual_conversion_rate']:.3f})")
    print(f"  ... and {len(test_ads) - 3} more")
    print()

    # Confirm
    response = input("Proceed with evaluation? [y/N]: ")
    if response.lower() != 'y':
        print("Evaluation cancelled.")
        sys.exit(0)

    print()
    print("Starting evaluation...")
    print("This will take approximately 10-30 seconds per ad")
    print()

    # Run evaluation
    results = await evaluate_all_ads(test_ads)

    # Save final results
    save_results(results, intermediate=False)

    # Summary
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print()
    print("=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    print(f"Successful predictions: {len(successful)}/{len(results)}")
    print(f"Failed predictions: {len(failed)}/{len(results)}")

    if successful:
        total_cost = sum(r["cost_usd"] for r in successful)
        avg_time = sum(r["processing_time"] for r in successful) / len(successful)
        print(f"Total cost: ${total_cost:.2f}")
        print(f"Average processing time: {avg_time:.1f}s per ad")

    if failed:
        print("\nFailed ads:")
        for r in failed:
            print(f"  - {r['image_path']}: {r['error']}")

    print()
    print(f"Results saved to: {RESULTS_PATH}")
    print("\nNext step: Run calculate_accuracy_visual_ads.py to compute validation metrics")


if __name__ == "__main__":
    asyncio.run(main())
