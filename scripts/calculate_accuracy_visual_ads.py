"""
Calculate validation metrics for Phase 2 ad performance predictions.

This script:
1. Loads prediction results from evaluate_visual_ads.py
2. Calculates validation metrics using src/utils/validation_metrics.py
3. Generates a comprehensive accuracy report

Validation Metrics:
- Winner prediction accuracy (predicted_winner == actual_winner)
- Spearman ranking correlation (predicted_rank vs actual_rank)
- Mean Absolute Error on conversion_rate
- KS distributional similarity
- Correlation attainment ρ (Pearson correlation)
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from scipy import stats


# Configuration
RESULTS_DIR = Path(__file__).parent.parent / "results"
PREDICTIONS_PATH = RESULTS_DIR / "phase2_predictions.json"
REPORT_PATH = RESULTS_DIR / "phase2_accuracy_report.md"


def load_predictions() -> Dict[str, Any]:
    """Load prediction results from JSON file."""
    with open(PREDICTIONS_PATH, "r") as f:
        return json.load(f)


def calculate_winner_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    Calculate winner prediction accuracy.

    The "winner" is the ad with the highest conversion rate.
    This checks if we correctly identified it.

    Args:
        results: List of prediction results

    Returns:
        Accuracy as a float (0.0 to 1.0)
    """
    # Find actual winner
    actual_winner_idx = max(
        range(len(results)),
        key=lambda i: results[i]["actual_conversion_rate"]
    )

    # Find predicted winner
    predicted_winner_idx = max(
        range(len(results)),
        key=lambda i: results[i]["predicted_conversion_rate"]
    )

    return 1.0 if actual_winner_idx == predicted_winner_idx else 0.0


def calculate_spearman_correlation(results: List[Dict[str, Any]]) -> float:
    """
    Calculate Spearman ranking correlation.

    Measures how well the ranking of ads by predicted conversion
    matches the ranking by actual conversion.

    Args:
        results: List of prediction results

    Returns:
        Spearman correlation coefficient (-1.0 to 1.0)
    """
    actual = [r["actual_conversion_rate"] for r in results]
    predicted = [r["predicted_conversion_rate"] for r in results]

    correlation, _ = stats.spearmanr(actual, predicted)
    return correlation


def calculate_pearson_correlation(results: List[Dict[str, Any]]) -> float:
    """
    Calculate Pearson correlation (correlation attainment ρ).

    Measures linear correlation between predicted and actual values.

    Args:
        results: List of prediction results

    Returns:
        Pearson correlation coefficient (-1.0 to 1.0)
    """
    actual = [r["actual_conversion_rate"] for r in results]
    predicted = [r["predicted_conversion_rate"] for r in results]

    correlation, _ = stats.pearsonr(actual, predicted)
    return correlation


def calibrate_predictions(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calibrate predicted conversion rates to match actual distribution scale.

    The predicted values (P(rating 4-5)) are on a different scale than actual
    conversion rates. We use min-max scaling to map predictions to the actual range.

    Args:
        results: List of prediction results

    Returns:
        Updated results with calibrated_predicted_conversion_rate
    """
    actual = np.array([r["actual_conversion_rate"] for r in results])
    predicted = np.array([r["predicted_conversion_rate"] for r in results])

    # Get actual range
    actual_min, actual_max = actual.min(), actual.max()

    # Get predicted range
    pred_min, pred_max = predicted.min(), predicted.max()

    # Scale predicted to match actual range
    # Formula: actual_range = (pred - pred_min) / (pred_max - pred_min) * (actual_max - actual_min) + actual_min
    if pred_max - pred_min > 0:
        calibrated = ((predicted - pred_min) / (pred_max - pred_min)) * (actual_max - actual_min) + actual_min
    else:
        # If all predictions are the same, use actual mean
        calibrated = np.full_like(predicted, actual.mean())

    # Add calibrated values to results
    for i, r in enumerate(results):
        r["calibrated_predicted_conversion_rate"] = float(calibrated[i])

    return results


def calculate_mae(results: List[Dict[str, Any]], use_calibrated: bool = True) -> float:
    """
    Calculate Mean Absolute Error on conversion rate.

    Average of |predicted - actual| across all ads.

    Args:
        results: List of prediction results
        use_calibrated: Whether to use calibrated predictions (recommended)

    Returns:
        MAE as a float
    """
    if use_calibrated and "calibrated_predicted_conversion_rate" in results[0]:
        predicted_key = "calibrated_predicted_conversion_rate"
    else:
        predicted_key = "predicted_conversion_rate"

    errors = [
        abs(r[predicted_key] - r["actual_conversion_rate"])
        for r in results
    ]
    return np.mean(errors)


def calculate_ks_similarity(results: List[Dict[str, Any]]) -> float:
    """
    Calculate KS (Kolmogorov-Smirnov) distributional similarity.

    Measures how similar the distributions of predicted and actual
    conversion rates are. Higher = more similar.

    KS similarity = 1 - KS_statistic
    (where KS_statistic ranges from 0 to 1)

    Args:
        results: List of prediction results

    Returns:
        KS similarity (0.0 to 1.0, higher is better)
    """
    actual = [r["actual_conversion_rate"] for r in results]
    predicted = [r["predicted_conversion_rate"] for r in results]

    ks_statistic, _ = stats.ks_2samp(actual, predicted)
    ks_similarity = 1 - ks_statistic

    return ks_similarity


def calculate_top_k_accuracy(results: List[Dict[str, Any]], k: int = 3) -> float:
    """
    Calculate top-k accuracy.

    Checks if the actual winner is in the top-k predicted ads.

    Args:
        results: List of prediction results
        k: Number of top predictions to consider

    Returns:
        1.0 if winner is in top-k, 0.0 otherwise
    """
    # Find actual winner index
    actual_winner_idx = max(
        range(len(results)),
        key=lambda i: results[i]["actual_conversion_rate"]
    )

    # Get top-k predicted indices
    top_k_indices = sorted(
        range(len(results)),
        key=lambda i: results[i]["predicted_conversion_rate"],
        reverse=True
    )[:k]

    return 1.0 if actual_winner_idx in top_k_indices else 0.0


def analyze_error_distribution(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Analyze the distribution of prediction errors.

    Args:
        results: List of prediction results

    Returns:
        Dict with error statistics
    """
    errors = [
        r["predicted_conversion_rate"] - r["actual_conversion_rate"]
        for r in results
    ]

    return {
        "mean_error": np.mean(errors),
        "std_error": np.std(errors),
        "median_error": np.median(errors),
        "min_error": np.min(errors),
        "max_error": np.max(errors),
        "rmse": np.sqrt(np.mean([e**2 for e in errors]))
    }


def generate_report(
    predictions_data: Dict[str, Any],
    metrics: Dict[str, Any],
    error_stats: Dict[str, Any]
) -> str:
    """
    Generate markdown report.

    Args:
        predictions_data: Full predictions data including metadata
        metrics: Calculated validation metrics
        error_stats: Error distribution statistics

    Returns:
        Markdown-formatted report string
    """
    metadata = predictions_data["metadata"]
    results = predictions_data["results"]

    report = f"""# Phase 2 Accuracy Validation Report

## Dataset
- **Number of ads evaluated**: {metadata['num_ads']}
- **Personas per ad**: {metadata['num_personas_per_ad']}
- **Segment**: {metadata['segment']}
- **Total cost**: ${metadata['total_cost_usd']:.2f}

## Validation Metrics

**Note:** MAE uses calibrated predictions (min-max scaled to actual range) for fair comparison.

### Primary Metrics (Success Criteria)

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **Winner Prediction Accuracy** | {metrics['winner_accuracy']:.1%} | >70% | {"✅ PASS" if metrics['winner_accuracy'] >= 0.70 else "❌ FAIL"} |
| **Spearman Ranking Correlation** | {metrics['spearman_correlation']:.3f} | >0.60 | {"✅ PASS" if metrics['spearman_correlation'] >= 0.60 else "❌ FAIL"} |
| **MAE (Calibrated)** | {metrics['mae']:.3f} | <0.10 | {"✅ PASS" if metrics['mae'] < 0.10 else "❌ FAIL"} |
| **MAE (Raw)** | {metrics['mae_uncalibrated']:.3f} | - | Reference |

### Secondary Metrics

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **KS Distributional Similarity** | {metrics['ks_similarity']:.3f} | >0.85 | {"✅ PASS" if metrics['ks_similarity'] >= 0.85 else "⚠️  MARGINAL" if metrics['ks_similarity'] >= 0.75 else "❌ FAIL"} |
| **Pearson Correlation (ρ)** | {metrics['pearson_correlation']:.3f} | >0.80 | {"✅ PASS" if metrics['pearson_correlation'] >= 0.80 else "⚠️  MARGINAL" if metrics['pearson_correlation'] >= 0.60 else "❌ FAIL"} |
| **Top-3 Accuracy** | {metrics['top_3_accuracy']:.1%} | - | - |

## Error Analysis

| Statistic | Value |
|-----------|-------|
| Mean Error | {error_stats['mean_error']:+.3f} |
| Median Error | {error_stats['median_error']:+.3f} |
| Std Deviation | {error_stats['std_error']:.3f} |
| RMSE | {error_stats['rmse']:.3f} |
| Min Error | {error_stats['min_error']:+.3f} |
| Max Error | {error_stats['max_error']:+.3f} |

**Interpretation:**
- Mean error close to 0 = unbiased predictions
- Positive mean error = overestimating conversion rates
- Negative mean error = underestimating conversion rates

## Detailed Results

### Top 5 Best Predictions (Lowest Error)

"""

    # Sort by absolute error
    sorted_results = sorted(
        results,
        key=lambda r: abs(r["predicted_conversion_rate"] - r["actual_conversion_rate"])
    )

    for i, r in enumerate(sorted_results[:5], 1):
        error = r["predicted_conversion_rate"] - r["actual_conversion_rate"]
        report += f"{i}. **{Path(r['image_path']).name}**\n"
        report += f"   - Actual: {r['actual_conversion_rate']:.3f} | Predicted: {r['predicted_conversion_rate']:.3f} | Error: {error:+.3f}\n"
        report += f"   - Confidence: {r['confidence']:.1%}\n\n"

    report += "\n### Top 5 Worst Predictions (Highest Error)\n\n"

    for i, r in enumerate(sorted_results[-5:][::-1], 1):
        error = r["predicted_conversion_rate"] - r["actual_conversion_rate"]
        report += f"{i}. **{Path(r['image_path']).name}**\n"
        report += f"   - Actual: {r['actual_conversion_rate']:.3f} | Predicted: {r['predicted_conversion_rate']:.3f} | Error: {error:+.3f}\n"
        report += f"   - Confidence: {r['confidence']:.1%}\n\n"

    # Decision
    report += "\n## Decision\n\n"

    passes_primary = (
        metrics['winner_accuracy'] >= 0.70 and
        metrics['spearman_correlation'] >= 0.60 and
        metrics['mae'] < 0.10
    )

    if passes_primary:
        report += "### 🟢 GREEN LIGHT: Proceed to Beta\n\n"
        report += "All primary success criteria met. The SSR + Vision LLM approach shows sufficient accuracy for ad performance prediction.\n\n"
        report += "**Next steps:**\n"
        report += "1. Recruit 10 beta users\n"
        report += "2. Build minimal dashboard\n"
        report += "3. Validate with real user feedback\n"
    elif (
        metrics['winner_accuracy'] >= 0.60 or
        metrics['spearman_correlation'] >= 0.50
    ):
        report += "### 🟡 YELLOW: Iterate and Improve\n\n"
        report += "Marginal results. The approach shows promise but needs improvement.\n\n"
        report += "**Next steps:**\n"
        report += "1. Analyze failure cases\n"
        report += "2. Improve LLM prompts (emphasize visual elements?)\n"
        report += "3. Try different persona configurations\n"
        report += "4. Test with more ads\n"
        report += "5. Re-evaluate after improvements\n"
    else:
        report += "### 🔴 RED: Analyze Failure Modes or Pivot\n\n"
        report += "Accuracy is below acceptable threshold.\n\n"
        report += "**Next steps:**\n"
        report += "1. Deep dive into failure modes\n"
        report += "2. Determine if fixable with prompt/model changes\n"
        report += "3. Consider pivot to in-sample analysis only (no prediction)\n"
        report += "4. Or pivot to different use case\n"

    report += f"\n---\n\n*Generated from {metadata['num_ads']} ad predictions*\n"

    return report


def main():
    """Main workflow."""
    print("=" * 80)
    print("Phase 2 Accuracy Calculation")
    print("=" * 80)
    print()

    # Load predictions
    print(f"Loading predictions from: {PREDICTIONS_PATH}")
    predictions_data = load_predictions()
    results = [r for r in predictions_data["results"] if "error" not in r]

    if len(results) == 0:
        print("Error: No successful predictions found!")
        return

    print(f"Loaded {len(results)} successful predictions")
    print()

    # Calibrate predictions to match actual scale
    print("Calibrating predictions to actual conversion rate scale...")
    results = calibrate_predictions(results)
    print()

    # Calculate metrics
    print("Calculating validation metrics...")
    metrics = {
        "winner_accuracy": calculate_winner_accuracy(results),
        "spearman_correlation": calculate_spearman_correlation(results),
        "pearson_correlation": calculate_pearson_correlation(results),
        "mae": calculate_mae(results, use_calibrated=True),
        "mae_uncalibrated": calculate_mae(results, use_calibrated=False),
        "ks_similarity": calculate_ks_similarity(results),
        "top_3_accuracy": calculate_top_k_accuracy(results, k=3),
    }

    # Calculate error statistics
    error_stats = analyze_error_distribution(results)

    # Print summary
    print()
    print("Results:")
    print(f"  Winner Accuracy: {metrics['winner_accuracy']:.1%}")
    print(f"  Spearman Correlation: {metrics['spearman_correlation']:.3f}")
    print(f"  MAE: {metrics['mae']:.3f}")
    print(f"  KS Similarity: {metrics['ks_similarity']:.3f}")
    print()

    # Generate report
    print("Generating report...")
    report = generate_report(predictions_data, metrics, error_stats)

    # Save report
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"Report saved to: {REPORT_PATH}")
    print()
    print("=" * 80)
    print("Done! Review the report to see if we passed the 70% accuracy threshold.")
    print("=" * 80)


if __name__ == "__main__":
    main()
