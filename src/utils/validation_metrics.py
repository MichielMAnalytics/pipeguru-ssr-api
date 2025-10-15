"""
Validation metrics for SSR predictions based on the paper:
"LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings"

These metrics are used to evaluate how well synthetic predictions match human data.
"""

from typing import List, Tuple

import numpy as np
from scipy import stats


def ks_distributional_similarity(pmf_pred: np.ndarray, pmf_actual: np.ndarray) -> float:
    """
    Calculate Kolmogorov-Smirnov distributional similarity between two PMFs.

    The paper uses KS similarity as: KS_sim = 1 - KS_dist
    Target performance: KS_sim > 0.85 (paper achieves ~0.88 with 6 reference sets)

    Args:
        pmf_pred: Predicted probability mass function (1D array)
        pmf_actual: Actual probability mass function (1D array)

    Returns:
        float: KS similarity score between 0 and 1 (higher is better)

    Example:
        >>> pred_pmf = np.array([0.1, 0.2, 0.3, 0.3, 0.1])
        >>> actual_pmf = np.array([0.05, 0.15, 0.35, 0.35, 0.1])
        >>> ks_sim = ks_distributional_similarity(pred_pmf, actual_pmf)
        >>> print(f"KS similarity: {ks_sim:.3f}")
    """
    # Ensure arrays are numpy
    pmf_pred = np.asarray(pmf_pred)
    pmf_actual = np.asarray(pmf_actual)

    # Normalize to ensure they're valid PMFs
    pmf_pred = pmf_pred / np.sum(pmf_pred)
    pmf_actual = pmf_actual / np.sum(pmf_actual)

    # Calculate CDFs
    cdf_pred = np.cumsum(pmf_pred)
    cdf_actual = np.cumsum(pmf_actual)

    # KS distance is the maximum absolute difference between CDFs
    ks_distance = np.max(np.abs(cdf_pred - cdf_actual))

    # Convert to similarity (1 = identical distributions)
    ks_similarity = 1.0 - ks_distance

    return float(ks_similarity)


def correlation_attainment(
    predictions: List[float],
    actuals: List[float],
    test_retest_reliability: float = None,
    num_bootstrap: int = 2000,
) -> Tuple[float, float]:
    """
    Calculate correlation attainment: how close predictions are to test-retest ceiling.

    The paper defines this as: ρ = R^xy / R^xx
    where R^xy is correlation between synthetic and real data,
    and R^xx is the test-retest reliability ceiling (human vs human retest).

    Target performance: ρ > 0.8 (paper achieves ~0.90 with demographics + SSR)

    Args:
        predictions: List of predicted values (e.g., mean purchase intents)
        actuals: List of actual values (same length as predictions)
        test_retest_reliability: Pre-computed test-retest R^xx. If None, will bootstrap.
        num_bootstrap: Number of bootstrap samples for estimating test-retest ceiling

    Returns:
        Tuple of (correlation_attainment, R_xy):
            - correlation_attainment: Ratio of achieved to maximum correlation (0-1)
            - R_xy: Pearson correlation between predictions and actuals

    Example:
        >>> predictions = [3.5, 3.8, 4.1, 3.2, 4.3]
        >>> actuals = [3.6, 3.7, 4.0, 3.3, 4.2]
        >>> rho, r_xy = correlation_attainment(predictions, actuals)
        >>> print(f"Correlation attainment: {rho:.1%}")
        >>> print(f"R^xy: {r_xy:.3f}")
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have same length")

    if len(predictions) < 2:
        raise ValueError("Need at least 2 data points for correlation")

    # Calculate R^xy: correlation between predictions and actuals
    r_xy = stats.pearsonr(predictions, actuals)[0]

    # Calculate or use provided test-retest reliability R^xx
    if test_retest_reliability is None:
        # Bootstrap estimate: split actuals randomly and calculate correlation
        r_xx_estimates = []
        n = len(actuals)
        half_n = n // 2

        for _ in range(num_bootstrap):
            # Randomly shuffle and split
            shuffled = np.random.permutation(actuals)
            test_half = shuffled[:half_n]
            control_half = shuffled[half_n : 2 * half_n]

            if len(test_half) > 1 and len(control_half) > 1:
                r_test_control = stats.pearsonr(test_half, control_half)[0]
                r_xx_estimates.append(r_test_control)

        r_xx = np.mean(r_xx_estimates)
    else:
        r_xx = test_retest_reliability

    # Correlation attainment
    if r_xx > 0:
        rho = r_xy / r_xx
    else:
        rho = 0.0

    return float(rho), float(r_xy)


def mean_absolute_error(predictions: List[float], actuals: List[float]) -> float:
    """
    Calculate mean absolute error between predictions and actuals.

    Args:
        predictions: List of predicted values
        actuals: List of actual values (same length as predictions)

    Returns:
        float: MAE score (lower is better)

    Example:
        >>> predictions = [3.5, 3.8, 4.1]
        >>> actuals = [3.6, 3.7, 4.0]
        >>> mae = mean_absolute_error(predictions, actuals)
        >>> print(f"MAE: {mae:.3f}")
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have same length")

    mae = np.mean(np.abs(predictions - actuals))
    return float(mae)


def winner_prediction_accuracy(
    predictions: List[float],
    actuals: List[float],
    labels: List[str] = None,
) -> Tuple[float, dict]:
    """
    Calculate winner prediction accuracy: can we identify which option performed best?

    This is useful for A/B testing scenarios where you want to know if the model
    can correctly identify the winning variant.

    Args:
        predictions: List of predicted values
        actuals: List of actual values (same length as predictions)
        labels: Optional labels for each prediction (for debugging)

    Returns:
        Tuple of (accuracy, details):
            - accuracy: 1.0 if predicted winner matches actual winner, 0.0 otherwise
            - details: Dict with predicted_winner, actual_winner, and their indices

    Example:
        >>> predictions = [3.5, 4.2, 3.8]  # Model predicts option 2 wins
        >>> actuals = [3.6, 4.1, 3.7]      # Option 2 actually won
        >>> labels = ["Ad A", "Ad B", "Ad C"]
        >>> accuracy, details = winner_prediction_accuracy(predictions, actuals, labels)
        >>> print(f"Accuracy: {accuracy:.0%}")
        >>> print(f"Details: {details}")
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have same length")

    if len(predictions) < 2:
        raise ValueError("Need at least 2 options to determine a winner")

    # Find indices of winners
    predicted_winner_idx = int(np.argmax(predictions))
    actual_winner_idx = int(np.argmax(actuals))

    # Check if they match
    accuracy = 1.0 if predicted_winner_idx == actual_winner_idx else 0.0

    # Build details dict
    details = {
        "predicted_winner_index": predicted_winner_idx,
        "actual_winner_index": actual_winner_idx,
        "predicted_winner_value": float(predictions[predicted_winner_idx]),
        "actual_winner_value": float(actuals[actual_winner_idx]),
    }

    if labels is not None:
        details["predicted_winner_label"] = labels[predicted_winner_idx]
        details["actual_winner_label"] = labels[actual_winner_idx]

    return accuracy, details


def ranking_correlation(
    predictions: List[float],
    actuals: List[float],
    method: str = "spearman",
) -> float:
    """
    Calculate ranking correlation between predictions and actuals.

    This measures how well the model ranks options relative to each other,
    which is often more important than absolute accuracy.

    Target performance: Spearman correlation > 0.6 (paper achieves 0.72-0.74)

    Args:
        predictions: List of predicted values
        actuals: List of actual values (same length as predictions)
        method: "spearman" or "kendall" correlation

    Returns:
        float: Correlation coefficient (-1 to 1, higher is better)

    Example:
        >>> predictions = [3.5, 4.2, 3.8, 4.5]
        >>> actuals = [3.6, 4.1, 3.7, 4.4]
        >>> corr = ranking_correlation(predictions, actuals)
        >>> print(f"Spearman correlation: {corr:.3f}")
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have same length")

    if len(predictions) < 2:
        raise ValueError("Need at least 2 data points for correlation")

    if method == "spearman":
        corr = stats.spearmanr(predictions, actuals)[0]
    elif method == "kendall":
        corr = stats.kendalltau(predictions, actuals)[0]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'kendall'")

    return float(corr)
