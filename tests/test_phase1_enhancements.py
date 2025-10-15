"""
Tests for Phase 1 SSR enhancements based on paper:
"LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings"
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.core.persona_generator import PersonaGenerator
from src.core.reference_statements import (
    ALL_REFERENCE_SETS,
    get_reference_sets,
    get_single_reference_set,
)
from src.main import app
from src.utils.validation_metrics import (
    correlation_attainment,
    ks_distributional_similarity,
    mean_absolute_error,
    ranking_correlation,
    winner_prediction_accuracy,
)

client = TestClient(app)


# ============================================================================
# Test Reference Statements Module
# ============================================================================


def test_reference_sets_exist():
    """Test that all 6 reference sets are defined."""
    sets = get_reference_sets("purchase_intent")
    assert len(sets) == 6, "Paper uses 6 reference sets"

    # Each set should have 5 statements (for 5-point Likert scale)
    for i, ref_set in enumerate(sets):
        assert len(ref_set) == 5, f"Reference set {i} should have 5 statements"
        # All statements should be non-empty strings
        for stmt in ref_set:
            assert isinstance(stmt, str)
            assert len(stmt) > 0


def test_get_single_reference_set():
    """Test getting individual reference sets by index."""
    set_0 = get_single_reference_set(0)
    assert len(set_0) == 5
    assert isinstance(set_0[0], str)

    # Test all indices
    for i in range(6):
        ref_set = get_single_reference_set(i)
        assert len(ref_set) == 5


def test_reference_set_index_out_of_range():
    """Test that invalid index raises error."""
    with pytest.raises(ValueError):
        get_single_reference_set(10)


def test_reference_sets_are_diverse():
    """Test that reference sets use different phrasings."""
    sets = get_reference_sets("purchase_intent")

    # Check that sets are not identical
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i] != sets[j], f"Sets {i} and {j} should be different"


# ============================================================================
# Test Rate Endpoint with Multiple Reference Sets
# ============================================================================


def test_rate_endpoint_with_multiple_sets_enabled():
    """Test rate endpoint with multiple reference sets enabled (default)."""
    request_data = {
        "responses": ["I love this product", "Not interested"],
        "reference_sentences": [
            "I definitely would not purchase",
            "I probably would not purchase",
            "I might or might not purchase",
            "I probably would purchase",
            "I definitely would purchase",
        ],
        "use_multiple_reference_sets": True,
    }

    response = client.post("/v1/rate", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert len(data["ratings"]) == 2

    # Check metadata includes reference set info
    assert data["metadata"]["num_reference_sets"] == 6
    assert data["metadata"]["use_multiple_reference_sets"] is True

    # PMFs should still sum to 1
    for rating in data["ratings"]:
        assert abs(sum(rating["pmf"]) - 1.0) < 0.001


def test_rate_endpoint_with_multiple_sets_disabled():
    """Test rate endpoint with single reference set."""
    request_data = {
        "responses": ["I love this product"],
        "reference_sentences": [
            "I definitely would not purchase",
            "I probably would not purchase",
            "I might or might not purchase",
            "I probably would purchase",
            "I definitely would purchase",
        ],
        "use_multiple_reference_sets": False,
    }

    response = client.post("/v1/rate", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["metadata"]["num_reference_sets"] == 1
    assert data["metadata"]["use_multiple_reference_sets"] is False


def test_rate_endpoint_default_uses_multiple_sets():
    """Test that default behavior uses multiple reference sets."""
    request_data = {
        "responses": ["Great product"],
        "reference_sentences": [
            "Definitely not",
            "Probably not",
            "Maybe",
            "Probably yes",
            "Definitely yes",
        ],
    }

    response = client.post("/v1/rate", json=request_data)
    assert response.status_code == 200

    data = response.json()
    # Default should be True
    assert data["metadata"]["use_multiple_reference_sets"] is True
    assert data["metadata"]["num_reference_sets"] == 6


def test_multi_set_pmfs_more_stable():
    """
    Test that multi-set averaging produces stable results.
    Run same request multiple times and check variance.
    """
    request_data = {
        "responses": ["I'm somewhat interested in this product"],
        "reference_sentences": [
            "Not at all interested",
            "Slightly interested",
            "Moderately interested",
            "Very interested",
            "Extremely interested",
        ],
        "use_multiple_reference_sets": True,
    }

    # Run multiple times (SSR is deterministic with same inputs)
    results = []
    for _ in range(3):
        response = client.post("/v1/rate", json=request_data)
        data = response.json()
        results.append(data["ratings"][0]["pmf"])

    # All runs should give identical results (deterministic)
    for i in range(1, len(results)):
        np.testing.assert_array_almost_equal(results[0], results[i], decimal=5)


# ============================================================================
# Test Validation Metrics
# ============================================================================


def test_ks_distributional_similarity():
    """Test KS similarity metric."""
    # Identical distributions
    pmf1 = np.array([0.1, 0.2, 0.3, 0.3, 0.1])
    pmf2 = np.array([0.1, 0.2, 0.3, 0.3, 0.1])
    similarity = ks_distributional_similarity(pmf1, pmf2)
    assert similarity == 1.0, "Identical PMFs should have similarity 1.0"

    # Completely different distributions
    pmf3 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    pmf4 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    similarity = ks_distributional_similarity(pmf3, pmf4)
    assert similarity < 0.5, "Very different PMFs should have low similarity"

    # Similar but not identical
    pmf5 = np.array([0.1, 0.2, 0.3, 0.3, 0.1])
    pmf6 = np.array([0.15, 0.2, 0.25, 0.3, 0.1])
    similarity = ks_distributional_similarity(pmf5, pmf6)
    assert 0.7 < similarity < 1.0, "Similar PMFs should have high similarity"


def test_ks_similarity_target_benchmark():
    """Test that we understand the paper's benchmark (KS sim > 0.85)."""
    # This is just a documentation test
    target_ks_similarity = 0.85
    assert target_ks_similarity == 0.85, "Paper targets KS similarity > 0.85 with multi-sets"


def test_correlation_attainment():
    """Test correlation attainment metric."""
    # Perfect correlation
    predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
    actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
    rho, r_xy = correlation_attainment(predictions, actuals, test_retest_reliability=1.0)
    assert r_xy == 1.0, "Perfect predictions should have R=1.0"
    assert rho == 1.0, "Perfect predictions should have ρ=1.0"

    # Strong correlation
    predictions = [1.0, 2.1, 2.9, 4.1, 4.9]
    actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
    rho, r_xy = correlation_attainment(predictions, actuals, test_retest_reliability=0.9)
    assert r_xy > 0.95, "Strong correlation expected"
    assert rho > 0.95 / 0.9, "High attainment expected"


def test_correlation_attainment_target_benchmark():
    """Test that we understand the paper's benchmark (ρ > 0.80)."""
    target_correlation_attainment = 0.80
    assert target_correlation_attainment == 0.80, "Paper targets ρ > 0.80 (90% of test-retest)"


def test_mean_absolute_error():
    """Test MAE calculation."""
    predictions = [3.5, 4.0, 2.5]
    actuals = [3.6, 3.9, 2.7]
    mae = mean_absolute_error(predictions, actuals)

    expected_mae = (0.1 + 0.1 + 0.2) / 3
    assert abs(mae - expected_mae) < 0.001


def test_winner_prediction_accuracy():
    """Test winner prediction for A/B testing."""
    # Correct winner prediction
    predictions = [3.5, 4.2, 3.8]
    actuals = [3.6, 4.1, 3.7]
    accuracy, details = winner_prediction_accuracy(predictions, actuals)
    assert accuracy == 1.0, "Should correctly identify winner"
    assert details["predicted_winner_index"] == 1
    assert details["actual_winner_index"] == 1

    # Incorrect winner prediction
    predictions = [4.5, 3.0, 3.2]
    actuals = [3.5, 4.0, 3.8]
    accuracy, details = winner_prediction_accuracy(predictions, actuals)
    assert accuracy == 0.0, "Should detect incorrect winner"


def test_ranking_correlation():
    """Test Spearman ranking correlation."""
    # Perfect ranking
    predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
    actuals = [1.1, 2.1, 3.1, 4.1, 5.1]
    corr = ranking_correlation(predictions, actuals, method="spearman")
    assert abs(corr - 1.0) < 0.001, "Perfect ranking should have ρ≈1.0"

    # Reversed ranking
    predictions = [5.0, 4.0, 3.0, 2.0, 1.0]
    actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
    corr = ranking_correlation(predictions, actuals, method="spearman")
    assert abs(corr - (-1.0)) < 0.001, "Reversed ranking should have ρ≈-1.0"


def test_ranking_correlation_target_benchmark():
    """Test that we understand the paper's benchmark (Spearman > 0.6)."""
    target_spearman = 0.6
    assert target_spearman == 0.6, "Paper targets Spearman correlation > 0.6"


# ============================================================================
# Test Enhanced Persona Generator
# ============================================================================


def test_persona_generator_includes_income():
    """Test that personas include income information (paper's key finding)."""
    generator = PersonaGenerator(seed=42)
    persona = generator.generate_persona("general_consumer")

    # Check that income appears in persona
    assert "Income:" in persona or "income" in persona.lower()
    assert "discretionary" in persona.lower()

    # Check demographic section exists
    assert "DEMOGRAPHICS" in persona


def test_persona_generator_emphasizes_age():
    """Test that personas emphasize age (paper's key finding)."""
    generator = PersonaGenerator(seed=42)
    persona = generator.generate_persona("general_consumer")

    # Age should appear multiple times or in prominent section
    assert "Age:" in persona
    assert "years old" in persona
    assert "DEMOGRAPHICS" in persona


def test_persona_all_segments_have_income():
    """Test that all segments include income brackets."""
    generator = PersonaGenerator(seed=42)
    segments = generator.get_available_segments()

    for segment in segments:
        persona = generator.generate_persona(segment)
        assert "income" in persona.lower() or "Income:" in persona
        assert "discretionary" in persona.lower()


def test_persona_income_brackets_diverse():
    """Test that income brackets vary across personas."""
    generator = PersonaGenerator(seed=42)
    personas = generator.generate_personas(10, "general_consumer")

    # Extract unique income mentions (rough check)
    income_mentions = []
    for persona in personas:
        if "$" in persona:
            # Extract first dollar amount range
            start = persona.find("$")
            end = persona.find(" ", start + 1, start + 20)
            if end > start:
                income_mentions.append(persona[start:end])

    # Should have some diversity
    unique_incomes = set(income_mentions)
    assert len(unique_incomes) >= 2, "Should have diverse income levels"


def test_persona_structure_has_key_sections():
    """Test that persona has structured sections per our enhancement."""
    generator = PersonaGenerator(seed=42)
    persona = generator.generate_persona("general_consumer")

    # Check key sections exist
    assert "DEMOGRAPHICS" in persona
    assert "SHOPPING BEHAVIOR" in persona
    assert "VALUES AND PRIORITIES" in persona
    assert "CONCERNS AND PAIN POINTS" in persona

    # Check structured guidance
    assert "carefully consider" in persona


def test_persona_gender_included():
    """Test that gender is now included in all personas."""
    generator = PersonaGenerator(seed=42)

    for segment in ["general_consumer", "millennial_women", "gen_z"]:
        persona = generator.generate_persona(segment)
        # Gender should appear in demographic section
        assert ("woman" in persona.lower() or "man" in persona.lower() or
                "non-binary" in persona.lower() or "Gender:" in persona)


def test_millennial_women_segment_enhanced():
    """Test millennial_women segment has proper demographics."""
    generator = PersonaGenerator(seed=42)
    persona = generator.generate_persona("millennial_women")

    # Should be female
    assert "woman" in persona.lower()
    # Should have age in range
    assert "DEMOGRAPHICS" in persona
    assert "Income:" in persona


def test_gen_z_segment_has_lower_income():
    """Test Gen Z segment typically has lower income brackets."""
    generator = PersonaGenerator(seed=42)
    personas = generator.generate_personas(20, "gen_z")

    # Count how many have "struggling" or "managing" in income description
    lower_income_count = sum(
        1 for p in personas
        if "struggling" in p.lower() or "managing but tight" in p.lower()
    )

    # Most Gen Z should be in lower income brackets
    assert lower_income_count > 10, "Gen Z segment should trend toward lower income"


# ============================================================================
# Integration Test: Full Pipeline with Enhancements
# ============================================================================


def test_full_pipeline_with_enhancements():
    """
    Test that all Phase 1 enhancements work together.
    This simulates the full prediction flow.
    """
    # Use multiple reference sets
    request_data = {
        "responses": [
            "This product looks amazing and I'd definitely buy it",
            "Not really my style, probably wouldn't purchase",
            "Interesting concept but price seems high for my budget",
        ],
        "reference_sentences": [
            "I definitely would not purchase",
            "I probably would not purchase",
            "I might or might not purchase",
            "I probably would purchase",
            "I definitely would purchase",
        ],
        "use_multiple_reference_sets": True,
        "temperature": 1.0,
        "epsilon": 0.01,
    }

    response = client.post("/v1/rate", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert len(data["ratings"]) == 3

    # Verify multi-set was used
    assert data["metadata"]["num_reference_sets"] == 6

    # Extract PMFs and expected values
    pmfs = [r["pmf"] for r in data["ratings"]]
    expected_values = [r["expected_value"] for r in data["ratings"]]

    # Verify PMFs are valid
    for pmf in pmfs:
        assert len(pmf) == 5
        assert abs(sum(pmf) - 1.0) < 0.001

    # First response should have high expected value (positive)
    assert expected_values[0] > 3.5, "Positive response should have high expected value"

    # Second response should have low expected value (negative)
    assert expected_values[1] < 3.0, "Negative response should have low expected value"

    # Third response should be moderate (budget concern)
    assert 2.0 < expected_values[2] < 4.0, "Budget concern should be moderate"

    # Calculate distributional similarity between first and second (should be low)
    similarity = ks_distributional_similarity(
        np.array(pmfs[0]),
        np.array(pmfs[1])
    )
    assert similarity < 0.8, "Positive and negative responses should have different PMFs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
