"""Test API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "PipeGuru SSR API"
    assert "version" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"


def test_rate_endpoint_success():
    """Test successful rating conversion."""
    request_data = {
        "responses": [
            "I really like this product",
            "Not sure about this",
        ],
        "reference_sentences": [
            "I definitely would not purchase",
            "I probably would not purchase",
            "I might or might not purchase",
            "I probably would purchase",
            "I definitely would purchase",
        ],
        "temperature": 1.0,
        "epsilon": 0.01,
    }

    response = client.post("/v1/rate", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert len(data["ratings"]) == 2

    # Check first rating
    rating = data["ratings"][0]
    assert rating["response"] == "I really like this product"
    assert len(rating["pmf"]) == 5
    assert abs(sum(rating["pmf"]) - 1.0) < 0.001  # PMF sums to 1
    assert 1 <= rating["most_likely_rating"] <= 5
    assert 0 <= rating["confidence"] <= 1
    assert rating["expected_value"] > 0

    # Check metadata
    assert data["metadata"]["num_responses"] == 2
    assert data["metadata"]["num_reference_sentences"] == 5
    assert "processing_time_ms" in data["metadata"]


def test_rate_endpoint_validation_error():
    """Test rating endpoint with invalid input."""
    request_data = {
        "responses": [],  # Empty list should fail validation
        "reference_sentences": ["sentence1"],
    }

    response = client.post("/v1/rate", json=request_data)
    assert response.status_code == 422  # Validation error


def test_rate_endpoint_with_single_response():
    """Test rating with single response."""
    request_data = {
        "responses": ["I love this product"],
        "reference_sentences": [
            "I definitely would not purchase",
            "I probably would not purchase",
            "I might or might not purchase",
            "I probably would purchase",
            "I definitely would purchase",
        ],
    }

    response = client.post("/v1/rate", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert len(data["ratings"]) == 1
    assert data["ratings"][0]["expected_value"] > 2.5  # Positive sentiment (lenient threshold)


def test_rate_endpoint_with_temperature():
    """Test rating with different temperature values."""
    request_data = {
        "responses": ["I like this"],
        "reference_sentences": [
            "Strongly disagree",
            "Disagree",
            "Neutral",
            "Agree",
            "Strongly agree",
        ],
        "temperature": 0.1,  # Sharp distribution
    }

    response = client.post("/v1/rate", json=request_data)
    assert response.status_code == 200

    data = response.json()
    # With low temperature, confidence should be high
    assert data["ratings"][0]["confidence"] > 0.5
