"""Tests for LLM client."""

import os

import pytest

from src.core.llm_client import LLMClient


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="Gemini API key not set"
)
async def test_llm_client_initialization():
    """Test LLM client can be initialized."""
    client = LLMClient()
    assert client.api_key is not None
    assert client.model == "gemini-2.5-flash"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="Gemini API key not set"
)
async def test_llm_connection():
    """Test connection to Gemini API."""
    client = LLMClient()
    result = await client.test_connection()
    assert result is True


def test_llm_client_requires_api_key():
    """Test that LLMClient raises error without API key."""
    # Temporarily remove API key from environment
    original_key = os.environ.pop("GEMINI_API_KEY", None)

    try:
        with pytest.raises(ValueError, match="GEMINI_API_KEY must be set"):
            LLMClient()
    finally:
        # Restore original key if it existed
        if original_key:
            os.environ["GEMINI_API_KEY"] = original_key


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="Gemini API key not set"
)
async def test_evaluate_ad_with_persona():
    """Test evaluating an ad with a persona (using a simple test image)."""
    # This test requires an actual API call, so it's skipped unless API key is set
    # In a real scenario, you'd use a mock image or test image

    client = LLMClient()

    # Create a simple test persona
    persona = """You are Emma, a 28-year-old marketing professional living in Chicago.
    You value sustainable products and are willing to pay a premium for quality.
    You shop online frequently and care about fast shipping."""

    # For testing, we'd need a real image. This is just a structure test.
    # In practice, you'd use a fixture image or mock the API call.

    # This would be a real test with an actual image:
    # response = await client.evaluate_ad_with_persona(
    #     ad_image_base64="<base64_image>",
    #     persona_description=persona,
    #     reference_context="We're measuring purchase intent from 1-5."
    # )
    # assert isinstance(response, str)
    # assert len(response) > 0

    # For now, just verify the method exists
    assert hasattr(client, "evaluate_ad_with_persona")
