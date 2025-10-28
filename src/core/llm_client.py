"""Google Gemini Vision client for evaluating ad images with personas."""

import base64
import os
from typing import Optional

from google import genai
from google.genai import types


class LLMClient:
    """Client for interacting with Google Gemini Vision API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: Google API key. If not provided, reads from GEMINI_API_KEY env var.
            model: Model to use. If not provided, reads from GEMINI_MODEL env var (default: gemini-2.5-flash).
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in environment or passed to constructor")

        # Read configuration from env vars or use provided values or defaults
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

        print(f"[LLM Client] Initialized with model: {self.model}")

    async def evaluate_ad_with_persona(
        self,
        ad_image_base64: str,
        persona_description: str,
    ) -> str:
        """
        Evaluate an ad image from the perspective of a synthetic persona.

        Args:
            ad_image_base64: Base64-encoded image data
            persona_description: Description of the synthetic persona

        Returns:
            str: Natural language response from the persona's perspective (qualitative feedback only)

        Note:
            The LLM is NOT shown the reference scale. It provides natural language feedback,
            which is then independently converted to quantitative scores using SSR methodology.
        """
        system_prompt = """You are roleplaying as a specific customer persona.
You will see an advertisement and respond naturally from that persona's perspective,
considering their demographics, values, preferences, and purchasing behavior."""

        user_prompt = f"""{persona_description}

You're scrolling through social media and see the following advertisement:

[IMAGE SHOWN ABOVE]

Consider:
1. Does this ad catch your attention?
2. Is the product relevant to your needs and interests?
3. Does it align with your values?
4. Is the price point (if shown) acceptable to you?
5. Would you click to learn more?
6. How likely are you to make a purchase?

Respond naturally and authentically from your perspective. Share your honest thoughts
about whether you'd be interested in this product and why or why not. Be specific about
what appeals to you or turns you off.

IMPORTANT: Do NOT state a numeric rating or score. Just describe your thoughts and feelings naturally."""

        try:
            # Convert base64 to bytes for Gemini
            image_bytes = base64.b64decode(ad_image_base64)

            # Create the content parts
            contents = [
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=image_bytes
                    )
                ),
                types.Part(text=user_prompt)
            ]

            # Generate content with system instruction (ASYNC!)
            # Disable automatic function calling (AFC) to improve performance
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                )
            )

            content = response.text

            # Handle empty responses
            if not content:

                # Check if it was cut off by max tokens
                if hasattr(response, 'candidates') and response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason and 'MAX_TOKENS' in str(finish_reason):
                        raise RuntimeError(
                            "Response truncated due to MAX_TOKENS limit. "
                            "This shouldn't happen with default settings. Check prompt length."
                        )

                raise RuntimeError("Empty response from Gemini API")

            return content

        except Exception as e:
            raise RuntimeError(f"Error calling Gemini API: {str(e)}")

    async def generate_text(self, prompt: str) -> str:
        """
        Generate text using Gemini (for persona generation, etc.).

        Args:
            prompt: Text prompt for generation

        Returns:
            str: Generated text

        Raises:
            RuntimeError: If generation fails
        """
        try:
            # Use the same async pattern as evaluate_ad_with_persona
            # Disable automatic function calling (AFC) to improve performance
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )

            content = response.text

            if not content:
                raise RuntimeError("Empty response from Gemini API")

            return content

        except Exception as e:
            raise RuntimeError(f"Error generating text with Gemini: {str(e)}")

    async def test_connection(self) -> bool:
        """
        Test the connection to Gemini API.

        Returns:
            bool: True if connection successful
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents="Hello",
            )
            return bool(response.text)
        except Exception:
            return False
