"""OpenAI GPT-4 Vision client for evaluating ad images with personas."""

import base64
import os
from typing import Optional

from openai import AsyncOpenAI


class LLMClient:
    """Client for interacting with OpenAI GPT-4 Vision API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the LLM client.

        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            model: Model to use. Default is gpt-4o (supports vision, cheaper than gpt-4-vision-preview).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed to constructor")

        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def evaluate_ad_with_persona(
        self,
        ad_image_base64: str,
        persona_description: str,
        reference_context: str,
    ) -> str:
        """
        Evaluate an ad image from the perspective of a synthetic persona.

        Args:
            ad_image_base64: Base64-encoded image data
            persona_description: Description of the synthetic persona
            reference_context: Context about what we're measuring (e.g., purchase intent)

        Returns:
            str: Natural language response from the persona's perspective
        """
        system_prompt = """You are roleplaying as a specific customer persona.
You will see an advertisement and respond naturally from that persona's perspective,
considering their demographics, values, preferences, and purchasing behavior."""

        user_prompt = f"""{persona_description}

You're scrolling through social media and see the following advertisement:

[IMAGE SHOWN ABOVE]

{reference_context}

Consider:
1. Does this ad catch your attention?
2. Is the product relevant to your needs and interests?
3. Does it align with your values?
4. Is the price point (if shown) acceptable to you?
5. Would you click to learn more?
6. How likely are you to make a purchase?

Respond naturally and authentically from your perspective. Share your honest thoughts
about whether you'd be interested in this product and why or why not. Be specific about
what appeals to you or turns you off."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{ad_image_base64}",
                                    "detail": "high"
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                max_tokens=300,
                temperature=0.7,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI API: {str(e)}")

    async def test_connection(self) -> bool:
        """
        Test the connection to OpenAI API.

        Returns:
            bool: True if connection successful
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            return bool(response.choices[0].message.content)
        except Exception:
            return False
