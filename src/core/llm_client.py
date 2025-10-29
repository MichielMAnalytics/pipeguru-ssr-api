"""Google Gemini Vision client for evaluating ad images with personas."""

import base64
import os
from typing import Optional

from google import genai
from google.genai import types

from src.core.placement_context import get_placement_context


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
        mime_type: str = "image/jpeg",
        brand_familiarity_instruction: Optional[str] = None,
        ad_placement: Optional[str] = None,
    ) -> str:
        """
        Evaluate an ad creative (image or video) from the perspective of a synthetic persona.

        Args:
            ad_image_base64: Base64-encoded image or video data
            persona_description: Description of the synthetic persona
            mime_type: MIME type of the creative (e.g., 'image/jpeg', 'video/mp4')
            brand_familiarity_instruction: Optional instruction about brand familiarity level
            ad_placement: Optional ad placement context (e.g., 'instagram_feed', 'tiktok_fyp')

        Returns:
            str: Natural language response from the persona's perspective (qualitative feedback only)

        Note:
            The LLM is NOT shown the reference scale. It provides natural language feedback,
            which is then independently converted to quantitative scores using SSR methodology.
        """
        # Determine media type for contextual prompting
        is_video = mime_type.startswith("video/")
        media_label = "video advertisement" if is_video else "advertisement"
        media_context = "[VIDEO SHOWN ABOVE]" if is_video else "[IMAGE SHOWN ABOVE]"

        system_prompt = """You are roleplaying as a specific customer persona.
You will see an advertisement and respond naturally from that persona's perspective,
considering their demographics, values, preferences, and purchasing behavior."""

        # Build user prompt with optional brand familiarity context
        user_prompt_parts = []

        # Add brand familiarity instruction first (if provided)
        if brand_familiarity_instruction:
            user_prompt_parts.append(brand_familiarity_instruction.strip())
            user_prompt_parts.append("\n---\n")

        # Add persona description
        user_prompt_parts.append(f"{persona_description}\n")

        # Add scenario with placement context
        placement_context = get_placement_context(ad_placement)
        user_prompt_parts.append(placement_context)
        user_prompt_parts.append(f" You see the following {media_label}:")

        user_prompt = "".join(user_prompt_parts)

        user_prompt += f"""

{media_context}

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

        # Log the full prompt for debugging
        debug_prompts = os.getenv("DEBUG_PROMPTS", "false").lower() == "true"

        if brand_familiarity_instruction:
            print(f"[LLM Client] ✓ Brand familiarity context included (length: {len(brand_familiarity_instruction)} chars)")

            if debug_prompts:
                print("\n" + "="*80)
                print("FULL BRAND FAMILIARITY INSTRUCTION:")
                print("="*80)
                print(brand_familiarity_instruction)
                print("="*80 + "\n")
            else:
                # Log first 300 chars of brand instruction to verify it's working
                preview = brand_familiarity_instruction[:300].replace('\n', ' ')
                print(f"[LLM Client]   Preview: {preview}...")
        else:
            print(f"[LLM Client] No brand familiarity context")

        print(f"[LLM Client] User prompt length: {len(user_prompt)} chars")

        if debug_prompts:
            print("\n" + "="*80)
            print("FULL USER PROMPT:")
            print("="*80)
            print(user_prompt)
            print("="*80 + "\n")

        try:
            # Convert base64 to bytes for Gemini
            image_bytes = base64.b64decode(ad_image_base64)

            # Create the content parts
            contents = [
                types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
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

    async def generate_qualitative_summary(
        self,
        persona_feedbacks: list[str],
        average_score: float,
        purchase_intent: float,
    ) -> str:
        """
        Generate a concise qualitative summary from all persona feedback.

        This is a single LLM call that synthesizes all persona responses
        into a 2-3 sentence summary highlighting common themes.

        Args:
            persona_feedbacks: List of qualitative feedback from all personas
            average_score: Average score across personas (1-5)
            purchase_intent: Predicted purchase intent (0-1)

        Returns:
            str: Concise summary (2-3 sentences)
        """
        # Truncate individual feedbacks to save tokens (keep first 150 chars each)
        truncated_feedbacks = [fb[:150] + "..." if len(fb) > 150 else fb for fb in persona_feedbacks]

        # Limit to max 20 samples for efficiency if there are many personas
        if len(truncated_feedbacks) > 20:
            # Sample evenly across the list
            step = len(truncated_feedbacks) // 20
            truncated_feedbacks = [truncated_feedbacks[i] for i in range(0, len(truncated_feedbacks), step)][:20]

        # Join with numbering for clarity
        feedbacks_text = "\n".join(f"{i+1}. {fb}" for i, fb in enumerate(truncated_feedbacks))

        prompt = f"""You are analyzing customer feedback for an advertisement.

Below are {len(persona_feedbacks)} persona responses (sample shown):

{feedbacks_text}

Quantitative metrics:
- Average score: {average_score:.1f}/5
- Purchase intent: {purchase_intent:.1%}

Write a 2-3 sentence summary that captures:
1. Common themes (what most personas agree on)
2. Key strengths (what works well)
3. Main concerns or weaknesses (if any)

Be concise and actionable. Focus on patterns, not individual opinions.

Summary:"""

        try:
            response_text = await self.generate_text(prompt)
            # Clean up the response (remove any leading "Summary:" prefix)
            summary = response_text.strip()
            if summary.lower().startswith("summary:"):
                summary = summary[8:].strip()
            return summary
        except Exception as e:
            # Fallback to a simple summary if LLM fails
            return f"Analysis based on {len(persona_feedbacks)} personas. Average score: {average_score:.1f}/5. Purchase intent: {purchase_intent:.1%}."

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
