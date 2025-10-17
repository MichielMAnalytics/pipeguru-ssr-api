"""Ad prediction orchestration logic."""

import asyncio
from typing import List, Tuple

import numpy as np
import polars as po
from semantic_similarity_rating import ResponseRater

from src.core.embeddings import GeminiEmbeddings
from src.core.llm_client import LLMClient
from src.core.persona_generator import PersonaGenerator
from src.core.reference_statements import get_reference_sets


class AdPredictor:
    """Orchestrates ad performance prediction using personas, LLM, and SSR."""

    def __init__(self):
        """Initialize the predictor with LLM client, embeddings, and persona generator."""
        self.llm_client = LLMClient()
        self.embeddings = GeminiEmbeddings()
        self.persona_generator = PersonaGenerator()

    async def predict_ad_performance(
        self,
        ad_image_base64: str,
        num_personas: int,
        segment: str,
        reference_sentences: List[str],
        temperature: float = 1.0,
        epsilon: float = 0.01,
        use_multiple_reference_sets: bool = True,
    ) -> dict:
        """
        Predict ad performance using synthetic personas and SSR.

        Args:
            ad_image_base64: Base64-encoded ad image
            num_personas: Number of personas to generate
            segment: Persona segment (general_consumer, millennial_women, gen_z)
            reference_sentences: Reference sentences for Likert scale
            temperature: SSR temperature parameter
            epsilon: SSR epsilon parameter
            use_multiple_reference_sets: Use 6 reference sets and average (recommended)

        Returns:
            dict: Prediction results with PMF, conversion rate, individual results, cost
        """
        # Step 1: Generate personas
        personas = self.persona_generator.generate_personas(
            num_personas=num_personas, segment=segment
        )

        # Step 2: Get LLM evaluations (in parallel for speed)
        reference_context = self._build_reference_context(reference_sentences)

        llm_responses = await self._get_llm_evaluations(
            ad_image_base64=ad_image_base64,
            personas=personas,
            reference_context=reference_context,
        )

        # DEBUG: Print first 3 LLM responses to diagnose identical predictions
        print(f"\n[DEBUG] First 3 LLM responses:")
        for i, resp in enumerate(llm_responses[:3], 1):
            print(f"  Response {i}: {resp[:200] if resp else 'NONE/EMPTY'}...")
        print(f"[DEBUG] All responses identical: {len(set(llm_responses)) == 1}")
        print(f"[DEBUG] Number of unique responses: {len(set(llm_responses))}")

        # DEBUG: Check first 2 personas
        print(f"\n[DEBUG] First 2 personas:")
        for i, persona in enumerate(personas[:2], 1):
            print(f"  Persona {i}: {persona[:200]}...")
        print()

        # Step 3: Convert to PMFs using SSR
        pmfs = self._convert_to_pmfs(
            llm_responses=llm_responses,
            reference_sentences=reference_sentences,
            temperature=temperature,
            epsilon=epsilon,
            use_multiple_reference_sets=use_multiple_reference_sets,
        )

        # Step 4: Calculate aggregate metrics
        aggregate_pmf = np.mean(pmfs, axis=0)
        predicted_conversion_rate = self._calculate_conversion_rate(aggregate_pmf)
        confidence = self._calculate_confidence(pmfs)

        # Step 5: Build individual results
        individual_results = []
        for i, (persona, llm_resp, pmf) in enumerate(
            zip(personas, llm_responses, pmfs), 1
        ):
            individual_results.append(
                {
                    "persona_id": i,
                    "persona_description": persona[:100] + "...",  # Truncate for response size
                    "llm_response": llm_resp[:200] + "...",  # Truncate
                    "pmf": [round(float(p), 4) for p in pmf],
                    "expected_value": round(
                        float(sum((i + 1) * p for i, p in enumerate(pmf))), 2
                    ),
                }
            )

        # Step 6: Calculate cost
        cost = self._calculate_cost(num_personas)

        return {
            "predicted_conversion_rate": round(predicted_conversion_rate, 4),
            "confidence": round(confidence, 4),
            "pmf_aggregate": [round(float(p), 4) for p in aggregate_pmf],
            "expected_value": round(
                float(sum((i + 1) * p for i, p in enumerate(aggregate_pmf))), 2
            ),
            "persona_results": individual_results,
            "cost": cost,
            "metadata": {
                "num_personas": num_personas,
                "segment": segment,
                "num_reference_sentences": len(reference_sentences),
            },
        }

    async def _get_llm_evaluations(
        self, ad_image_base64: str, personas: List[str], reference_context: str
    ) -> List[str]:
        """Get LLM evaluations for all personas with rate limiting."""
        # Limit concurrent requests to avoid rate limits (200K tokens/min)
        # Each request ~1200 tokens, so 5 concurrent = ~6K tokens, safe margin
        semaphore = asyncio.Semaphore(5)

        async def eval_with_limit(persona):
            async with semaphore:
                return await self.llm_client.evaluate_ad_with_persona(
                    ad_image_base64=ad_image_base64,
                    persona_description=persona,
                    reference_context=reference_context,
                )

        tasks = [eval_with_limit(persona) for persona in personas]
        responses = await asyncio.gather(*tasks)
        return responses

    def _convert_to_pmfs(
        self,
        llm_responses: List[str],
        reference_sentences: List[str],
        temperature: float,
        epsilon: float,
        use_multiple_reference_sets: bool = True,
    ) -> np.ndarray:
        """
        Convert LLM responses to PMFs using SSR with Gemini embeddings.

        Args:
            llm_responses: LLM text responses to convert
            reference_sentences: Reference sentences for Likert scale (used when single set)
            temperature: Temperature parameter
            epsilon: Epsilon parameter
            use_multiple_reference_sets: Whether to use 6 sets and average (recommended)

        Returns:
            numpy array of PMFs
        """
        num_points = len(reference_sentences)

        # Determine which reference sets to use
        if use_multiple_reference_sets:
            # Use 6 reference sets from paper for better stability (KS sim ~0.88 vs ~0.72)
            reference_sentences_list = get_reference_sets("purchase_intent")
        else:
            # Use provided reference sentences only
            reference_sentences_list = [reference_sentences]

        # Generate embeddings for LLM responses (once, reused for all reference sets)
        print(f"[AdPredictor] Generating embeddings for {len(llm_responses)} LLM responses...")
        llm_response_embeddings = self.embeddings.encode(llm_responses)

        # Collect PMFs from all reference sets
        all_pmfs_per_response = [[] for _ in llm_responses]

        for ref_idx, ref_sentences in enumerate(reference_sentences_list):
            print(f"[AdPredictor] Processing reference set {ref_idx + 1}/{len(reference_sentences_list)}...")

            # Generate embeddings for reference sentences
            ref_embeddings = self.embeddings.encode(ref_sentences)

            # Create reference DataFrame with embeddings
            df = po.DataFrame(
                {
                    "id": [f"set_{ref_idx}"] * num_points,
                    "int_response": list(range(1, num_points + 1)),
                    "sentence": ref_sentences,
                    "embedding": ref_embeddings.tolist(),  # Add embeddings column
                }
            )

            # Initialize ResponseRater in embedding mode (no HuggingFace model needed!)
            rater = ResponseRater(df, embeddings_column="embedding")

            # Get PMFs for this reference set (passing embeddings, not text)
            pmfs_for_set = rater.get_response_pmfs(
                reference_set_id=f"set_{ref_idx}",
                llm_responses=llm_response_embeddings,  # Pass embeddings instead of text
                temperature=temperature,
                epsilon=epsilon,
            )

            # Collect PMFs for each response
            for response_idx, pmf in enumerate(pmfs_for_set):
                all_pmfs_per_response[response_idx].append(pmf)

        # Average PMFs across all reference sets for each response
        averaged_pmfs = np.array([np.mean(pmf_list, axis=0) for pmf_list in all_pmfs_per_response])

        return averaged_pmfs

    def _build_reference_context(self, reference_sentences: List[str]) -> str:
        """Build reference context string for LLM prompt."""
        context = f"We're measuring purchase intent on a {len(reference_sentences)}-point scale:\n"
        for i, sentence in enumerate(reference_sentences, 1):
            context += f"{i} = {sentence}\n"
        return context

    def _calculate_conversion_rate(self, pmf: np.ndarray) -> float:
        """
        Calculate predicted conversion rate from PMF.

        For a 5-point scale, we consider ratings 4-5 as "likely to convert".
        """
        # Sum probabilities for top 2 ratings
        if len(pmf) >= 4:
            conversion_rate = float(np.sum(pmf[-2:]))
        else:
            # For shorter scales, take top rating
            conversion_rate = float(pmf[-1])

        return conversion_rate

    def _calculate_confidence(self, pmfs: np.ndarray) -> float:
        """
        Calculate confidence score based on agreement across personas.

        Higher confidence = personas agree more on the rating.
        """
        # Calculate standard deviation of expected values
        expected_values = [sum((i + 1) * p for i, p in enumerate(pmf)) for pmf in pmfs]
        std_dev = float(np.std(expected_values))

        # Convert to confidence (lower std = higher confidence)
        # Normalize to 0-1 scale (assuming max std ~2 for 5-point scale)
        max_std = 2.0
        confidence = max(0.0, 1.0 - (std_dev / max_std))

        return confidence

    def _calculate_cost(self, num_personas: int) -> dict:
        """Calculate estimated cost for the prediction."""
        # Gemini 2.5 Flash costs approximately $0.001-0.002 per call
        cost_per_call = 0.0015  # Average estimate
        total_cost = num_personas * cost_per_call

        return {
            "llm_calls": num_personas,
            "estimated_cost_usd": round(total_cost, 2),
        }
