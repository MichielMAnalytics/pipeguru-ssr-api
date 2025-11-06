"""Ad prediction orchestration logic."""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as po
from semantic_similarity_rating import ResponseRater

from src.core.embeddings import GeminiEmbeddings
from src.core.llm_client import LLMClient
from src.core.persona_generator import PersonaGenerator
from src.core.reference_statements import get_reference_sets


class AdPredictor:
    """Orchestrates ad performance prediction using personas, LLM, and SSR."""

    def __init__(
        self,
        embeddings_dir: Path = Path("default_embeddings"),
        max_concurrent_llm_calls: int | None = None,
    ):
        """
        Initialize the predictor with LLM client, embeddings, and persona generator.

        Args:
            embeddings_dir: Directory containing pre-computed reference embeddings
            max_concurrent_llm_calls: Maximum concurrent LLM API calls
                - Free tier: Use 3-5 (15 RPM limit)
                - Paid tier: Use 50-100 (1000 RPM limit)
                - Default: Read from MAX_CONCURRENT_LLM_CALLS env var, or 50
        """
        self.llm_client = LLMClient()
        self.embeddings = GeminiEmbeddings()
        self.persona_generator = PersonaGenerator()

        # Concurrency control - read from env or use default
        if max_concurrent_llm_calls is None:
            max_concurrent_llm_calls = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "50"))
        self.max_concurrent_llm_calls = max_concurrent_llm_calls

        # Cache for pre-computed reference embeddings
        self._reference_embeddings_cache: Dict[str, np.ndarray] = {}
        self._embeddings_dir = embeddings_dir

        # Load reference embeddings at initialization
        self._load_reference_embeddings()

    def _load_reference_embeddings(self):
        """
        Load pre-computed reference embeddings from disk.

        This eliminates the need to regenerate embeddings on every request,
        saving ~10 seconds and 30 API calls per request.
        """
        manifest_path = self._embeddings_dir / "manifest.json"

        if not manifest_path.exists():
            return

        # Load manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Verify model matches
        model_name = self.embeddings.model
        if manifest["model"] != model_name:
            return

        # Load embeddings for each reference set
        for filename in manifest["files"]:
            filepath = self._embeddings_dir / filename
            if not filepath.exists():
                self._reference_embeddings_cache.clear()
                return

            # Extract set index from filename (e.g., "..._set_0.npy" -> 0)
            set_idx = int(filename.split("_set_")[1].split(".")[0])
            cache_key = f"purchase_intent_set_{set_idx}"

            # Load numpy array
            embeddings_array = np.load(filepath)
            self._reference_embeddings_cache[cache_key] = embeddings_array

    async def analyze_creative(
        self,
        creative_base64: str,
        personas: List[str],
        reference_sentences: List[str],
        temperature: float = 1.0,
        epsilon: float = 0.01,
        use_multiple_reference_sets: bool = True,
        mime_type: str | None = None,
        brand_context: str | None = None,
        brand_familiarity_distribution: dict | str | None = None,
        brand_familiarity_seed: int | None = None,
        ad_placement: str | None = None,
    ) -> dict:
        """
        Analyze creative (image or video) with specific personas.

        Returns qualitative + quantitative feedback per persona, plus aggregates.

        Args:
            creative_base64: Base64-encoded ad creative (image or video)
            personas: List of persona descriptions
            reference_sentences: Reference sentences for Likert scale
            temperature: SSR temperature parameter
            epsilon: SSR epsilon parameter
            use_multiple_reference_sets: Use 6 reference sets and average (recommended)
            mime_type: Optional MIME type (auto-detected if not provided)
            brand_context: Optional brand context/background information
            brand_familiarity_distribution: Optional distribution (preset name or custom dict)
            brand_familiarity_seed: Optional random seed for reproducibility
            ad_placement: Optional ad placement context (e.g., 'instagram_feed', 'tiktok_fyp')

        Returns:
            dict: {
                "persona_results": [PersonaAnalysisResult],
                "aggregate": AggregateResults,
                "metadata": {...}
            }
        """
        # Auto-detect MIME type if not provided
        if mime_type is None:
            from src.utils.mime_detector import detect_mime_type_from_base64
            mime_type, media_category = detect_mime_type_from_base64(creative_base64)
        else:
            from src.utils.mime_detector import validate_mime_type
            mime_type, media_category = validate_mime_type(mime_type)

        # Step 0: Generate brand familiarity instructions per persona (if applicable)
        brand_familiarity_instructions = None
        if brand_context and brand_familiarity_distribution:
            from src.core.brand_familiarity import (
                BrandFamiliarityDistribution,
                generate_brand_familiarity_instructions,
            )

            # Parse distribution (could be preset name or custom dict)
            if isinstance(brand_familiarity_distribution, str):
                # It's a preset name
                distribution = BrandFamiliarityDistribution.get_preset(brand_familiarity_distribution)
            else:
                # It's a custom distribution - convert string keys to ints
                distribution = {int(k): v for k, v in brand_familiarity_distribution.items()}

            # Assign familiarity levels to each persona
            # Use deterministic mode if seed is None (exact percentages)
            # Use random mode if seed is provided (for reproducibility with variation)
            familiarity_levels = BrandFamiliarityDistribution.assign_levels_to_personas(
                num_personas=len(personas),
                distribution=distribution,
                seed=brand_familiarity_seed,
                deterministic=(brand_familiarity_seed is None),
            )

            # Get unique levels that need to be generated
            unique_levels = sorted(set(familiarity_levels))

            # Generate ONLY the required familiarity level instructions in ONE LLM call
            instructions_by_level = await generate_brand_familiarity_instructions(
                brand_context=brand_context,
                required_levels=unique_levels,
                llm_client=self.llm_client,
            )

            # Map each persona to their appropriate instruction
            brand_familiarity_instructions = [
                instructions_by_level.get(level, "")
                for level in familiarity_levels
            ]

        # Step 1: Get LLM evaluations (in parallel for speed)
        llm_responses = await self._get_llm_evaluations(
            ad_image_base64=creative_base64,
            personas=personas,
            mime_type=mime_type,
            brand_familiarity_instructions=brand_familiarity_instructions,
            ad_placement=ad_placement,
        )

        # Step 2: Convert to PMFs using SSR
        pmfs = self._convert_to_pmfs(
            llm_responses=llm_responses,
            reference_sentences=reference_sentences,
            temperature=temperature,
            epsilon=epsilon,
            use_multiple_reference_sets=use_multiple_reference_sets,
        )

        # Step 3: Build individual results with FULL qualitative feedback
        persona_results = []
        for i, (persona, llm_resp, pmf) in enumerate(zip(personas, llm_responses, pmfs), 1):
            most_likely = int(np.argmax(pmf) + 1)  # 1-indexed
            expected_val = sum((j + 1) * p for j, p in enumerate(pmf))
            rating_certainty = float(np.max(pmf))

            result = {
                "persona_id": i,
                "persona_description": persona,  # FULL persona, not truncated
                "qualitative_feedback": llm_resp,  # FULL response, not truncated!
                "quantitative_score": most_likely,
                "expected_value": round(expected_val, 2),
                "pmf": [round(float(p), 4) for p in pmf],
                "rating_certainty": round(rating_certainty, 4),
            }

            # Add brand familiarity level if applicable
            if brand_familiarity_instructions:
                result["brand_familiarity_level"] = familiarity_levels[i - 1]

            persona_results.append(result)

        # Step 4: Calculate aggregates
        aggregate_pmf = np.mean(pmfs, axis=0)
        avg_score = float(np.mean([r["expected_value"] for r in persona_results]))
        purchase_intent = self._calculate_purchase_intent(aggregate_pmf)
        persona_agreement = self._calculate_persona_agreement(pmfs)

        # Step 4.5: Generate qualitative summary (single LLM call)
        qualitative_summary = await self.llm_client.generate_qualitative_summary(
            persona_feedbacks=llm_responses,
            average_score=avg_score,
            purchase_intent=purchase_intent,
        )

        aggregate = {
            "average_score": round(avg_score, 2),
            "predicted_purchase_intent": round(purchase_intent, 4),
            "pmf_aggregate": [round(float(p), 4) for p in aggregate_pmf],
            "persona_agreement": round(persona_agreement, 4),
            "qualitative_summary": qualitative_summary,
        }

        # Step 5: Metadata
        metadata = {
            "num_personas": len(personas),
            "llm_calls": len(personas),
            "llm_model": "gemini-2.5-flash",
            "media_type": media_category,
            "mime_type": mime_type,
        }

        # Add brand familiarity metadata if applicable
        if brand_familiarity_instructions:
            from collections import Counter
            level_counts = Counter(familiarity_levels)
            metadata["brand_familiarity"] = {
                "enabled": True,
                "distribution_used": brand_familiarity_distribution if isinstance(brand_familiarity_distribution, str) else "custom",
                "seed": brand_familiarity_seed,
                "level_distribution": {
                    f"level_{level}": {
                        "count": level_counts[level],
                        "percentage": round(level_counts[level] / len(personas) * 100, 1),
                        "label": {
                            1: "Never heard",
                            2: "Vaguely aware",
                            3: "Familiar",
                            4: "Very familiar",
                            5: "Brand advocate"
                        }[level]
                    }
                    for level in sorted(level_counts.keys())
                }
            }
        else:
            metadata["brand_familiarity"] = {
                "enabled": False
            }

        return {
            "persona_results": persona_results,
            "aggregate": aggregate,
            "metadata": metadata,
        }

    async def _get_llm_evaluations(
        self,
        ad_image_base64: str,
        personas: List[str],
        mime_type: str,
        brand_familiarity_instructions: List[str] | None = None,
        ad_placement: str | None = None,
    ) -> List[str]:
        """
        Get LLM evaluations for all personas with rate limiting.

        The LLM is NOT shown the reference scale to avoid anchoring bias.
        It provides natural language feedback only.

        Gemini 2.5 Flash rate limits:
        - Free tier: 15 RPM, 1M TPM
        - Paid tier: 1000 RPM, 4M TPM

        Concurrency is controlled by max_concurrent_llm_calls set in __init__.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_llm_calls)

        print(f"[AdPredictor] Processing {len(personas)} personas with max {self.max_concurrent_llm_calls} concurrent calls...")

        async def eval_with_limit(persona, brand_instruction):
            async with semaphore:
                return await self.llm_client.evaluate_ad_with_persona(
                    ad_image_base64=ad_image_base64,
                    persona_description=persona,
                    mime_type=mime_type,
                    brand_familiarity_instruction=brand_instruction,
                    ad_placement=ad_placement,
                )

        # Pair each persona with its brand familiarity instruction (if any)
        if brand_familiarity_instructions:
            tasks = [
                eval_with_limit(persona, instruction)
                for persona, instruction in zip(personas, brand_familiarity_instructions)
            ]
        else:
            tasks = [eval_with_limit(persona, None) for persona in personas]

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

            # Try to load from cache first, otherwise generate embeddings
            cache_key = f"purchase_intent_set_{ref_idx}"
            if cache_key in self._reference_embeddings_cache:
                ref_embeddings = self._reference_embeddings_cache[cache_key]
                print(f"[AdPredictor]   ✓ Using cached embeddings (saved 1 API call)")
            else:
                # Generate embeddings for reference sentences (fallback)
                print(f"[AdPredictor]   Generating embeddings (cache miss)...")
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

    def _calculate_purchase_intent(self, pmf: np.ndarray) -> float:
        """
        Calculate predicted purchase intent from PMF.

        For a 5-point scale, we consider ratings 4-5 as "likely to purchase".
        """
        # Sum probabilities for top 2 ratings
        if len(pmf) >= 4:
            purchase_intent = float(np.sum(pmf[-2:]))
        else:
            # For shorter scales, take top rating
            purchase_intent = float(pmf[-1])

        return purchase_intent

    def _calculate_persona_agreement(self, pmfs: np.ndarray) -> float:
        """
        Calculate persona agreement score based on consensus across personas.

        Higher agreement = personas have more consensus on the rating.
        Lower agreement = personas have divergent opinions.

        This measures how much personas agree with each other, not individual certainty.
        """
        # Calculate standard deviation of expected values
        expected_values = [sum((i + 1) * p for i, p in enumerate(pmf)) for pmf in pmfs]
        std_dev = float(np.std(expected_values))

        # Convert to agreement (lower std = higher agreement)
        # Normalize to 0-1 scale (assuming max std ~2 for 5-point scale)
        max_std = 2.0
        agreement = max(0.0, 1.0 - (std_dev / max_std))

        return agreement

