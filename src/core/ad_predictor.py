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

        print(f"[AdPredictor] Initialized with max {self.max_concurrent_llm_calls} concurrent LLM calls")

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
            print(f"[AdPredictor] Warning: No pre-computed embeddings found at {manifest_path}")
            print(f"[AdPredictor] Run 'python scripts/generate_reference_embeddings.py' to generate them")
            print(f"[AdPredictor] Falling back to runtime embedding generation (slower)")
            return

        # Load manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Verify model matches
        model_name = self.embeddings.model
        if manifest["model"] != model_name:
            print(f"[AdPredictor] Warning: Embedding model mismatch!")
            print(f"  Expected: {model_name}")
            print(f"  Found in cache: {manifest['model']}")
            print(f"[AdPredictor] Falling back to runtime embedding generation")
            return

        # Load embeddings for each reference set
        print(f"[AdPredictor] Loading {manifest['num_sets']} pre-computed reference embeddings...")
        for filename in manifest["files"]:
            filepath = self._embeddings_dir / filename
            if not filepath.exists():
                print(f"[AdPredictor] Warning: Missing file {filename}, skipping cache")
                self._reference_embeddings_cache.clear()
                return

            # Extract set index from filename (e.g., "..._set_0.npy" -> 0)
            set_idx = int(filename.split("_set_")[1].split(".")[0])
            cache_key = f"purchase_intent_set_{set_idx}"

            # Load numpy array
            embeddings_array = np.load(filepath)
            self._reference_embeddings_cache[cache_key] = embeddings_array

        print(f"[AdPredictor] ✓ Loaded {len(self._reference_embeddings_cache)} reference embeddings from cache")
        print(f"[AdPredictor] This saves ~30 API calls per request!")

    async def analyze_creative(
        self,
        creative_base64: str,
        personas: List[str],
        reference_sentences: List[str],
        temperature: float = 1.0,
        epsilon: float = 0.01,
        use_multiple_reference_sets: bool = True,
    ) -> dict:
        """
        Analyze creative with specific personas.

        Returns qualitative + quantitative feedback per persona, plus aggregates.

        Args:
            creative_base64: Base64-encoded ad image
            personas: List of persona descriptions
            reference_sentences: Reference sentences for Likert scale
            temperature: SSR temperature parameter
            epsilon: SSR epsilon parameter
            use_multiple_reference_sets: Use 6 reference sets and average (recommended)

        Returns:
            dict: {
                "persona_results": [PersonaAnalysisResult],
                "aggregate": AggregateResults,
                "metadata": {...}
            }
        """
        # Step 1: Get LLM evaluations (in parallel for speed)
        # Note: We do NOT pass reference_context to LLM to avoid anchoring bias
        llm_responses = await self._get_llm_evaluations(
            ad_image_base64=creative_base64,
            personas=personas,
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

            persona_results.append({
                "persona_id": i,
                "persona_description": persona,  # FULL persona, not truncated
                "qualitative_feedback": llm_resp,  # FULL response, not truncated!
                "quantitative_score": most_likely,
                "expected_value": round(expected_val, 2),
                "pmf": [round(float(p), 4) for p in pmf],
                "rating_certainty": round(rating_certainty, 4),
            })

        # Step 4: Calculate aggregates
        aggregate_pmf = np.mean(pmfs, axis=0)
        avg_score = float(np.mean([r["expected_value"] for r in persona_results]))
        purchase_intent = self._calculate_purchase_intent(aggregate_pmf)
        persona_agreement = self._calculate_persona_agreement(pmfs)

        aggregate = {
            "average_score": round(avg_score, 2),
            "predicted_purchase_intent": round(purchase_intent, 4),
            "pmf_aggregate": [round(float(p), 4) for p in aggregate_pmf],
            "persona_agreement": round(persona_agreement, 4),
        }

        # Step 5: Metadata
        metadata = {
            "num_personas": len(personas),
            "llm_calls": len(personas),
            "llm_model": "gemini-2.5-flash",
        }

        return {
            "persona_results": persona_results,
            "aggregate": aggregate,
            "metadata": metadata,
        }

    async def _get_llm_evaluations(
        self, ad_image_base64: str, personas: List[str]
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

        async def eval_with_limit(persona):
            async with semaphore:
                return await self.llm_client.evaluate_ad_with_persona(
                    ad_image_base64=ad_image_base64,
                    persona_description=persona,
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

