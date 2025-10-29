"""Brand familiarity context generator for persona instructions."""

import random
from typing import Dict, List, Literal, Optional

from src.core.llm_client import LLMClient


class BrandFamiliarityDistribution:
    """Manages distribution of brand familiarity levels across personas."""

    # Preset distributions for common scenarios
    PRESETS: Dict[str, Dict[int, float]] = {
        "uniform": {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        "new_brand": {1: 0.70, 2: 0.20, 3: 0.08, 4: 0.02, 5: 0.0},
        "emerging_brand": {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.08, 5: 0.02},
        "established_brand": {1: 0.10, 2: 0.20, 3: 0.40, 4: 0.20, 5: 0.10},
        "popular_brand": {1: 0.05, 2: 0.15, 3: 0.30, 4: 0.35, 5: 0.15},
        "cult_brand": {1: 0.50, 2: 0.20, 3: 0.10, 4: 0.10, 5: 0.10},
    }

    @staticmethod
    def validate_distribution(distribution: Dict[int, float]) -> None:
        """
        Validate that a distribution is properly formatted.

        Args:
            distribution: Dictionary mapping familiarity level (1-5) to percentage (0-1)

        Raises:
            ValueError: If distribution is invalid
        """
        # Check all keys are 1-5
        for level in distribution.keys():
            if level not in [1, 2, 3, 4, 5]:
                raise ValueError(f"Invalid familiarity level: {level}. Must be 1-5.")

        # Check percentages sum to ~1.0 (allow 0.99-1.01 for rounding)
        total = sum(distribution.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Distribution percentages must sum to 1.0 (got {total:.2f})")

        # Check all percentages are 0-1
        for level, pct in distribution.items():
            if not (0 <= pct <= 1):
                raise ValueError(f"Invalid percentage {pct} for level {level}. Must be 0-1.")

    @staticmethod
    def get_preset(preset_name: str) -> Dict[int, float]:
        """
        Get a preset distribution by name.

        Args:
            preset_name: Name of preset (uniform, new_brand, emerging_brand, established_brand, popular_brand, cult_brand)

        Returns:
            Distribution dictionary

        Raises:
            ValueError: If preset name not found
        """
        if preset_name not in BrandFamiliarityDistribution.PRESETS:
            available = ", ".join(BrandFamiliarityDistribution.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        return BrandFamiliarityDistribution.PRESETS[preset_name].copy()

    @staticmethod
    def assign_levels_to_personas(
        num_personas: int,
        distribution: Dict[int, float],
        seed: int | None = None,
        deterministic: bool = False,
    ) -> List[int]:
        """
        Assign familiarity levels to personas based on distribution.

        Args:
            num_personas: Number of personas to assign levels to
            distribution: Dictionary mapping familiarity level (1-5) to percentage (0-1)
            seed: Optional random seed for reproducibility (ignored if deterministic=True)
            deterministic: If True, assigns exact percentages without randomness

        Returns:
            List of familiarity levels (one per persona)

        Example:
            >>> # Random assignment
            >>> assign_levels_to_personas(10, {1: 0.3, 2: 0.5, 3: 0.2})
            [2, 1, 2, 3, 2, 2, 1, 2, 1, 2]  # ~3 at level 1, ~5 at level 2, ~2 at level 3

            >>> # Deterministic assignment
            >>> assign_levels_to_personas(10, {1: 0.3, 2: 0.5, 3: 0.2}, deterministic=True)
            [1, 1, 1, 2, 2, 2, 2, 2, 3, 3]  # Exactly 3, 5, 2
        """
        BrandFamiliarityDistribution.validate_distribution(distribution)

        if deterministic:
            # Deterministic assignment: assign exact percentages
            assigned_levels = []
            sorted_levels = sorted(distribution.keys())

            for level in sorted_levels:
                count = round(num_personas * distribution[level])
                assigned_levels.extend([level] * count)

            # Handle rounding issues - adjust to exactly num_personas
            while len(assigned_levels) < num_personas:
                # Add one more of the most common level
                assigned_levels.append(sorted_levels[0])
            while len(assigned_levels) > num_personas:
                # Remove one of the most common level
                assigned_levels.pop()

            return assigned_levels
        else:
            # Random assignment with optional seed
            if seed is not None:
                random.seed(seed)

            # Create weighted population
            levels = []
            weights = []
            for level in sorted(distribution.keys()):
                levels.append(level)
                weights.append(distribution[level])

            # Assign levels based on distribution
            assigned_levels = random.choices(levels, weights=weights, k=num_personas)

            return assigned_levels


async def generate_brand_familiarity_instructions(
    brand_context: str,
    required_levels: List[int],
    llm_client: Optional[LLMClient] = None,
) -> Dict[int, str]:
    """
    Generate brand familiarity instructions for specified levels.

    Level 1 is hardcoded (no brand info). Levels 2-5 are generated via LLM in a single call.

    Args:
        brand_context: Comprehensive brand information (identity, values, products, etc.)
        required_levels: List of familiarity levels to generate (e.g., [1, 2, 3])
        llm_client: Optional LLMClient instance (creates new one if not provided)

    Returns:
        Dict mapping familiarity level to instruction text

    Example:
        >>> context = "Upfront is a Dutch sports nutrition company..."
        >>> instructions = await generate_brand_familiarity_instructions(context, [1, 2, 3])
        >>> # Returns {1: "...", 2: "...", 3: "..."}
    """
    if not brand_context or not brand_context.strip():
        return {}

    if not required_levels:
        return {}

    # Validate levels
    for level in required_levels:
        if level < 1 or level > 5:
            raise ValueError(f"Invalid familiarity level {level}. Must be 1-5.")

    # Sort and deduplicate
    required_levels = sorted(set(required_levels))

    instructions = {}

    # Level 1 is always the same - hardcode it
    LEVEL_1_INSTRUCTION = """You have never heard of this brand before. This is your first time seeing anything from them. You don't recognize the name, logo, or any of their products. You have no prior opinions or knowledge about them."""

    if 1 in required_levels:
        instructions[1] = LEVEL_1_INSTRUCTION
        print(f"[Brand Familiarity] Level 1 instruction: hardcoded (no LLM call needed)")

    # Filter out level 1 from LLM generation
    levels_needing_llm = [l for l in required_levels if l > 1]

    # If only level 1 was requested, return immediately
    if not levels_needing_llm:
        return instructions

    # Create LLM client if not provided
    if llm_client is None:
        llm_client = LLMClient()

    # Define level descriptions
    level_descriptions = {
        1: "NEVER heard of this brand - zero knowledge, complete first-time exposure",
        2: "VAGUELY AWARE - seen once or twice, superficial recognition only",
        3: "FAMILIAR - knows what they do, general understanding of positioning",
        4: "VERY FAMILIAR - has engaged with brand, purchased before, clear opinions",
        5: "BRAND ADVOCATE - deeply loyal, extensive knowledge, emotional connection"
    }

    level_rules = {
        1: "Include NO brand information. They evaluate purely on the ad creative itself.",
        2: "Include only 1-2 surface-level facts (e.g., 'seen their products at stores', 'know they sell X category')",
        3: "Include basic brand identity, what they offer, general positioning (20-30% of context)",
        4: "Include detailed knowledge - products, values, personal experience mentions (60-70% of context)",
        5: "Include full context - they know everything and are emotionally invested (100% of context)"
    }

    # Build dynamic prompt for only levels needing LLM (2-5)
    levels_list = ", ".join([f"{l}" for l in levels_needing_llm])

    prompt = f"""You are helping create realistic customer personas for ad testing. Given comprehensive brand information, generate brand familiarity contexts for ONLY the following levels: {levels_list}

FULL BRAND CONTEXT:
{brand_context}

REQUIRED LEVELS TO GENERATE:
"""

    for level in levels_needing_llm:
        prompt += f"- Level {level}: {level_descriptions[level]}\n"

    prompt += f"""
CRITICAL RULES TO PREVENT DATA LEAKAGE:
"""

    for level in levels_needing_llm:
        prompt += f"- Level {level}: {level_rules[level]}\n"

    prompt += f"""
FORMAT YOUR RESPONSE WITH THESE EXACT SECTIONS (one per required level):

"""

    # Add template for each level needing LLM (skip level 1 - it's hardcoded)
    for level in levels_needing_llm:
        prompt += f"""=== LEVEL {level} ===
[Write a natural, factual statement about what this person knows about the brand at level {level}. Write in second person ("You...") as if describing their actual situation. Do NOT include prescriptive instructions like "your response should reflect" - just state what they know/don't know.]

"""

    prompt += "Generate the required levels now:"

    # Call LLM to generate contexts for levels 2-5
    print(f"[Brand Familiarity] Generating levels {levels_needing_llm} via LLM...")
    response = await llm_client.generate_text(prompt)

    # Parse the response into individual levels
    sections = response.split("=== LEVEL ")

    for section in sections[1:]:  # Skip first empty split
        try:
            # Extract level number and content
            level_str, content = section.split(" ===\n", 1)
            level = int(level_str.strip())
            instructions[level] = content.strip()
        except (ValueError, IndexError) as e:
            print(f"[Brand Familiarity] Warning: Failed to parse level from section: {e}")
            continue

    # Validate we got all required levels (including hardcoded level 1)
    if len(instructions) != len(required_levels):
        print(f"[Brand Familiarity] Warning: Expected {len(required_levels)} levels, got {len(instructions)}")
        missing = set(required_levels) - set(instructions.keys())
        if missing:
            print(f"[Brand Familiarity] Missing levels: {missing}")

    print(f"[Brand Familiarity] ✓ Generated instructions for levels: {sorted(instructions.keys())}")
    return instructions
