"""
Reference statement sets for Semantic Similarity Rating (SSR).

Based on the paper "LLMs Reproduce Human Purchase Intent via Semantic Similarity
Elicitation of Likert Ratings" which uses 6 reference sets and averages the PMFs
for improved stability (KS similarity ~0.88 vs ~0.72 for single set).

Each set contains 5 statements corresponding to a 5-point Likert scale for purchase intent.
"""

from typing import List

# Reference Set 1: Direct likelihood statements
REFERENCE_SET_1 = [
    "It's rather unlikely I'd buy it.",
    "I probably wouldn't buy it.",
    "I might or might not buy it.",
    "I'd probably buy it.",
    "It's very likely I'd buy it."
]

# Reference Set 2: Definitive statements
REFERENCE_SET_2 = [
    "I definitely would not purchase this.",
    "I would not purchase this.",
    "I'm unsure if I would purchase this.",
    "I would purchase this.",
    "I definitely would purchase this."
]

# Reference Set 3: Interest-based statements
REFERENCE_SET_3 = [
    "I'm not interested in purchasing this at all.",
    "I'm not very interested in purchasing this.",
    "I have some interest in purchasing this.",
    "I'm quite interested in purchasing this.",
    "I'm very interested in purchasing this."
]

# Reference Set 4: Action-oriented statements
REFERENCE_SET_4 = [
    "I would never buy this product.",
    "I don't think I'd buy this product.",
    "I could see myself buying this product.",
    "I would likely buy this product.",
    "I would absolutely buy this product."
]

# Reference Set 5: Probability-focused statements
REFERENCE_SET_5 = [
    "There's no chance I'd purchase this.",
    "There's a low chance I'd purchase this.",
    "There's a moderate chance I'd purchase this.",
    "There's a good chance I'd purchase this.",
    "There's a very high chance I'd purchase this."
]

# Reference Set 6: Conversational statements
REFERENCE_SET_6 = [
    "No, I wouldn't buy this.",
    "Probably not, doesn't seem right for me.",
    "Maybe, I'd have to think about it.",
    "Yes, I think I'd buy this.",
    "Absolutely, I'd definitely buy this."
]

# All reference sets
ALL_REFERENCE_SETS = [
    REFERENCE_SET_1,
    REFERENCE_SET_2,
    REFERENCE_SET_3,
    REFERENCE_SET_4,
    REFERENCE_SET_5,
    REFERENCE_SET_6
]


def get_reference_sets(question_type: str = "purchase_intent") -> List[List[str]]:
    """
    Get reference statement sets for a given question type.

    Args:
        question_type: Type of question. Currently only "purchase_intent" is supported.

    Returns:
        List of reference statement sets, where each set contains 5 statements
        corresponding to ratings 1-5 on a Likert scale.

    Raises:
        ValueError: If question_type is not supported.
    """
    if question_type == "purchase_intent":
        return ALL_REFERENCE_SETS
    else:
        raise ValueError(f"Unsupported question type: {question_type}")


def get_single_reference_set(set_index: int = 0, question_type: str = "purchase_intent") -> List[str]:
    """
    Get a specific reference statement set by index.

    Args:
        set_index: Index of the reference set (0-5).
        question_type: Type of question. Currently only "purchase_intent" is supported.

    Returns:
        List of 5 reference statements for the specified set.

    Raises:
        ValueError: If set_index is out of range or question_type is not supported.
    """
    sets = get_reference_sets(question_type)
    if not 0 <= set_index < len(sets):
        raise ValueError(f"set_index must be between 0 and {len(sets) - 1}")
    return sets[set_index]
