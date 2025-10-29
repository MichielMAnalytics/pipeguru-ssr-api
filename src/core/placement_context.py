"""Ad placement context configurations for different platforms and placements."""

from typing import Dict, Literal

# Valid placement options
PlacementType = Literal[
    "instagram_feed",
    "instagram_stories",
    "instagram_reels",
    "instagram_explore",
    "tiktok_fyp",
    "tiktok_following",
    "google_search",
    "google_display",
    "google_youtube",
    "google_shopping",
]

# Mapping of placement to user context description
PLACEMENT_CONTEXTS: Dict[str, str] = {
    # Instagram placements
    "instagram_feed": "You're scrolling through your Instagram feed, casually browsing posts from friends and accounts you follow.",
    "instagram_stories": "You're tapping through Instagram Stories, quickly viewing short-lived content from people you follow.",
    "instagram_reels": "You're watching Instagram Reels, swiping through short entertaining videos.",
    "instagram_explore": "You're browsing Instagram's Explore page, discovering new content and accounts.",

    # TikTok placements
    "tiktok_fyp": "You're scrolling through TikTok's For You Page, watching short videos curated for you.",
    "tiktok_following": "You're watching videos from creators you follow on TikTok.",

    # Google placements
    "google_search": "You just searched on Google and are reviewing the search results.",
    "google_display": "You're browsing a website when you notice this display ad.",
    "google_youtube": "You're about to watch a YouTube video when this ad starts playing.",
    "google_shopping": "You're comparing products on Google Shopping.",
}

# Default context when no placement is specified
DEFAULT_CONTEXT = "You're scrolling through social media."


def get_placement_context(placement: str | None) -> str:
    """
    Get the user context description for a given ad placement.

    Args:
        placement: Ad placement identifier (e.g., "instagram_feed", "tiktok_fyp")

    Returns:
        User context description string

    Example:
        >>> get_placement_context("tiktok_fyp")
        "You're scrolling through TikTok's For You Page, watching short videos curated for you."

        >>> get_placement_context(None)
        "You're browsing online and see the following advertisement:"
    """
    if placement is None:
        return DEFAULT_CONTEXT

    return PLACEMENT_CONTEXTS.get(placement, DEFAULT_CONTEXT)


def validate_placement(placement: str) -> bool:
    """
    Validate if a placement string is supported.

    Args:
        placement: Ad placement identifier

    Returns:
        True if valid, False otherwise

    Example:
        >>> validate_placement("instagram_feed")
        True

        >>> validate_placement("invalid_placement")
        False
    """
    return placement in PLACEMENT_CONTEXTS
