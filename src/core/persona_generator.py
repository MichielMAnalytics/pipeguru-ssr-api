"""Persona validation helpers."""

from typing import Dict, List, Optional


class PersonaGenerator:
    """Helper utilities for persona validation."""

    @staticmethod
    def validate_persona(description: str) -> dict:
        """
        Validate a persona description.

        Args:
            description: Persona description text

        Returns:
            dict: {
                "valid": bool,
                "length": int,
                "warnings": Optional[List[str]]  # Suggestions for improvement
            }
        """
        warnings = []

        # Minimum length check
        if len(description) < 50:
            return {
                "valid": False,
                "length": len(description),
                "warnings": ["Persona too short (minimum 50 characters)"],
            }

        # Check for key elements (optional suggestions)
        description_lower = description.lower()

        if "age" not in description_lower and "year" not in description_lower:
            warnings.append("Consider including age for better predictions (age is a strong signal)")

        if "income" not in description_lower and "earn" not in description_lower and "$" not in description:
            warnings.append("Consider including income level for better predictions (income is a strong signal)")

        if "shop" not in description_lower and "purchase" not in description_lower and "buy" not in description_lower:
            warnings.append("Consider including shopping behavior/frequency")

        return {
            "valid": True,
            "length": len(description),
            "warnings": warnings if warnings else None,
        }
