"""Persona generation for ad prediction."""

import random
from typing import Dict, List, Optional


class PersonaGenerator:
    """Generate synthetic customer personas for ad testing."""

    # Persona templates for different segments
    SEGMENT_TEMPLATES = {
        "general_consumer": {
            "names": ["Emma", "Sarah", "Michael", "David", "Jessica", "Chris", "Amanda", "Ryan"],
            "ages": range(25, 55),
            "occupations": [
                "marketing professional",
                "software engineer",
                "teacher",
                "sales manager",
                "designer",
                "accountant",
                "consultant",
                "nurse",
            ],
            "locations": ["Chicago", "New York", "Los Angeles", "Seattle", "Austin", "Boston"],
            "values": [
                "quality",
                "sustainability",
                "convenience",
                "value for money",
                "brand reputation",
            ],
            "pain_points": [
                "limited time",
                "budget constraints",
                "finding trustworthy brands",
                "fast shipping",
            ],
            "shopping_frequency": [
                "weekly",
                "bi-weekly",
                "monthly",
                "a few times per year",
            ],
            "price_sensitivity": ["low", "medium", "high"],
        },
        "millennial_women": {
            "names": ["Emma", "Sophia", "Olivia", "Ava", "Isabella", "Mia", "Charlotte"],
            "ages": range(25, 40),
            "occupations": [
                "marketing manager",
                "UX designer",
                "social media manager",
                "product manager",
                "HR specialist",
            ],
            "locations": ["San Francisco", "New York", "Chicago", "Seattle", "Austin"],
            "values": [
                "sustainability",
                "ethical brands",
                "quality",
                "social impact",
                "authenticity",
            ],
            "pain_points": [
                "busy schedule",
                "finding authentic brands",
                "sustainability concerns",
                "work-life balance",
            ],
            "shopping_frequency": ["weekly", "bi-weekly"],
            "price_sensitivity": ["low", "medium"],
        },
        "gen_z": {
            "names": ["Zoe", "Madison", "Harper", "Ella", "Avery", "Liam", "Noah"],
            "ages": range(18, 27),
            "occupations": [
                "student",
                "junior designer",
                "content creator",
                "barista",
                "retail associate",
            ],
            "locations": ["Los Angeles", "Miami", "Portland", "Denver", "Nashville"],
            "values": [
                "authenticity",
                "social consciousness",
                "experiences",
                "self-expression",
                "community",
            ],
            "pain_points": [
                "limited budget",
                "avoiding greenwashing",
                "finding unique products",
                "social proof",
            ],
            "shopping_frequency": ["monthly", "a few times per year"],
            "price_sensitivity": ["high", "medium"],
        },
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize persona generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def generate_persona(
        self,
        segment: str = "general_consumer",
        custom_attributes: Optional[Dict] = None,
    ) -> str:
        """
        Generate a single synthetic persona.

        Args:
            segment: Persona segment (general_consumer, millennial_women, gen_z)
            custom_attributes: Optional dict to override specific attributes

        Returns:
            str: Persona description as natural language
        """
        if segment not in self.SEGMENT_TEMPLATES:
            raise ValueError(
                f"Unknown segment: {segment}. Available: {list(self.SEGMENT_TEMPLATES.keys())}"
            )

        template = self.SEGMENT_TEMPLATES[segment]

        # Sample attributes
        name = random.choice(template["names"])
        age = random.choice(list(template["ages"]))
        occupation = random.choice(template["occupations"])
        location = random.choice(template["locations"])
        values = random.sample(template["values"], k=min(2, len(template["values"])))
        pain_points = random.sample(
            template["pain_points"], k=min(2, len(template["pain_points"]))
        )
        shopping_freq = random.choice(template["shopping_frequency"])
        price_sens = random.choice(template["price_sensitivity"])

        # Apply custom attributes if provided
        if custom_attributes:
            name = custom_attributes.get("name", name)
            age = custom_attributes.get("age", age)
            occupation = custom_attributes.get("occupation", occupation)
            location = custom_attributes.get("location", location)
            values = custom_attributes.get("values", values)
            pain_points = custom_attributes.get("pain_points", pain_points)

        # Build persona description
        persona = f"""You are {name}, a {age}-year-old {occupation} living in {location}.

You shop online {shopping_freq} and have {price_sens} price sensitivity.

Your values and priorities when shopping:
- You value {values[0]} and {values[1]}

Your main concerns and pain points:
- {pain_points[0]}
- {pain_points[1]}

When evaluating products, you carefully consider whether they align with your values and needs.
You're {"willing to pay a premium for quality" if price_sens == "low" else "budget-conscious and look for good deals" if price_sens == "high" else "balanced between quality and price"}."""

        return persona

    def generate_personas(
        self,
        num_personas: int,
        segment: str = "general_consumer",
        custom_attributes: Optional[Dict] = None,
    ) -> List[str]:
        """
        Generate multiple synthetic personas.

        Args:
            num_personas: Number of personas to generate
            segment: Persona segment
            custom_attributes: Optional dict to override specific attributes

        Returns:
            List[str]: List of persona descriptions
        """
        return [
            self.generate_persona(segment=segment, custom_attributes=custom_attributes)
            for _ in range(num_personas)
        ]

    def get_available_segments(self) -> List[str]:
        """Get list of available persona segments."""
        return list(self.SEGMENT_TEMPLATES.keys())
