"""Persona generation for ad prediction."""

import random
from typing import Dict, List, Optional


class PersonaGenerator:
    """Generate synthetic customer personas for ad testing."""

    # Persona templates for different segments
    # Based on paper findings: AGE and INCOME have strongest signal
    SEGMENT_TEMPLATES = {
        "general_consumer": {
            "names": ["Emma", "Sarah", "Michael", "David", "Jessica", "Chris", "Amanda", "Ryan"],
            "ages": range(25, 55),
            "genders": ["woman", "man"],
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
            # Paper used income levels - adding explicit income brackets
            "income_brackets": [
                {"level": "struggling financially", "range": "$20k-35k", "discretionary": "very limited"},
                {"level": "managing but tight budget", "range": "$35k-50k", "discretionary": "limited"},
                {"level": "comfortable but mindful", "range": "$50k-75k", "discretionary": "moderate"},
                {"level": "financially comfortable", "range": "$75k-100k", "discretionary": "good"},
                {"level": "financially secure", "range": "$100k+", "discretionary": "significant"},
            ],
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
            "genders": ["woman"],
            "occupations": [
                "marketing manager",
                "UX designer",
                "social media manager",
                "product manager",
                "HR specialist",
            ],
            "locations": ["San Francisco", "New York", "Chicago", "Seattle", "Austin"],
            "income_brackets": [
                {"level": "managing but tight budget", "range": "$35k-50k", "discretionary": "limited"},
                {"level": "comfortable but mindful", "range": "$50k-75k", "discretionary": "moderate"},
                {"level": "financially comfortable", "range": "$75k-100k", "discretionary": "good"},
                {"level": "financially secure", "range": "$100k+", "discretionary": "significant"},
            ],
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
            "genders": ["woman", "man", "non-binary"],
            "occupations": [
                "student",
                "junior designer",
                "content creator",
                "barista",
                "retail associate",
            ],
            "locations": ["Los Angeles", "Miami", "Portland", "Denver", "Nashville"],
            "income_brackets": [
                {"level": "struggling financially", "range": "$20k-35k", "discretionary": "very limited"},
                {"level": "managing but tight budget", "range": "$35k-50k", "discretionary": "limited"},
                {"level": "comfortable but mindful", "range": "$50k-75k", "discretionary": "moderate"},
            ],
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

        # Sample attributes - AGE and INCOME are most important per paper
        age = random.choice(list(template["ages"]))
        income_bracket = random.choice(template["income_brackets"])
        gender = random.choice(template["genders"])
        name = random.choice(template["names"])
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
            gender = custom_attributes.get("gender", gender)
            occupation = custom_attributes.get("occupation", occupation)
            location = custom_attributes.get("location", location)
            values = custom_attributes.get("values", values)
            pain_points = custom_attributes.get("pain_points", pain_points)
            income_bracket = custom_attributes.get("income_bracket", income_bracket)

        # Build persona description - EMPHASIZE AGE and INCOME per paper findings
        persona = f"""You are {name}, a {age}-year-old {gender} who works as a {occupation} in {location}.

DEMOGRAPHICS (important for purchase intent):
- Age: {age} years old
- Income: {income_bracket['range']} annually, {income_bracket['level']}
- Discretionary spending: {income_bracket['discretionary']}
- Gender: {gender}
- Location: {location}

SHOPPING BEHAVIOR:
You shop online {shopping_freq} and have {price_sens} price sensitivity. Given your income level ({income_bracket['level']}), you have {income_bracket['discretionary']} discretionary spending for non-essentials.

VALUES AND PRIORITIES:
- You value {values[0]} and {values[1]}

CONCERNS AND PAIN POINTS:
- {pain_points[0]}
- {pain_points[1]}

When evaluating products, carefully consider:
1. Whether the price fits your budget ({income_bracket['level']})
2. Whether it aligns with your age group's needs and preferences
3. Whether it matches your values and priorities

You're {"willing to invest in quality products" if price_sens == "low" else "very budget-conscious and look for good deals" if price_sens == "high" else "balanced between quality and price"}."""

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
