"""Simple test script for persona generation."""

from src.core.persona_generator import PersonaGenerator

# Test 1: Generate personas
print("=" * 60)
print("TEST 1: Persona Generation")
print("=" * 60)

gen = PersonaGenerator(seed=42)
personas = gen.generate_personas(num_personas=2, segment="millennial_women")

for i, persona in enumerate(personas, 1):
    print(f"\nPersona {i}:")
    print(persona)
    print()

print("✓ Personas generated successfully!")
print()

# Test 2: Available segments
print("=" * 60)
print("TEST 2: Available Segments")
print("=" * 60)
segments = gen.get_available_segments()
print(f"Available segments: {segments}")
print("✓ Segments listed successfully!")
