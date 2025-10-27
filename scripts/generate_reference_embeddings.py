"""
Generate and save reference statement embeddings to disk.

This script pre-computes embeddings for all reference statement sets and saves them
as .npy files in the default_embeddings/ directory. This eliminates the need to
regenerate these embeddings on every API request.

Usage:
    python scripts/generate_reference_embeddings.py
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.core.embeddings import GeminiEmbeddings
from src.core.reference_statements import get_reference_sets

# Load environment variables from .env file
load_dotenv()


def generate_and_save_embeddings(
    output_dir: Path = Path("default_embeddings"),
    question_type: str = "purchase_intent",
    force: bool = False,
):
    """
    Generate embeddings for all reference sets and save to disk.

    Args:
        output_dir: Directory to save embedding files
        question_type: Type of question ("purchase_intent")
        force: If True, regenerate even if files exist
    """
    print("=" * 80)
    print("Generating Reference Statement Embeddings")
    print("=" * 80)
    print()

    # Initialize embeddings client
    embeddings = GeminiEmbeddings()
    model_name = embeddings.model.replace("/", "_")  # Sanitize for filename
    dimensions = embeddings.output_dimensionality

    print(f"Model: {embeddings.model}")
    print(f"Dimensions: {dimensions}")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all reference sets
    reference_sets = get_reference_sets(question_type)
    print(f"Found {len(reference_sets)} reference sets for '{question_type}'")
    print()

    # Generate and save embeddings for each set
    saved_files = []
    for idx, ref_sentences in enumerate(reference_sets):
        # Generate filename
        filename = f"{model_name}_{dimensions}d_{question_type}_set_{idx}.npy"
        filepath = output_dir / filename

        # Skip if exists (unless force=True)
        if filepath.exists() and not force:
            print(f"✓ Set {idx + 1}/{len(reference_sets)}: Already exists, skipping")
            saved_files.append(filename)
            continue

        print(f"Processing set {idx + 1}/{len(reference_sets)}...")
        print(f"  Statements: {ref_sentences}")

        # Generate embeddings
        embeddings_array = embeddings.encode(ref_sentences)

        # Save to disk
        np.save(filepath, embeddings_array)
        saved_files.append(filename)

        print(f"  ✓ Saved to: {filename}")
        print(f"  Shape: {embeddings_array.shape}")
        print()

    # Create manifest file with metadata
    manifest = {
        "model": embeddings.model,
        "dimensions": dimensions,
        "question_type": question_type,
        "num_sets": len(reference_sets),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": saved_files,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 80)
    print("✓ All embeddings generated successfully!")
    print("=" * 80)
    print()
    print(f"Saved {len(saved_files)} embedding files:")
    for filename in saved_files:
        print(f"  - {filename}")
    print()
    print(f"Manifest saved to: {manifest_path}")
    print()
    print("These embeddings will be loaded automatically by AdPredictor.")
    print()


def verify_embeddings(output_dir: Path = Path("default_embeddings")):
    """
    Verify that saved embeddings can be loaded correctly.

    Args:
        output_dir: Directory containing embedding files
    """
    print("=" * 80)
    print("Verifying Saved Embeddings")
    print("=" * 80)
    print()

    # Load manifest
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"✗ Error: Manifest file not found at {manifest_path}")
        return False

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print(f"Manifest loaded:")
    print(f"  Model: {manifest['model']}")
    print(f"  Dimensions: {manifest['dimensions']}")
    print(f"  Question type: {manifest['question_type']}")
    print(f"  Number of sets: {manifest['num_sets']}")
    print(f"  Generated at: {manifest['generated_at']}")
    print()

    # Load and verify each embedding file
    all_valid = True
    for idx, filename in enumerate(manifest["files"]):
        filepath = output_dir / filename

        if not filepath.exists():
            print(f"✗ Set {idx}: File not found - {filename}")
            all_valid = False
            continue

        try:
            embeddings_array = np.load(filepath)
            expected_shape = (5, manifest["dimensions"])

            if embeddings_array.shape == expected_shape:
                print(f"✓ Set {idx}: Valid - shape {embeddings_array.shape}")
            else:
                print(f"✗ Set {idx}: Invalid shape - expected {expected_shape}, got {embeddings_array.shape}")
                all_valid = False

        except Exception as e:
            print(f"✗ Set {idx}: Error loading - {e}")
            all_valid = False

    print()
    if all_valid:
        print("✓ All embeddings verified successfully!")
    else:
        print("✗ Some embeddings failed verification")

    print()
    return all_valid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and save reference statement embeddings"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if files exist",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing embeddings, don't generate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("default_embeddings"),
        help="Output directory for embeddings (default: default_embeddings)",
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_embeddings(args.output_dir)
    else:
        generate_and_save_embeddings(
            output_dir=args.output_dir,
            force=args.force,
        )

        # Verify after generation
        print()
        verify_embeddings(args.output_dir)
