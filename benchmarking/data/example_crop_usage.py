#!/usr/bin/env python3
"""Example usage of CROP-dataset loader.

This script demonstrates common usage patterns for the crop_dataset_loader module.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarking.data.crop_dataset_loader import (
    load_crop_qa,
    save_qa_items,
    load_qa_items,
    print_statistics,
)


def example_basic_loading():
    """Example 1: Basic dataset loading."""
    print("\n" + "="*60)
    print("Example 1: Basic Loading")
    print("="*60)

    print("\nLoading 50 items from CROP-dataset...")
    try:
        items = load_crop_qa(limit=50)
        print(f"Loaded {len(items)} items")

        if items:
            # Show first item
            item = items[0]
            print(f"\nFirst item:")
            print(f"  ID: {item.id}")
            print(f"  Question: {item.question[:100]}...")
            print(f"  Answer: {item.answer[:100]}...")
            print(f"  Crop: {item.crop_type}")
            print(f"  Language: {item.language}")
            print(f"  Context: {item.context[:50]}..." if item.context else "  Context: (none)")
    except Exception as e:
        print(f"Failed: {e}")
        print("Note: Dataset may have schema issues. See CROP_LOADER_STATUS.md")


def example_filtered_loading():
    """Example 2: Filtered loading by crop and language."""
    print("\n" + "="*60)
    print("Example 2: Filtered Loading")
    print("="*60)

    print("\nLoading rice and corn questions in English...")
    try:
        items = load_crop_qa(
            limit=100,
            crops=['rice', 'corn'],
            lang='en'
        )
        print(f"Loaded {len(items)} items")

        # Count by crop
        from collections import Counter
        crops = Counter(item.crop_type for item in items)
        print(f"\nCrop distribution:")
        for crop, count in crops.items():
            print(f"  {crop}: {count}")

    except Exception as e:
        print(f"Failed: {e}")


def example_save_and_load():
    """Example 3: Save and load processed data."""
    print("\n" + "="*60)
    print("Example 3: Save and Load")
    print("="*60)

    output_file = Path("temp_example_crop.jsonl")

    try:
        # Load from HuggingFace
        print("\nLoading from HuggingFace...")
        items = load_crop_qa(limit=20)
        print(f"Loaded {len(items)} items")

        # Save to file
        print(f"\nSaving to {output_file}...")
        save_qa_items(items, output_file)

        # Load from file
        print(f"\nLoading from {output_file}...")
        loaded_items = load_qa_items(output_file)
        print(f"Loaded {len(loaded_items)} items from file")

        # Verify
        assert len(items) == len(loaded_items)
        print("\nVerification: OK")

    except Exception as e:
        print(f"Failed: {e}")
    finally:
        # Cleanup
        if output_file.exists():
            output_file.unlink()
            print(f"\nCleaned up {output_file}")


def example_statistics():
    """Example 4: Print dataset statistics."""
    print("\n" + "="*60)
    print("Example 4: Statistics")
    print("="*60)

    try:
        print("\nLoading dataset for statistics...")
        items = load_crop_qa(limit=200)

        print_statistics(items)

    except Exception as e:
        print(f"Failed: {e}")


def example_integration_with_sourcedoc():
    """Example 5: Convert to SourceDoc format."""
    print("\n" + "="*60)
    print("Example 5: Integration with SourceDoc")
    print("="*60)

    try:
        # This would require the actual SourceDoc import
        # from core.Models.Schemas import SourceDoc

        print("\nLoading dataset...")
        items = load_crop_qa(limit=10)

        # Convert to SourceDoc format (example structure)
        print("\nConverting to SourceDoc format...")
        docs = []
        for item in items:
            doc_dict = {
                'id': item.id,
                'text': item.answer,
                'metadata': {
                    'question': item.question,
                    'crop_type': item.crop_type,
                    'language': item.language,
                    'context': item.context,
                }
            }
            docs.append(doc_dict)

        print(f"Converted {len(docs)} items to SourceDoc format")
        print(f"\nExample doc:")
        if docs:
            print(f"  ID: {docs[0]['id']}")
            print(f"  Text: {docs[0]['text'][:50]}...")
            print(f"  Metadata keys: {list(docs[0]['metadata'].keys())}")

    except Exception as e:
        print(f"Failed: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" CROP-DATASET LOADER USAGE EXAMPLES")
    print("="*70)

    example_basic_loading()
    example_filtered_loading()
    example_save_and_load()
    example_statistics()
    example_integration_with_sourcedoc()

    print("\n" + "="*70)
    print(" EXAMPLES COMPLETED")
    print("="*70)
    print("\nFor more information, see:")
    print("  - README_CROP.md")
    print("  - CROP_LOADER_STATUS.md")
    print("  - crop_dataset_loader.py")
    print()


if __name__ == "__main__":
    main()
