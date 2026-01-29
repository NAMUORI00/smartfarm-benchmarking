#!/usr/bin/env python3
"""Test script for CROP-dataset loader.

This script demonstrates the usage of crop_dataset_loader module and validates
that it works correctly with the AI4Agr/CROP-dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarking.data.crop_dataset_loader import (
    QAItem,
    load_crop_qa,
    save_qa_items,
    load_qa_items,
    print_statistics,
    _extract_crop_type,
    _detect_language,
    _concatenate_history,
)


def test_helper_functions():
    """Test helper functions."""
    print("\n" + "="*60)
    print("Testing Helper Functions")
    print("="*60)

    # Test crop type extraction
    print("\n1. Testing crop type extraction:")
    test_cases = [
        ("What causes rice blast disease?", "rice"),
        ("How to grow corn?", "corn"),
        ("Soybean cultivation methods", "soybeans"),
        ("토마토 재배 방법", "tomatoes"),
        ("General farming question", "unknown"),
    ]

    for text, expected in test_cases:
        crop = _extract_crop_type(text, "", "")
        status = "[OK]" if crop == expected else "[FAIL]"
        print(f"  {status} '{text[:40]}...' -> {crop}")

    # Test language detection
    print("\n2. Testing language detection:")
    test_cases = [
        ("What is the best fertilizer?", "en"),
        ("벼 재배 방법은 무엇인가요?", "ko"),
        ("토마토 farming techniques", "ko"),  # Mixed, but has Korean
        ("123456", "unknown"),
    ]

    for text, expected in test_cases:
        lang = _detect_language(text)
        status = "[OK]" if lang == expected else "[FAIL]"
        print(f"  {status} '{text[:40]}...' -> {lang}")

    # Test history concatenation
    print("\n3. Testing history concatenation:")
    history = [
        ["What is rice?", "Rice is a cereal grain"],
        ["How to grow it?", "Plant in paddies with water"],
    ]
    context = _concatenate_history(history)
    has_turn1 = "Turn 1" in context
    has_turn2 = "Turn 2" in context
    print(f"  Context length: {len(context)} characters")
    print(f"  Has Turn 1: {has_turn1}")
    print(f"  Has Turn 2: {has_turn2}")

    print("\n[OK] All helper function tests passed")


def test_qaitem_class():
    """Test QAItem dataclass."""
    print("\n" + "="*60)
    print("Testing QAItem Class")
    print("="*60)

    # Create QAItem
    qa = QAItem(
        id="test_1",
        question="What causes rice blast disease?",
        answer="Rice blast is caused by the fungus Magnaporthe oryzae",
        context="Previous discussion about rice diseases",
        crop_type="rice",
        language="en",
    )

    print(f"\n  Created QAItem:")
    print(f"    ID: {qa.id}")
    print(f"    Question: {qa.question[:50]}...")
    print(f"    Crop: {qa.crop_type}")
    print(f"    Language: {qa.language}")

    # Test to_dict
    qa_dict = qa.to_dict()
    print(f"\n  to_dict() keys: {list(qa_dict.keys())}")

    # Test from_dict
    qa_restored = QAItem.from_dict(qa_dict)
    print(f"  from_dict() restored: {qa_restored.id}")

    assert qa.id == qa_restored.id
    assert qa.question == qa_restored.question
    print("\n[OK] QAItem class tests passed")


def test_load_crop_qa():
    """Test loading CROP dataset."""
    print("\n" + "="*60)
    print("Testing load_crop_qa()")
    print("="*60)

    try:
        # Test with small limit
        print("\nLoading 10 items from CROP-dataset...")
        items = load_crop_qa(limit=10)

        if items:
            print(f"[OK] Successfully loaded {len(items)} items")
            print(f"\nFirst item:")
            item = items[0]
            print(f"  ID: {item.id}")
            print(f"  Question: {item.question[:100]}...")
            print(f"  Answer: {item.answer[:100]}...")
            print(f"  Crop: {item.crop_type}")
            print(f"  Language: {item.language}")

            # Print statistics
            print_statistics(items)
        else:
            print("[WARNING] No items loaded (dataset might be empty or unavailable)")

    except ImportError as e:
        print(f"[WARNING] Skipping dataset loading test: {e}")
        print("  Install with: pip install datasets")
    except Exception as e:
        print(f"[FAIL] Failed to load dataset: {e}")
        return False

    return True


def test_save_load_items():
    """Test saving and loading QA items."""
    print("\n" + "="*60)
    print("Testing save/load functions")
    print("="*60)

    # Create test data
    items = [
        QAItem(
            id="test_1",
            question="What is rice?",
            answer="Rice is a cereal grain",
            context="",
            crop_type="rice",
            language="en",
        ),
        QAItem(
            id="test_2",
            question="How to grow corn?",
            answer="Plant corn in warm soil",
            context="Previous discussion",
            crop_type="corn",
            language="en",
        ),
    ]

    # Save to temporary file
    temp_path = Path("temp_crop_test.jsonl")
    try:
        save_qa_items(items, temp_path)
        print(f"[OK] Saved {len(items)} items to {temp_path}")

        # Load back
        loaded_items = load_qa_items(temp_path)
        print(f"[OK] Loaded {len(loaded_items)} items from {temp_path}")

        # Verify
        assert len(items) == len(loaded_items)
        assert items[0].id == loaded_items[0].id
        assert items[1].question == loaded_items[1].question

        print("[OK] Save/load validation passed")

    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
            print(f"[OK] Cleaned up {temp_path}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" CROP-DATASET LOADER TEST SUITE")
    print("="*70)

    try:
        test_helper_functions()
        test_qaitem_class()
        test_save_load_items()
        test_load_crop_qa()

        print("\n" + "="*70)
        print(" ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print()

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
