#!/usr/bin/env python3
"""CROP-dataset loader for agricultural QA benchmark.

Downloads and processes the AI4Agr/CROP-dataset from HuggingFace, providing
structured QA items with support for subset filtering and caching.

NOTE: The CROP-dataset may have schema inconsistencies in some files (specifically
with the 'history' field mixing list and non-list values). This loader attempts
to handle these issues gracefully, but some splits may fail to load completely.

Usage:
    python -m benchmarking.data.crop_dataset_loader --output data/crop --limit 5000
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QAItem:
    """Structured QA item from CROP dataset."""

    id: str
    question: str
    answer: str
    context: str
    crop_type: str
    language: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> QAItem:
        """Create QAItem from dictionary."""
        return cls(**data)


def _extract_crop_type(instruction: str, input_text: str, output_text: str) -> str:
    """Extract crop type from instruction/input/output text.

    Args:
        instruction: Instruction text
        input_text: Input text
        output_text: Output text

    Returns:
        Detected crop type or 'unknown'
    """
    # Define crop keywords
    crop_keywords = {
        'rice': ['rice', 'paddy', '벼', '쌀'],
        'corn': ['corn', 'maize', '옥수수'],
        'soybeans': ['soybean', 'soy', '콩', '대두'],
        'tomatoes': ['tomato', '토마토'],
        'wheat': ['wheat', '밀'],
        'potato': ['potato', '감자'],
        'pepper': ['pepper', 'chili', '고추'],
        'cucumber': ['cucumber', '오이'],
        'strawberry': ['strawberry', '딸기'],
    }

    # Combine all text for searching
    combined_text = f"{instruction} {input_text} {output_text}".lower()

    # Search for crop keywords
    for crop_type, keywords in crop_keywords.items():
        for keyword in keywords:
            if keyword.lower() in combined_text:
                return crop_type

    return 'unknown'


def _detect_language(text: str) -> str:
    """Detect language from text.

    Args:
        text: Text to analyze

    Returns:
        Language code ('en', 'ko', or 'unknown')
    """
    # Simple heuristic: check for Korean characters
    if any('\uac00' <= char <= '\ud7a3' for char in text):
        return 'ko'
    elif any('a' <= char.lower() <= 'z' for char in text):
        return 'en'
    return 'unknown'


def _concatenate_history(history: List[List[str]]) -> str:
    """Concatenate dialogue history into context string.

    Args:
        history: List of [question, answer] pairs, or None/empty

    Returns:
        Formatted context string
    """
    if not history or not isinstance(history, list):
        return ""

    parts = []
    try:
        for i, turn in enumerate(history, 1):
            # Handle different history formats
            if not isinstance(turn, (list, tuple)) or len(turn) < 2:
                continue
            q, a = turn[0], turn[1]
            if not q and not a:
                continue
            parts.append(f"[Turn {i}]")
            parts.append(f"Q: {q}")
            parts.append(f"A: {a}")
            parts.append("")
    except (TypeError, ValueError, IndexError) as e:
        logger.warning(f"Failed to parse history: {e}")
        return ""

    return "\n".join(parts).strip()


def load_crop_qa(
    limit: Optional[int] = None,
    crops: Optional[List[str]] = None,
    lang: Optional[str] = None,
    cache_dir: Optional[str] = None,
    split: str = "train",
) -> List[QAItem]:
    """Load CROP-dataset from HuggingFace with optional filtering.

    Args:
        limit: Maximum number of items to load
        crops: List of crop types to filter (e.g., ['rice', 'corn'])
        lang: Language filter ('en', 'ko', or None for all)
        cache_dir: Directory to cache downloaded dataset
        split: Dataset split to load (default: 'train')

    Returns:
        List of QAItem objects

    Raises:
        ImportError: If datasets library is not installed
        Exception: If dataset loading fails
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Install with: pip install datasets")
        raise ImportError(
            "The 'datasets' library is required. "
            "Install it with: pip install datasets"
        )

    logger.info(f"Loading CROP-dataset from HuggingFace (split={split})...")

    try:
        # Try to load the dataset, but handle schema issues
        dataset = load_dataset(
            "AI4Agr/CROP-dataset",
            split=split,
            cache_dir=cache_dir,
        )
    except Exception as e:
        error_msg = str(e)
        # Check if it's the known history field schema issue
        if "cannot mix list and non-list" in error_msg or "history" in error_msg:
            logger.warning(f"Dataset has schema issues with history field: {e}")
            logger.info("Attempting to load with verification_mode='no_checks'...")
            try:
                dataset = load_dataset(
                    "AI4Agr/CROP-dataset",
                    split=split,
                    cache_dir=cache_dir,
                    verification_mode='no_checks',
                )
                logger.info("Successfully loaded with verification_mode='no_checks'")
            except Exception as e2:
                logger.error(f"Failed to load CROP-dataset even with no_checks: {e2}")
                raise
        else:
            logger.error(f"Failed to load CROP-dataset: {e}")
            raise

    logger.info(f"Loaded {len(dataset)} raw examples")

    qa_items: List[QAItem] = []

    for idx, item in enumerate(dataset):
        try:
            # Extract fields with type safety
            instruction = str(item.get("instruction", ""))
            input_text = str(item.get("input", ""))
            output_text = str(item.get("output", ""))
            system = str(item.get("system", ""))
            history = item.get("history", [])

            # Skip if essential fields are missing
            if not instruction and not output_text:
                continue

            # Build question text
            if input_text:
                question = f"{instruction}\n{input_text}".strip()
            else:
                question = instruction.strip()

            # Build context from history
            context = _concatenate_history(history)
            if system and not context:
                context = system

            # Extract metadata
            crop_type = _extract_crop_type(instruction, input_text, output_text)
            language = _detect_language(question + output_text)
        except Exception as e:
            logger.warning(f"Failed to process item {idx}: {e}")
            continue

        # Apply filters
        if crops and crop_type not in crops:
            continue

        if lang and language != lang:
            continue

        # Create QA item
        qa_item = QAItem(
            id=f"crop_{idx}",
            question=question,
            answer=output_text,
            context=context,
            crop_type=crop_type,
            language=language,
        )

        qa_items.append(qa_item)

        # Check limit
        if limit and len(qa_items) >= limit:
            logger.info(f"Reached limit of {limit} items")
            break

    logger.info(
        f"Filtered to {len(qa_items)} QA items "
        f"(crops={crops}, lang={lang}, limit={limit})"
    )

    return qa_items


def save_qa_items(items: List[QAItem], output_path: Path) -> None:
    """Save QA items to JSONL file.

    Args:
        items: List of QAItem objects
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(items)} items to {output_path}")


def load_qa_items(input_path: Path) -> List[QAItem]:
    """Load QA items from JSONL file.

    Args:
        input_path: Path to JSONL file

    Returns:
        List of QAItem objects
    """
    items = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(QAItem.from_dict(data))

    logger.info(f"Loaded {len(items)} items from {input_path}")
    return items


def print_statistics(items: List[QAItem]) -> None:
    """Print statistics about QA items.

    Args:
        items: List of QAItem objects
    """
    from collections import Counter

    crop_counts = Counter(item.crop_type for item in items)
    lang_counts = Counter(item.language for item in items)

    print("\n" + "="*60)
    print("CROP-DATASET STATISTICS")
    print("="*60)
    print(f"Total items: {len(items)}")
    print()
    print("Crop distribution:")
    for crop, count in crop_counts.most_common():
        print(f"  {crop}: {count}")
    print()
    print("Language distribution:")
    for lang, count in lang_counts.most_common():
        print(f"  {lang}: {count}")
    print("="*60)
    print()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load and process CROP-dataset from HuggingFace"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/crop",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of items to load",
    )
    parser.add_argument(
        "--crops",
        nargs="+",
        default=None,
        help="Filter by crop types (e.g., rice corn soybeans tomatoes)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "ko"],
        default=None,
        help="Filter by language",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics about loaded data",
    )

    return parser.parse_args()


def main() -> None:
    """Main CLI entry point."""
    args = parse_args()

    # Set default cache directory
    if args.cache_dir is None:
        output_dir = Path(args.output)
        args.cache_dir = str(output_dir / "raw")

    # Load dataset
    items = load_crop_qa(
        limit=args.limit,
        crops=args.crops,
        lang=args.lang,
        cache_dir=args.cache_dir,
        split=args.split,
    )

    # Print statistics if requested
    if args.stats or not items:
        print_statistics(items)

    if not items:
        logger.warning("No items loaded. Adjust filters or check dataset.")
        return

    # Save processed data
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "crop_qa.jsonl"
    save_qa_items(items, output_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
