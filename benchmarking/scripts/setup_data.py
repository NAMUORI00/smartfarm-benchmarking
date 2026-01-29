#!/usr/bin/env python3
"""Data preparation and validation script for benchmarking.

This script validates and optionally downloads required datasets:
- wasabi_en_ko_parallel.jsonl (corpus)
- wasabi_qa_dataset.jsonl (QA pairs)

Usage:
    python -m benchmarking.scripts.setup_data [OPTIONS]
    
    # Validate existing data
    python -m benchmarking.scripts.setup_data --validate
    
    # Download missing data from HuggingFace
    python -m benchmarking.scripts.setup_data --download
    
    # Show data statistics
    python -m benchmarking.scripts.setup_data --stats
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()


@dataclass
class DatasetInfo:
    """Dataset metadata and statistics."""
    path: Path
    exists: bool
    line_count: int = 0
    valid_json: bool = False
    sample_fields: List[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.sample_fields is None:
            self.sample_fields = []


def validate_jsonl(path: Path, max_lines: int = 1000) -> DatasetInfo:
    """Validate JSONL file format and collect statistics.
    
    Args:
        path: Path to JSONL file
        max_lines: Maximum lines to validate (for performance)
    
    Returns:
        DatasetInfo with validation results
    """
    info = DatasetInfo(path=path, exists=path.exists())
    
    if not info.exists:
        info.error = "File not found"
        return info
    
    try:
        line_count = 0
        sample_fields = set()
        
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if i == 0:
                        sample_fields = set(data.keys())
                    line_count += 1
                except json.JSONDecodeError as e:
                    info.error = f"Invalid JSON at line {i + 1}: {e}"
                    info.line_count = line_count
                    return info
                
                if i >= max_lines:
                    # Count remaining lines without parsing
                    remaining = sum(1 for _ in f if _.strip())
                    line_count += remaining
                    break
        
        info.line_count = line_count
        info.valid_json = True
        info.sample_fields = sorted(sample_fields)
        
    except Exception as e:
        info.error = str(e)
    
    return info


def get_corpus_stats(path: Path) -> Dict:
    """Get detailed corpus statistics.
    
    Args:
        path: Path to corpus JSONL file
    
    Returns:
        Dictionary with corpus statistics
    """
    stats = {
        "total_documents": 0,
        "total_chars": 0,
        "avg_doc_length": 0,
        "has_korean": 0,
        "has_english": 0,
        "crops": {},
    }
    
    if not path.exists():
        return stats
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                stats["total_documents"] += 1
                
                # Text length
                text = data.get("text_ko") or data.get("text", "")
                stats["total_chars"] += len(text)
                
                # Language detection (simple)
                if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in text):
                    stats["has_korean"] += 1
                if data.get("text_en") or any(c.isascii() and c.isalpha() for c in text):
                    stats["has_english"] += 1
                
                # Crop metadata
                metadata = data.get("metadata", {})
                crop = metadata.get("crop") or metadata.get("작물")
                if crop:
                    stats["crops"][crop] = stats["crops"].get(crop, 0) + 1
                    
            except json.JSONDecodeError:
                continue
    
    if stats["total_documents"] > 0:
        stats["avg_doc_length"] = stats["total_chars"] // stats["total_documents"]
    
    return stats


def get_qa_stats(path: Path) -> Dict:
    """Get detailed QA dataset statistics.
    
    Args:
        path: Path to QA JSONL file
    
    Returns:
        Dictionary with QA statistics
    """
    stats = {
        "total_questions": 0,
        "categories": {},
        "complexities": {},
        "avg_question_length": 0,
        "avg_answer_length": 0,
        "has_source_ids": 0,
    }
    
    if not path.exists():
        return stats
    
    total_q_len = 0
    total_a_len = 0
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                stats["total_questions"] += 1
                
                # Question/answer lengths
                question = data.get("question", "")
                answer = data.get("answer", "")
                total_q_len += len(question)
                total_a_len += len(answer)
                
                # Category
                category = data.get("category", "unknown")
                stats["categories"][category] = stats["categories"].get(category, 0) + 1
                
                # Complexity
                complexity = data.get("complexity", "unknown")
                stats["complexities"][complexity] = stats["complexities"].get(complexity, 0) + 1
                
                # Source IDs
                if data.get("source_ids"):
                    stats["has_source_ids"] += 1
                    
            except json.JSONDecodeError:
                continue
    
    if stats["total_questions"] > 0:
        stats["avg_question_length"] = total_q_len // stats["total_questions"]
        stats["avg_answer_length"] = total_a_len // stats["total_questions"]
    
    return stats


def download_from_huggingface(repo_id: str, filename: str, output_dir: Path) -> bool:
    """Download dataset from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        filename: File to download
        output_dir: Output directory
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"Downloading {filename} from {repo_id}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        
        print(f"Downloaded to: {downloaded_path}")
        return True
        
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def print_validation_result(info: DatasetInfo, name: str) -> None:
    """Print validation result for a dataset."""
    if not info.exists:
        print(f"  ✗ {name}: NOT FOUND")
        print(f"    Path: {info.path}")
        return
    
    if not info.valid_json:
        print(f"  ✗ {name}: INVALID")
        print(f"    Error: {info.error}")
        return
    
    print(f"  ✓ {name}: {info.line_count} records")
    print(f"    Path: {info.path}")
    print(f"    Fields: {', '.join(info.sample_fields[:5])}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Data preparation and validation for benchmarking"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing data files",
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing data from HuggingFace",
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed data statistics",
    )
    
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to corpus file",
    )
    
    parser.add_argument(
        "--qa",
        type=Path,
        default=None,
        help="Path to QA dataset file",
    )
    
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="NAMUORI00/smartfarm-wasabi-dataset",
        help="HuggingFace repository for download",
    )
    
    args = parser.parse_args()

    # Use Settings-based path resolution
    from core.Config.Settings import settings

    corpus_path = args.corpus or Path(settings.get_corpus_path())
    qa_path = args.qa or Path(settings.get_qa_dataset_path())
    
    # If no action specified, default to validate
    if not any([args.validate, args.download, args.stats]):
        args.validate = True
    
    # Validate
    if args.validate:
        print("\n=== Data Validation ===\n")
        
        corpus_info = validate_jsonl(corpus_path)
        print_validation_result(corpus_info, "Corpus")
        
        qa_info = validate_jsonl(qa_path)
        print_validation_result(qa_info, "QA Dataset")
        
        print()
        
        if not corpus_info.valid_json or not qa_info.valid_json:
            print("Validation FAILED. Use --download to fetch missing data.")
            return 1
        else:
            print("Validation PASSED.")
    
    # Download
    if args.download:
        print("\n=== Data Download ===\n")
        
        if not corpus_path.exists():
            success = download_from_huggingface(
                args.hf_repo,
                "wasabi_en_ko_parallel.jsonl",
                corpus_path.parent,
            )
            if not success:
                return 1
        else:
            print(f"Corpus already exists: {corpus_path}")
        
        if not qa_path.exists():
            success = download_from_huggingface(
                args.hf_repo,
                "wasabi_qa_dataset.jsonl",
                qa_path.parent,
            )
            if not success:
                return 1
        else:
            print(f"QA dataset already exists: {qa_path}")
    
    # Statistics
    if args.stats:
        print("\n=== Corpus Statistics ===\n")
        corpus_stats = get_corpus_stats(corpus_path)
        print(f"  Total documents: {corpus_stats['total_documents']}")
        print(f"  Total characters: {corpus_stats['total_chars']:,}")
        print(f"  Avg doc length: {corpus_stats['avg_doc_length']} chars")
        print(f"  Korean docs: {corpus_stats['has_korean']}")
        print(f"  English docs: {corpus_stats['has_english']}")
        if corpus_stats['crops']:
            print(f"  Crops: {corpus_stats['crops']}")
        
        print("\n=== QA Dataset Statistics ===\n")
        qa_stats = get_qa_stats(qa_path)
        print(f"  Total questions: {qa_stats['total_questions']}")
        print(f"  Avg question length: {qa_stats['avg_question_length']} chars")
        print(f"  Avg answer length: {qa_stats['avg_answer_length']} chars")
        print(f"  With source IDs: {qa_stats['has_source_ids']}")
        if qa_stats['categories']:
            print(f"  Categories: {qa_stats['categories']}")
        if qa_stats['complexities']:
            print(f"  Complexities: {qa_stats['complexities']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
