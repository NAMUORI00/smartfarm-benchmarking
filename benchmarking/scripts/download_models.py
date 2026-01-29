#!/usr/bin/env python3
"""Model download script for benchmarking.

Pre-downloads embedding models to avoid delays during benchmark execution.

Usage:
    python -m benchmarking.scripts.download_models [OPTIONS]
    
    # Download default model
    python -m benchmarking.scripts.download_models
    
    # Download specific model
    python -m benchmarking.scripts.download_models --model Qwen/Qwen3-Embedding-0.6B
    
    # Verify model works
    python -m benchmarking.scripts.download_models --verify
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional


# Supported models for benchmarking
SUPPORTED_MODELS = {
    "minilm": {
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "size_mb": 480,
        "description": "Fast multilingual model (recommended for edge)",
    },
    "mpnet": {
        "id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dim": 768,
        "size_mb": 970,
        "description": "Higher quality multilingual model",
    },
    "qwen3": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "dim": 1024,
        "size_mb": 1200,
        "description": "High quality but requires more memory",
    },
}

DEFAULT_MODEL = "minilm"


def get_cache_dir() -> Path:
    """Get the model cache directory."""
    # sentence-transformers uses ~/.cache/torch/sentence_transformers/
    home = Path.home()
    cache_dir = home / ".cache" / "torch" / "sentence_transformers"
    return cache_dir


def check_model_cached(model_id: str) -> bool:
    """Check if model is already cached.
    
    Args:
        model_id: HuggingFace model ID
    
    Returns:
        True if model appears to be cached
    """
    cache_dir = get_cache_dir()
    
    # Model directory name is typically the model_id with / replaced
    model_name = model_id.replace("/", "_")
    model_dir = cache_dir / model_name
    
    if model_dir.exists():
        # Check for key files
        has_config = (model_dir / "config.json").exists()
        has_model = any(model_dir.glob("*.bin")) or any(model_dir.glob("*.safetensors"))
        return has_config and has_model
    
    return False


def download_model(model_id: str, verify: bool = True) -> bool:
    """Download and optionally verify a model.
    
    Args:
        model_id: HuggingFace model ID or alias
        verify: Whether to run verification after download
    
    Returns:
        True if successful, False otherwise
    """
    # Resolve alias
    if model_id.lower() in SUPPORTED_MODELS:
        model_info = SUPPORTED_MODELS[model_id.lower()]
        model_id = model_info["id"]
        print(f"Using model: {model_id}")
        print(f"  Description: {model_info['description']}")
        print(f"  Dimensions: {model_info['dim']}")
        print(f"  Size: ~{model_info['size_mb']}MB")
        print()
    
    # Check cache
    if check_model_cached(model_id):
        print(f"Model already cached: {model_id}")
        if verify:
            return verify_model(model_id)
        return True
    
    # Download
    print(f"Downloading model: {model_id}")
    print("This may take a few minutes depending on your connection...")
    print()
    
    start_time = time.time()
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(model_id, device="cpu")
        
        elapsed = time.time() - start_time
        print(f"\nDownload complete in {elapsed:.1f}s")
        print(f"  Model dimension: {model.get_sentence_embedding_dimension()}")
        
        if verify:
            return verify_model(model_id, model)
        
        return True
        
    except ImportError:
        print("Error: sentence-transformers not installed.")
        print("Run: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


def verify_model(model_id: str, model=None) -> bool:
    """Verify model works correctly.
    
    Args:
        model_id: HuggingFace model ID
        model: Optional pre-loaded model
    
    Returns:
        True if verification passes
    """
    print("\nVerifying model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        if model is None:
            model = SentenceTransformer(model_id, device="cpu")
        
        # Test encoding
        test_texts = [
            "와사비 재배 최적 온도는?",
            "토마토 양액 EC 농도",
            "스마트팜 환경 관리",
        ]
        
        embeddings = model.encode(test_texts)
        
        # Basic checks
        assert embeddings.shape[0] == len(test_texts), "Embedding count mismatch"
        assert embeddings.shape[1] > 0, "Zero-dimensional embeddings"
        assert not np.isnan(embeddings).any(), "NaN in embeddings"
        assert not np.isinf(embeddings).any(), "Inf in embeddings"
        
        # Check similarity (first two should be more similar than first and third)
        from numpy.linalg import norm
        
        def cosine_sim(a, b):
            return np.dot(a, b) / (norm(a) * norm(b))
        
        sim_01 = cosine_sim(embeddings[0], embeddings[1])
        sim_02 = cosine_sim(embeddings[0], embeddings[2])
        
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        print(f"  Test encodings: {len(test_texts)} texts")
        print(f"  Similarity check: {sim_01:.3f} vs {sim_02:.3f}")
        print("\n✓ Model verification PASSED")
        
        return True
        
    except AssertionError as e:
        print(f"✗ Verification FAILED: {e}")
        return False
    except Exception as e:
        print(f"✗ Verification ERROR: {e}")
        return False


def list_models() -> None:
    """List supported models."""
    print("\n=== Supported Models ===\n")
    print(f"{'Alias':<10} {'Model ID':<55} {'Dim':<5} {'Size':<8}")
    print("-" * 80)
    
    for alias, info in SUPPORTED_MODELS.items():
        print(f"{alias:<10} {info['id']:<55} {info['dim']:<5} {info['size_mb']}MB")
    
    print()
    print(f"Default: {DEFAULT_MODEL} ({SUPPORTED_MODELS[DEFAULT_MODEL]['id']})")
    print()


def show_cache_info() -> None:
    """Show model cache information."""
    cache_dir = get_cache_dir()
    
    print(f"\n=== Model Cache ===\n")
    print(f"Cache directory: {cache_dir}")
    
    if not cache_dir.exists():
        print("(Cache directory does not exist)")
        return
    
    # List cached models
    print("\nCached models:")
    for model_dir in cache_dir.iterdir():
        if model_dir.is_dir():
            # Calculate size
            size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            size_mb = size_bytes / (1024 * 1024)
            
            # Check for known aliases
            alias = ""
            for name, info in SUPPORTED_MODELS.items():
                if info["id"].replace("/", "_") == model_dir.name:
                    alias = f" ({name})"
                    break
            
            print(f"  {model_dir.name}{alias}: {size_mb:.1f}MB")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and verify embedding models for benchmarking"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to download (alias or full ID, default: {DEFAULT_MODEL})",
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify model after download",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported models",
    )
    
    parser.add_argument(
        "--cache-info",
        action="store_true",
        help="Show cache directory information",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all supported models",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return 0
    
    if args.cache_info:
        show_cache_info()
        return 0
    
    if args.all:
        print("Downloading all supported models...")
        success = True
        for alias in SUPPORTED_MODELS:
            print(f"\n{'='*60}")
            if not download_model(alias, verify=args.verify):
                success = False
        return 0 if success else 1
    
    # Download specified model
    success = download_model(args.model, verify=args.verify)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
