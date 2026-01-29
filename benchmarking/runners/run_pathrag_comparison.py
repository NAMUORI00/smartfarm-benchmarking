"""Wrapper script to run PathRAG comparison test with proper imports."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import and run test
from benchmarking.tests.test_pathrag_comparison import test_direct_search

if __name__ == "__main__":
    test_direct_search()
