<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-28 | Updated: 2026-01-28 -->

# utils/

Shared utility functions for benchmarking infrastructure.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package initialization |
| `experiment_utils.py` | Common RAG evaluation utilities (query loading, API calls, metrics aggregation) |

## experiment_utils.py

Provides reusable utilities for experiment runners.

### Key Functions

**Query Management**

```python
def load_queries(path: Path | None) -> List[Dict[str, Any]]:
    """Load evaluation queries from JSONL file.

    Args:
        path: Path to JSONL file, or None for fallback queries

    Returns:
        List of query dicts with id, question, category, etc.
    """
```

**API Communication**

```python
def call_query(
    base_url: str,
    question: str,
    ranker: str,
    top_k: int
) -> Dict[str, Any]:
    """Call RAG /query endpoint.

    Args:
        base_url: HTTP base URL (e.g., http://localhost:41177)
        question: Query text
        ranker: Ranker type (none, llm, etc.)
        top_k: Number of results to retrieve

    Returns:
        Response dict with documents and latency
    """
```

**Statistics Aggregation**

```python
def aggregate_results(
    all_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate per-query results to dataset statistics.

    Computes mean, stddev, min, max for each metric.

    Args:
        all_results: List of per-query result dicts

    Returns:
        Aggregated statistics dict
    """
```

### Constants

```python
DEFAULT_BASE_URL = "http://127.0.0.1:41177"

# Fallback test queries (when no input file provided)
FALLBACK_QUERIES = [
    {"id": "q1", "category": "온도", "question": "토마토 온실 생육 최적 온도는?"},
    {"id": "q2", "category": "양액", "question": "파프리카 양액 EC 기준을 알려줘"},
    {"id": "q3", "category": "병해충", "question": "딸기 흰가루병 초기 증상과 대처법은?"},
    {"id": "q4", "category": "재배일정", "question": "상추 파종부터 수확까지 재배 일정을 정리해줘"},
]
```

## For AI Agents

### Using Experiment Utils

**In experiment runners:**

```python
from benchmarking.utils.experiment_utils import (
    load_queries,
    call_query,
    aggregate_results
)

# Load queries
queries = load_queries(Path("scripts/data/smartfarm_eval.jsonl"))

# Run queries
all_results = []
for query in queries:
    result = call_query(
        base_url="http://127.0.0.1:41177",
        question=query["question"],
        ranker="none",
        top_k=4
    )
    all_results.append(result)

# Aggregate
summary = aggregate_results(all_results)
```

### Common Patterns

**Batch evaluation loop:**
```python
results_by_ranker = {}
for ranker in ["none", "llm"]:
    results = []
    for query in queries:
        result = call_query(base_url, query["question"], ranker, k=4)
        results.append(result)
    results_by_ranker[ranker] = aggregate_results(results)
```

**Multi-config testing:**
```python
configs = [
    {"ranker": "none", "top_k": 4},
    {"ranker": "llm", "top_k": 4},
    {"ranker": "none", "top_k": 8},
]

for config in configs:
    results = []
    for query in queries:
        result = call_query(
            base_url,
            query["question"],
            config["ranker"],
            config["top_k"]
        )
        results.append(result)
    summary = aggregate_results(results)
    save_results(f"{config['ranker']}_k{config['top_k']}.json", summary)
```

### Adding New Utilities

1. **For common operations**, add function to `experiment_utils.py`
2. **Document parameters** with type hints
3. **Include docstring** with Args, Returns, Raises sections
4. **Add to imports** in `__init__.py` if widely used

Example:

```python
def compute_category_metrics(
    results: List[Dict[str, Any]],
    queries: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """Group results by question category and compute metrics per group.

    Args:
        results: Per-query result dicts
        queries: Original query list (for category labels)

    Returns:
        Dict mapping category -> metric dict
    """
    metrics_by_category = {}
    for category in set(q["category"] for q in queries):
        category_results = [
            r for r, q in zip(results, queries)
            if q["category"] == category
        ]
        metrics_by_category[category] = aggregate_results(category_results)
    return metrics_by_category
```

## Conventions

- All functions have explicit type hints (Python 3.9+)
- Docstrings follow Google style (Args, Returns, Raises)
- JSON I/O functions handle UTF-8 explicitly
- Error handling uses descriptive messages
- Timeout for API calls: 30 seconds (adjustable)
- Null/None handling explicit (no silent failures)

## Integration Points

- **Imported by**: All experiment runners (beir_benchmark.py, pathrag_lt_benchmark.py, ragas_eval.py, etc.)
- **Used for**: Query loading, API communication, result aggregation
- **Outputs to**: JSON files in `output/experiments/`
- **Consumed by**: Metric computers and reporters

## API Response Format

Expected response structure from `/query` endpoint:

```json
{
  "question": "토마토 온실 생육 최적 온도는?",
  "documents": [
    {
      "doc_id": "doc_001",
      "title": "토마토 온실 재배",
      "content": "...",
      "score": 0.87
    }
  ],
  "latency_ms": 245,
  "top_k": 4,
  "ranker": "none"
}
```

## Error Handling

**Network errors:**
- Retries with exponential backoff (3 attempts)
- Raises `ConnectionError` if all attempts fail

**Invalid queries:**
- Returns empty document list
- Includes error message in response

**Timeout:**
- Defaults to 30 seconds per query
- Configurable via `QUERY_TIMEOUT_SEC`

## Notes

- Utility functions are **synchronous** (not async)
- Network calls include latency measurement
- Query loading tolerates missing files (fallback queries)
- Aggregation functions assume non-empty input lists
- All text handling is UTF-8 (SmartFarm Korean content)
