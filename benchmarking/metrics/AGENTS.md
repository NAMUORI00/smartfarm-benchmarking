<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-28 | Updated: 2026-01-28 -->

# metrics/

Evaluation metrics for RAG performance assessment.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package initialization |
| `retrieval_metrics.py` | Standard IR metrics (Precision@K, Recall@K, MRR, NDCG@K, Hit Rate) |
| `qa_metrics.py` | Question-answering quality metrics (answer relevance, faithfulness, etc.) |
| `domain_metrics.py` | Domain-specific metrics for SmartFarm (crop-specific performance) |

## retrieval_metrics.py

Standard information retrieval evaluation metrics.

### Available Metrics

| Metric | Formula | Range | Use Case |
|--------|---------|-------|----------|
| **Precision@K** | relevant_in_top_k / k | [0, 1] | Fraction of top-K results that are relevant |
| **Recall@K** | relevant_retrieved / total_relevant | [0, 1] | Fraction of all relevant docs that appear in top-K |
| **MRR** | 1 / (rank_of_first_relevant + 1) | [0, 1] | Rank position of first correct result |
| **NDCG@K** | DCG@K / IDCG@K | [0, 1] | Ranking quality with graded relevance |
| **Hit Rate@K** | 1 if relevant in top-K else 0 | {0, 1} | Binary: was anything relevant found? |

### RetrievalMetrics Class

```python
from benchmarking.metrics.retrieval_metrics import RetrievalMetrics

metrics = RetrievalMetrics(k_values=[1, 4, 5, 10])

# Compute for single query
result = metrics.compute_all(
    retrieved=["doc_1", "doc_5", "doc_12", "doc_3"],
    relevant={"doc_1", "doc_5"},
    relevance_scores={"doc_1": 3, "doc_5": 2, "doc_12": 0}
)
# {'mrr': 1.0, 'precision@4': 0.5, 'recall@4': 1.0,
#  'hit_rate@4': 1.0, 'ndcg@4': 0.75, ...}

# Aggregate across queries
all_results = [result1, result2, result3, ...]
summary = metrics.aggregate(all_results)
# {'mean_precision@4': 0.45, 'mean_recall@4': 0.72, ...}
```

## qa_metrics.py

Question-answering specific evaluation metrics.

| Metric | Purpose |
|--------|---------|
| Answer Relevance | Does retrieved context answer the question? |
| Faithfulness | Is the answer grounded in the retrieved documents? |
| RAGAS Metrics | Combined F1-style score for retrieval quality |

## domain_metrics.py

Domain-specific metrics for SmartFarm agricultural data.

| Metric | Purpose |
|--------|---------|
| Crop-Specific Recall | Recall per crop type (토마토, 파프리카, etc.) |
| Category Performance | Precision/Recall by question category (온도, 양액, etc.) |
| Complexity-Aware NDCG | NDCG weighted by question difficulty |
| Seasonal Relevance | Accuracy for time-sensitive queries |

### Example Usage

```python
from benchmarking.metrics.domain_metrics import CropAnalyzer

analyzer = CropAnalyzer(relevance_judgments)
crop_metrics = analyzer.analyze_by_crop(results)
# {'tomato': {'precision': 0.85, 'recall': 0.78},
#  'paprika': {'precision': 0.82, 'recall': 0.75}}

category_metrics = analyzer.analyze_by_category(results)
# {'온도': {'precision': 0.88, 'mrr': 0.92},
#  '양액': {'precision': 0.81, 'mrr': 0.85}}
```

## For AI Agents

### Adding Custom Metrics

1. **Create metric function:**
```python
# In metrics/custom_metrics.py
def my_metric(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute custom metric.

    Args:
        retrieved: Ranked list of doc IDs
        relevant: Set of relevant doc IDs
        k: Number of top results to consider

    Returns:
        Metric score in [0, 1] or appropriate range
    """
    # Implementation
    return score
```

2. **Register in RetrievalMetrics:**
```python
# In retrieval_metrics.py compute_all()
results[f"custom_metric@{k}"] = my_metric(retrieved, relevant, k)
```

3. **Update aggregation:**
```python
# In metrics classes
custom_values = [r.get("custom_metric@4", 0) for r in all_results]
results["mean_custom_metric@4"] = sum(custom_values) / len(custom_values)
```

### Metric K-Values

Primary evaluation uses **K=4** per paper section 5.1.3:
```python
# In config/benchmark_config.yaml or code
k_values: [1, 4, 5, 10]  # K=4 emphasized in results
```

### Computing Metrics for Experiments

**In beir_benchmark.py:**
```python
from benchmarking.metrics.retrieval_metrics import RetrievalMetrics

metrics = RetrievalMetrics()
for query_id, question in qa_pairs:
    retrieved_docs = retriever.search(question, k=4)
    doc_ids = [d.id for d in retrieved_docs]

    result = metrics.compute_all(
        retrieved=doc_ids,
        relevant=ground_truth[query_id]["relevant_ids"],
        relevance_scores=ground_truth[query_id]["scores"]
    )
    all_results.append(result)

summary = metrics.aggregate(all_results)
```

## Conventions

- All metrics normalize to [0, 1] when possible
- NDCG uses DCG formula with log2(i+2) discounting
- MRR returns 0 if no relevant doc found
- Metric names lowercase with underscores (e.g., `precision_at_k`)
- K-dependent metrics suffixed with `@k` (e.g., `ndcg@4`)
- Aggregated metrics prefixed with `mean_` (e.g., `mean_recall@4`)
- Per-query results stored as dict; per-dataset as dict of means

## Usage in Paper

Results from these metrics populate paper tables:

- **Table 1**: Baseline comparison (Precision@4, Recall@4, MRR, NDCG@4)
- **Table 2**: Ablation study (MRR focus, delta from base)
- **Table 3**: Edge performance (latency, memory - from edge_benchmark)
- **Figures**: Per-category analysis using domain_metrics

## Notes

- All metrics are query-level first, aggregated per-dataset
- Relevance judgments are 0-3 scale (0=not relevant, 3=highly relevant)
- NDCG@4 uses ideal ranking from full ground truth
- Hit rate@K useful for "did we find anything" baseline
- MRR emphasizes ranking quality (position of first correct result)
