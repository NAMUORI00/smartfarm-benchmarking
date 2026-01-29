<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-28 | Updated: 2026-01-28 -->

# baselines/

Baseline retriever implementations for performance comparison in ablation studies.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package initialization |
| `dense_only.py` | FAISS-based dense semantic search only (embedding similarity) |
| `sparse_only.py` | BM25/TF-IDF lexical search only (keyword matching) |
| `naive_hybrid.py` | Simple weighted sum of dense + sparse scores (fixed alpha=0.5) |
| `rrf_hybrid.py` | Reciprocal Rank Fusion - merges rankings without weighting |
| `lightrag.py` | LightRAG integration for graph-based retrieval comparison |

## Architecture

All baselines inherit from `BaseRetriever` protocol:
```python
class BaseRetriever(Protocol):
    def search(self, q: str, k: int = 4) -> List[SourceDoc]: ...
```

### Baseline Characteristics

**DenseOnlyRetriever**
- Uses FAISS embedding index
- No sparse search, no graph traversal
- Baseline for "embedding-only" experimental condition

**SparseOnlyRetriever**
- Uses BM25 term matching
- No dense embeddings
- Baseline for "lexical-only" experimental condition

**NaiveHybridRetriever**
- Combines dense + sparse with fixed weights (α=0.5)
- Score = α × dense_score + (1-α) × sparse_score
- Naive fusion baseline for hybrid methods

**RRFHybridRetriever**
- Reciprocal Rank Fusion: `score = 1/(rank+60)`
- Combines rankings without tuning weights
- Parameter-free fusion method for comparison

**LightRAGRetriever**
- Graph-based retrieval using entity relationships
- Builds dynamic entity graphs from corpus
- Comparison with knowledge graph methods

## For AI Agents

### Adding New Baselines
1. Implement `BaseRetriever` protocol with `search(q, k)` method
2. Place in `baselines/` directory with clear name
3. Add to `benchmark_config.yaml` under `experiments.baseline.baselines`
4. Import in experiment runners (e.g., `beir_benchmark.py`, `pathrag_lt_benchmark.py`)

### Key Constraints
- All baselines must return top-K results in rank order
- Document relevance must be scored [0, 1] or comparable scale
- Cold start time measured in `ete_latency_benchmark.py`
- Results aggregated by `PaperResultsReporter.py`

### Testing Baselines
```bash
python -c "from benchmarking.baselines import DenseOnlyRetriever; ..."
python -m benchmarking.experiments.beir_benchmark
```

## Conventions

- Baselines compared on: Precision@4, Recall@4, MRR, NDCG@4
- K=4 is primary metric per paper section 5.1.3
- Results output to `output/` directory
- Best performers highlighted in LaTeX tables by `PaperResultsReporter.py`

## Usage in Experiments

### BEIR Benchmark
```bash
python -m benchmarking.experiments.beir_benchmark \
    --dataset smartfarm \
    --output-dir output/beir_results
```

### PathRAG Benchmark
```bash
python -m benchmarking.experiments.pathrag_lt_benchmark \
    --corpus path/to/corpus.jsonl \
    --qa-file path/to/qa_dataset.jsonl \
    --output-dir output/pathrag_results
```

## Notes

- Dense baseline requires embedding model (config: `models.embedding.model_id`)
- Sparse baseline requires indexed vocabulary (built on-the-fly or cached)
- Hybrid baselines assume both dense and sparse indices are available
- Results feed into paper Tables 1 and comparisons in Section 5.1
