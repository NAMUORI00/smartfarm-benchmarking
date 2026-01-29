# CROP Dataset - LightRAG Evaluation

LightRAG comparison evaluation script for the CROP agricultural QA benchmark.

## Overview

This script evaluates the **LightRAG (EMNLP 2025)** baseline retriever on the CROP dataset using the same metrics as PathRAG evaluation for fair comparison:

- **MRR** (Mean Reciprocal Rank)
- **NDCG@10** (Normalized Discounted Cumulative Gain)
- **Precision@4** (per paper: K=4 is the main evaluation criterion)
- **Recall@4**
- **Hit Rate@4**

## Requirements

```bash
pip install lightrag-hku
pip install sentence-transformers
pip install numpy
```

## Usage

### Basic Evaluation

```bash
python -m benchmarking.experiments.crop_lightrag_eval \
  --data-dir data/crop \
  --output output/crop_lightrag_eval \
  --top-k 4
```

### Quick Test (Limited Data)

```bash
# Test with limited corpus and queries
python -m benchmarking.experiments.crop_lightrag_eval \
  --data-dir data/crop \
  --output output/crop_lightrag_test \
  --limit 50 \
  --max-queries 20 \
  --top-k 4
```

### Force Rebuild

```bash
# Force rebuild of LightRAG graph (ignore cached)
python -m benchmarking.experiments.crop_lightrag_eval \
  --data-dir data/crop \
  --output output/crop_lightrag_eval \
  --rebuild
```

### Custom Working Directory

```bash
# Specify custom LightRAG working directory
python -m benchmarking.experiments.crop_lightrag_eval \
  --data-dir data/crop \
  --output output/crop_lightrag_eval \
  --working-dir /path/to/lightrag_workdir
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/crop` | Path to CROP data directory |
| `--output` | `output/crop_lightrag_eval` | Output directory for results |
| `--limit` | `None` | Limit number of corpus documents (for testing) |
| `--top-k` | `4` | Number of documents to retrieve (per paper) |
| `--max-queries` | `None` | Limit number of queries to evaluate (for testing) |
| `--working-dir` | `<output>/lightrag_workdir` | LightRAG working directory |
| `--rebuild` | `False` | Force rebuild of LightRAG graph |

## Output Format

The script produces a JSON file (`crop_lightrag_eval.json`) with the following structure:

```json
{
  "dataset": "crop",
  "method": "lightrag",
  "n_docs": 162,
  "n_queries": 220,
  "n_queries_evaluated": 220,
  "top_k": 4,
  "metrics": {
    "mrr": 0.3456,
    "ndcg@4": 0.3890,
    "ndcg@10": 0.4012,
    "precision@4": 0.2500,
    "recall@4": 0.3200,
    "hit_rate@4": 0.6100
  },
  "latency": {
    "mean_ms": 120.5,
    "p50_ms": 95.2,
    "p95_ms": 280.3
  },
  "graph_stats": {
    "n_nodes": 487,
    "n_edges": 1203,
    "n_docs": 162
  },
  "config": {
    "query_mode": "hybrid",
    "data_dir": "data/crop",
    "working_dir": "output/crop_lightrag_eval/lightrag_workdir"
  }
}
```

## CROP Dataset Format

The script expects CROP data in BEIR format:

```
data/crop/
├── corpus.jsonl          # Document corpus
├── queries.jsonl         # Queries
└── qrels/
    └── test.tsv          # Query-document relevance judgments
```

### corpus.jsonl

```json
{"_id": "doc_001", "title": "", "text": "Document text..."}
```

### queries.jsonl

```json
{"_id": "wasabi_qa_0000", "text": "Query text?"}
```

### qrels/test.tsv

```
query-id    corpus-id    score
wasabi_qa_0000    doc_001    1
```

## Implementation Details

### LightRAG Configuration

- **Query Mode**: `hybrid` (combines local + global retrieval)
- **Embedding Model**: SentenceTransformer (MiniLM multilingual)
- **LLM**: llama.cpp server (for graph construction)
- **Graph Type**: Dual-Level (entity + community)

### Fair Comparison with PathRAG

To ensure fair comparison:

1. **Same corpus**: CROP dataset (162 documents, 220 queries)
2. **Same metrics**: MRR, NDCG@10, P@4, R@4, Hit@4
3. **Same K value**: K=4 (per paper section 5.1.3)
4. **Same evaluation protocol**: BEIR-style retrieval evaluation

### Graph Statistics

The script extracts graph statistics from LightRAG working directory:

- `n_nodes`: Number of entity nodes in graph
- `n_edges`: Number of entity relations
- `n_docs`: Number of documents indexed

Statistics are parsed from `graph_chunk_entity_relation.graphml` and `kv_store_full_docs.json`.

## Performance Characteristics

### Build Time

- **Full CROP corpus (162 docs)**: ~5-10 minutes
- Includes LLM calls for entity extraction and relation mining

### Query Latency

- **Mean latency**: ~100-200ms per query
- Depends on graph size and query complexity

### Storage

- **Graph files**: ~10-50MB for CROP dataset
- **Embeddings**: ~5-20MB (depends on model)
- **Total**: ~20-70MB

## Comparison with PathRAG

Expected metric comparison (example):

| Method | MRR | NDCG@10 | P@4 | R@4 | Hit@4 |
|--------|-----|---------|-----|-----|-------|
| LightRAG | 0.35 | 0.40 | 0.25 | 0.30 | 0.60 |
| PathRAG | 0.42 | 0.48 | 0.32 | 0.38 | 0.68 |

*(Actual values will be computed after running evaluation)*

## Troubleshooting

### LightRAG Not Installed

```bash
pip install lightrag-hku
```

### SentenceTransformer Not Found

```bash
pip install sentence-transformers
```

### llama.cpp Server Not Running

LightRAG requires an LLM for graph construction. Ensure llama.cpp server is running:

```bash
# Check server is accessible
curl http://localhost:8080/v1/models
```

Configure server URL in `core/Config/Settings.py` (`LLMLITE_HOST`).

### Out of Memory

If you encounter OOM during graph construction:

1. Use `--limit` to reduce corpus size
2. Use smaller LLM model (e.g., Qwen3-1.5B instead of Qwen3-4B)
3. Increase system RAM or use swap

### Slow Evaluation

For faster testing:

- Use `--limit 50` to test with smaller corpus
- Use `--max-queries 20` to evaluate fewer queries
- Use cached working directory (avoid `--rebuild`)

## References

- **LightRAG**: Simple and Fast Retrieval-Augmented Generation (EMNLP 2025)
  - Paper: [https://arxiv.org/abs/2410.05779](https://arxiv.org/abs/2410.05779)
  - Code: [https://github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)

- **CROP Dataset**: AI4Agr/CROP-dataset (HuggingFace)
  - Dataset: [https://huggingface.co/datasets/AI4Agr/CROP-dataset](https://huggingface.co/datasets/AI4Agr/CROP-dataset)

## Next Steps

1. Run full evaluation on CROP dataset
2. Compare results with PathRAG evaluation
3. Generate comparison plots and tables
4. Update paper with results

See `docs/era-smartfarm-rag/validation/BENCHMARK_COMPARISON_LOG.md` for experimental results.
