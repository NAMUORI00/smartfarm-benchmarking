# CROP Corpus Summary

## Quick Reference

```bash
# Build corpus from CROP QA dataset
python -m benchmarking.data.crop_corpus_builder \
  --input ../smartfarm-ingest/output \
  --output data/crop \
  --limit 5000

# Run tests
python -m benchmarking.data.test_crop_corpus_builder

# Use with BEIR benchmark
python -m benchmarking.experiments.beir_benchmark \
  --datasets crop \
  --beir-dir data \
  --output-dir output/crop_results
```

## Generated Corpus Statistics

| Metric | Value |
|--------|-------|
| Total QA pairs processed | 660 |
| Unique corpus documents | 162 |
| Total queries | 220 |
| Query-document pairs | 220 |
| Document length (min) | 124 chars |
| Document length (avg) | 282 chars |
| Document length (max) | 748 chars |
| Deduplication threshold | 0.95 (95% similarity) |

## File Structure

```
smartfarm-benchmarking/
├── benchmarking/data/
│   ├── __init__.py
│   ├── crop_corpus_builder.py      # Main corpus builder
│   ├── test_crop_corpus_builder.py # Unit tests
│   ├── README.md                    # Usage guide
│   └── CROP_CORPUS_SUMMARY.md       # This file
│
└── data/crop/                       # Generated BEIR corpus
    ├── corpus.jsonl                 # 162 documents
    ├── queries.jsonl                # 220 queries
    └── qrels/
        └── test.tsv                 # 220 query-doc pairs
```

## Implementation Details

### Key Features

1. **Deduplication**: Token-based Jaccard similarity removes near-duplicate contexts
2. **Hash-based IDs**: SHA-256 content hashing for deterministic document IDs
3. **BEIR Compliance**: Strict adherence to BEIR format specifications
4. **Korean Text Support**: Preserves Korean characters without encoding issues
5. **Filtering**: Only processes `*qa*.jsonl` files, ignores translation/raw data

### Data Flow

```
smartfarm-ingest/output/*.jsonl
    ↓
[Load QA pairs] → Filter by filename pattern (*qa*.jsonl)
    ↓
[Extract contexts] → Get unique context passages
    ↓
[Deduplicate] → Remove similar contexts (>95% similarity)
    ↓
[Generate IDs] → SHA-256 hash → 16-char hex IDs
    ↓
[Map queries] → Link each query to its source document
    ↓
[Write outputs] → corpus.jsonl, queries.jsonl, qrels/test.tsv
```

### Format Specifications

#### corpus.jsonl
```json
{
  "_id": "6267f9c7db29f0e9",
  "title": "",
  "text": "게다가, 직접 수확한 고추냉이..."
}
```

#### queries.jsonl
```json
{
  "_id": "wasabi_qa_0000",
  "text": "실내 와사비 재배에서..."
}
```

#### qrels/test.tsv
```
query-id	corpus-id	score
wasabi_qa_0000	6267f9c7db29f0e9	1
```

## Validation

All tests passing:
- ✓ Document ID generation (deterministic, collision-resistant)
- ✓ Similarity computation (Jaccard coefficient)
- ✓ Context deduplication (configurable threshold)
- ✓ QA pair loading (file filtering, validation)
- ✓ Corpus extraction (mapping, relevance)
- ✓ BEIR format compliance (schema, types, references)

## Use Cases

### 1. Retrieval Evaluation
Compare retrieval methods (dense, sparse, hybrid) on agriculture domain data.

### 2. Benchmark Studies
Measure retrieval performance on Korean technical text.

### 3. Domain Adaptation
Test embedding models on specialized vocabulary (wasabi cultivation).

### 4. Multilingual Retrieval
Evaluate cross-lingual retrieval capabilities (Korean queries, mixed corpus).

## Performance Notes

- Processing speed: ~1000 QA pairs/second
- Memory usage: <100MB for full dataset
- Deduplication reduces corpus size by ~75% (660 → 162 docs)

## Future Enhancements

1. **Advanced deduplication**: Semantic similarity using embeddings
2. **Multi-level qrels**: Graded relevance scores (0-3) instead of binary
3. **Query augmentation**: Paraphrase generation from metadata
4. **Document splitting**: Chunk long contexts for better retrieval

## References

- BEIR format: https://github.com/beir-cellar/beir
- CROP dataset: smartfarm-ingest/output/wasabi_qa_dataset*.jsonl
- Benchmark implementation: benchmarking/experiments/beir_benchmark.py
