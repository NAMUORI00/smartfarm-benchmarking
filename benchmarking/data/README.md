# CROP Dataset Corpus Builder

This module extracts unique context passages from CROP-dataset QA pairs and generates BEIR-compatible corpus files for retrieval evaluation.

## Overview

The CROP corpus builder processes QA pairs from the CROP-dataset (wasabi agriculture domain) and creates:
- **corpus.jsonl**: Unique deduplicated context documents
- **queries.jsonl**: Questions from QA pairs
- **qrels/test.tsv**: Query-document relevance judgments

## Usage

### Basic Usage

```bash
python -m benchmarking.data.crop_corpus_builder \
  --input ../smartfarm-ingest/output \
  --output data/crop \
  --limit 5000
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input` | Path | Required | Input directory containing JSONL files with QA pairs |
| `--output` | Path | Required | Output directory for corpus, queries, and qrels |
| `--limit` | int | None | Limit number of QA pairs to process (default: all) |
| `--dedup-threshold` | float | 0.95 | Similarity threshold for deduplication (0.0-1.0) |

### Input Format

The builder expects JSONL files with the following structure:

```json
{
  "id": "wasabi_qa_0000",
  "question": "실내 와사비 재배에서...",
  "answer": "실내 와사비 재배...",
  "context": "게다가, 직접 수확한 고추냉이...",
  "category": "환경조건",
  "complexity": "basic",
  "source_ids": ["web_source#c24"],
  "metadata": {...}
}
```

Only files matching the pattern `*qa*.jsonl` are processed.

### Output Format

#### corpus.jsonl
BEIR-compatible corpus documents:

```json
{"_id": "6267f9c7db29f0e9", "title": "", "text": "document context..."}
```

#### queries.jsonl
BEIR-compatible queries:

```json
{"_id": "wasabi_qa_0000", "text": "question text..."}
```

#### qrels/test.tsv
Tab-separated relevance judgments with header:

```
query-id	corpus-id	score
wasabi_qa_0000	6267f9c7db29f0e9	1
```

## Features

### Deduplication

The builder uses token-based Jaccard similarity to deduplicate similar contexts:
- Default threshold: 0.95 (95% similarity)
- Adjustable via `--dedup-threshold`
- Reduces corpus size while maintaining coverage

### Document IDs

Document IDs are generated using SHA-256 content hashing:
- Deterministic (same content = same ID)
- Compact (16 hex characters)
- Collision-resistant

### Query-Document Mapping

Each query is mapped to the document from which its answer was derived:
- Binary relevance (score = 1)
- Single document per query (extracted from QA pair context)

## Statistics

Example statistics from full CROP dataset:

```
Loaded: 660 QA pairs
Unique documents: 162 (after deduplication)
Queries: 220
Query-document pairs: 220
```

## Integration with BEIR

The generated corpus can be used directly with BEIR benchmark tools:

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader

corpus, queries, qrels = GenericDataLoader(
    data_folder="data/crop"
).load(split="test")
```

Or with the smartfarm-benchmarking BEIR benchmark:

```bash
python -m benchmarking.experiments.beir_benchmark \
  --datasets crop \
  --beir-dir data \
  --output-dir output/crop_results
```

## File Structure

```
data/crop/
├── corpus.jsonl          # Deduplicated context documents
├── queries.jsonl         # Questions from QA pairs
└── qrels/
    └── test.tsv          # Query-document relevance judgments
```

## Notes

- The builder uses the `context` field as the document text (not `answer` or `output`)
- Empty contexts are automatically filtered out
- Non-QA JSONL files (e.g., translation files) are ignored
- Korean text is preserved without transliteration
