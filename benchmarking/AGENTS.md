<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-28 | Updated: 2026-01-28 -->

# benchmarking/

RAG evaluation framework for experiments, metrics, and reporting.

## STRUCTURE

```
benchmarking/
├── run_benchmark.py      # Main entry point
├── experiments/          # Experiment runners
│   ├── beir_benchmark.py
│   ├── pathrag_lt_benchmark.py
│   ├── hybrid_pathrag_lt_benchmark.py
│   ├── llm_graph_benchmark.py
│   ├── lightrag_comparison.py
│   ├── lightrag_graph_test.py
│   ├── seed_strategy_benchmark.py
│   ├── seedbench_comparison.py
│   ├── ete_latency_benchmark.py
│   ├── RagExperimentRunner.py
│   ├── batch_eval_rag.py
│   ├── ragas_eval.py
│   └── eval_chunking_configs.py
├── baselines/            # Baseline implementations
│   ├── dense_only.py
│   ├── sparse_only.py
│   ├── naive_hybrid.py
│   ├── rrf_hybrid.py
│   ├── adaptive_hybrid.py
│   └── lightrag.py
├── metrics/              # Evaluation metrics
│   ├── retrieval_metrics.py
│   ├── qa_metrics.py
│   └── domain_metrics.py
├── reporters/            # Result formatters
│   ├── PaperResultsReporter.py
│   ├── AblationReporter.py
│   ├── FigureGenerator.py
│   ├── ChunkingExperimentReporter.py
│   └── ProfilingExperimentReporter.py
├── config/               # Experiment configs (YAML)
├── data/                 # Test datasets
│   └── smartfarm_eval.jsonl
├── docker/               # Containerized benchmarking
├── runners/              # Task runners
├── scripts/              # Utility scripts
├── tests/                # Test suite
└── utils/                # Helper utilities
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Run BEIR benchmark | `experiments/beir_benchmark.py` |
| PathRAG-LT evaluation | `experiments/pathrag_lt_benchmark.py` |
| Hybrid PathRAG tests | `experiments/hybrid_pathrag_lt_benchmark.py` |
| LLM graph experiments | `experiments/llm_graph_benchmark.py` |
| Seed strategy tests | `experiments/seed_strategy_benchmark.py` |
| LightRAG comparison | `experiments/lightrag_comparison.py` |
| RAGAS evaluation | `experiments/ragas_eval.py` |
| Add new baseline | `baselines/` → implement `BaseRetriever` |
| Add new metric | `metrics/` → follow existing pattern |
| Generate paper tables | `reporters/PaperResultsReporter.py` |

## COMMANDS

```bash
# Run full benchmark suite
python benchmarking/run_benchmark.py

# Run BEIR benchmark
python benchmarking/experiments/beir_benchmark.py

# Run PathRAG-LT benchmark
python benchmarking/experiments/pathrag_lt_benchmark.py

# Run RAGAS evaluation
python benchmarking/experiments/ragas_eval.py

# Generate paper figures
python benchmarking/reporters/FigureGenerator.py
```

## CONVENTIONS

- Experiments output to `output/YYYY-MM-DD_HHMMSS/`
- Results in JSON + Markdown format
- Reporters generate LaTeX-ready tables

## NOTES

- Uses `smartfarm_eval.jsonl` dataset from `data/`
- Docker setup in `docker/` for reproducible evaluation
- Results feed into workspace `docs/` (paper/validation)
