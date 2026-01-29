<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-28 | Updated: 2026-01-28 -->

# runners/

Benchmark execution scripts for orchestrating experiments on various environments.

## Key Files
| File | Description |
|------|-------------|
| `RunAllExperiments.sh` | Bash: Run full pipeline (chunking, batch eval, multi-experiment, A/B comparison) |
| `EdgeProfilingRunner.sh` | Bash: Profile on edge device with ablation mode support |
| `run_edge_experiments.sh` | Bash: Lightweight edge benchmark runner |
| `RunOrinBenchmarksDocker.sh` | Bash: Execute benchmarks in Docker on Jetson Orin |
| `RunOrinRagDocker.sh` | Bash: Run RAG server + benchmarks in Docker |
| `RunBeirBenchmark.sh` | Bash: BEIR 표준 벤치마크 데이터셋 평가 실행 |

## RunAllExperiments.sh

Orchestrates the full experimental pipeline.

### Prerequisites

1. **FastAPI server running:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 41177
   ```

2. **Evaluation JSONL file:**
   ```
   scripts/data/smartfarm_eval.jsonl
   ```

3. **Python environment activated:**
   ```bash
   source venv/bin/activate  # or conda activate env
   ```

### Usage

**Basic run:**
```bash
bash runners/RunAllExperiments.sh \
  --host http://127.0.0.1:41177 \
  --input scripts/data/smartfarm_eval.jsonl \
  --out-dir /tmp/era_rag_results
```

**With ablation mode (PathRAG/Cache combinations):**
```bash
bash runners/RunAllExperiments.sh \
  --host http://127.0.0.1:41177 \
  --input scripts/data/smartfarm_eval.jsonl \
  --out-dir /tmp/era_rag_ablation_runs \
  --ablate-path-cache
```

### Execution Steps

1. **Chunking Configuration Evaluation**
   - Tests CHUNK_SIZE/STRIDE combinations
   - Outputs: `{OUT_DIR}/chunking/chunking_eval_results.json`
   - Reporter: Generates `chunking_summary.md`

2. **Batch Evaluation**
   - Runs with `ranker=none` and `ranker=llm`
   - Measures latency and success rate
   - Outputs: `{OUT_DIR}/batch/batch_*.json`

3. **Multi-Experiment Runner**
   - Runs across multiple ranker/top_k configs
   - Aggregates results
   - Outputs: `{OUT_DIR}/experiments/*.json`

### Output Structure

```
/tmp/era_rag_results/
├── chunking/
│   ├── chunking_eval_results.json
│   └── chunking_summary.md
├── batch/
│   ├── batch_none_top4.json
│   └── batch_llm_top4.json
└── experiments/
    └── *.json
```

## EdgeProfilingRunner.sh

Specializes in edge device profiling with ablation support.

### Features

- **Cold start timing**: Measures initialization overhead
- **Steady-state latency**: Query latency after warmup
- **Memory profiling**: Peak memory during operations
- **Throughput**: Queries per second (QPS)
- **Ablation mode**: Test component combinations

### Usage

```bash
bash runners/EdgeProfilingRunner.sh \
  --host http://192.168.1.100:41177 \  # Edge device IP
  --input scripts/data/smartfarm_eval.jsonl \
  --out-dir /tmp/edge_results \
  --ablate-path-cache  # Optional: test component combinations
```

### Ablation Mode Details

In ablation mode, tests all combinations:
- ENABLE_PATHRAG: true/false
- ENABLE_CACHE: true/false

Results grouped by:
```
/tmp/edge_results/
├── path_on_cache_on/
│   └── batch/
├── path_on_cache_off/
│   └── batch/
├── path_off_cache_on/
│   └── batch/
└── path_off_cache_off/
    └── batch/
```

## run_edge_experiments.sh

Lightweight alternative for quick edge benchmarking.

### Usage

```bash
bash runners/run_edge_experiments.sh \
  --device-ip 192.168.1.100 \
  --port 41177 \
  --iterations 50
```

### Measurements

- Query latency p50, p95, p99
- Memory overhead
- Cold start time

## RunOrinBenchmarksDocker.sh

Docker-based benchmark execution on Jetson Orin.

### Prerequisites

- **Docker installed** on Jetson
- **Image built**: See `docker/Dockerfile`
- **GPU accessible** from container

### Usage

```bash
bash runners/RunOrinBenchmarksDocker.sh \
  --image smartfarm-search:orin-latest \
  --input /data/smartfarm_eval.jsonl \
  --output /results
```

### How It Works

1. Builds benchmark Docker image (if needed)
2. Runs container with GPU access
3. Mounts data volumes for corpus/QA
4. Captures results on host system

## RunOrinRagDocker.sh

End-to-end: Runs RAG server + benchmarks in Docker.

### Usage

```bash
bash runners/RunOrinRagDocker.sh \
  --docker-compose-file docker/docker-compose.yml \
  --input /data/smartfarm_eval.jsonl \
  --output /results
```

### Services

- **api**: FastAPI RAG server
- **benchmark**: Experiment runner (waits for API ready)
- **results**: Volume mount for outputs

## RunBeirBenchmark.sh

BEIR 표준 벤치마크 데이터셋 평가를 실행합니다.

### Features

- 다양한 BEIR 데이터셋 지원 (arguana, fiqa, nfcorpus, scidocs, scifact 등)
- 자동 데이터셋 다운로드 및 캐싱
- 표준 IR 메트릭 평가 (NDCG@10, MAP, Recall@100 등)

### Usage

```bash
bash runners/RunBeirBenchmark.sh \
  --datasets arguana,fiqa,nfcorpus \
  --output-dir /tmp/beir_results
```

### Supported Datasets

| Dataset | Domain | Queries |
|---------|--------|---------|
| arguana | Argument retrieval | 1,406 |
| fiqa | Financial QA | 648 |
| nfcorpus | Medical/Nutrition | 323 |
| scidocs | Scientific | 1,000 |
| scifact | Scientific fact | 300 |

### Output Structure

```
/tmp/beir_results/
├── arguana/
│   ├── results.json
│   └── metrics_summary.md
├── fiqa/
│   └── ...
└── combined_results.json
```

## For AI Agents

### Adding New Runners

1. Create bash script in `runners/` directory
2. Include standard argument parsing (--host, --input, --output)
3. Validate prerequisites
4. Run Python experiment scripts with stderr/stdout logging
5. Aggregate results to output directory
6. Make executable: `chmod +x runners/my_runner.sh`

### Environment Variables

Common variables used by runners:

| Variable | Purpose |
|----------|---------|
| `ENABLE_PATHRAG` | Toggle PathRAG component |
| `ENABLE_CACHE` | Toggle caching layer |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `OMP_NUM_THREADS` | Parallelism |

### Integration with CI/CD

Runners are idempotent and suitable for automation:
```yaml
# Example: GitHub Actions
- name: Run RAG Benchmarks
  run: bash runners/RunAllExperiments.sh \
    --host http://localhost:41177 \
    --input test_data.jsonl \
    --out-dir results
```

### Logging and Debugging

All scripts support piping to log files:
```bash
bash runners/RunAllExperiments.sh ... 2>&1 | tee benchmark.log
```

## Conventions

- Scripts exit with code 0 on success, non-zero on failure
- All paths can be relative or absolute
- Output directories created automatically (`mkdir -p`)
- JSON results compatible with Python reporters
- Timestamps use `date +%s` (Unix epoch) for ordering
- Verbose mode available via `--verbose` flag

## Notes

- Server must be running before benchmark script
- Ablation mode significantly increases total runtime (4x factor)
- Edge profiling requires target device with port access
- Docker runners require Docker daemon running
- Results feed into `benchmarking/reporters/` for formatting
