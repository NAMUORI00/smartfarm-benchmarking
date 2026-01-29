<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-28 | Updated: 2026-01-28 -->

# config/

Experiment configuration files in YAML format.

## Key Files
| File | Description |
|------|-------------|
| `benchmark_config.yaml` | Master configuration for all experiments |

## Configuration Structure

The main `benchmark_config.yaml` defines:

### Data Section
- Paths to corpus and QA datasets (relative to project root)
- Minimum validation thresholds (docs, QA pairs)

### Experiments Section
Enables/disables and configures each experiment type:

| Experiment | RQ | Config Keys |
|-------------|-----|------------|
| `baseline` | RQ1 | `enabled`, `top_k`, `baselines` list |
| `ablation` | RQ2 | `enabled`, `top_k`, `configs` (all/specific) |
| `domain` | RQ3 | `enabled`, `top_k`, `analyze_by` (category/complexity) |
| `edge` | RQ4 | `enabled`, `warmup_iterations`, `benchmark_iterations`, `measure` list |

### Models Section
- `embedding.model_id`: Sentence-Transformer or equivalent model
- `embedding.device`: auto/cpu/cuda device selection

### Output Section
- `dir`: Base output directory path
- `timestamp_subdir`: Create timestamped run folders
- `formats`: JSON, Markdown output
- `latex`: Generate LaTeX tables for paper
- `create_latest_link`: Symlink to latest results

### Runtime Section
- `seed`: Random seed for reproducibility (default: 42)
- `verbose`: Logging verbosity
- `progress_bar`: Show progress bars
- `continue_on_error`: Resilience across experiments

## For AI Agents

### Modifying Configurations
1. Edit relevant section in `benchmark_config.yaml`
2. Run validation: `python -m benchmarking.run_benchmark --validate-config`
3. Check output paths are writable and data files exist
4. Re-run experiments with updated settings

### Experiment Selection Examples

**Run only baseline comparison:**
```yaml
experiments:
  baseline:
    enabled: true
  ablation:
    enabled: false
  domain:
    enabled: false
  edge:
    enabled: false
```

**Test with subset of configs:**
```yaml
experiments:
  ablation:
    configs:
      - base_naive
      - +dat
      - full_proposed
```

**Edge benchmark with different target:**
```yaml
experiments:
  edge:
    benchmark_iterations: 100  # Higher for accuracy
    measure:
      - query_latency
      - memory_usage
```

### Configuration Validation
Before running experiments, verify:
- [ ] `data.corpus` path exists and is readable
- [ ] `data.qa_dataset` path exists and is readable
- [ ] `models.embedding.model_id` is valid HuggingFace model
- [ ] `output.dir` parent directory is writable
- [ ] All experiment `enabled` flags are set correctly

## Conventions

- Config is YAML format (NOT TOML, NOT JSON)
- Paths are relative to `smartfarm-benchmarking/` project root
- All paths use forward slashes `/` (universal)
- Comments use `#` and explain non-obvious settings
- Default values should work for most use cases

## Usage

### In Experiment Code
```python
from benchmarking.utils.config import load_benchmark_config

config = load_benchmark_config("benchmarking/config/benchmark_config.yaml")
top_k = config["experiments"]["baseline"]["top_k"]
```

### Via Command Line
```bash
python -m benchmarking.run_benchmark --config benchmarking/config/benchmark_config.yaml
```

## Notes

- Config is source of truth for experiment settings
- Changes persist across runs (use git to track)
- Docker setup may override certain paths (see `docker/benchmark_config.yaml`)
- Model downloads happen automatically on first use
