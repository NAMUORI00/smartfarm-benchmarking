<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-28 | Updated: 2026-01-28 -->

# reporters/

Result formatters for paper tables, figures, and reports.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package initialization |
| `PaperResultsReporter.py` | Generate LaTeX tables and figure data for paper submission |
| `AblationReporter.py` | Format ablation study results (component contribution analysis) |
| `FigureGenerator.py` | Create visualization-ready JSON for charts and plots |
| `ChunkingExperimentReporter.py` | Report chunking configuration evaluation |
| `ProfilingExperimentReporter.py` | Report edge device profiling results |

## PaperResultsReporter.py

Main reporter for paper figures and tables.

### Generated Outputs

| Output | Format | Purpose |
|--------|--------|---------|
| `table1_baseline.tex` | LaTeX | Table 1: Baseline Comparison (P@4, R@4, MRR, NDCG@4) |
| `table2_ablation.tex` | LaTeX | Table 2: Ablation Study (component contributions with deltas) |
| `table3_edge.tex` | LaTeX | Table 3: Edge Performance (cold start, latency percentiles, throughput) |
| `figure_data.json` | JSON | Bar chart data, line charts, latency distributions |
| `experiment_results.md` | Markdown | Human-readable summary of all results |

### LaTeX Table Features

- **Best values highlighted** with `\textbf{}` for visual emphasis
- **Method names** formatted for paper readability (e.g., "Dense-only", "RRF Hybrid")
- **Metric columns** aligned with decimal points
- **Symbols** for checkmarks (✓) and delta indicators (Δ)

### Usage

```python
from benchmarking.reporters import PaperResultsReporter
from pathlib import Path

reporter = PaperResultsReporter(
    experiments_dir=Path("output/experiments"),
    output_dir=Path("output/paper")
)
reporter.generate_all()
```

### Expected Input Structure

```
output/experiments/
├── baselines/
│   ├── baseline_results.json
│   └── baseline_summary.json
├── ablation/
│   ├── ablation_results.json
│   └── ablation_summary.json
├── edge/
│   ├── edge_benchmark_results.json
│   └── edge_benchmark_summary.json
└── data/
    ├── benchmark_queries.json
    └── ground_truth.json
```

### Markdown Report Format

Auto-generated markdown includes:
- Baseline comparison table
- Ablation study progression
- Edge performance metrics
- Summary statistics

## AblationReporter.py

Specializes in ablation study formatting.

### Key Features

- **Configuration tracking**: Identifies which components enabled/disabled
- **Progressive improvement**: Shows delta from base configuration
- **Component matrix**: DAT + Ontology combinations
- **Statistical summaries**: Mean and stddev per config

### Configuration Mappings

```
base_naive                    → Baseline (Naive Hybrid, α=0.5)
+dat                         → + Dynamic Alpha Tuning
+dat+onto                    → + Ontology Matching
full_proposed                → Full Proposed Method
```

## FigureGenerator.py

Creates JSON data for visualization.

### Figure Types Generated

**Figure 1: Baseline Comparison**
```json
{
  "methods": ["dense_only", "sparse_only", "naive_hybrid", "rrf_hybrid", "proposed"],
  "metrics": {
    "precision": [0.65, 0.58, 0.72, 0.74, 0.83],
    "recall": [0.72, 0.64, 0.78, 0.79, 0.88],
    "mrr": [0.70, 0.62, 0.75, 0.77, 0.85],
    "ndcg": [0.73, 0.65, 0.76, 0.78, 0.86]
  }
}
```

**Figure 2: Ablation Progression**
```json
{
  "configs": ["base", "+dat", "+dat+onto", "full"],
  "mrr": [0.75, 0.77, 0.80, 0.85],
  "delta_mrr": [0.0, 0.02, 0.05, 0.10]
}
```

**Figure 3: Latency Distribution**
```json
{
  "percentiles": ["p50", "p75", "p90", "p95", "p99"],
  "values": [45.2, 62.1, 78.5, 92.3, 156.7]
}
```

## For AI Agents

### Running Reporters

**Generate all paper outputs:**
```bash
python -m benchmarking.reporters.PaperResultsReporter \
    --experiments-dir output/experiments \
    --output-dir output/paper
```

**Output to specific directory:**
```bash
python -m benchmarking.reporters.PaperResultsReporter \
    --experiments-dir output/experiments/2026-01-27_142530 \
    --output-dir output/paper/latest
```

### Customizing Reporter Output

1. **Modify table column order:**
   - Edit `method_order` list in `generate_baseline_table()`
   - Edit `config_order` list in `generate_ablation_table()`

2. **Change LaTeX formatting:**
   - Edit `\begin{table}...\end{table}` strings
   - Update `\caption{}` and `\label{}` tags
   - Modify decimal precision in format strings (`.4f`, `.2f`)

3. **Add new metrics to figures:**
   - Update `figure_data["baseline_comparison"]["metrics"]` dict
   - Ensure metric exists in summary JSON files

### Integration with Paper

Generated files go to:
```
output/paper/
├── table1_baseline.tex        ← \include{} in main.tex
├── table2_ablation.tex        ← \include{} in main.tex
├── table3_edge.tex            ← \include{} in main.tex
├── figure_data.json           ← Feed to plotting scripts
└── experiment_results.md      ← Reference for paper text
```

## ChunkingExperimentReporter.py

Reports chunking configuration evaluation.

- Compares CHUNK_SIZE / STRIDE combinations
- Shows impact on retrieval metrics
- Helps identify optimal configuration

## ProfilingExperimentReporter.py

Reports edge device profiling.

- Cold start time breakdown
- Per-operation latency
- Memory allocation patterns
- Throughput measurements

## Conventions

- All LaTeX is production-ready (no editing needed)
- Metric ordering: Precision → Recall → MRR → NDCG
- Configs ordered by addition of components
- Best values bolded (helps reviewers spot improvements)
- Numbers formatted to 4 decimal places (sufficient precision)
- Markdown includes both numbers and brief interpretation

## Output File Naming

| File | Purpose |
|------|---------|
| `table*.tex` | Include directly in paper |
| `figure_data.json` | Process with matplotlib/gnuplot |
| `experiment_results.md` | Reference during writing |

## Notes

- All reporters assume JSON input from experiments
- Reporter gracefully handles missing data (outputs "% No data available")
- Delta calculations in ablation use base_naive as reference
- LaTeX tables include `\toprule`, `\midrule`, `\bottomrule` (booktabs style)
- Markdown report serves as human-readable alternative to LaTeX
