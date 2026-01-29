#!/usr/bin/env python3
"""ERA-SmartFarm-RAG Unified Benchmark Runner

This module provides a single entry point for running all benchmarking experiments
defined in the paper: baseline comparison, ablation study, domain analysis, and
edge performance benchmarks.

Usage:
    python -m benchmarking.run_benchmark [OPTIONS]
    
    # Run all experiments with default config
    python -m benchmarking.run_benchmark
    
    # Run specific experiments only
    python -m benchmarking.run_benchmark --only baseline,ablation
    
    # Use custom config file
    python -m benchmarking.run_benchmark --config path/to/config.yaml

Options:
    --config PATH       Config file path (default: benchmarking/config/benchmark_config.yaml)
    --output PATH       Output directory (default: from config or output/experiments)
    --only EXPERIMENTS  Comma-separated list: baseline,ablation,domain,edge
    --device DEVICE     Force device: cpu, cuda, or auto (default: auto)
    --verbose           Enable verbose logging
    --dry-run           Validate config and data without running experiments
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from benchmarking.bootstrap import ensure_search_on_path, resolve_ingest_root, resolve_workspace_root


@dataclass
class BenchmarkConfig:
    """Benchmark configuration container."""
    
    # Data paths
    corpus_path: Path
    qa_path: Path
    
    # Output settings
    output_dir: Path
    timestamp_subdir: bool = True
    output_formats: List[str] = field(default_factory=lambda: ["json", "markdown"])
    latex_output: bool = False
    
    # Experiment flags
    run_baseline: bool = True
    run_ablation: bool = True
    run_domain: bool = True
    run_edge: bool = True
    
    # Experiment settings
    top_k: int = 4
    ablation_configs: str = "all"
    edge_warmup: int = 5
    edge_iterations: int = 50
    
    # Runtime settings
    device: str = "auto"
    seed: int = 42
    verbose: bool = False
    continue_on_error: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: Path, overrides: Optional[Dict] = None) -> "BenchmarkConfig":
        """Load configuration from YAML file with optional overrides."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        # Resolve paths relative to project root
        project_root = config_path.parent.parent.parent
        ensure_search_on_path()
        from core.Config.Settings import settings

        def _expand_env(val: str) -> str:
            """Expand ${VAR} and ${VAR:-default} patterns in string."""
            def replacer(m):
                var = m.group(1)
                default = m.group(2) if m.group(2) is not None else ""
                return os.environ.get(var, default)
            return re.sub(r'\$\{([^}:]+)(?::-([^}]*))?\}', replacer, val)

        data_cfg = cfg.get("data", {})

        # Resolve paths: config value > env var > settings default
        raw_corpus = data_cfg.get("corpus", "")
        raw_qa = data_cfg.get("qa_dataset", "")

        if raw_corpus:
            expanded = _expand_env(str(raw_corpus))
            corpus_path = Path(expanded) if Path(expanded).is_absolute() else project_root / expanded
        else:
            workspace_root = resolve_workspace_root()
            ingest_root = resolve_ingest_root(workspace_root)
            candidate = (ingest_root / "output" / "wasabi_en_ko_parallel.jsonl") if ingest_root else None
            corpus_path = candidate if candidate and candidate.exists() else Path(settings.get_corpus_path())

        if raw_qa:
            expanded = _expand_env(str(raw_qa))
            qa_path = Path(expanded) if Path(expanded).is_absolute() else project_root / expanded
        else:
            workspace_root = resolve_workspace_root()
            ingest_root = resolve_ingest_root(workspace_root)
            candidate = (ingest_root / "output" / "wasabi_qa_dataset.jsonl") if ingest_root else None
            qa_path = candidate if candidate and candidate.exists() else Path(settings.get_qa_dataset_path())
        
        output_cfg = cfg.get("output", {})
        output_dir = project_root / output_cfg.get("dir", "output/experiments")
        
        exp_cfg = cfg.get("experiments", {})
        runtime_cfg = cfg.get("runtime", {})
        models_cfg = cfg.get("models", {})
        
        config = cls(
            corpus_path=corpus_path.resolve(),
            qa_path=qa_path.resolve(),
            output_dir=output_dir.resolve(),
            timestamp_subdir=output_cfg.get("timestamp_subdir", True),
            output_formats=output_cfg.get("formats", ["json", "markdown"]),
            latex_output=output_cfg.get("latex", False),
            run_baseline=exp_cfg.get("baseline", {}).get("enabled", True),
            run_ablation=exp_cfg.get("ablation", {}).get("enabled", True),
            run_domain=exp_cfg.get("domain", {}).get("enabled", True),
            run_edge=exp_cfg.get("edge", {}).get("enabled", True),
            top_k=exp_cfg.get("baseline", {}).get("top_k", 4),
            ablation_configs=exp_cfg.get("ablation", {}).get("configs", "all"),
            edge_warmup=exp_cfg.get("edge", {}).get("warmup_iterations", 5),
            edge_iterations=exp_cfg.get("edge", {}).get("benchmark_iterations", 50),
            device=models_cfg.get("device", "auto"),
            seed=runtime_cfg.get("seed", 42),
            verbose=runtime_cfg.get("verbose", False),
            continue_on_error=runtime_cfg.get("continue_on_error", True),
        )
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key) and value is not None:
                    setattr(config, key, value)
        
        return config


class BenchmarkRunner:
    """Unified benchmark runner for all experiments."""
    
    EXPERIMENTS = ["baseline", "ablation", "domain", "edge"]
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.run_dir: Optional[Path] = None
        
        # Set random seed for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
    
    def _print_banner(self) -> None:
        """Print startup banner."""
        print("=" * 64)
        print("  ERA-SmartFarm-RAG Benchmark Suite v1.0")
        print("=" * 64)
        print()
    
    def _print_section(self, step: int, total: int, title: str) -> None:
        """Print section header."""
        print(f"[{step}/{total}] {title}")
    
    def _print_status(self, message: str, success: bool = True) -> None:
        """Print status message."""
        symbol = "+" if success else "x"
        print(f"  {symbol} {message}")
    
    def _setup_output_dir(self) -> Path:
        """Create and return output directory."""
        if self.config.timestamp_subdir:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            run_dir = self.config.output_dir / timestamp
        else:
            run_dir = self.config.output_dir
        
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create latest symlink
        latest_link = self.config.output_dir / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.exists():
            pass  # Don't overwrite non-symlink
        else:
            try:
                latest_link.symlink_to(run_dir.name)
            except OSError:
                pass  # Symlinks may not work on Windows without admin
        
        return run_dir
    
    def _validate_environment(self) -> bool:
        """Validate Python environment and dependencies."""
        self._print_section(1, 5, "Environment Check")
        
        # Python version
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self._print_status(f"Python {py_version}")
        
        # Required packages
        required = ["numpy", "faiss", "sklearn", "yaml"]
        missing = []
        for pkg in required:
            try:
                __import__(pkg.replace("-", "_"))
            except ImportError:
                missing.append(pkg)
        
        if missing:
            self._print_status(f"Missing packages: {', '.join(missing)}", success=False)
            return False
        self._print_status("Required packages installed")
        
        # Device
        device = self._detect_device()
        self._print_status(f"Device: {device}")
        
        print()
        return True
    
    def _detect_device(self) -> str:
        """Detect available device."""
        if self.config.device != "auto":
            return self.config.device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    
    def _validate_data(self) -> bool:
        """Validate data files exist and are readable."""
        self._print_section(2, 5, "Data Validation")
        
        # Check corpus
        if not self.config.corpus_path.exists():
            self._print_status(f"Corpus not found: {self.config.corpus_path}", success=False)
            return False
        
        corpus_lines = sum(1 for _ in open(self.config.corpus_path, encoding="utf-8"))
        self._print_status(f"Corpus: {corpus_lines} documents ({self.config.corpus_path.name})")
        
        # Check QA dataset
        if not self.config.qa_path.exists():
            self._print_status(f"QA dataset not found: {self.config.qa_path}", success=False)
            return False
        
        qa_lines = sum(1 for _ in open(self.config.qa_path, encoding="utf-8"))
        self._print_status(f"QA Dataset: {qa_lines} questions ({self.config.qa_path.name})")
        
        print()
        return True
    
    def _prepare_models(self) -> bool:
        """Prepare embedding models."""
        self._print_section(3, 5, "Model Preparation")
        
        try:
            # Import here to avoid startup delay
            from core.Services.Retrieval.Embeddings import EmbeddingRetriever
            
            # Just check if model can be loaded (will use cache if available)
            retriever = EmbeddingRetriever()
            retriever._load_model()
            
            self._print_status(f"Embedding model: {retriever.model_id} (dim={retriever.dim})")
            print()
            return True
            
        except Exception as e:
            self._print_status(f"Model loading failed: {e}", success=False)
            print()
            return False
    
    def _run_experiments(self) -> Dict[str, Any]:
        """Run all enabled experiments."""
        self._print_section(4, 5, "Running Experiments")
        
        results = {}
        experiments = []
        
        if self.config.run_baseline:
            experiments.append(("Baseline Comparison", "baseline", self._run_baseline))
        if self.config.run_ablation:
            experiments.append(("Ablation Study", "ablation", self._run_ablation))
        if self.config.run_domain:
            experiments.append(("Domain Analysis", "domain", self._run_domain))
        if self.config.run_edge:
            experiments.append(("Edge Benchmark", "edge", self._run_edge))
        
        for name, key, runner in experiments:
            print(f"  {name}...", end=" ", flush=True)
            start = time.time()
            
            try:
                result = runner()
                elapsed = time.time() - start
                results[key] = result
                print(f"done [{elapsed:.1f}s]")
            except Exception as e:
                print(f"FAILED: {e}")
                if not self.config.continue_on_error:
                    raise
                results[key] = {"error": str(e)}
        
        print()
        return results
    
    def _run_baseline(self) -> Dict[str, Any]:
        """Run baseline comparison experiment."""
        from benchmarking.experiments.baseline_comparison import BaselineExperiment
        
        experiment = BaselineExperiment(
            corpus_path=self.config.corpus_path,
            qa_path=self.config.qa_path,
            output_dir=self.run_dir / "baseline",
            top_k=self.config.top_k,
        )
        return experiment.run_experiment()
    
    def _run_ablation(self) -> Dict[str, Any]:
        """Run ablation study experiment."""
        from benchmarking.experiments.ablation_study import AblationExperiment
        
        experiment = AblationExperiment(
            corpus_path=self.config.corpus_path,
            qa_path=self.config.qa_path,
            output_dir=self.run_dir / "ablation",
            top_k=self.config.top_k,
        )
        return experiment.run_experiment()
    
    def _run_domain(self) -> Dict[str, Any]:
        """Run domain analysis experiment."""
        from benchmarking.experiments.domain_analysis import DomainAnalysis
        
        experiment = DomainAnalysis(
            corpus_path=self.config.corpus_path,
            qa_path=self.config.qa_path,
            output_dir=self.run_dir / "domain",
            top_k=self.config.top_k,
        )
        return experiment.run_analysis()
    
    def _run_edge(self) -> Dict[str, Any]:
        """Run edge performance benchmark."""
        from benchmarking.experiments.edge_benchmark import EdgeBenchmark
        
        experiment = EdgeBenchmark(
            corpus_path=self.config.corpus_path,
            qa_path=self.config.qa_path,
            output_dir=self.run_dir / "edge",
            warmup_queries=self.config.edge_warmup,
            n_runs=self.config.edge_iterations // 50 if self.config.edge_iterations >= 50 else 1,
        )
        return experiment.run_benchmark()
    
    def _generate_reports(self, results: Dict[str, Any]) -> None:
        """Generate output reports."""
        self._print_section(5, 5, "Generating Reports")
        
        # Save JSON results
        if "json" in self.config.output_formats:
            for exp_name, exp_results in results.items():
                if "error" not in exp_results:
                    json_path = self.run_dir / f"{exp_name}_summary.json"
                    # Results are already saved by individual experiments
                    self._print_status(str(json_path))
        
        # Generate markdown report
        if "markdown" in self.config.output_formats:
            report_path = self.run_dir / "BENCHMARK_REPORT.md"
            self._generate_markdown_report(results, report_path)
            self._print_status(str(report_path))
        
        print()
    
    def _generate_markdown_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Generate unified markdown report."""
        lines = [
            "# ERA-SmartFarm-RAG Benchmark Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Seed**: {self.config.seed}",
            "",
            "---",
            "",
        ]
        
        # Baseline results
        if "baseline" in results and "error" not in results["baseline"]:
            lines.extend([
                "## RQ1: Baseline Comparison",
                "",
                "| Method | P@4 | R@4 | MRR | NDCG@4 |",
                "|--------|-----|-----|-----|--------|",
            ])
            summary = results["baseline"].get("summary", {})
            for method, metrics in summary.items():
                p4 = metrics.get("precision_at_k", {}).get("mean", 0)
                r4 = metrics.get("recall_at_k", {}).get("mean", 0)
                mrr = metrics.get("mrr", {}).get("mean", 0)
                ndcg = metrics.get("ndcg", {}).get("mean", 0)
                lines.append(f"| {method} | {p4:.3f} | {r4:.3f} | {mrr:.3f} | {ndcg:.3f} |")
            lines.extend(["", "---", ""])
        
        # Ablation results
        if "ablation" in results and "error" not in results["ablation"]:
            lines.extend([
                "## RQ2: Ablation Study",
                "",
                "| Configuration | MRR | NDCG@4 | Delta MRR |",
                "|---------------|-----|--------|-----------|",
            ])
            summary = results["ablation"].get("summary", {})
            base_mrr = summary.get("base_naive", {}).get("mrr", {}).get("mean", 0)
            for config_name, metrics in summary.items():
                mrr = metrics.get("mrr", {}).get("mean", 0)
                ndcg = metrics.get("ndcg", {}).get("mean", 0)
                delta = mrr - base_mrr if config_name != "base_naive" else 0
                delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}" if delta < 0 else "--"
                lines.append(f"| {config_name} | {mrr:.3f} | {ndcg:.3f} | {delta_str} |")
            lines.extend(["", "---", ""])
        
        # Edge performance results
        if "edge" in results and "error" not in results["edge"]:
            lines.extend([
                "## RQ4: Edge Performance",
                "",
            ])
            summary = results["edge"].get("summary", {})
            if "query_latency_ms" in summary:
                lat = summary["query_latency_ms"]
                lines.extend([
                    f"- **Query Latency (p50)**: {lat.get('p50', 0):.1f} ms",
                    f"- **Query Latency (p95)**: {lat.get('p95', 0):.1f} ms",
                    f"- **Query Latency (p99)**: {lat.get('p99', 0):.1f} ms",
                ])
            if "memory_mb" in summary:
                lines.append(f"- **Memory Usage**: {summary['memory_mb']:.1f} MB")
            if "throughput_qps" in summary:
                lines.append(f"- **Throughput**: {summary['throughput_qps']:.1f} QPS")
            lines.extend(["", "---", ""])
        
        # Write report
        output_path.write_text("\n".join(lines), encoding="utf-8")
    
    def _print_summary(self, elapsed: float) -> None:
        """Print final summary."""
        print("=" * 64)
        print(f"  COMPLETE | Total: {elapsed:.0f}s | Results: {self.run_dir}/")
        print("=" * 64)
    
    def run(self) -> int:
        """Run the full benchmark suite.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        self._print_banner()
        self.start_time = time.time()
        
        # Setup output directory
        self.run_dir = self._setup_output_dir()
        
        # Validation steps
        if not self._validate_environment():
            return 1
        
        if not self._validate_data():
            return 1
        
        if not self._prepare_models():
            return 1
        
        # Run experiments
        self.results = self._run_experiments()
        
        # Generate reports
        self._generate_reports(self.results)
        
        # Print summary
        elapsed = time.time() - self.start_time
        self._print_summary(elapsed)
        
        return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ERA-SmartFarm-RAG Unified Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments with default config
  python -m benchmarking.run_benchmark

  # Run specific experiments only
  python -m benchmarking.run_benchmark --only baseline,ablation

  # Use custom config and output directory
  python -m benchmarking.run_benchmark --config custom.yaml --output results/

  # Dry run (validate only)
  python -m benchmarking.run_benchmark --dry-run
        """,
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmarking/config/benchmark_config.yaml"),
        help="Configuration file path (default: benchmarking/config/benchmark_config.yaml)",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (overrides config)",
    )
    
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated experiments to run: baseline,ablation,domain,edge",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Force device selection (overrides config)",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and data without running experiments",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Find project root
    if args.config.is_absolute():
        config_path = args.config
    else:
        # Try relative to current directory first
        config_path = Path.cwd() / args.config
        if not config_path.exists():
            # Try relative to this file's location
            config_path = Path(__file__).parent.parent / args.config
    
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    # Build overrides from command line
    overrides = {}
    
    if args.output:
        overrides["output_dir"] = args.output.resolve()
    
    if args.device:
        overrides["device"] = args.device
    
    if args.verbose:
        overrides["verbose"] = True
    
    if args.only:
        experiments = [e.strip().lower() for e in args.only.split(",")]
        overrides["run_baseline"] = "baseline" in experiments
        overrides["run_ablation"] = "ablation" in experiments
        overrides["run_domain"] = "domain" in experiments
        overrides["run_edge"] = "edge" in experiments
    
    # Load config
    try:
        config = BenchmarkConfig.from_yaml(config_path, overrides)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Dry run mode
    if args.dry_run:
        print("Dry run mode - validating configuration...")
        print(f"  Config: {config_path}")
        print(f"  Corpus: {config.corpus_path}")
        print(f"  QA Dataset: {config.qa_path}")
        print(f"  Output: {config.output_dir}")
        print(f"  Experiments: baseline={config.run_baseline}, ablation={config.run_ablation}, "
              f"domain={config.run_domain}, edge={config.run_edge}")
        return 0
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
