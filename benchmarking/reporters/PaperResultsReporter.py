#!/usr/bin/env python3
"""Paper Results Reporter

논문용 표와 그림을 생성하는 리포터.

생성되는 결과물:
1. Table 1: Baseline comparison results (LaTeX)
2. Table 2: Ablation study results (LaTeX)
3. Table 3: Edge performance metrics (LaTeX)
4. Figure data: JSON for plotting

사용 예시:
    python -m benchmarking.reporters.PaperResultsReporter \
        --experiments-dir output/experiments \
        --output-dir output/paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class PaperResultsReporter:
    """논문용 결과 리포터"""

    def __init__(self, experiments_dir: Path, output_dir: Path):
        self.experiments_dir = experiments_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 결과 로드
        self.baseline_summary = self._load_json("baselines/baseline_summary.json")
        self.ablation_summary = self._load_json("ablation/ablation_summary.json")
        self.edge_summary = self._load_json("edge/edge_benchmark_summary.json")
        self.edge_full = self._load_json("edge/edge_benchmark_results.json")

    def _load_json(self, rel_path: str) -> Optional[Dict]:
        """JSON 파일 로드"""
        path = self.experiments_dir / rel_path
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def generate_baseline_table(self) -> str:
        """Table 1: Baseline Comparison (LaTeX)"""
        if not self.baseline_summary:
            return "% No baseline data available"

        latex = r"""
\begin{table}[t]
\centering
\caption{Retrieval Performance Comparison of Baseline Methods}
\label{tab:baseline}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{P@4} & \textbf{R@4} & \textbf{MRR} & \textbf{NDCG@4} \\
\midrule
"""
        # 메서드 순서 정의
        method_order = ["dense_only", "sparse_only", "naive_hybrid", "rrf_hybrid", "proposed"]
        method_names = {
            "dense_only": "Dense-only",
            "sparse_only": "Sparse-only (TF-IDF)",
            "naive_hybrid": r"Naive Hybrid ($\alpha=0.5$)",
            "rrf_hybrid": r"RRF Hybrid",
            "proposed": r"Proposed (HybridDAT)",
        }

        best = {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0}
        for method in method_order:
            if method not in self.baseline_summary:
                continue
            data = self.baseline_summary[method]
            best["precision_at_k"] = max(best["precision_at_k"], data["precision_at_k"]["mean"])
            best["recall_at_k"] = max(best["recall_at_k"], data["recall_at_k"]["mean"])
            best["mrr"] = max(best["mrr"], data["mrr"]["mean"])
            best["ndcg"] = max(best["ndcg"], data["ndcg"]["mean"])

        for method in method_order:
            if method not in self.baseline_summary:
                continue
            data = self.baseline_summary[method]
            name = method_names.get(method, method)
            p_at_k = data["precision_at_k"]["mean"]
            r_at_k = data["recall_at_k"]["mean"]
            mrr = data["mrr"]["mean"]
            ndcg = data["ndcg"]["mean"]

            def fmt(value: float, metric: str) -> str:
                if abs(value - best[metric]) < 1e-9:
                    return f"\\textbf{{{value:.4f}}}"
                return f"{value:.4f}"

            latex += (
                f"{name} & {fmt(p_at_k, 'precision_at_k')} & {fmt(r_at_k, 'recall_at_k')} & "
                f"{fmt(mrr, 'mrr')} & {fmt(ndcg, 'ndcg')} \\\\\n"
            )

        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_ablation_table(self) -> str:
        """Table 2: Ablation Study (LaTeX)"""
        if not self.ablation_summary:
            return "% No ablation data available"

        latex = r"""
\begin{table}[t]
\centering
\caption{Ablation Study: Component Contribution Analysis}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{DAT} & \textbf{MRR} & \textbf{$\Delta$MRR} \\
\midrule
"""
        config_order = ["base_naive", "+dat", "+dat+onto", "full_proposed"]
        config_names = {
            "base_naive": "Base (Naive Hybrid)",
            "+dat": "+ Dynamic Alpha",
            "+dat+onto": "+ Ontology Matching",
            "full_proposed": r"\textbf{Full Proposed}",
        }

        base_mrr = self.ablation_summary.get("base_naive", {}).get("mrr", {}).get("mean", 0)

        for config in config_order:
            if config not in self.ablation_summary:
                continue
            data = self.ablation_summary[config]
            name = config_names.get(config, config)
            cfg = data.get("config", {})
            dat = r"\checkmark" if cfg.get("use_dat") else ""
            mrr = data["mrr"]["mean"]
            delta = mrr - base_mrr

            if config == "full_proposed":
                latex += f"{name} & {dat} & \\textbf{{{mrr:.4f}}} & \\textbf{{+{delta:.4f}}} \\\\\n"
            elif config == "base_naive":
                latex += f"{name} & {dat} & {mrr:.4f} & -- \\\\\n"
            else:
                latex += f"{name} & {dat} & {mrr:.4f} & +{delta:.4f} \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_edge_table(self) -> str:
        """Table 3: Edge Performance (LaTeX)"""
        if not self.edge_summary and not self.edge_full:
            return "% No edge benchmark data available"

        data = self.edge_summary or {}
        full = self.edge_full or {}

        cold_start = data.get("cold_start_time_s", full.get("cold_start", {}).get("total_cold_start_time_s", 0))
        index_mem = data.get("index_memory_mb", full.get("cold_start", {}).get("total_memory_increase_mb", 0))
        p50 = data.get("query_latency_p50_ms", full.get("query_latency", {}).get("latency_ms", {}).get("p50", 0))
        p95 = data.get("query_latency_p95_ms", full.get("query_latency", {}).get("latency_ms", {}).get("p95", 0))
        p99 = data.get("query_latency_p99_ms", full.get("query_latency", {}).get("latency_ms", {}).get("p99", 0))
        qps = data.get("qps", full.get("query_latency", {}).get("qps", 0))

        latex = r"""
\begin{table}[t]
\centering
\caption{Edge Device Performance Metrics (8GB RAM Target)}
\label{tab:edge}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
"""
        latex += f"Cold Start Time & {cold_start:.2f} s \\\\\n"
        latex += f"Index Memory & {index_mem:.1f} MB \\\\\n"
        latex += f"Query Latency (p50) & {p50:.1f} ms \\\\\n"
        latex += f"Query Latency (p95) & {p95:.1f} ms \\\\\n"
        latex += f"Query Latency (p99) & {p99:.1f} ms \\\\\n"
        latex += f"Throughput & {qps:.1f} QPS \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_figure_data(self) -> Dict[str, Any]:
        """그래프용 데이터 생성"""
        figure_data = {}

        # Figure 1: Baseline comparison bar chart data
        if self.baseline_summary:
            figure_data["baseline_comparison"] = {
                "methods": [],
                "metrics": {
                    "precision": [],
                    "recall": [],
                    "mrr": [],
                    "ndcg": [],
                }
            }
            for method in ["dense_only", "sparse_only", "naive_hybrid", "rrf_hybrid", "proposed"]:
                if method in self.baseline_summary:
                    data = self.baseline_summary[method]
                    figure_data["baseline_comparison"]["methods"].append(method)
                    figure_data["baseline_comparison"]["metrics"]["precision"].append(data["precision_at_k"]["mean"])
                    figure_data["baseline_comparison"]["metrics"]["recall"].append(data["recall_at_k"]["mean"])
                    figure_data["baseline_comparison"]["metrics"]["mrr"].append(data["mrr"]["mean"])
                    figure_data["baseline_comparison"]["metrics"]["ndcg"].append(data["ndcg"]["mean"])

        # Figure 2: Ablation cumulative improvement
        if self.ablation_summary:
            figure_data["ablation_improvement"] = {
                "configs": [],
                "mrr": [],
                "delta_mrr": [],
            }
            base_mrr = self.ablation_summary.get("base_naive", {}).get("mrr", {}).get("mean", 0)
            for config in ["base_naive", "+dat", "+dat+onto", "+dat+crop", "+dat+dedup", "full_proposed"]:
                if config in self.ablation_summary:
                    mrr = self.ablation_summary[config]["mrr"]["mean"]
                    figure_data["ablation_improvement"]["configs"].append(config)
                    figure_data["ablation_improvement"]["mrr"].append(mrr)
                    figure_data["ablation_improvement"]["delta_mrr"].append(mrr - base_mrr)

        # Figure 3: Latency distribution
        if self.edge_full:
            latency = self.edge_full.get("query_latency", {}).get("latency_ms", {})
            figure_data["latency_percentiles"] = {
                "percentiles": ["p50", "p75", "p90", "p95", "p99"],
                "values": [
                    latency.get("p50", 0),
                    latency.get("p75", 0),
                    latency.get("p90", 0),
                    latency.get("p95", 0),
                    latency.get("p99", 0),
                ]
            }

        return figure_data

    def generate_all(self):
        """모든 결과물 생성"""
        print("[PaperResultsReporter] Generating paper results...")

        # LaTeX 표 생성
        tables = {
            "table1_baseline.tex": self.generate_baseline_table(),
            "table2_ablation.tex": self.generate_ablation_table(),
            "table3_edge.tex": self.generate_edge_table(),
        }

        for filename, content in tables.items():
            path = self.output_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Generated: {path}")

        # 그래프 데이터 생성
        figure_data = self.generate_figure_data()
        figure_path = self.output_dir / "figure_data.json"
        with open(figure_path, "w", encoding="utf-8") as f:
            json.dump(figure_data, f, ensure_ascii=False, indent=2)
        print(f"  Generated: {figure_path}")

        # 통합 Markdown 보고서
        md_report = self._generate_markdown_report()
        md_path = self.output_dir / "experiment_results.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"  Generated: {md_path}")

        print("[PaperResultsReporter] Done!")

    def _generate_markdown_report(self) -> str:
        """Markdown 형식 보고서 생성"""
        md = "# Wasabi RAG Experiment Results\n\n"

        # Baseline Comparison
        md += "## 1. Baseline Comparison\n\n"
        if self.baseline_summary:
            md += "| Method | P@4 | R@4 | MRR | NDCG@4 |\n"
            md += "|--------|-----|-----|-----|--------|\n"
            for method in ["dense_only", "sparse_only", "naive_hybrid", "rrf_hybrid", "proposed"]:
                if method in self.baseline_summary:
                    data = self.baseline_summary[method]
                    md += f"| {method} | {data['precision_at_k']['mean']:.4f} | {data['recall_at_k']['mean']:.4f} | {data['mrr']['mean']:.4f} | {data['ndcg']['mean']:.4f} |\n"
        else:
            md += "*No baseline data available*\n"

        # Ablation Study
        md += "\n## 2. Ablation Study\n\n"
        if self.ablation_summary:
            base_mrr = self.ablation_summary.get("base_naive", {}).get("mrr", {}).get("mean", 0)
            md += "| Configuration | MRR | ΔMRR |\n"
            md += "|---------------|-----|------|\n"
            for config in ["base_naive", "+dat", "+dat+onto", "full_proposed"]:
                if config in self.ablation_summary:
                    mrr = self.ablation_summary[config]["mrr"]["mean"]
                    delta = mrr - base_mrr
                    md += f"| {config} | {mrr:.4f} | {'+' if delta >= 0 else ''}{delta:.4f} |\n"
        else:
            md += "*No ablation data available*\n"

        # Edge Benchmark
        md += "\n## 3. Edge Performance\n\n"
        if self.edge_summary:
            md += f"- **Cold Start Time**: {self.edge_summary.get('cold_start_time_s', 'N/A'):.2f}s\n"
            md += f"- **Index Memory**: {self.edge_summary.get('index_memory_mb', 'N/A'):.1f} MB\n"
            md += f"- **Query Latency (p50)**: {self.edge_summary.get('query_latency_p50_ms', 'N/A'):.1f} ms\n"
            md += f"- **Query Latency (p95)**: {self.edge_summary.get('query_latency_p95_ms', 'N/A'):.1f} ms\n"
            md += f"- **Throughput**: {self.edge_summary.get('qps', 'N/A'):.1f} QPS\n"
        else:
            md += "*No edge benchmark data available*\n"

        return md


def main():
    parser = argparse.ArgumentParser(description="Paper Results Reporter")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("output/experiments"),
        help="실험 결과 디렉토리",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/paper"),
        help="논문 결과물 출력 디렉토리",
    )
    args = parser.parse_args()

    reporter = PaperResultsReporter(
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    reporter.generate_all()


if __name__ == "__main__":
    main()
