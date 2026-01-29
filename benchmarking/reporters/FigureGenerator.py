#!/usr/bin/env python3
"""Figure Generator for Paper

논문용 그래프를 생성하는 스크립트.

생성되는 그래프:
1. Figure 1: Baseline comparison bar chart
2. Figure 2: Ablation study cumulative improvement
3. Figure 3: Latency percentile distribution
4. Figure 4: Category-wise performance heatmap
5. Figure 5: Memory scaling plot

사용 예시:
    python -m benchmarking.reporters.FigureGenerator \
        --experiments-dir output/experiments \
        --output-dir output/paper/figures

요구사항:
    pip install matplotlib seaborn pandas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib not installed. Install with: pip install matplotlib")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class FigureGenerator:
    """논문용 그래프 생성기"""

    # 스타일 설정
    COLORS = {
        "dense_only": "#1f77b4",
        "sparse_only": "#ff7f0e",
        "naive_hybrid": "#2ca02c",
        "rrf_hybrid": "#9467bd",
        "proposed": "#d62728",
    }

    def __init__(self, experiments_dir: Path, output_dir: Path):
        self.experiments_dir = experiments_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 결과 로드
        self.baseline_summary = self._load_json("baselines/baseline_summary.json")
        self.ablation_summary = self._load_json("ablation/ablation_summary.json")
        self.edge_full = self._load_json("edge/edge_benchmark_results.json")
        self.domain_results = self._load_json("domain/domain_analysis_results.json")

        # 한글 폰트 설정 (Windows)
        self._setup_fonts()

    def _load_json(self, rel_path: str) -> Optional[Dict]:
        """JSON 파일 로드"""
        path = self.experiments_dir / rel_path
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _setup_fonts(self):
        """한글 폰트 설정"""
        if not HAS_MATPLOTLIB:
            return

        # 한글 폰트 후보
        font_candidates = [
            "Malgun Gothic",  # Windows
            "NanumGothic",    # Linux/Mac with Nanum
            "AppleGothic",    # Mac
            "DejaVu Sans",    # Fallback
        ]

        for font in font_candidates:
            try:
                fm.findfont(font, fallback_to_default=False)
                plt.rcParams["font.family"] = font
                break
            except Exception:
                continue

        plt.rcParams["axes.unicode_minus"] = False

    def generate_baseline_comparison(self) -> Optional[str]:
        """Figure 1: Baseline comparison bar chart"""
        if not HAS_MATPLOTLIB or not self.baseline_summary:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        methods = ["dense_only", "sparse_only", "naive_hybrid", "rrf_hybrid", "proposed"]
        method_labels = ["Dense-only", "Sparse-only", "Naive Hybrid", "RRF Hybrid", "Proposed"]
        metrics = ["precision_at_k", "recall_at_k", "mrr", "ndcg"]
        metric_labels = ["P@4", "R@4", "MRR", "NDCG@4"]

        x = np.arange(len(metrics))
        width = 0.16

        for i, (method, label) in enumerate(zip(methods, method_labels)):
            if method not in self.baseline_summary:
                continue
            data = self.baseline_summary[method]
            values = [data[m]["mean"] for m in metrics]
            bars = ax.bar(x + i * width, values, width, label=label, color=self.COLORS.get(method, f"C{i}"))

        ax.set_ylabel("Score")
        ax.set_title("Retrieval Performance Comparison")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(metric_labels)
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        output_path = self.output_dir / "fig1_baseline_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Generated: {output_path}")
        return str(output_path)

    def generate_ablation_chart(self) -> Optional[str]:
        """Figure 2: Ablation study cumulative improvement"""
        if not HAS_MATPLOTLIB or not self.ablation_summary:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        configs = ["base_naive", "+dat", "+dat+onto", "full_proposed"]
        config_labels = ["Base", "+DAT", "+Onto", "Full"]

        mrr_values = []
        for config in configs:
            if config in self.ablation_summary:
                mrr_values.append(self.ablation_summary[config]["mrr"]["mean"])

        if not mrr_values:
            return None

        x = np.arange(len(mrr_values))
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(mrr_values)))

        bars = ax.bar(x, mrr_values, color=colors, edgecolor="black", linewidth=0.5)

        # 값 표시
        for bar, val in zip(bars, mrr_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=10)

        ax.set_ylabel("MRR")
        ax.set_title("Ablation Study: Component Contribution")
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels[:len(mrr_values)], rotation=45, ha="right")
        ax.set_ylim(0, max(mrr_values) * 1.15)
        ax.grid(axis="y", alpha=0.3)

        output_path = self.output_dir / "fig2_ablation_study.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Generated: {output_path}")
        return str(output_path)

    def generate_latency_chart(self) -> Optional[str]:
        """Figure 3: Latency percentile distribution"""
        if not HAS_MATPLOTLIB or not self.edge_full:
            return None

        latency = self.edge_full.get("query_latency", {}).get("latency_ms", {})
        if not latency:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))

        percentiles = ["p50", "p75", "p90", "p95", "p99"]
        labels = ["p50", "p75", "p90", "p95", "p99"]
        values = [latency.get(p, 0) for p in percentiles]

        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(percentiles)))
        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)

        # 값 표시
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                   f"{val:.1f}", ha="center", va="bottom", fontsize=10)

        ax.set_ylabel("Latency (ms)")
        ax.set_xlabel("Percentile")
        ax.set_title("Query Latency Distribution")
        ax.grid(axis="y", alpha=0.3)

        output_path = self.output_dir / "fig3_latency_distribution.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Generated: {output_path}")
        return str(output_path)

    def generate_category_heatmap(self) -> Optional[str]:
        """Figure 4: Category-wise performance heatmap"""
        if not HAS_MATPLOTLIB or not self.domain_results:
            return None

        by_category = self.domain_results.get("by_category", {})
        if not by_category:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        categories = list(by_category.keys())
        metrics = ["mrr", "ndcg"]

        data = []
        for cat in categories:
            row = [by_category[cat][m]["mean"] for m in metrics]
            data.append(row)

        data = np.array(data)

        if HAS_SEABORN:
            sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu",
                       xticklabels=["MRR", "NDCG@4"],
                       yticklabels=categories, ax=ax)
        else:
            im = ax.imshow(data, cmap="YlGnBu", aspect="auto")
            ax.set_xticks(np.arange(len(metrics)))
            ax.set_yticks(np.arange(len(categories)))
            ax.set_xticklabels(["MRR", "NDCG@4"])
            ax.set_yticklabels(categories)
            for i in range(len(categories)):
                for j in range(len(metrics)):
                    ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center")
            plt.colorbar(im, ax=ax)

        ax.set_title("Performance by Query Category")

        output_path = self.output_dir / "fig4_category_heatmap.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Generated: {output_path}")
        return str(output_path)

    def generate_memory_scaling(self) -> Optional[str]:
        """Figure 5: Memory scaling plot"""
        if not HAS_MATPLOTLIB or not self.edge_full:
            return None

        memory_scaling = self.edge_full.get("memory_scaling", {})
        if not memory_scaling:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))

        n_docs = []
        memory_mb = []

        for key, data in sorted(memory_scaling.items()):
            n_docs.append(data["n_documents"])
            memory_mb.append(data["memory_mb"])

        ax.plot(n_docs, memory_mb, "o-", color="#2ca02c", linewidth=2, markersize=8)

        ax.set_xlabel("Number of Documents")
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title("Memory Scaling with Document Count")
        ax.grid(alpha=0.3)

        output_path = self.output_dir / "fig5_memory_scaling.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Generated: {output_path}")
        return str(output_path)

    def generate_all(self):
        """모든 그래프 생성"""
        if not HAS_MATPLOTLIB:
            print("[Error] matplotlib not installed. Cannot generate figures.")
            print("Install with: pip install matplotlib seaborn")
            return

        print("[FigureGenerator] Generating figures...")

        generated = []

        fig1 = self.generate_baseline_comparison()
        if fig1:
            generated.append(fig1)

        fig2 = self.generate_ablation_chart()
        if fig2:
            generated.append(fig2)

        fig3 = self.generate_latency_chart()
        if fig3:
            generated.append(fig3)

        fig4 = self.generate_category_heatmap()
        if fig4:
            generated.append(fig4)

        fig5 = self.generate_memory_scaling()
        if fig5:
            generated.append(fig5)

        print(f"[FigureGenerator] Generated {len(generated)} figures")

        # 생성 목록 저장
        manifest_path = self.output_dir / "figures_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"figures": generated}, f, indent=2)
        print(f"[FigureGenerator] Manifest saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Figure Generator for Paper")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("output/experiments"),
        help="실험 결과 디렉토리",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/paper/figures"),
        help="그래프 출력 디렉토리",
    )
    args = parser.parse_args()

    generator = FigureGenerator(
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )
    generator.generate_all()


if __name__ == "__main__":
    main()
