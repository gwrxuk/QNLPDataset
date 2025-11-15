#!/usr/bin/env python3
"""
Plot collapse-run quantum NLP metric charts (bar + violin with 95% CI).
Data source: fast_qiskit_with_collapse_*_analysis_results.csv (pre-collapse metrics).
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_data(project_root: Path) -> pd.DataFrame:
    """Load collapse analysis results (AI + journalist) and tag labels."""
    ai_path = (
        project_root
        / "20251112_collapose"
        / "fast_qiskit_with_collapse_ai_analysis_results.csv"
    )
    journalist_path = (
        project_root
        / "20251112_collapose"
        / "fast_qiskit_with_collapse_journalist_analysis_results.csv"
    )

    if not ai_path.exists() or not journalist_path.exists():
        raise FileNotFoundError("找不到 collapse 版本的分析結果，請先執行 fast_qiskit_with_collapse_analyzer。")

    ai_df = pd.read_csv(ai_path)
    journalist_df = pd.read_csv(journalist_path)

    ai_df = ai_df.copy()
    ai_df["source"] = "AI"
    ai_df["label"] = ai_df["source"] + "｜" + ai_df["field"]

    journalist_df = journalist_df.copy()
    journalist_df["source"] = "記者"
    journalist_df["label"] = journalist_df["source"] + "｜" + journalist_df["field"]

    combined = pd.concat([ai_df, journalist_df], ignore_index=True)
    return combined


def summarize_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for each label."""
    stats = (
        df.groupby("label")["value"]
        .agg(["count", "mean", "std", "min", "max", "median"])
        .reset_index()
        .sort_values("label")
    )
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci_half_width"] = 1.96 * stats["sem"]
    stats["ci_low"] = stats["mean"] - stats["ci_half_width"]
    stats["ci_high"] = stats["mean"] + stats["ci_half_width"]
    return stats


def plot_bar(stats: pd.DataFrame, display_name: str, output_path: Path) -> None:
    """Create bar chart showing mean ± 95% CI for each category."""
    sns.set_theme(style="whitegrid", font="Arial Unicode MS")
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        stats["label"],
        stats["mean"],
        yerr=stats["ci_half_width"],
        capsize=5,
        color=["#4e79a7", "#f28e2c", "#76b7b2", "#e15759", "#59a14f"],
    )

    ax.set_ylabel(display_name)
    y_low = stats["ci_low"].min()
    y_high = stats["ci_high"].max()
    margin = max(0.02, (y_high - y_low) * 0.1)
    ax.set_ylim(y_low - margin, y_high + margin)
    ax.set_title(f"{display_name}：平均與95%信賴區間")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    for bar, mean in zip(bars, stats["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + margin * 0.05,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_distribution(metric_df: pd.DataFrame, stats: pd.DataFrame, display_name: str, output_path: Path) -> None:
    """Create violin/strip plot with 95% CI markers."""
    sns.set_theme(style="whitegrid", font="Arial Unicode MS")
    fig, ax = plt.subplots(figsize=(10, 5))
    order = stats["label"].tolist()

    sns.violinplot(
        data=metric_df,
        x="label",
        y="value",
        order=order,
        ax=ax,
        inner="quartile",
        cut=0,
        palette="Set2",
    )

    sns.stripplot(
        data=metric_df,
        x="label",
        y="value",
        order=order,
        ax=ax,
        color="black",
        alpha=0.2,
        size=2,
        jitter=0.1,
    )

    error_legend_shown = False
    for idx, label in enumerate(order):
        row = stats[stats["label"] == label].iloc[0]
        ax.errorbar(
            idx,
            row["mean"],
            yerr=row["ci_half_width"],
            fmt="o",
            color="#d62728",
            capsize=5,
            label="平均 ± 95% CI" if not error_legend_shown else None,
            zorder=5,
        )
        error_legend_shown = True

    ax.set_ylabel(display_name)
    ax.set_title(f"{display_name}分佈 (Violin + Strip + 95% CI)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    df = load_data(project_root)
    metrics_info = [
        ("von_neumann_entropy_pre", "馮·紐曼熵"),
        ("superposition_strength_pre", "量子疊加強度"),
        ("quantum_coherence_pre", "量子相干性"),
        ("semantic_interference", "語義干涉"),
        ("frame_competition_pre", "框架競爭強度"),
        ("emotional_intensity", "情緒強度"),
        ("multiple_reality_strength_pre", "多重現實強度"),
    ]

    output_dir = project_root / "20251112_collapose" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, display_name in metrics_info:
        if metric not in df.columns:
            print(f"⚠️ 指標 {metric} 不存在於數據中，跳過。")
            continue

        metric_df = df[["label", "source", "field", metric]].copy()
        metric_df = metric_df.rename(columns={metric: "value"})

        stats = summarize_metric(metric_df)

        stats_path = output_dir / f"{metric}_stats.csv"
        stats.to_csv(stats_path, index=False, encoding="utf-8")

        print(f"\n📊 {display_name} 描述統計：")
        print(
            stats.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else f"{x}",
            )
        )

        bar_path = output_dir / f"{metric}_bar.png"
        dist_path = output_dir / f"{metric}_distribution.png"

        plot_bar(stats, display_name, bar_path)
        plot_distribution(metric_df, stats, display_name, dist_path)

        print(f"✅ {display_name} 長條圖已輸出: {bar_path}")
        print(f"✅ {display_name} 分佈圖已輸出: {dist_path}")


if __name__ == "__main__":
    main()

