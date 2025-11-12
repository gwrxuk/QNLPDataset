#!/usr/bin/env python3
"""
Generate visualisations for the entropy-based multi-frame analysis.

Outputs (saved alongside this script):
  - competition_by_field.png  : mean frame competition per field (CNA vs AI)
  - frame_probability_heatmap.png : heatmap of mean frame probabilities for both datasets
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CNA_PATH = BASE_DIR / "cna_multi_frame_analysis.csv"
AI_PATH = BASE_DIR / "ai_multi_frame_analysis.csv"
FRAME_SUMMARY_PATH = BASE_DIR / "frame_probability_compare.csv"

plt.rcParams["font.family"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]


def plot_competition_by_field(cna_df: pd.DataFrame, ai_df: pd.DataFrame) -> None:
    """Plot mean entropy-based competition for each field."""
    cna_field = (
        cna_df.groupby("field")["frame_competition"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
    )
    ai_field = (
        ai_df.groupby("field")["frame_competition"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    all_fields = list(cna_field["field"]) + list(ai_field["field"])
    means = list(cna_field["mean"]) + list(ai_field["mean"])
    colors = ["#1f77b4"] * len(cna_field) + ["#ff7f0e"] * len(ai_field)
    labels = [f"CNA {field}" for field in cna_field["field"]] + [
        f"AI {field}" for field in ai_field["field"]
    ]

    ax.bar(labels, means, color=colors)
    ax.set_ylabel("Mean frame competition (entropy-based)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Frame competition by field (CNA vs AI)")
    ax.tick_params(axis="x", labelrotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i, value in enumerate(means):
        ax.text(i, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    output_path = BASE_DIR / "competition_by_field.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved {output_path}")


def plot_frame_probability_heatmap(summary_df: pd.DataFrame) -> None:
    """Create a heatmap comparing mean frame probabilities per dataset."""
    pivot = summary_df.pivot(index="frame", columns="dataset", values="mean_probability")
    pivot = pivot.loc[sorted(pivot.index)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=pivot.values.max())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean frame probability comparison (CNA vs AI)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label="Mean probability")
    fig.tight_layout()
    output_path = BASE_DIR / "frame_probability_heatmap.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved {output_path}")


def load_frame_summary() -> pd.DataFrame:
    """Load and reshape frame probability summary."""
    df = pd.read_csv(FRAME_SUMMARY_PATH)
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "frame": row["frame"],
                "dataset": "CNA",
                "mean_probability": row["cna_mean_prob"],
            }
        )
        records.append(
            {
                "frame": row["frame"],
                "dataset": "AI",
                "mean_probability": row["ai_mean_prob"],
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    if not CNA_PATH.exists() or not AI_PATH.exists() or not FRAME_SUMMARY_PATH.exists():
        raise FileNotFoundError("Required analysis files not found in 20251112/. Run multi_frame_analysis.py first.")

    cna_df = pd.read_csv(CNA_PATH)
    ai_df = pd.read_csv(AI_PATH)
    frame_summary = load_frame_summary()

    plot_competition_by_field(cna_df, ai_df)
    plot_frame_probability_heatmap(frame_summary)


if __name__ == "__main__":
    main()

