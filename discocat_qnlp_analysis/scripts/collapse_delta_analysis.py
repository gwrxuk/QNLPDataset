#!/usr/bin/env python3
"""Analyze frame competition deltas and visualize collapse impacts."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '20251112_collapose'))


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def summarize_deltas(df: pd.DataFrame) -> Dict[str, float]:
    deltas = df['frame_competition_delta'].dropna()
    if deltas.empty:
        return {"count": 0}
    return {
        "count": int(deltas.count()),
        "mean": float(deltas.mean()),
        "std": float(deltas.std()),
        "min": float(deltas.min()),
        "max": float(deltas.max()),
        "median": float(deltas.median()),
        "p10": float(deltas.quantile(0.10)),
        "p90": float(deltas.quantile(0.90)),
    }


def plot_histogram(ai_df: pd.DataFrame, journalist_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(10, 6))
    bins = np.linspace(-0.5, 0.5, 61)
    plt.hist(ai_df['frame_competition_delta'].dropna(), bins=bins, alpha=0.6, label='AI Generated', color='#1f77b4')
    plt.hist(journalist_df['frame_competition_delta'].dropna(), bins=bins, alpha=0.6, label='Journalist Written', color='#ff7f0e')
    plt.axvline(0.0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Frame Competition Δ (Pre - Post)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Frame Competition Collapse Effects')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_scatter(df: pd.DataFrame, label: str, output_path: str) -> None:
    subset = df.dropna(subset=['frame_competition_pre', 'frame_competition_post'])
    if subset.empty:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(subset['frame_competition_pre'], subset['frame_competition_post'], alpha=0.4, s=20)
    max_val = max(subset['frame_competition_pre'].max(), subset['frame_competition_post'].max())
    plt.plot([0, max_val], [0, max_val], linestyle='--', color='black', linewidth=1)
    plt.xlabel('Pre-Collapse Frame Competition')
    plt.ylabel('Post-Collapse Frame Competition')
    plt.title(f'{label}: Pre vs Post Frame Competition')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze collapse deltas and produce visualizations')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory containing collapse results')
    parser.add_argument('--ai-prefix', type=str, default='fast_qiskit_with_collapse_ai_analysis', help='Prefix for AI results file')
    parser.add_argument('--journalist-prefix', type=str, default='fast_qiskit_with_collapse_journalist_analysis', help='Prefix for journalist results file')

    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    ai_path = os.path.join(output_dir, f'{args.ai_prefix}_results.csv')
    journalist_path = os.path.join(output_dir, f'{args.journalist_prefix}_results.csv')

    ai_df = load_dataset(ai_path)
    journalist_df = load_dataset(journalist_path)

    ai_summary = summarize_deltas(ai_df)
    journalist_summary = summarize_deltas(journalist_df)

    summary = {
        'output_dir': output_dir,
        'ai_results': ai_path,
        'journalist_results': journalist_path,
        'ai_frame_competition_delta': ai_summary,
        'journalist_frame_competition_delta': journalist_summary,
    }

    summary_path = os.path.join(output_dir, 'frame_competition_delta_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    histogram_path = os.path.join(output_dir, 'frame_competition_delta_histogram.png')
    plot_histogram(ai_df, journalist_df, histogram_path)

    plot_scatter(ai_df, 'AI Generated', os.path.join(output_dir, 'frame_competition_ai_pre_vs_post.png'))
    plot_scatter(journalist_df, 'Journalist Written', os.path.join(output_dir, 'frame_competition_journalist_pre_vs_post.png'))

    print('✅ Collapse delta analysis complete!')
    print(f'🧾 Summary: {summary_path}')
    print(f'📊 Histogram: {histogram_path}')


if __name__ == '__main__':
    main()
