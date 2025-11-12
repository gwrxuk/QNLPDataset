#!/usr/bin/env python3
"""
Comprehensive frame analysis for CNA and AI datasets (2025-11-12 batch).

This script expands the earlier reform-focused framing model by incorporating a
broader catalogue of lexical frames (accountability, justice, victim support,
economic impact, public sentiment, safety, policy, communication strategy,
corporate governance, labour conditions, plus reform itself).  For each text
segment we compute frame intensities, normalised probabilities, competition
scores, and dominant-frame diagnostics.

Outputs (written to the 20251112 directory):
  - cna_multi_frame_analysis.csv
  - ai_multi_frame_analysis.csv
  - multi_frame_summary.json
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import jieba
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Frame lexicons
# ---------------------------------------------------------------------------

FRAME_LEXICONS: Dict[str, Tuple[str, ...]] = {
    "reform": (
        "改革", "革新", "變革", "改善", "提升", "優化", "改進", "調整", "整頓", "重啟", "更新",
    ),
    "accountability": (
        "問責", "究責", "責任", "監督", "檢討", "懲處", "懲戒", "追究", "道歉", "負責",
        "處分", "紀律", "裁罰", "修正",
    ),
    "justice": (
        "司法", "法院", "法庭", "檢方", "檢察官", "起訴", "判決", "訴訟", "違法", "違規",
        "法規", "刑責", "刑期", "法律", "裁定", "羈押", "偵辦",
    ),
    "victim_support": (
        "受害者", "被害人", "受害", "受害人家屬", "求助", "支援", "關懷", "陪伴", "保護",
        "救助", "輔導", "安置", "慰問", "協助", "伸張", "援助",
    ),
    "public_sentiment": (
        "民眾", "輿論", "社會", "聲援", "抗議", "遊行", "請願", "連署", "群眾", "網友",
        "批評", "聲浪", "關注", "反彈", "支持", "呼籲",
    ),
    "economic": (
        "經濟", "成本", "投資", "營收", "利潤", "市場", "股價", "財務", "商機", "產業",
        "收益", "支出", "資金", "就業", "財政", "預算", "估值", "併購",
    ),
    "safety": (
        "安全", "防護", "保護", "風險", "危機", "危害", "保安", "預防", "守則", "監控",
        "檢測", "保障", "通報", "警戒", "防範", "管控", "緊急",
    ),
    "communication": (
        "聲明", "公告", "說明", "澄清", "記者會", "回應", "表示", "公開", "發言", "發布",
        "報告", "告知", "說法", "簡報", "揭露",
    ),
    "policy": (
        "政策", "法案", "制度", "規範", "措施", "指引", "方案", "規定", "管理", "標準",
        "草案", "規劃", "計畫", "方案", "申請", "流程",
    ),
    "corporate_governance": (
        "董事長", "總經理", "高層", "管理層", "企業文化", "公司", "品牌", "總部", "主管",
        "董事會", "經營", "營運", "人資", "人事", "政策會", "內部", "部門", "團隊",
    ),
    "labour": (
        "員工", "同事", "職場", "勞工", "工作", "人力", "培訓", "福利", "職員", "職務",
        "雇主", "受僱者", "職工", "班表", "輪班", "職涯",
    ),
}

FRAME_ORDER: Tuple[str, ...] = tuple(FRAME_LEXICONS.keys())


# ---------------------------------------------------------------------------
# Helper dataclasses & functions
# ---------------------------------------------------------------------------

@dataclass
class FrameMetrics:
    frame_counts: Dict[str, int]
    frame_probs: Dict[str, float]
    competition_entropy: float
    competition_kl: float
    normalised_entropy: float
    von_neumann_entropy: float
    active_frames: int
    dominant_frame: str
    dominant_probability: float


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [token.strip() for token in jieba.lcut(text) if token.strip()]


def score_frames(tokens: Iterable[str]) -> Dict[str, int]:
    counts = {frame: 0 for frame in FRAME_ORDER}
    for token in tokens:
        for frame, lexicon in FRAME_LEXICONS.items():
            if token in lexicon:
                counts[frame] += 1
    return counts


def compute_metrics(frame_counts: Dict[str, int]) -> FrameMetrics:
    total_hits = sum(frame_counts.values())
    if total_hits == 0:
        frame_probs = {frame: 0.0 for frame in FRAME_ORDER}
        return FrameMetrics(
            frame_counts=frame_counts,
            frame_probs=frame_probs,
            competition_entropy=0.0,
            competition_kl=0.0,
            normalised_entropy=0.0,
            von_neumann_entropy=0.0,
            active_frames=0,
            dominant_frame="none",
            dominant_probability=0.0,
        )

    frame_probs = {frame: count / total_hits for frame, count in frame_counts.items()}
    positive_probs = [prob for prob in frame_probs.values() if prob > 0]
    active_frames = len(positive_probs)

    if active_frames > 1:
        uniform_prob = 1.0 / active_frames
        kl_divergence = sum(prob * math.log2(prob / uniform_prob) for prob in positive_probs)
        max_kl = math.log2(active_frames)
        competition_kl = 1.0 - min(1.0, kl_divergence / max_kl)
    else:
        competition_kl = 0.0

    if active_frames > 1:
        von_neumann_entropy = -sum(prob * math.log2(prob) for prob in positive_probs)
        normalised_entropy = von_neumann_entropy / math.log2(active_frames)
    else:
        von_neumann_entropy = 0.0
        normalised_entropy = 0.0

    # Density matrix is diagonal with probabilities on the diagonal
    density_matrix = np.diag([frame_probs[frame] for frame in FRAME_ORDER])
    eigenvalues = np.diag(density_matrix)
    positive_eigs = [val for val in eigenvalues if val > 1e-12]
    if positive_eigs:
        entropy_bits = -sum(val * math.log2(val) for val in positive_eigs)
        competition_entropy = min(1.0, entropy_bits * 0.5)
    else:
        entropy_bits = 0.0
        competition_entropy = 0.0

    dominant_frame, dominant_probability = max(frame_probs.items(), key=lambda item: item[1])

    return FrameMetrics(
        frame_counts=frame_counts,
        frame_probs=frame_probs,
        competition_entropy=competition_entropy,
        competition_kl=competition_kl,
        normalised_entropy=normalised_entropy,
        von_neumann_entropy=von_neumann_entropy,
        active_frames=active_frames,
        dominant_frame=dominant_frame,
        dominant_probability=dominant_probability,
    )


def analyse_text_record(record_id: int, field: str, text: str, source: str) -> Dict[str, object]:
    tokens = tokenize(text)
    frame_counts = score_frames(tokens)
    metrics = compute_metrics(frame_counts)

    result: Dict[str, object] = {
        "source": source,
        "record_index": record_id,
        "field": field,
        "token_count": len(tokens),
        "frame_competition": metrics.competition_entropy,
        "frame_competition_kl": metrics.competition_kl,
        "frame_entropy": metrics.normalised_entropy,
        "von_neumann_entropy": metrics.von_neumann_entropy,
        "active_frames": metrics.active_frames,
        "dominant_frame": metrics.dominant_frame,
        "dominant_probability": metrics.dominant_probability,
        "original_text": text,
    }

    for frame in FRAME_ORDER:
        result[f"count_{frame}"] = metrics.frame_counts[frame]
        result[f"prob_{frame}"] = metrics.frame_probs[frame]

    return result


def analyse_dataframe(df: pd.DataFrame, text_columns: Iterable[str], source: str) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        for column in text_columns:
            if column not in row or pd.isna(row[column]):
                continue
            text = str(row[column]).strip()
            if not text:
                continue
            record = analyse_text_record(idx, column, text, source)
            records.append(record)
    return pd.DataFrame(records)


def summarise_results(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    metric_columns = [
        "frame_competition",
        "frame_competition_kl",
        "frame_entropy",
        "von_neumann_entropy",
        "active_frames",
        "dominant_probability",
    ]
    for metric in metric_columns:
        if metric in df.columns:
            series = df[metric]
            summary[metric] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
            }
    for frame in FRAME_ORDER:
        prob_col = f"prob_{frame}"
        if prob_col in df.columns:
            series = df[prob_col]
            summary[f"{frame}_probability"] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
    return summary


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    cna_path = base_dir / "cna.csv"
    ai_path = base_dir / "dataseet.xlsx"

    if not cna_path.exists():
        raise FileNotFoundError(f"CNA dataset not found at {cna_path}")
    if not ai_path.exists():
        raise FileNotFoundError(f"AI dataset workbook not found at {ai_path}")

    print("📥 Loading datasets...")
    cna_df = pd.read_csv(cna_path)
    ai_df = pd.read_excel(ai_path)

    print("🔎 Analysing CNA corpus with expanded frame catalogue...")
    cna_results = analyse_dataframe(
        cna_df,
        text_columns=("title", "content"),
        source="CNA",
    )
    cna_output_path = base_dir / "cna_multi_frame_analysis.csv"
    cna_results.to_csv(cna_output_path, index=False, encoding="utf-8-sig")
    print(f"✅ CNA frame analysis saved to {cna_output_path}")

    print("🔎 Analysing AI corpus with expanded frame catalogue...")
    ai_results = analyse_dataframe(
        ai_df,
        text_columns=("新聞標題", "影片對話", "影片描述"),
        source="AI_Generated",
    )
    ai_output_path = base_dir / "ai_multi_frame_analysis.csv"
    ai_results.to_csv(ai_output_path, index=False, encoding="utf-8-sig")
    print(f"✅ AI frame analysis saved to {ai_output_path}")

    summary = {
        "CNA": summarise_results(cna_results),
        "AI_Generated": summarise_results(ai_results),
        "frame_catalogue": FRAME_ORDER,
    }

    summary_path = base_dir / "multi_frame_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"📄 Summary statistics written to {summary_path}")


if __name__ == "__main__":
    jieba.initialize()
    main()

