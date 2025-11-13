#!/usr/bin/env python3
"""Fast Qiskit analyzer with multiple quantum collapse modes.

This script extends the original fast analyzer by providing:
  • Post-readout collapse (default) using Monte Carlo sampling.
  • Mid-circuit collapse that performs a simulated measurement before
    continuing the circuit evolution.
  • Optional decoherence/noise knob that mixes the empirical distribution
    with uniform noise before sampling.

Outputs are stored in a configurable directory (default:
``../20251112_collapose``) with filenames that encode the chosen collapse
mode and noise settings so multiple experiments can coexist.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.random import default_rng
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import jieba
import jieba.posseg as pseg
import warnings

warnings.filterwarnings("ignore")

# Configure jieba dictionary if available
DICT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dict.txt.big"))
if os.path.exists(DICT_PATH):
    jieba.set_dictionary(DICT_PATH)


class FastQiskitAnalyzerWithCollapse:
    """Fast Qiskit analyzer supporting multiple collapse strategies."""

    def __init__(
        self,
        collapse_mode: str = "post_readout",
        shots: int = 2048,
        noise_level: float = 0.0,
    ) -> None:
        print("🚀 初始化具坍縮模擬的快速Qiskit量子分析器...")

        valid_modes = {"post_readout", "mid_circuit", "none"}
        if collapse_mode not in valid_modes:
            raise ValueError(f"Unsupported collapse_mode: {collapse_mode}")

        self.collapse_mode = collapse_mode
        self.shots = max(1, int(shots))
        self.noise_level = float(np.clip(noise_level, 0.0, 1.0))
        self.enable_collapse = collapse_mode != "none"
        self.rng = default_rng()

        self.category_map = {
            "N": 0.25,  # 名詞
            "V": 0.5,   # 動詞
            "A": 0.75,  # 形容詞
            "P": 1.0,   # 介詞
            "D": 0.3,   # 副詞
            "M": 0.6,   # 數詞
            "Q": 0.8,   # 量詞
            "R": 0.4,   # 代詞
        }

        self.positive_words = {
            "成功", "获得", "优秀", "突破", "创新", "发展", "改善", "提升", "荣获", "卓越", "领先", "进步"
        }
        self.negative_words = {
            "失败", "问题", "困难", "危机", "冲突", "争议", "批评", "质疑", "担忧", "下降", "减少", "损失"
        }

        print("✅ 具坍縮模擬的快速Qiskit量子分析器初始化完成")

    # ------------------------------------------------------------------
    # Circuit construction helpers
    # ------------------------------------------------------------------
    def create_simple_quantum_circuit(self, words: List[str], pos_tags: List[str]) -> QuantumCircuit:
        """Construct the simplified quantum circuit used for text encoding."""
        num_qubits = min(4, max(2, len(set(pos_tags))))
        circuit = QuantumCircuit(num_qubits)

        for qubit in range(num_qubits):
            circuit.h(qubit)

        pos_counts: Dict[str, int] = {}
        for pos in pos_tags:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        for idx, (pos, count) in enumerate(list(pos_counts.items())[:num_qubits]):
            if pos in self.category_map:
                angle = self.category_map[pos] * (count / len(pos_tags)) * np.pi / 4
                circuit.ry(angle, idx)

        if num_qubits > 1:
            circuit.cx(0, 1)

        return circuit

    # ------------------------------------------------------------------
    # Probability utilities
    # ------------------------------------------------------------------
    @staticmethod
    def compute_probability_metrics(probabilities: np.ndarray) -> Dict[str, float]:
        """Compute entropy, superposition, coherence, and frame competition."""
        epsilon = 1e-12
        probabilities = np.clip(probabilities, 0.0, 1.0)
        total = probabilities.sum()
        if total > 0:
            probabilities = probabilities / total

        von_neumann_entropy = float(-np.sum(probabilities * np.log2(probabilities + epsilon)))
        superposition_strength = float(4.0 * np.sum(probabilities * (1.0 - probabilities)))
        quantum_coherence = float(1.0 - np.sum(probabilities ** 2))

        if len(probabilities) > 1:
            uniform_prob = 1.0 / len(probabilities)
            kl_div = float(np.sum(probabilities * np.log2((probabilities + epsilon) / uniform_prob)))
            max_kl = np.log2(len(probabilities))
            frame_competition = float(max(0.0, 1.0 - (kl_div / max_kl))) if max_kl > 0 else 0.0
        else:
            frame_competition = 0.0

        return {
            "von_neumann_entropy": von_neumann_entropy,
            "superposition_strength": superposition_strength,
            "quantum_coherence": quantum_coherence,
            "frame_competition": frame_competition,
        }

    @staticmethod
    def calculate_semantic_interference(words: List[str]) -> float:
        """Compute semantic interference based on word frequency variance."""
        word_counts: Dict[str, int] = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        if not word_counts:
            return 0.0

        values = np.array(list(word_counts.values()), dtype=float)
        return float(np.var(values) / len(words))

    def calculate_reality_strength(
        self,
        superposition_strength: float,
        semantic_interference: float,
        frame_competition: float,
        emotional_intensity: float,
    ) -> float:
        """Combine metrics into a multiple-reality strength indicator."""
        reality_strength = (
            superposition_strength * 0.4
            + semantic_interference * 0.3
            + frame_competition * 0.2
            + emotional_intensity * 0.1
        )
        return float(reality_strength)

    @staticmethod
    def counts_to_probabilities(counts: Dict[str, int], num_qubits: int) -> np.ndarray:
        """Convert measurement counts into a probability vector."""
        total_shots = sum(counts.values())
        probabilities = np.zeros(2 ** num_qubits)
        if total_shots == 0:
            return probabilities

        for bitstring, freq in counts.items():
            cleaned = bitstring.replace(" ", "")
            idx = int(cleaned[::-1], 2)
            probabilities[idx] = freq / total_shots
        return probabilities

    def apply_noise(self, probabilities: np.ndarray) -> np.ndarray:
        """Blend probabilities with uniform noise controlled by noise_level."""
        total = probabilities.sum()
        if total == 0:
            uniform = np.ones_like(probabilities) / len(probabilities)
            return uniform

        probabilities = probabilities / total
        if self.noise_level <= 0:
            return probabilities

        uniform = np.ones_like(probabilities) / len(probabilities)
        noisy = (1.0 - self.noise_level) * probabilities + self.noise_level * uniform
        noisy_sum = noisy.sum()
        if noisy_sum > 0:
            noisy /= noisy_sum
        return noisy

    def sample_counts_from_probabilities(self, probabilities: np.ndarray, num_qubits: int) -> Dict[str, int]:
        """Sample bitstrings according to the provided probability distribution."""
        probabilities = self.apply_noise(probabilities)
        if probabilities.sum() == 0:
            probabilities = np.ones_like(probabilities) / len(probabilities)

        indices = self.rng.choice(len(probabilities), size=self.shots, p=probabilities)
        counts: Dict[str, int] = {}
        for idx in indices:
            bitstring = format(int(idx), f"0{num_qubits}b")[::-1]
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Mid-circuit simulation helpers
    # ------------------------------------------------------------------
    def measure_and_collapse(self, statevector: Statevector, qubit_index: int) -> Tuple[int, Statevector]:
        """Measure a qubit and return the outcome plus the collapsed state."""
        data = statevector.data.copy()
        num_states = len(data)
        indices = np.arange(num_states)
        mask = (indices >> qubit_index) & 1

        prob_zero = float(np.sum(np.abs(data[mask == 0]) ** 2))
        if prob_zero < 0:
            prob_zero = 0.0
        prob_zero = min(max(prob_zero, 0.0), 1.0)

        random_draw = self.rng.random()
        outcome = 0 if random_draw < prob_zero else 1
        kept_mask = mask == outcome

        collapsed = np.zeros_like(data)
        collapsed[kept_mask] = data[kept_mask]
        norm = np.linalg.norm(collapsed)
        if norm > 0:
            collapsed /= norm
        else:
            collapsed[0] = 1.0

        return outcome, Statevector(collapsed)

    @staticmethod
    def build_mid_circuit_followup(num_qubits: int, outcome: int) -> QuantumCircuit:
        """Create a follow-up circuit conditioned on the mid-circuit outcome."""
        follow_up = QuantumCircuit(num_qubits)
        base_angle = np.pi / 6
        modifier = 1.5 if outcome == 1 else 0.75
        follow_up.ry(base_angle * modifier, 0)
        if num_qubits > 1:
            follow_up.cx(0, 1)
        if num_qubits > 2:
            follow_up.ry(base_angle * 0.5, 2)
        return follow_up

    def sample_mid_circuit_counts(self, base_state: Statevector, num_qubits: int) -> Dict[str, int]:
        """Simulate mid-circuit measurement by Monte Carlo sampling."""
        counts: Dict[str, int] = {}
        for _ in range(self.shots):
            sv = Statevector(base_state.data.copy())
            outcome, collapsed_sv = self.measure_and_collapse(sv, 0)
            follow_up = self.build_mid_circuit_followup(num_qubits, outcome)
            final_sv = collapsed_sv.evolve(follow_up)
            probabilities = np.abs(final_sv.data) ** 2
            probabilities = self.apply_noise(probabilities)
            idx = self.rng.choice(len(probabilities), p=probabilities)
            bitstring = format(int(idx), f"0{num_qubits}b")[::-1]
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Core analysis routine
    # ------------------------------------------------------------------
    def fast_quantum_analysis(
        self,
        text: str,
        field_name: str = "text",
    ) -> Dict[str, Any] | None:
        """Run fast quantum NLP analysis with optional collapse simulation."""
        if not text or len(text.strip()) == 0:
            return None

        words: List[str] = []
        pos_tags: List[str] = []
        for word, flag in pseg.cut(text):
            token = word.strip()
            if token:
                words.append(token)
                pos_tags.append(flag)

        if not words:
            return None

        try:
            base_circuit = self.create_simple_quantum_circuit(words, pos_tags)
            statevector = Statevector.from_instruction(base_circuit)
            probabilities_pre = np.abs(statevector.data) ** 2

            metrics_pre = self.compute_probability_metrics(probabilities_pre)

            semantic_interference = self.calculate_semantic_interference(words)
            positive_count = sum(1 for w in words if w in self.positive_words)
            negative_count = sum(1 for w in words if w in self.negative_words)
            emotional_intensity = (positive_count + negative_count) / len(words)

            reality_strength_pre = self.calculate_reality_strength(
                metrics_pre["superposition_strength"],
                semantic_interference,
                metrics_pre["frame_competition"],
                emotional_intensity,
            )

            collapse_metrics: Dict[str, float] = {}
            reality_strength_post = None
            collapse_counts: Dict[str, int] | None = None

            if self.enable_collapse:
                if self.collapse_mode == "post_readout":
                    collapse_counts = self.sample_counts_from_probabilities(
                        probabilities_pre,
                        base_circuit.num_qubits,
                    )
                elif self.collapse_mode == "mid_circuit":
                    collapse_counts = self.sample_mid_circuit_counts(statevector, base_circuit.num_qubits)
                else:  # collapse_mode == "none" falls through
                    collapse_counts = None

            if collapse_counts:
                probabilities_post = self.counts_to_probabilities(collapse_counts, base_circuit.num_qubits)
                collapse_metrics = self.compute_probability_metrics(probabilities_post)
                reality_strength_post = self.calculate_reality_strength(
                    collapse_metrics["superposition_strength"],
                    semantic_interference,
                    collapse_metrics["frame_competition"],
                    emotional_intensity,
                )

            result: Dict[str, Any] = {
                "field": field_name,
                "original_text": text[:100] + "..." if len(text) > 100 else text,
                "word_count": len(words),
                "unique_words": len(set(words)),
                "categorical_diversity": len(set(pos_tags)),
                "quantum_circuit_qubits": base_circuit.num_qubits,
                "semantic_interference": float(semantic_interference),
                "emotional_intensity": float(emotional_intensity),
                "analysis_version": "fast_qiskit_with_collapse_v2.0",
                "collapse_mode": self.collapse_mode,
                "collapse_shots": self.shots if self.enable_collapse else 0,
                "noise_level": self.noise_level,
            }

            result.update(
                {
                    "von_neumann_entropy_pre": metrics_pre["von_neumann_entropy"],
                    "superposition_strength_pre": metrics_pre["superposition_strength"],
                    "quantum_coherence_pre": metrics_pre["quantum_coherence"],
                    "frame_competition_pre": metrics_pre["frame_competition"],
                    "multiple_reality_strength_pre": reality_strength_pre,
                }
            )

            if collapse_counts:
                result.update(
                    {
                        "von_neumann_entropy_post": collapse_metrics.get("von_neumann_entropy"),
                        "superposition_strength_post": collapse_metrics.get("superposition_strength"),
                        "quantum_coherence_post": collapse_metrics.get("quantum_coherence"),
                        "frame_competition_post": collapse_metrics.get("frame_competition"),
                        "multiple_reality_strength_post": float(reality_strength_post)
                        if reality_strength_post is not None
                        else None,
                        "frame_competition_delta": metrics_pre["frame_competition"]
                        - collapse_metrics.get("frame_competition", 0.0),
                        "collapse_unique_outcomes": len(collapse_counts),
                    }
                )
            else:
                result.update(
                    {
                        "von_neumann_entropy_post": None,
                        "superposition_strength_post": None,
                        "quantum_coherence_post": None,
                        "frame_competition_post": None,
                        "multiple_reality_strength_post": None,
                        "frame_competition_delta": None,
                        "collapse_unique_outcomes": 0,
                    }
                )

            return result

        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  量子分析失敗，回退到經典估算: {str(exc)[:60]}...")
            return self.classical_fallback(words, pos_tags, field_name, text)

    def classical_fallback(
        self,
        words: List[str],
        pos_tags: List[str],
        field_name: str,
        text: str,
    ) -> Dict[str, Any]:
        """Fallback to classical metrics if quantum simulation fails."""
        word_counts: Dict[str, int] = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        probabilities = np.array([count / len(words) for count in word_counts.values()])
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()

        metrics = self.compute_probability_metrics(probabilities)
        semantic_interference = self.calculate_semantic_interference(words)

        positive_count = sum(1 for w in words if w in self.positive_words)
        negative_count = sum(1 for w in words if w in self.negative_words)
        emotional_intensity = (positive_count + negative_count) / len(words)

        reality_strength = self.calculate_reality_strength(
            metrics["superposition_strength"],
            semantic_interference,
            metrics["frame_competition"],
            emotional_intensity,
        )

        return {
            "field": field_name,
            "original_text": text[:100] + "..." if len(text) > 100 else text,
            "word_count": len(words),
            "unique_words": len(set(words)),
            "categorical_diversity": len(set(pos_tags)),
            "quantum_circuit_qubits": 0,
            "semantic_interference": float(semantic_interference),
            "emotional_intensity": float(emotional_intensity),
            "analysis_version": "fast_classical_collapse_fallback_v1.0",
            "collapse_mode": "fallback",
            "collapse_shots": 0,
            "noise_level": self.noise_level,
            "von_neumann_entropy_pre": metrics["von_neumann_entropy"],
            "superposition_strength_pre": metrics["superposition_strength"],
            "quantum_coherence_pre": metrics["quantum_coherence"],
            "frame_competition_pre": metrics["frame_competition"],
            "multiple_reality_strength_pre": reality_strength,
            "von_neumann_entropy_post": None,
            "superposition_strength_post": None,
            "quantum_coherence_post": None,
            "frame_competition_post": None,
            "multiple_reality_strength_post": None,
            "frame_competition_delta": None,
            "collapse_unique_outcomes": 0,
        }

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------
    def process_record_batch(self, records: List[Dict], record_type: str) -> List[Dict[str, Any]]:
        """Batch-process records for a specific dataset type."""
        results: List[Dict[str, Any]] = []

        for record in records:
            record_id = record.get("id", 0)

            if record_type == "ai":
                fields = ["新聞標題", "影片對話", "影片描述"]
                for field in fields:
                    if field in record and record[field]:
                        analysis = self.fast_quantum_analysis(record[field], field)
                        if analysis:
                            analysis["record_id"] = record_id
                            analysis["data_source"] = "AI_Generated"
                            results.append(analysis)
            elif record_type == "journalist":
                mapping = {"title": "新聞標題", "content": "新聞內容"}
                for original_field, mapped_field in mapping.items():
                    if original_field in record and record[original_field]:
                        analysis = self.fast_quantum_analysis(record[original_field], mapped_field)
                        if analysis:
                            analysis["record_id"] = record_id
                            analysis["data_source"] = "Journalist_Written"
                            results.append(analysis)

        return results


def ensure_output_dir(path: str | None) -> str:
    """Create and return the absolute output directory for this run."""
    target = path or os.path.join("..", "20251112_collapose")
    abs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), target))
    os.makedirs(abs_dir, exist_ok=True)
    return abs_dir


def save_results_with_summary(
    df: pd.DataFrame,
    output_dir: str,
    filename_prefix: str,
) -> Tuple[str, str]:
    """Persist detailed results and summary statistics."""
    csv_path = os.path.join(output_dir, f"{filename_prefix}_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    numeric_columns = [
        "von_neumann_entropy_pre",
        "superposition_strength_pre",
        "quantum_coherence_pre",
        "frame_competition_pre",
        "multiple_reality_strength_pre",
        "von_neumann_entropy_post",
        "superposition_strength_post",
        "quantum_coherence_post",
        "frame_competition_post",
        "multiple_reality_strength_post",
        "frame_competition_delta",
        "semantic_interference",
        "emotional_intensity",
    ]

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for field in df["field"].unique():
        field_df = df[df["field"] == field]
        field_stats: Dict[str, Dict[str, float]] = {}
        for col in numeric_columns:
            if col in field_df.columns and not field_df[col].isnull().all():
                field_stats[col] = {
                    "mean": float(field_df[col].mean()),
                    "std": float(field_df[col].std()),
                    "min": float(field_df[col].min()),
                    "max": float(field_df[col].max()),
                }
        summary[field] = field_stats

    overall_stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_columns:
        if col in df.columns and not df[col].isnull().all():
            overall_stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
    summary["overall"] = overall_stats

    summary_path = os.path.join(output_dir, f"{filename_prefix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return csv_path, summary_path


def resolve_prefix(base: str, mode: str, noise_level: float) -> str:
    """Build a filename prefix encoding mode and noise configuration."""
    prefix = base
    if mode != "post_readout":
        prefix = f"{prefix}_{mode}"
    if noise_level > 0:
        noise_tag = f"noise{int(noise_level * 100):02d}"
        prefix = f"{prefix}_{noise_tag}"
    return prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast Qiskit analyzer with collapse options")
    parser.add_argument("--shots", type=int, default=4096, help="Number of measurement shots for collapse sampling")
    parser.add_argument(
        "--collapse-mode",
        choices=["post_readout", "mid_circuit", "none"],
        default="post_readout",
        help="Collapse strategy to apply",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="Noise level (0-1) mixed with uniform distribution prior to sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Relative or absolute directory for outputs (default: ../20251112_collapose)",
    )

    args = parser.parse_args()

    print("🚀 開始快速Qiskit量子分析（含坍縮模擬）...")
    start_time = time.time()

    output_dir = ensure_output_dir(args.output_dir)
    print(f"📁 輸出目錄: {output_dir}")

    analyzer = FastQiskitAnalyzerWithCollapse(
        collapse_mode=args.collapse_mode,
        shots=args.shots,
        noise_level=args.noise_level,
    )

    base_prefix = resolve_prefix("fast_qiskit_with_collapse", analyzer.collapse_mode, analyzer.noise_level)

    # AI dataset analysis
    print("\n📊 分析AI生成新聞（含坍縮）...")
    ai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dataseet.xlsx"))
    if os.path.exists(ai_path):
        ai_df = pd.read_excel(ai_path)
        print(f"✅ 載入AI資料: {len(ai_df)} 筆")

        ai_records = []
        for idx, record in ai_df.iterrows():
            record_dict = record.to_dict()
            record_dict["id"] = idx
            ai_records.append(record_dict)

        ai_results = analyzer.process_record_batch(ai_records, "ai")
        if ai_results:
            ai_results_df = pd.DataFrame(ai_results)
            ai_prefix = f"{base_prefix}_ai_analysis"
            ai_results_csv, ai_summary_json = save_results_with_summary(ai_results_df, output_dir, ai_prefix)
            print(f"💾 AI分析結果: {ai_results_csv}")
            print(f"🧾 AI統計摘要: {ai_summary_json}")
        else:
            print("⚠️ 未產生任何AI分析結果")
    else:
        print(f"❌ 找不到AI資料集: {ai_path}")

    # Journalist dataset analysis
    print("\n👨‍💼 分析記者撰寫新聞（含坍縮）...")
    journalist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cna.csv"))
    if os.path.exists(journalist_path):
        journalist_df = pd.read_csv(journalist_path)
        print(f"✅ 載入記者資料: {len(journalist_df)} 筆")

        journalist_records = []
        for idx, record in journalist_df.iterrows():
            record_dict = record.to_dict()
            record_dict["id"] = idx
            journalist_records.append(record_dict)

        journalist_results = analyzer.process_record_batch(journalist_records, "journalist")
        if journalist_results:
            journalist_results_df = pd.DataFrame(journalist_results)
            journalist_prefix = f"{base_prefix}_journalist_analysis"
            journalist_results_csv, journalist_summary_json = save_results_with_summary(
                journalist_results_df,
                output_dir,
                journalist_prefix,
            )
            print(f"💾 記者分析結果: {journalist_results_csv}")
            print(f"🧾 記者統計摘要: {journalist_summary_json}")
        else:
            print("⚠️ 未產生任何記者分析結果")
    else:
        print(f"❌ 找不到記者資料集: {journalist_path}")

    end_time = time.time()
    elapsed = end_time - start_time

    print("\n✅ 坍縮量子分析完成!")
    print(f"⏱️  總耗時: {elapsed:.2f} 秒")
    print(f"📁 結果目錄: {output_dir}")


if __name__ == "__main__":
    main()
