## Title

Entropy-Calibrated Frame Competition for DisCoCat Quantum NLP: A Multi-Corpus Analysis with Circuit-Level Integration

## Authors

J. Liu, et al.

## Abstract

We present an entropy-calibrated framing analysis that quantifies semantic competition across journalist-authored and AI-generated news corpora within a DisCoCat quantum NLP workflow. By aligning the classical pre-processing pipeline with the canonical frame competition definition `FrameCompetition = min(1, S(ρ) × 0.5)`, we generate per-segment density matrices whose von Neumann entropy guides ancilla scheduling in quantum circuits. The approach yields reproducible competition metrics, visual diagnostics, and competition-conditioned circuit exemplars, supplying a fully documented bridge between lexical framing and quantum semantic composition.

## 1. Introduction

Categorical compositional semantics (DisCoCat) offers a principled map from grammatical reductions to linear-algebraic meaning representation. Recent QNLP studies require accurate framing diagnostics to determine ancilla usage and entangling patterns in the circuit back end. Prior iterations relied on KL-divergence heuristics, potentially misaligned with quantum entropy semantics. Here, we retrofit the entire pipeline—data analysis, visualization, and circuit construction—to the entropy-based competition strength defined in our methodology documentation. Key contributions include:

- Recomputing frame competition via von Neumann entropy for every CNA and AI segment (`multi_frame_analysis.py`).
- Visualizing segment-level competition disparities via `competition_by_field.png` and `frame_probability_heatmap.png`.
- Propagating the metric into quantum analyzers and competition-aware circuit diagrams (`qiskit_discocat_competition_{low,medium,high}.png`).

## 2. Related Work

1. **Framing in QNLP.** Previous work employed KL-normalized scores to approximate frame competition; however, they diverged from the formal density-matrix definition introduced in our technical methodology.
2. **Entropy in quantum circuits.** Von Neumann entropy is the standard for assessing superposition magnitude—critical when calibrating ancilla amplitudes and entangling gates.
3. **DisCoCat pipelines.** Integrating classical metrics with diagrammatic circuits has been discussed, but a direct, reproducible mapping from frame entropy to circuit parameters remained unresolved.

## 3. Data

- **CNA corpus:** 40 journalist-written items (`cna.csv`) processed into 20 titles and 20 full contents.
- **AI corpus:** 298 generated stories with separate headline (`新聞標題`), dialogue (`影片對話`), and description (`影片描述`), yielding 894 text segments.
- **Frame lexicon:** 11 categories (reform, accountability, justice, victim support, public sentiment, economic, safety, communication, policy, corporate governance, labour) compiled in `multi_frame_analysis.py`.

## 4. Methods

### 4.1 Pipeline Overview

1. **Tokenization:** Jieba segmentation without stop words.
2. **Frame scoring:** Word-count aggregation aligned with the 11-frame lexicon.

   - Reform (改革, 革新, 變革, 改善, 提升, 優化, 改進, 調整, 整頓, 重啟, 更新)
   - Accountability (問責, 究責, 監督, 檢討, 懲處, 懲戒, 追究, 道歉, 負責, 處分, 紀律, 裁罰, 修正)
   - Justice (司法, 法院, 法庭, 檢方, 檢察官, 起訴, 判決, 訴訟, 違法, 違規, 法規, 刑責, 刑期, 法律, 裁定, 羈押, 偵辦)
   - Victim support (受害者, 被害人, 受害, 受害人家屬, 求助, 支援, 關懷, 陪伴, 保護, 救助, 輔導, 安置, 慰問, 協助, 伸張, 援助)
   - Public sentiment (民眾, 輿論, 社會, 聲援, 抗議, 遊行, 請願, 連署, 群眾, 網友, 批評, 聲浪, 關注, 反彈, 支持, 呼籲)
   - Economic impact (經濟, 成本, 投資, 營收, 利潤, 市場, 股價, 財務, 商機, 產業, 收益, 支出, 資金, 就業, 財政, 預算, 估值, 併購)
   - Safety & risk (安全, 防護, 保護, 風險, 危機, 危害, 保安, 預防, 守則, 監控, 檢測, 保障, 通報, 警戒, 防範, 管控, 緊急)
   - Communication strategy (聲明, 公告, 說明, 澄清, 記者會, 回應, 表示, 公開, 發言, 發布, 報告, 告知, 說法, 簡報, 揭露)
   - Policy & governance (政策, 法案, 制度, 規範, 措施, 指引, 方案, 規定, 管理, 標準, 草案, 規劃, 計畫, 方案, 申請, 流程)
   - Corporate governance (董事長, 總經理, 高層, 管理層, 企業文化, 公司, 品牌, 總部, 主管, 董事會, 經營, 營運, 人資, 人事, 政策會, 內部, 部門, 團隊)
   - Labour & workplace (員工, 同事, 職場, 勞工, 工作, 人力, 培訓, 福利, 職員, 職務, 雇主, 受僱者, 職工, 班表, 輪班, 職涯)

3. **Probability normalization:** Conversion to per-frame probabilities relative to total frame hits.
4. **Entropy calculation:** Construction of a diagonal density matrix, von Neumann entropy `S(ρ)`, competition score `min(1, S(ρ) × 0.5)`.
5. **Outputs:** `cna_multi_frame_analysis.csv`, `ai_multi_frame_analysis.csv`, `multi_frame_summary.json`, `frame_probability_compare.csv`.

```python
# core excerpt from multi_frame_analysis.py
def compute_metrics(frame_counts: Dict[str, int]) -> FrameMetrics:
    total_hits = sum(frame_counts.values())
    if total_hits == 0:
        frame_probs = {frame: 0.0 for frame in FRAME_ORDER}
        return FrameMetrics(...)

    frame_probs = {frame: count / total_hits for frame, count in frame_counts.items()}
    density_matrix = np.diag([frame_probs[frame] for frame in FRAME_ORDER])
    positive_eigs = [val for val in np.diag(density_matrix) if val > 1e-12]
    entropy_bits = -sum(val * math.log2(val) for val in positive_eigs)
    competition_entropy = min(1.0, entropy_bits * 0.5)
    ...
```

### 4.2 Tooling Updates

- `multi_frame_analysis.py` – main analysis engine, exports entropy-based competition and KL reference.
- `multi_frame_visualizations.py` – generates `competition_by_field.png` and `frame_probability_heatmap.png`.
- `qiskit_quantum_analyzer.py`, `cna_quantum_analyzer.py`, `quantum_frame_analyzer.py` – receive the entropy metric, providing both `frame_competition` and `frame_competition_kl`.
- `qiskit_discocat_circuit.py` – implements competition-conditioned circuit generation.

### 4.3 Circuit Mapping

We map competition values to circuit parameters via `competition_to_gate_schedule`:

- Lexical rotation gain: `0.4 + 0.6 × competition`.
- Ancilla amplitude: `competition × π/2`.
- Context amplitude: `competition × π/3`.
- Entangler toggles: `competition > 0.35` triggers cross CRZ; `competition > 0.65` enables triple controls (CCX/CRX).
- Example circuits rendered for low (0.05), medium (0.45), and high (0.85) competition levels.

### 4.4 Integrating Entropy into Analyzers

```python
# scripts/qiskit_quantum_analyzer.py
eigenvals = np.linalg.eigvals(density_matrix)
von_neumann_entropy = float(-np.sum(eigenvals * np.log2(eigenvals + 1e-12)))
competition_entropy = float(min(1.0, von_neumann_entropy * 0.5))
if len(probabilities_filtered) > 1:
    uniform_prob = 1.0 / len(probabilities_filtered)
    kl_divergence = np.sum(probabilities_filtered * np.log2((probabilities_filtered + 1e-12) / uniform_prob))
    max_kl = np.log2(len(probabilities_filtered))
    frame_competition_kl = float(1.0 - min(1.0, kl_divergence / max_kl))
else:
    frame_competition_kl = 0.0
```

```python
# scripts/cna_quantum_analyzer.py
total_entropy = -np.sum(valid_probs * np.log2(valid_probs + 1e-12))
metrics['frame_competition'] = float(min(1.0, total_entropy * 0.5))
if len(valid_probs) > 1:
    uniform_prob = 1.0 / len(valid_probs)
    kl_divergence = np.sum(valid_probs * np.log2((valid_probs + 1e-12) / uniform_prob))
    max_kl = np.log2(len(valid_probs))
    metrics['frame_competition_kl'] = float(1.0 - min(1.0, kl_divergence / max_kl))
else:
    metrics['frame_competition_kl'] = 0.0
```

```python
# scripts/quantum_frame_analyzer.py
metrics['frame_competition'] = float(min(1.0, entropy_val * 0.5))
if len(valid_probs) > 1:
    uniform_prob = 1.0 / len(valid_probs)
    kl_divergence = np.sum(valid_probs * np.log2((valid_probs + 1e-12) / uniform_prob))
    max_kl = np.log2(len(valid_probs))
    metrics['frame_competition_kl'] = float(1.0 - min(1.0, kl_divergence / max_kl))
else:
    metrics['frame_competition_kl'] = 0.0
```

### 4.5 Circuit Mapping (Extended)

The blend between analytics and circuit control is implemented in `build_competition_conditioned_circuit`:

## 5. Results

### 5.1 Dataset-Level Statistics

| Metric | CNA (n = 40) | AI (n = 894) |
| --- | --- | --- |
| Mean frame competition | 0.422 | 0.522 |
| Median frame competition | 0.230 | 0.649 |
| Mean von Neumann entropy (bits) | 0.844 | 1.043 |
| Mean active frames | 2.18 | 2.80 |

High competition is predominantly observed in AI dialogue/description segments; CNA contents show strong competition, while titles, irrespective of source, often collapse to a single frame.

### 5.2 Field-Level Differences

| Dataset / Field | Segments | Mean competition | Median |
| --- | --- | --- | --- |
| CNA titles | 20 | 0.025 | 0.000 |
| CNA contents | 20 | 0.819 | 0.950 |
| AI headlines | 298 | 0.030 | 0.000 |
| AI dialogues | 298 | 0.798 | 0.953 |
| AI descriptions | 298 | 0.738 | 0.807 |

The `competition_by_field.png` visualization corroborates these metrics, highlighting that long-form narratives sustain higher competition than concise headlines.

### 5.3 Frame Probability Patterns

`frame_probability_heatmap.png` reveals:

- CNA emphasizes communication and corporate governance frames, with moderate contributions from policy and public sentiment.
- AI-generated text distributes probability mass across public sentiment, justice, economic, and policy frames.
- Reform vocabulary appears marginally in both datasets, indicating the necessity of an expanded lexicon.

### 5.4 Circuit Exemplars

`qiskit_discocat_competition_{low,medium,high}.png` shows ancilla adjustments:

- **Low competition:** Minimal ancilla rotations, no triple-control gates.
- **Medium competition:** Cross CRZ enabled, modest ancilla amplitudes.
- **High competition:** Full ancilla superpositions, CCX/CRX interplay, reflecting heightened frame interference.

## 6. Discussion

1. **Entropy alignment ensures interpretability:** By matching the formal frame competition definition, we avoid heuristic mismatches between classical pre-processing and quantum semantics.
2. **Segment sensitivity:** Titles are effectively classical; multi-sentence contents produce high-competition states requiring richer quantum treatment.
3. **Circuit readiness:** The new mapping yields a direct control knob—frame competition drives ancilla amplitude and entanglement depth, bridging analytics and circuit execution.
4. **KL metrics remain useful:** Retained as `frame_competition_kl`, they offer regression checks and continuity with earlier experiments.

## 7. Limitations and Future Work

- **Lexicon scope:** Additional frames (e.g., geopolitical, technological) could be added to capture domain-specific narratives.
- **Temporal dynamics:** Competition could be tracked across time slices to observe framing evolution.
- **Hardware calibration:** Mapping competition to physical gate parameters will require noise-aware calibration on specific quantum devices.
- **Integration with DisCoCat workflow:** Further work will feed per-segment competition scores directly into sentence-level circuit builders and ancilla scheduling.

## 8. Conclusion

We close the loop between classical framing analytics and DisCoCat quantum circuits by recalibrating frame competition through von Neumann entropy. The resulting datasets, visual diagnostics, and competition-conditioned circuits provide an audit-ready, academically rigorous foundation for future QNLP experiments.

## Appendix A: Reproducibility Checklist

1. `python3 20251112/multi_frame_analysis.py`
2. `python3 20251112/multi_frame_visualizations.py`
3. `python3 information_society/paper_visuals/qiskit_discocat_circuit.py`
4. Verify outputs (`multi_frame_summary.json`, `frame_probability_compare.csv`, PNG files).
5. Consult updated documentation (`multi_frame_analysis_report.md`, `technical_methodology_detailed_explanation.md`).


