## Abstract

We present a comprehensive framing analysis of two corpora collected for the DisCoCat QNLP study: 40 journalist-written reports from the Central News Agency (CNA) and 894 AI-generated items spanning news headlines, dialogue transcripts, and video descriptions.  Using the expanded multi-frame lexicon introduced on 12 November 2025, we quantify frame prevalence, competition, and dominance.  The analysis reveals substantive differences in frame diversity and salience between journalist and AI texts, with AI output exhibiting higher frame competition and a stronger presence of public-sentiment, justice, and economic narratives.

## 1. Datasets

- **CNA corpus**  
  `20251112/cna.csv` (40 records) containing `title` and `content`.

- **AI corpus**  
  `20251112/dataseet.xlsx` (298 news items) with three textual fields: `新聞標題`, `影片對話`, `影片描述`.  After treating each field as an individual document, the total AI sample reaches 894 segments.

Both datasets were copied into `20251112/` to faciliate reproducible analysis with a frozen lexicon and tooling snapshot.

## 2. Methodology

### 2.1 Frame catalogue

The analysis covers 11 frame families with associated lexicons:

1. **Reform** – "改革", "革新", "變革", …  
2. **Accountability** – "問責", "究責", "監督", …  
3. **Justice** – "司法", "法院", "檢察官", …  
4. **Victim support** – "受害者", "救助", "輔導", …  
5. **Public sentiment** – "民眾", "輿論", "抗議", …  
6. **Economic impact** – "經濟", "營收", "股價", …  
7. **Safety & risk** – "安全", "風險", "防範", …  
8. **Communication strategy** – "聲明", "澄清", "記者會", …  
9. **Policy & governance** – "政策", "制度", "措施", …  
10. **Corporate governance** – "董事長", "高層", "企業文化", …  
11. **Labour & workplace** – "員工", "職場", "勞工", …

### 2.2 Pipeline

1. **Tokenisation** – Jieba segmentation with stopword pruning performed on each text segment.  
2. **Frame scoring** – For each token, counts are accumulated across matching frame lexicons.  
3. **Probability normalisation** – Frame frequencies are converted into probabilities relative to the total frame hits in the segment.  
4. **Quantum-aligned competition** – Construct a diagonal density matrix from frame probabilities, compute the von Neumann entropy `S(ρ) = -Tr(ρ log₂ ρ)`, and evaluate `FrameCompetition = min(1.0, S(ρ) × 0.5)` in accordance with `calculate_frame_competition(self, density_matrix)`.  The KL-based score is retained separately as `frame_competition_kl`.  
5. **Dominance metrics** – Record the dominant frame and its probability; count the number of frames with non-zero probability.  
6. **Aggregation** – Generate summary tables with per-frame averages (`frame_probability_compare.csv`) and dataset-level statistics (`multi_frame_summary.json`).

The implementation is contained in `20251112/multi_frame_analysis.py`.  Results are written to:

- `cna_multi_frame_analysis.csv`  
- `ai_multi_frame_analysis.csv`  
- `multi_frame_summary.json`  
- `frame_probability_compare.csv`

All files reside in the `20251112/` directory.

## 3. Descriptive Statistics

### 3.1 Dataset-level metrics

| Metric | CNA (n = 40) | AI (n = 894) |
| --- | --- | --- |
| Mean frame competition (entropy-based) | 0.422 | 0.522 |
| Std. frame competition | 0.449 | 0.427 |
| Median frame competition | 0.230 | 0.649 |
| Max frame competition | 1.000 | 1.000 |
| Mean von Neumann entropy (bits) | 0.844 | 1.043 |
| Mean active frames (per segment) | 2.18 | 2.80 |
| Median active frames | 1.50 | 3.00 |
| Mean dominant probability | 0.264 | 0.421 |

**Interpretation:** Entropy-based competition tracks the balance of frame probabilities. Values near one indicate richer superposition states, while near-zero values correspond to single-frame dominance.

### 3.2 Field-level behaviour

| Dataset / Field | Segments | Mean competition | Median | Min | Max |
| --- | --- | --- | --- | --- | --- |
| CNA `title` | 20 | 0.025 | 0.000 | 0.000 | 0.500 |
| CNA `content` | 20 | 0.819 | 0.950 | 0.000 | 1.000 |
| AI `新聞標題` | 298 | 0.030 | 0.000 | 0.000 | 0.792 |
| AI `影片對話` | 298 | 0.798 | 0.953 | 0.000 | 1.000 |
| AI `影片描述` | 298 | 0.738 | 0.807 | 0.000 | 1.000 |

**Observation:** Headlines tend to invoke one dominant frame (competition ≈ 0), while article bodies, dialogues, and descriptions often reach competition ≥ 0.8, indicating multiple frames in superposition.

### 3.3 Dominant frame distribution

Top dominant frames per dataset:

- **CNA:**  
  - `communication` (5 instances)  
  - `safety` (3)  
  - `public_sentiment` (3)  
  - `justice` (3)  
  - `corporate_governance` (2)  
  - `reform`, `labour`, `policy`, `accountability` (1 each)  
  - 18 segments register no frame hits (dominant frame = `none`).

- **AI-generated:**  
  - `public_sentiment` (197 instances)  
  - `justice` (105)  
  - `economic` (97)  
  - `policy` (74)  
  - `communication` (57)  
  - `accountability` (44)  
  - `safety` (40)  
  - `labour` (23)  
  - `reform` (19)  
  - `victim_support` (16)  
  - `corporate_governance` (15)  
  - `victim_support`, `reform`, `labour`, etc., cover the remaining cases, while 203 segments yield no frame hits.

### 3.4 Frame probability comparison

Derived from `frame_probability_compare.csv` (mean probabilities and counts):

| Frame | CNA mean prob. | AI mean prob. | CNA mean count | AI mean count |
| --- | --- | --- | --- | --- |
| Reform | 0.011 | 0.027 | 0.150 | 0.245 |
| Accountability | 0.042 | 0.042 | 0.175 | 0.355 |
| Justice | 0.027 | 0.086 | 0.150 | 0.727 |
| Victim support | 0.018 | 0.031 | 0.150 | 0.234 |
| Public sentiment | 0.057 | 0.182 | 0.650 | 1.289 |
| Economic | 0.039 | 0.082 | 0.600 | 0.667 |
| Safety | 0.042 | 0.066 | 0.400 | 0.507 |
| Communication | 0.134 | 0.094 | 1.000 | 0.625 |
| Policy | 0.059 | 0.101 | 0.750 | 0.880 |
| Corporate governance | 0.077 | 0.039 | 0.650 | 0.348 |
| Labour | 0.044 | 0.031 | 0.375 | 0.268 |

**Observations:**

- Both journalist and AI segments exhibit multiple frames within individual documents, but entropy-derived competition clarifies whether those frames share probability mass or collapse to a dominant perspective.
- CNA materials frequently express `communication` and `corporate_governance` frames; high competition emerges only when supplementary frames (policy, public sentiment) carry comparable weights.
- AI texts, especially dialogues and descriptions, often combine `public_sentiment`, `justice`, `economic`, and `policy` cues. High competition coincides with balanced probabilities across these frames.
- The raw presence of multiple frame hits does not guarantee high competition; the von Neumann entropy isolates cases where frames genuinely compete.

> Visual aids: `competition_by_field.png` summarises entropy-based competition per field, and `frame_probability_heatmap.png` compares mean frame probabilities for CNA versus AI segments. Both are generated via `multi_frame_visualizations.py`.

> Circuit exemplars: `qiskit_discocat_competition_low|medium|high.png` (from `qiskit_discocat_circuit.py`) show how ancilla rotations and entangling gates scale with the entropy-based competition score, providing a direct visual bridge between frame analytics and circuit design.

## 4. Illustrative Examples

### Example A – CNA `communication` dominant (low competition)

- **Segment:** News title relaying a corporate statement.  
- **Frame counts:** `communication` > other frames; minimal auxiliary hits.  
- **Metrics:** frame_competition ≈ 0.00, dominant frame probability ≈ 1.0.  
- **Interpretation:** The headline collapses onto a single frame.

### Example B – CNA `public_sentiment` dominant (moderate competition)

- **Segment:** Coverage of civic response to a policy move.  
- **Frame counts:** `public_sentiment`, `policy`, and `safety` all present.  
- **Metrics:** frame_competition ≈ 0.55, three active frames.  
- **Interpretation:** Civic and policy cues share probability mass, yielding noticeable competition.

### Example C – AI `justice` dominant (medium-high competition)

- **Segment:** AI-generated headline highlighting legal action.  
- **Frame counts:** `justice` strongest; `accountability` and `public_sentiment` secondary.  
- **Metrics:** frame_competition ≈ 0.65, active frames = 3, dominant probability ≈ 0.45.  
- **Interpretation:** Multiple legal/response frames coexist, though justice remains primary.

### Example D – AI `public_sentiment` dominant with high competition

- **Segment:** Generated dialogue describing multi-party reactions.  
- **Frame counts:** `public_sentiment`, `policy`, `economic`, and `communication` all substantial.  
- **Metrics:** frame_competition ≈ 0.95, active frames = 4, dominant probability ≈ 0.27.  
- **Interpretation:** Several frames contribute nearly evenly, producing a high-entropy superposition.

## 5. Discussion

1. **Frame diversity vs. competition** – Merely counting frames is insufficient; the entropy-based score highlights whether probability mass is balanced or concentrated.  
2. **Segment-specific behaviour** – Headlines typically collapse to one frame (low competition), whereas contents, dialogues, and descriptions frequently sustain high competition (≥ 0.8), indicating strong multi-frame superposition.  
3. **Reform frame re-evaluated** – Reform vocabulary often co-occurs with accountability, policy, or public sentiment; the entropy metric captures when these lenses balance within a single segment.  
4. **Implications for quantum modelling** – Per-segment entropy should guide amplitude assignments on ancillary qubits: high-competition segments warrant richer superpositions in the frame register, while low-competition headlines can be modelled with near-classical states.

## 6. Reproducibility Checklist

1. Copy datasets into `20251112/`.  
2. Run `python3 multi_frame_analysis.py`.  
3. Inspect generated CSV/JSON outputs and verify summary tables.  
4. Use this Markdown report (`multi_frame_analysis_report.md`) as the narrative companion.

## 7. Conclusion

The expanded multi-frame analysis shows that both journalist and AI corpora display varying degrees of intra-document frame competition: some segments collapse onto a single framing, while others sustain balanced multi-frame mixes.  These findings will feed into subsequent quantum-semantic modelling steps, informing how multiple framing registers should be encoded on ancillary qubits within the DisCoCat circuit, with per-document competition guiding amplitude assignments.

