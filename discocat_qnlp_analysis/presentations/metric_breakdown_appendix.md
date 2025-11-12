# Appendix: Quantum Metric Breakdown

## 1. Multiple Reality Strength (MR)
- **Definition**: Measures breadth of concurrent interpretive frames captured in the quantum state. Computed from weighted superposition amplitudes in `scripts/final_discocat_analyzer.py`.
- **AI Generated** (from `results/full_qiskit_ai_analysis_results.csv`):
  - Titles: **1.7001**
  - Dialogues: **1.7054**
  - Descriptions: **1.7033**
- **CNA News** (from `results/cna_final_discocat_analysis_results.csv`):
  - Titles: **0.7477**
  - Content: **0.6457**
- **Interpretation**: AI outputs sustain richer multi-frame superposition; CNA content favors a dominant narrative thread.

## 2. Frame Competition (C)
- **Formula**: \( C = -\frac{\operatorname{Tr}(\rho \log_2 \rho)}{\log_2 N} \)
  - ρ: density matrix, N: Hilbert space dimension (`paper/frame_competition_illustration_fixed.py`).
- **AI Generated**: Perfect competition **1.0000** across all sections.
- **CNA News**:
  - Titles: **0.9985**
  - Content: **0.9173**
- **Interpretation**: AI maintains equal weighting of frames; CNA content enforces frame hierarchy in long-form articles.

## 3. Von Neumann Entropy (S)
- **Formula**: \( S = -\operatorname{Tr}(\rho \log_2 \rho) \)
- **AI Generated**: ~**4.0000** (Titles), **4.0000** (Dialogues), **3.9966** (Descriptions).
- **CNA News**: **3.4378** (Titles), **7.3508** (Content).
- **Interpretation**: CNA content exhibits higher informational diversity; AI outputs are uniformly dense but capped by circuit configuration.

## 4. Semantic Interference (SI)
- **Computation**: Phase variance of statevector amplitudes (`scripts/final_discocat_analyzer.py`).
- **AI Generated**: **0.0014** (Titles), **0.0178** (Dialogues), **0.0106** (Descriptions).
- **CNA News**: **0.0008** (Titles), **0.0177** (Content).
- **Interpretation**: Both corpora remain neutral; AI dialogues and CNA content display slight emotional intensity peaks.

## 5. Additional Metrics (where available)
- **Grammatical Superposition**: Captures simultaneous grammatical roles (mean ≈ **0.33** for AI; **0.21** CNA — computed in `results/discocat_quantum_analysis_detailed.csv`).
- **Compositional Entanglement**: Average entangling gate contribution (AI ≈ **0.58**, CNA ≈ **0.41**) derived from controlled rotations in `scripts/discocat_qnlp_analyzer.py`.
- **Categorical Diversity**: Unique POS categories per article (AI median **7**, CNA median **6**), reported in segmentation summaries.

## 6. Sentiment & Emotion Model Metrics
- **Binary Sentiment (quantum_sentiment_predictor.py)**:
  - Accuracy: **0.9840** (Random Forest/SVM/LogReg)
  - Neutral dominance: **98.6%** of samples
- **Four-Emotion Classification (paper_based_sentiment_analyzer.py)**:
  - Accuracy: **0.9572** (Quantum Logistic Regression)
  - Distribution: Anger **27%**, Fear **25%**, Happiness **25%**, Sadness **23%**

## 7. CNA Section-Level Highlights
- **Titles**: High competition (0.9985), moderate MR (0.7477), moderate entropy (3.4378)
- **Content**: Lower competition (0.9173), lower MR (0.6457), high entropy (7.3508), SI aligned with AI dialogues (~0.0177)

## 8. AI Section-Level Highlights
- **Titles**: MR **1.7001**, C **1.0000**, SI **0.0014**, S **3.9966**
- **Dialogues**: MR **1.7054**, C **1.0000**, SI **0.0178**, S **4.0000**
- **Descriptions**: MR **1.7033**, C **1.0000**, SI **0.0106**, S **4.0000**

---
**Usage**: Incorporate these figures into slides 14–24 of `quantum_nlp_conference_script.md` for quantitative depth.
