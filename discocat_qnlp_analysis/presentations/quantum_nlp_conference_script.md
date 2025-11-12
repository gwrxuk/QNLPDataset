# Conference Presentation Script: Quantum NLP Analysis of AI and CNA News

## Slide 1 – Title: Quantum NLP Insights into AI and CNA News
- Presenter: [Your Name], [Affiliation]
- Event: [Conference Name], [Date]
- Project: DisCoCat Quantum NLP Emotional Tone & Sentiment Analysis

**Speaker Notes**
- Welcome the audience and introduce yourself.
- Frame the talk as the culmination of a multi-month deep dive into quantum natural language processing (QNLP) applied to Chinese-language news.
- Mention collaboration across code, documentation, and visualization artifacts in the repository.

## Slide 2 – Agenda
- Motivation & research questions
- Datasets and preprocessing pipeline
- Quantum circuit construction & metrics
- Key findings: AI vs. CNA comparisons
- Sentiment & emotion prediction reproductions
- Implications, limitations, and future work

**Speaker Notes**
- Provide a roadmap so attendees know what to expect over 30 slides.
- Emphasize that we will move from context to implementation to results.

## Slide 3 – Motivation
- AI-generated news proliferating across platforms
- Need to measure framing, tone, and emotional balance rigorously
- Desire for transparent quantum-native methodology

**Speaker Notes**
- Highlight concerns about AI news framing multiple realities simultaneously.
- Stress why quantum formalisms (superposition, entanglement) align naturally with linguistic ambiguity.

## Slide 4 – Research Questions
- How do AI and CNA news differ in frame competition and entropy?
- Can DisCoCat + Qiskit produce reproducible emotional tone metrics?
- How do quantum-inspired sentiment and emotion models perform on our datasets?

**Speaker Notes**
- Present each question as a thread we will resolve throughout the presentation.
- Reference `paper/technical_methodology_detailed_explanation.md` for the full methodological traceability.

## Slide 5 – Data Sources Overview
- AI-generated corpus: 894 items (298 titles, 298 dialogues, 298 descriptions)
- CNA professional corpus: 40 articles (titles + content)
- Metadata & processed outputs in `results/`

**Speaker Notes**
- Cite the CSV files: `full_qiskit_ai_analysis_results.csv` and `cna_final_discocat_analysis_results.csv`.
- Mention that CNA dataset validates external applicability.

## Slide 6 – CNA Dataset Highlights
- 20 titles + 20 corresponding full articles
- Segmented via jieba, POS-tagged, mapped into DisCoCat categories
- Results summarized in `technical_methodology_detailed_explanation.md` §6.10

**Speaker Notes**
- Note neutral tone characteristics and high informational density in CNA content.
- Stress the small but high-quality sample size.

## Slide 7 – AI Dataset Highlights
- 894 AI outputs from video pipeline (titles, dialogues, descriptions)
- Uniform high frame competition (1.0) across sections
- Stored analytics in `full_qiskit_ai_analysis_results.csv`

**Speaker Notes**
- Point out scale advantage: 894 entries support robust statistical comparison.
- Mention reliance on Qiskit quantum simulator for metrics.

## Slide 8 – Preprocessing Pipeline
- Chinese text segmentation (jieba)
- POS tagging mapped to 9 DisCoCat categories (N, V, A, D, P, R, C, F, X)
- Category statistics stored in segmentation CSVs (`results/*segmentation.csv`)

**Speaker Notes**
- Walk through how segmentation feeds qubit assignment.
- Reference `scripts/discocat_segmentation.py` for automation.

## Slide 9 – Grammatical-to-Qubit Mapping
- Each grammatical category mapped to dedicated qubit index
- Example: Noun→q0, Verb→q1, Function/Filler→q2, etc.
- Frequency-based rotation scaling per category

**Speaker Notes**
- Use the density matrix example from `paper/density_matrix_calculation_example.md` (N,N,N,F,V,...) to illustrate mapping.
- Emphasize dynamic rotation angles from word frequency and emotional weights.

## Slide 10 – Quantum Circuit Architecture
- Base superposition via Hadamard gates on all qubits
- Category-specific `RY` and `RZ` rotations encode semantics
- Entangling gates (`CX`, `CRZ`) capture compositional structure

**Speaker Notes**
- Reference visualization script `paper/quantum_circuit_visualization.py` and generated PNGs.
- Describe how superposition reflects simultaneous interpretive frames.

## Slide 11 – Quantum Gate Sequence Breakdown
- Initialize |0⟩⊗n → Hadamard → category rotations
- Controlled operations for compositional entanglement
- Frame competition and semantic ambiguity subcircuits

**Speaker Notes**
- Point attendees to `scripts/discocat_qnlp_analyzer.py` for implementation.
- Stress that gates were tuned to lexical and syntactic cues.

## Slide 12 – Density Matrix Construction
- Statevector obtained from Qiskit simulator
- Density matrix ρ = |ψ⟩⟨ψ|
- Example: 3-qubit (N,F,V) leads to N=8 (2³) dimensional ρ

**Speaker Notes**
- Summarize detailed computation from `paper/density_matrix_calculation_example.md`.
- Explain why N=2ⁿ (n qubits) underpins entropy normalization.

## Slide 13 – Key Quantum Metrics Overview
- Multiple Reality Strength
- Frame Competition (normalized von Neumann entropy)
- Von Neumann Entropy (bits)
- Semantic Interference (phase variance proxy)

**Speaker Notes**
- Reinforce that each metric derives from ρ and state phases.
- Mention formulas consolidated in §6 of `technical_methodology_detailed_explanation.md`.

## Slide 14 – Multiple Reality Strength
- Captures breadth of simultaneous interpretive frames
- AI: ~1.70 across sections vs CNA: 0.65–0.75
- Indicates richer, multi-frame positioning in AI content

**Speaker Notes**
- Interpret as evidence of AI producing multi-perspective narratives.
- Clarify analyzer scale differences (Qiskit vs DisCoCat pipelines).

## Slide 15 – Frame Competition Metric
- C = -Tr(ρ log₂ ρ) / log₂ N
- AI sections: 1.0000 (perfect competition)
- CNA: 0.9985 (titles) / 0.9173 (content)

**Speaker Notes**
- Explain that lower CNA content competition suggests more structured framing.
- Reference visuals from `paper/frame_competition_illustration_fixed.py`.

## Slide 16 – Von Neumann Entropy Insights
- Measures information density of quantum state
- AI: ~4.0 across sections (max for configuration)
- CNA: 3.44 (titles) vs 7.35 (content)

**Speaker Notes**
- Emphasize CNA content’s richer informational mix.
- Discuss implications for journalistic depth vs AI uniformity.

## Slide 17 – Semantic Interference Patterns
- Proxy for emotional intensity via phase variance
- AI dialogue highest: 0.0178 vs CNA content: 0.0177
- Both corpora largely neutral with subtle variance

**Speaker Notes**
- Highlight neutrality in news tone; AI exhibits slightly more variation in dialogues.
- Mention metric derived in `scripts/final_discocat_analyzer.py`.

## Slide 18 – Density Matrix Case Study
- Text: "麥當勞性侵案後改革 董事長發聲承諾改善"
- 9 tokens → mapped to N, F, V categories
- 8 qubits used to encode superposition & interactions

**Speaker Notes**
- Summarize steps from segmentation to final metrics.
- Use this slide to demystify computation for non-quantum audience.

## Slide 19 – Sentiment Prediction Reproduction (Binary)
- Quantum-inspired sentiment scores (positive/negative/neutral)
- Dataset: 934 samples (AI + CNA)
- Model accuracy: 0.984 (Random Forest/SVM/LogReg)

**Speaker Notes**
- Reference `quantum_sentiment_predictor.py` and `sentiment_analysis_report.md`.
- Note class imbalance (98.6% neutral) reflecting journalistic tone.

## Slide 20 – Emotion Classification (Paper-Based)
- Four emotions: happiness, fear, anger, sadness
- Quantum Logistic Regression accuracy: 0.9572
- Balanced distribution: anger (27%), fear (25%), happiness (25%), sadness (23%)

**Speaker Notes**
- Highlight replication of paper methodology (`paper_based_sentiment_analyzer.py`).
- Stress success in extending DisCoCat pipeline to multi-class emotion tasks.

## Slide 21 – AI vs CNA Metric Comparison Table
- Counts & metric means from §6.11
- AI Titles/Dialogues/Descriptions: MR≈1.70, C=1.0000, S≈0.01, S_interf≤0.018
- CNA Titles/Content: MR≈0.75/0.65, C≈0.999/0.917, Entropy≈3.44/7.35

**Speaker Notes**
- Display reconstructed table with values from methodology document.
- Emphasize relative patterns rather than absolute scales.

## Slide 22 – Frame Competition Interpretation
- AI perfect competition → uniform framing
- CNA titles near perfect; content drops due to narrative focus
- Suggests professional curation yields dominant framing in long-form

**Speaker Notes**
- Connect to editorial practices and contextual storytelling in CNA articles.
- Discuss implications for audience interpretation and trust.

## Slide 23 – Entropy & Information Density Takeaways
- CNA content entropy 7.35 >> AI 4.0 → richer detail
- AI uniform entropy = consistent but potentially shallow coverage
- Highlights complementarity of AI summaries and human deep dives

**Speaker Notes**
- Encourage hybrid workflows leveraging strengths of both sources.

## Slide 24 – Emotional Tone Distribution
- AI dialogues slightly higher interference → more emotion
- CNA remains tightly neutral across sections
- Supports hypothesis of AI modeling emotional variance for engagement

**Speaker Notes**
- Reference quantitative differences from analysis table and sentiment results.
- Suggest targeted guardrails for AI emotional expression.

## Slide 25 – Visualization Portfolio
- Density matrix workflow diagrams (`paper/density_matrix_*.png`)
- Quantum circuit diagrams (`quantum_circuit_diagram.png`)
- Frame competition illustrations (`frame_competition_*`)

**Speaker Notes**
- Encourage attendees to review generated visuals for training and documentation.
- Mention font-rendering solutions implemented (fallback fonts, ASCII symbols).

## Slide 26 – Tooling & Reproducibility
- Scripts in `scripts/` and `paper/` orchestrating analysis & visuals
- Results stored in `results/` and `sentiment_prediction_from_paper/`
- Automation via Qiskit simulators and Python pipelines

**Speaker Notes**
- Highlight reproducible workflow: segmentation → circuit → metrics → reports.
- Mention fixes for earlier runtime issues (font glyphs, import errors).

## Slide 27 – Key Findings Summary
- AI content: high frame competition, multiple realities, uniform entropy
- CNA content: richer entropy, structured framing, sustained neutrality
- Quantum sentiment & emotion models perform robustly (≥95% accuracy)

**Speaker Notes**
- Synthesize numerical insights into strategic takeaways.
- Emphasize contributions to quantum-assisted media analysis.

## Slide 28 – Implications for Practice
- Media watchdogs can audit AI framing with quantum metrics
- Newsrooms can benchmark AI tools against human-authored standards
- Quantum pipelines unlock nuanced tone monitoring in Chinese-language corpora

**Speaker Notes**
- Discuss potential integrations into editorial QA, regulatory oversight, and AI governance.

## Slide 29 – Limitations & Future Work
- Class imbalance in sentiment labels (98.6% neutral)
- Need for larger CNA sample and real quantum hardware runs
- Planned extensions: richer lexicons, temporal analysis, human-in-the-loop validation

**Speaker Notes**
- Acknowledge constraints candidly to build credibility.
- Invite collaboration on future research directions.

## Slide 30 – Closing & Q&A
- Recap: Quantum lens reveals framing differentials between AI and CNA news
- All assets available in repository under `/paper`, `/results`, `/presentations`
- Questions & discussion

**Speaker Notes**
- Thank the audience.
- Open the floor for questions, referencing the documentation for deeper dives.
