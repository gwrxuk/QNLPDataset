## Abstract

This note documents the effective qubit budget employed in the DisCoCat-based quantum natural language processing (QNLP) pipeline developed for the information-society case study. By auditing all source modules that construct quantum circuits, we confirm that the core sentence-level analysis operates on a four-qubit lexical register augmented by optional ancillae that are invoked only when discourse-level interference tests are required. We detail the grammatical mapping that motivates the four lexical wires, the operational role of ancilla qubits, and the empirical outputs obtained from representative sentences.

## 1. Introduction

Quantum interpretations of categorical compositional semantics routinely face scaling concerns: as the grammar of a sentence grows, so does the potential demand for qubits. To ensure methodological transparency, we catalogue the qubit usage embedded in our codebase and describe the circuit layout that underpins all experimental results reported in the accompanying paper. The objective is to provide a definitive, reproducible statement of the qubit counts used for each sentence, together with the rationale for employing ancilla qubits (`q4+`) in specific analytical scenarios.

## 2. Methods

### 2.1 Code audit procedure

We inspected every Python module under the project root that instantiates a quantum circuit or visualises a circuit diagram. Two families of scripts are directly relevant to the DisCoCat pipeline: (i) the Matplotlib-based illustrators in `information_society/paper_visuals/`, which document the canonical flow of gates; and (ii) the Qiskit-based artefacts in both the `information_society/paper_visuals/` and `paper/` directories, which either render circuits or simulate frame-competition metrics. Ancillary sentiment-classification scripts were also reviewed to ensure there is no additional hidden qubit usage.

### 2.2 Lexical register

The core circuit, visualised in `information_society/paper_visuals/quantum_circuit_visualization.py`, declares four qubit lines whose labels and grammatical roles are fixed across all figures:

- `q0`: topical noun phrase (sentence subject),
- `q1`: main predicate or verb tensor,
- `q2`: adjectival or adverbial modifier slot,
- `q3`: functional/discourse marker (determiners, particles, connectors).

This four-wire register corresponds to the minimal categorical structure needed to encode the compositional semantics of the news sentences under study. Each sentence is processed independently; unused slots remain idle in the `|0⟩` state, preserving uniform circuit width without introducing spurious gates.

### 2.3 Ancilla allocation

Optional ancilla qubits (`q4`, `q5`, …) are introduced in two contexts:

1. **Frame-competition probes.** The Qiskit routines in `information_society/paper_visuals/qiskit_discocat_circuit.py` construct two circuits. The baseline lexical circuit (`build_lexical_encoding_circuit`) allocates a fifth qubit as a staging ancilla for swapping functional content, whereas the frame-competition circuit (`build_frame_competition_circuit`) extends the register to six qubits to accommodate contextual comparison and controlled operations such as `CCX` and `CRX`.
2. **Category enumerations.** Analytical scripts in the `paper/` directory (e.g., `qubit_count_clarification.py`, `qubit_assignment_validation.py`) map the full set of nine DisCoCat categories to qubits `q0`–`q8`. These scripts are diagnostically oriented—they reveal which categories are present in a sentence and therefore which qubits would be active should we upscale the physical register. In practice, only the four lexical qubits are exercised during the core analysis runs.

Ancilla qubits begin in `|0⟩` and are only entangled with the lexical register when the experiment requires it (e.g., measuring discourse interference). If the analysis for a sentence does not trigger frame-competition routines, the ancillae remain idle and can be excluded entirely.

## 3. Sentence-level execution

Each sentence in the corpus is handled by a dedicated circuit instance. The procedure is as follows:

1. **Initialisation:** `H` gates are applied to the active lexical wires (`q0`–`q3`) to prepare a uniform superposition, granting headroom for semantic rotations.
2. **Semantic parameterisation:** Category-specific rotations (`RY` and `RZ`) encode the amplitudes extracted from lexical features (frequency, sentiment, contextual weighting). The angles are derived from pre-processing pipelines that compute word-level metrics.
3. **Compositional entanglement:** Controlled gates (`CX`, `CRZ`, sometimes `SWAP`) capture the grammatical flow dictated by the pregroup reductions. For example, the subject noun conditions the verb via `CX(q0 → q1)`, while modifiers interface with functional slots through `CRZ` operations.
4. **Optional ancilla interaction:** When frame competition is tested, ancilla qubits receive rotations representing competing frames (e.g., political vs. economic narrative) and participate in controlled entangling operations (`CCX`, `CRX`) with the lexical register.
5. **Measurement:** `measure_all` records classical outcomes for every active qubit. Sentence-level metrics (entropy, interference) are computed from the resulting statevector or density matrix.

The steps above are repeated for each sentence; no circuit spans multiple sentences, ensuring that qubit counts scale with grammatical complexity rather than paragraph length.

## 4. Results

### 4.1 Core circuit width

Across all illustrative and production scripts tied to DisCoCat semantics, we observed that sentences with the full noun–verb–modifier–function structure activate four qubits. Sentences lacking a modifier or functional component leave the corresponding qubits idle, effectively lowering the active width to three or fewer while keeping the register consistent.

### 4.2 Ancilla utilisation

Frame-competition experiments employ one or two ancilla qubits. The additional wires are visible in the Qiskit-rendered diagrams (`qiskit_discocat_lexical_circuit.png`, `qiskit_discocat_frame_competition.png`). In model runs where frame competition is disabled, these ancillae are omitted, and the circuit reverts to the four-qubit baseline.

### 4.3 Diagnostic scripts

Scripts that map the nine DisCoCat categories onto qubits (`q0`–`q8`) confirm that, for the sentences analysed, only categories N, V, A, and F/D are populated. Consequently, physical instantiations never require more than four lexical qubits, although the diagnostic pipeline is ready to scale should future corpora introduce richer grammatical diversity.

## 5. Discussion

The audit demonstrates that the DisCoCat implementation satisfies resource-conscious design principles. By constraining the active register to four lexical qubits per sentence and reserving ancilla qubits for specialised probes, the approach balances expressive power with near-term quantum hardware limitations. The explicit mapping between grammatical roles and qubit indices also aids interpretability, making it straightforward to trace how linguistic features influence quantum amplitudes. Future work may explore adaptive qubit allocation strategies for longitudinal discourse analysis, but the current pipeline already maintains a tight qubit budget that aligns with NISQ-era constraints.

## 6. Conclusion

We provide a comprehensive, code-backed explanation of qubit usage in the core DisCoCat analysis. Every sentence is encoded on four principal qubits corresponding to key grammatical slots, and ancilla qubits are invoked only for targeted frame-competition experiments. This documentation should enable reviewers and collaborators to reproduce the reported circuit layouts and verify that the qubit counts reported in the manuscript are accurate.

