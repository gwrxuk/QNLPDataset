## Qubit Usage Across the Codebase

- **Baseline DisCoCat circuit (`information_society/paper_visuals/quantum_circuit_visualization.py`)**  
  - Draws exactly four quantum wires: `q0` for the topical noun, `q1` for the main verb, `q2` for modifiers, `q3` for functional/discourse roles.  
  - This reflects the canonical lexical register used in the interactive figures.

- **Qiskit renderings (`information_society/paper_visuals/qiskit_discocat_circuit.py`)**  
  - `build_lexical_encoding_circuit()` allocates five qubits (`QuantumCircuit(5, 5)`), matching the four lexical slots plus one ancilla.  
  - `build_frame_competition_circuit()` expands to six qubits (`QuantumCircuit(6, 6)`) to capture two ancillae during frame-competition scenarios.  
  - These circuits illustrate how ancilla qubits appear when we model discourse-level adjustments.

- **Token-driven qubit planners (paper folder)**  
  - `paper/qubit_count_clarification.py` maps the nine DisCoCat categories \(N, V, A, D, P, R, C, F, X\) onto qubits `q0`–`q8`, then reports which ones are active for the sentence.  
  - `paper/qubit_assignment_validation.py` cross-checks the same mapping against actual POS tags.  
  - `paper/frame_competition_calculation_example.py` sets `num_qubits = min(8, max(3, len(unique_categories) + 2))`, so the circuit adjusts to grammar demand but never exceeds eight qubits.

- **Other illustrative scripts (`paper/frame_competition_*.py`)**  
  - Many of these diagrams use compact toy registers (often `n_qubits = 3`) to visualise the entropy and frame metrics without overwhelming detail.

- **Sentiment classifiers (information_society/sentiment_prediction + sentiment_prediction_from_paper)**  
  - Both pipelines state a 4-qubit ansatz for the Qiskit-based sentiment model, separate from the DisCoCat grammar circuits.

- **Key takeaway**  
  - The core DisCoCat analysis relies on four lexical qubits and optional ancillae.  
  - Supporting scripts demonstrate how the register can scale (up to eight or nine slots) when we enumerate all DisCoCat categories or stage advanced frame-competition experiments.  
  - All code paths stay within modest qubit counts, aligning with the narrative presented in `information_society/qubit_number.md`.

