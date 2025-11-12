## Qubit Allocation Across Sentences

- **Sentence-centric circuits**  
  Each sentence is converted into its own quantum circuit. The grammar reduction for that sentence decides how many lexical wires (qubits) we need and how they must interact.

- **Token-to-qubit mapping**  
  Nouns, verbs, adjectives, function words, and discourse markers are encoded on distinct qubits. The compact closed functor maps each part-of-speech tensor into a Hilbert space factor, so the number of qubits scales with the grammatical arity of the words in the sentence.

- **Ancilla and structural qubits**  
  Additional qubits appear when the circuit needs helper systems for frame competition, sentence-level context, or measurement staging. We budget ancillae per sentence, not across the entire paragraph.

- **Paragraph handling**  
  A paragraph is processed sentence by sentence. After finishing one sentence circuit we measure or store its state vector/density matrix, then start a new circuit for the next sentence. There is no shared set of qubits spanning multiple sentences unless we deliberately introduce paragraph-level entanglement.

- **Why the per-sentence reset?**  
  Resetting ensures grammatical composition stays local and avoids unintended correlations between sentences. It also keeps the circuit width manageable: long paragraphs with many sentences do not force a runaway growth in qubit count, because each circuit only needs to honor its own sentence structure.

- **Canonical register in this study**  
  For the illustrative sentences we analyse, we deploy four primary lexical qubits:  
  - `q0` for the topical noun phrase (subject noun)  
  - `q1` for the main predicate/verb tensor  
  - `q2` for adjectival or adverbial modifiers  
  - `q3` as a functional slot covering determiners, particles, or discourse connectors  
  On top of these, we introduce ancilla qubits (`q4+`) when we stage frame-competition experiments or need scratch space for controlled rotations. When a sentence lacks a category (e.g., no adjective), its corresponding qubit is initialised and left idle, effectively reducing the active width for that circuit instance.

- **Comparing sentence outputs**  
  Once every sentence has been evaluated, we can aggregate their measurement results or density matrices to compute discourse-level metrics (e.g., framing bias trends). These are classical post-processing steps; they do not require us to keep all sentence qubits alive simultaneously.

- **Practical guideline**  
  Estimate qubit count by summing the wires required for the lexical items in one sentence plus any ancillae needed for semantic interference experiments. Repeat that estimation for each sentence to understand the total workload, but remember that circuits run sequentially, not in parallel on a single register.

