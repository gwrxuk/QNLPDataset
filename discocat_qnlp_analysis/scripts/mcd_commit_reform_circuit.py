#!/usr/bin/env python3
"""
Construct and render a Qiskit circuit for the sentence 「麥當勞 承諾 改革」 using 3 qubits.

Qubit mapping (left-to-right):
    q0 -> n     (麥當勞)
    q1 -> s     (句子 / verb anchor)
    q2 -> n     (改革)
"""

from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "20251112_collapose"
    / "mcd_commit_reform_circuit.png"
)


def build_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3, name="麥當勞承諾改革")
    q_mcd, q_s, q_reform = range(3)

    # Step 1: Initialize superposition for each noun (semantic basis)
    qc.h(q_mcd)
    qc.h(q_reform)

    # Step 2: Encode the verb '承諾' as an entangling gate binding nouns and sentence qubit
    qc.cx(q_mcd, q_s)
    qc.cx(q_reform, q_s)

    # Step 3: Apply contextual rotations (encoding semantics)
    qc.ry(0.6, q_mcd)
    qc.ry(1.2, q_reform)
    qc.crz(0.8, q_mcd, q_s)
    qc.crz(1.0, q_reform, q_s)

    # Step 4: Collapse composition — simulate meaning reduction n⊗n→s
    qc.cz(q_mcd, q_reform)
    qc.cz(q_s, q_reform)

    qc.measure_all()

    return qc


def main() -> None:
    circuit = build_circuit()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig = circuit.draw(output="mpl")
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Circuit saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

