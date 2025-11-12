#!/usr/bin/env python3
"""
Generate DisCoCat-inspired quantum circuit diagrams with Qiskit.

Creates two illustrative circuits:
1. A lexical encoding circuit showing the register layout used in the study.
2. A frame-competition circuit highlighting contextual ancilla qubits.

The resulting figures are saved as PNG files in the paper_visuals directory.
"""

import os
from typing import Dict

import numpy as np

from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

OUTPUT_DIR = "/Users/junghualiu/case/a2a/qnlp/discocat_qnlp_analysis/information_society/paper_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_lexical_encoding_circuit() -> QuantumCircuit:
    """Create the baseline lexical encoding circuit used for a single sentence."""
    qc = QuantumCircuit(5, 5, name="LexicalEncoding")

    noun_theta = 0.65
    verb_theta = 1.1
    modifier_phi = 0.9
    frame_phi = 0.4

    qc.h(0)
    qc.h(2)

    qc.ry(noun_theta, 0)
    qc.ry(verb_theta, 1)
    qc.rz(modifier_phi, 2)
    qc.ry(frame_phi, 3)

    qc.cx(0, 1)
    qc.crz(0.6, 1, 2)
    qc.cx(2, 3)
    qc.swap(3, 4)

    qc.barrier()
    qc.measure_all(add_bits=False)

    return qc


def build_frame_competition_circuit() -> QuantumCircuit:
    """Create a circuit that introduces ancilla-driven frame competition dynamics."""
    qc = QuantumCircuit(6, 6, name="FrameCompetition")

    lex_angles = {
        "noun": 0.55,
        "verb": 1.25,
        "modifier": 0.35,
        "function": 0.75,
    }

    qc.h(range(4))

    qc.ry(lex_angles["noun"], 0)
    qc.ry(lex_angles["verb"], 1)
    qc.rz(lex_angles["modifier"], 2)
    qc.ry(lex_angles["function"], 3)

    qc.ry(0.9, 4)
    qc.ry(0.6, 5)

    qc.cx(0, 1)
    qc.crz(0.4, 1, 2)
    qc.crz(0.3, 2, 3)
    qc.cx(3, 4)
    qc.ccx(4, 5, 1)
    qc.crx(0.45, 5, 0)

    qc.barrier()
    qc.measure_all(add_bits=False)

    return qc


def scale_angle(base_angle: float, competition: float, gain: float = 1.0) -> float:
    """Scale rotation angles according to competition strength."""
    return base_angle * (0.5 + gain * competition)


def competition_to_gate_schedule(competition: float) -> Dict[str, float]:
    """Map entropy-based competition ∈ [0,1] to gate parameters."""
    competition = np.clip(competition, 0.0, 1.0)
    schedule = {
        "lexical_gain": 0.4 + 0.6 * competition,
        "ancilla_amp": competition * np.pi / 2,
        "context_amp": competition * np.pi / 3,
        "include_triple": competition > 0.65,
        "include_cross": competition > 0.35,
    }
    return schedule


def build_competition_conditioned_circuit(competition: float, label: str) -> QuantumCircuit:
    """
    Build a frame-aware circuit where ancilla interactions respond to the
    entropy-derived competition score (see calculate_frame_competition).
    """
    sched = competition_to_gate_schedule(competition)

    qc = QuantumCircuit(6, 6, name=f"FrameCompetition_{label}")

    base_angles = {
        "noun": 0.45,
        "verb": 0.95,
        "modifier": 0.35,
        "function": 0.60,
    }

    qc.h(range(4))

    qc.ry(scale_angle(base_angles["noun"], competition, sched["lexical_gain"]), 0)
    qc.ry(scale_angle(base_angles["verb"], competition, sched["lexical_gain"]), 1)
    qc.rz(scale_angle(base_angles["modifier"], competition, sched["lexical_gain"]), 2)
    qc.ry(scale_angle(base_angles["function"], competition, sched["lexical_gain"]), 3)

    qc.ry(sched["ancilla_amp"], 4)
    qc.ry(sched["context_amp"], 5)

    qc.cx(0, 1)
    qc.crz(0.25 + 0.35 * competition, 1, 2)
    if sched["include_cross"]:
        qc.crz(0.2 + 0.25 * competition, 2, 3)
    qc.cx(3, 4)

    if sched["include_triple"]:
        qc.ccx(4, 5, 2)
        qc.crx(0.2 + 0.5 * competition, 5, 0)
    else:
        qc.cx(4, 5)
        qc.crx(0.1 + 0.3 * competition, 5, 1)

    qc.barrier(label=f"Competition={competition:.2f}")
    qc.measure_all(add_bits=False)

    return qc


def save_circuit_diagram(circuit: QuantumCircuit, filename: str) -> None:
    """Render the circuit with Qiskit and save the diagram as a PNG."""
    path = os.path.join(OUTPUT_DIR, filename)
    circuit_drawer(
        circuit,
        output="mpl",
        fold=-1,
        interactive=False,
        style="iqp",
        filename=path,
    )
    print(f"Saved {filename}")


def main():
    print("Building Qiskit-based DisCoCat circuits...")

    lexical_circuit = build_lexical_encoding_circuit()
    frame_comp_circuit = build_frame_competition_circuit()
    competition_examples = {
        "low": 0.05,
        "medium": 0.45,
        "high": 0.85,
    }

    save_circuit_diagram(lexical_circuit, "qiskit_discocat_lexical_circuit.png")
    save_circuit_diagram(frame_comp_circuit, "qiskit_discocat_frame_competition.png")
    for label, value in competition_examples.items():
        circuit = build_competition_conditioned_circuit(value, label)
        save_circuit_diagram(
            circuit,
            f"qiskit_discocat_competition_{label}.png",
        )

    print("All circuits generated.")


if __name__ == "__main__":
    main()

