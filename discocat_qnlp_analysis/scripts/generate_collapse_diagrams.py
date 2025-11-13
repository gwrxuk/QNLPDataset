#!/usr/bin/env python3
"""Generate string diagram and quantum circuit visuals for collapse modes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Statevector

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "20251112_collapose"
OUTPUT_DIR.mkdir(exist_ok=True)



def draw_collapse_string_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    color = '#1a2650'
    accent = '#2c3e75'

    top_y = 0.88
    cup_y1 = 0.65
    cup_y2 = 0.48
    collapse_center_y = 0.32
    cap_y = 0.2
    output_y = 0.07

    lexicals = [
        ('Word 1', 0.18),
        ('Word 2', 0.50),
        ('Word 3', 0.82),
    ]

    def draw_wire(x: float, y0: float, y1: float, lw: float = 3.0) -> None:
        ax.plot([x, x], [y0, y1], color=color, linewidth=lw)

    def draw_cup(x_left: float, x_right: float, y: float, depth: float = 0.07) -> float:
        verts = [
            (x_left, y),
            (x_left, y - depth),
            (x_right, y - depth),
            (x_right, y),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        patch = PathPatch(MplPath(verts, codes), linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(patch)
        return (x_left + x_right) / 2

    def draw_cap(x_left: float, x_right: float, y: float, height: float = 0.07) -> None:
        verts = [
            (x_left, y),
            (x_left, y + height),
            (x_right, y + height),
            (x_right, y),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        patch = PathPatch(MplPath(verts, codes), linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(patch)

    def draw_leg(x_start: float, y_start: float, x_end: float, y_end: float) -> None:
        ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=2)

    for label, x in lexicals:
        circle = Circle((x, top_y), 0.015, facecolor=color, edgecolor=color)
        ax.add_patch(circle)
        draw_wire(x, top_y, cup_y1)
        ax.text(x, top_y + 0.045, label, ha='center', va='bottom', fontsize=12, fontweight='bold', color=color)
        ax.text(x, top_y - 0.05, 'lexical leg', ha='center', va='top', fontsize=9, color=accent)

    # extend third lexical leg to second cup height
    draw_wire(lexicals[2][1], cup_y1, cup_y2)

    # first cup: headline + dialogue
    mid1_x = draw_cup(lexicals[0][1], lexicals[1][1], cup_y1)
    draw_wire(mid1_x, cup_y1, cup_y2 + 0.02)
    ax.text(
        (lexicals[0][1] + lexicals[1][1]) / 2,
        cup_y1 - 0.11,
        "cup (⊔) composes Word 1\nand Word 2",
        ha='center',
        va='center',
        fontsize=9,
        color=accent,
    )

    # second cup: combine result with description
    mid2_x = draw_cup(mid1_x, lexicals[2][1], cup_y2, depth=0.08)
    draw_wire(mid2_x, cup_y2, collapse_center_y + 0.08)
    ax.text(
        (mid1_x + lexicals[2][1]) / 2,
        cup_y2 - 0.12,
        "cup (⊔) brings in Word 3\ncontext",
        ha='center',
        va='center',
        fontsize=9,
        color=accent,
    )

    # collapse region
    collapse_box = Rectangle((mid2_x - 0.11, collapse_center_y - 0.07), 0.22, 0.14,
                              facecolor='#e8eefc', edgecolor=color, linewidth=2)
    ax.add_patch(collapse_box)
    ax.text(mid2_x, collapse_center_y + 0.02, 'Collapse', ha='center', va='center', fontsize=12,
            fontweight='bold', color=color)
    ax.text(mid2_x, collapse_center_y - 0.035, 'post-readout / mid-circuit',
            ha='center', va='center', fontsize=9, color=accent)

    # wire exiting collapse
    draw_wire(mid2_x, collapse_center_y - 0.07, cap_y + 0.02)

    # cap leading to metrics
    cap_left = mid2_x - 0.13
    cap_right = mid2_x + 0.13
    draw_cap(cap_left, cap_right, cap_y, height=0.08)
    draw_wire(cap_left, cap_y, cap_y - 0.04, lw=2.5)
    draw_wire(cap_right, cap_y, cap_y - 0.04, lw=2.5)

    metrics_left_x = cap_left - 0.14
    metrics_right_x = cap_right + 0.14
    draw_leg(cap_left, cap_y - 0.04, metrics_left_x, output_y + 0.015)
    draw_leg(cap_right, cap_y - 0.04, metrics_right_x, output_y + 0.015)

    ax.text(
        metrics_left_x,
        output_y,
        "Quantum metrics\n(frame competition, entropy)",
        ha='center',
        va='top',
        fontsize=9,
        color=color,
    )
    ax.text(
        metrics_right_x,
        output_y,
        "Classical exports\n(JSON / CSV summaries)",
        ha='center',
        va='top',
        fontsize=9,
        color=color,
    )

    ax.text(0.5, 0.97, 'Sentence (tokens)', ha='center', va='bottom', fontsize=11, color=accent)
    ax.text(0.5, 0.94, 'DisCoCat-style collapse flow', ha='center', va='center', fontsize=14,
            fontweight='bold', color=color)
    ax.text(
        0.5,
        0.04,
        "Cups (⊔) compose lexical legs; the cap (⊓) realises metrics after collapse.",
        ha='center',
        va='center',
        fontsize=10,
        color=accent,
    )

    output_path = OUTPUT_DIR / 'collapse_string_diagram.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def build_post_readout_circuit() -> QuantumCircuit:
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "c")
    qc = QuantumCircuit(qreg, creg, name="post_readout")

    qc.h(qreg)
    qc.ry(0.6, qreg[0])
    qc.ry(0.4, qreg[1])
    qc.ry(0.2, qreg[2])
    qc.cx(qreg[0], qreg[1])
    qc.cx(qreg[1], qreg[2])
    qc.barrier()
    qc.measure(qreg, creg)
    return qc


def build_mid_circuit_collapse_circuit() -> QuantumCircuit:
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "c")
    qc = QuantumCircuit(qreg, creg, name="mid_circuit")

    qc.h(qreg)
    qc.ry(0.6, qreg[0])
    qc.ry(0.4, qreg[1])
    qc.ry(0.2, qreg[2])
    qc.cx(qreg[0], qreg[1])

    qc.barrier()
    qc.measure(qreg[0], creg[0])
    qc.reset(qreg[0])
    qc.ry(0.3, qreg[0]).c_if(creg[0], 0)
    qc.ry(-0.3, qreg[0]).c_if(creg[0], 1)

    qc.barrier()
    qc.cx(qreg[0], qreg[2])
    qc.cx(qreg[1], qreg[2])

    qc.barrier()
    qc.measure(qreg[1], creg[1])
    qc.measure(qreg[2], creg[2])
    return qc

def draw_pre_collapse_string_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    color = '#1a2650'
    accent = '#2c3e75'

    top_y = 0.88
    cup_y1 = 0.65
    cup_y2 = 0.48
    superposition_y = 0.33
    cap_y = 0.2
    output_y = 0.07

    lexicals = [
        ('Word 1', 0.18),
        ('Word 2', 0.50),
        ('Word 3', 0.82),
    ]

    def draw_wire(x: float, y0: float, y1: float, lw: float = 3.0) -> None:
        ax.plot([x, x], [y0, y1], color=color, linewidth=lw)

    def draw_cup(x_left: float, x_right: float, y: float, depth: float = 0.07) -> float:
        verts = [
            (x_left, y),
            (x_left, y - depth),
            (x_right, y - depth),
            (x_right, y),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        patch = PathPatch(MplPath(verts, codes), linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(patch)
        return (x_left + x_right) / 2

    def draw_cap(x_left: float, x_right: float, y: float, height: float = 0.07) -> None:
        verts = [
            (x_left, y),
            (x_left, y + height),
            (x_right, y + height),
            (x_right, y),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        patch = PathPatch(MplPath(verts, codes), linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(patch)

    def draw_leg(x_start: float, y_start: float, x_end: float, y_end: float) -> None:
        ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=2)

    for label, x in lexicals:
        circle = Circle((x, top_y), 0.015, facecolor=color, edgecolor=color)
        ax.add_patch(circle)
        draw_wire(x, top_y, cup_y1)
        ax.text(x, top_y + 0.045, label, ha='center', va='bottom', fontsize=12, fontweight='bold', color=color)
        ax.text(x, top_y - 0.05, 'lexical leg', ha='center', va='top', fontsize=9, color=accent)

    draw_wire(lexicals[2][1], cup_y1, cup_y2)

    mid1_x = draw_cup(lexicals[0][1], lexicals[1][1], cup_y1)
    draw_wire(mid1_x, cup_y1, cup_y2 + 0.02)
    ax.text(
        (lexicals[0][1] + lexicals[1][1]) / 2,
        cup_y1 - 0.11,
        "cup (⊔) composes Word 1\nand Word 2",
        ha='center',
        va='center',
        fontsize=9,
        color=accent,
    )

    mid2_x = draw_cup(mid1_x, lexicals[2][1], cup_y2, depth=0.08)
    draw_wire(mid2_x, cup_y2, superposition_y + 0.04)
    ax.text(
        (mid1_x + lexicals[2][1]) / 2,
        cup_y2 - 0.12,
        "cup (⊔) brings in Word 3\ncontext",
        ha='center',
        va='center',
        fontsize=9,
        color=accent,
    )

    ax.text(
        mid2_x,
        superposition_y,
        "Coherent superposition\n(no collapse yet)",
        ha='center',
        va='center',
        fontsize=10,
        color=accent,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#eef2ff', edgecolor=color, linewidth=1.5),
    )

    draw_wire(mid2_x, superposition_y - 0.05, cap_y + 0.02)

    cap_left = mid2_x - 0.13
    cap_right = mid2_x + 0.13
    draw_cap(cap_left, cap_right, cap_y, height=0.08)
    draw_wire(cap_left, cap_y, cap_y - 0.04, lw=2.5)
    draw_wire(cap_right, cap_y, cap_y - 0.04, lw=2.5)

    metrics_left_x = cap_left - 0.14
    metrics_right_x = cap_right + 0.14
    draw_leg(cap_left, cap_y - 0.04, metrics_left_x, output_y + 0.015)
    draw_leg(cap_right, cap_y - 0.04, metrics_right_x, output_y + 0.015)

    ax.text(
        metrics_left_x,
        output_y,
        "Quantum amplitudes\n(pre-collapse)",
        ha='center',
        va='top',
        fontsize=9,
        color=color,
    )
    ax.text(
        metrics_right_x,
        output_y,
        "Classical snapshots\n(statevector, tables)",
        ha='center',
        va='top',
        fontsize=9,
        color=color,
    )

    ax.text(0.5, 0.97, 'Sentence (tokens)', ha='center', va='bottom', fontsize=11, color=accent)
    ax.text(0.5, 0.94, 'DisCoCat-style pre-collapse flow', ha='center', va='center', fontsize=14,
            fontweight='bold', color=color)
    ax.text(
        0.5,
        0.04,
        "Legs compose via cups; no collapse factor has been applied yet.",
        ha='center',
        va='center',
        fontsize=10,
        color=accent,
    )

    output_path = OUTPUT_DIR / 'pre_collapse_string_diagram.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path







def build_pre_collapse_circuit() -> QuantumCircuit:
    qreg = QuantumRegister(3, "q")
    qc = QuantumCircuit(qreg, name="pre_collapse")
    qc.h(qreg)
    qc.ry(0.6, qreg[0])
    qc.ry(0.4, qreg[1])
    qc.ry(0.2, qreg[2])
    qc.cx(qreg[0], qreg[1])
    qc.cx(qreg[1], qreg[2])
    return qc


def save_statevector_amplitudes(qc: QuantumCircuit, filename: str) -> Path:
    state = Statevector.from_instruction(qc)
    amplitudes = state.data
    probabilities = np.abs(amplitudes) ** 2
    num_qubits = len(qc.qubits)
    bitstrings = [format(i, f"0{num_qubits}b") for i in range(len(probabilities))]

    path = OUTPUT_DIR / filename
    fig, ax = plt.subplots(figsize=(max(6, len(bitstrings) * 0.6), 4))
    bars = ax.bar(bitstrings, probabilities, color="#4c6ef5")
    ax.set_ylim(0, max(probabilities) * 1.1 if probabilities.size else 1)
    ax.set_xlabel('Basis state')
    ax.set_ylabel('Probability')
    ax.set_title('Pre-collapse statevector probabilities')
    ax.grid(axis='y', alpha=0.3)

    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{prob:.3f}",
                ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path
def save_circuit_diagram(qc: QuantumCircuit, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    qc.draw(output="mpl", filename=str(path))
    return path


def main() -> None:
    collapse_string_diagram_path = draw_collapse_string_diagram()
    pre_string_diagram_path = draw_pre_collapse_string_diagram()
    pre_circuit = build_pre_collapse_circuit()
    post_circuit = build_post_readout_circuit()
    mid_circuit = build_mid_circuit_collapse_circuit()

    pre_circuit_path = save_circuit_diagram(pre_circuit, "pre_collapse_circuit.png")
    pre_state_path = save_statevector_amplitudes(pre_circuit, "pre_collapse_probabilities.png")
    post_path = save_circuit_diagram(post_circuit, "post_readout_circuit.png")
    mid_path = save_circuit_diagram(mid_circuit, "mid_circuit_collapse_circuit.png")

    print("Generated diagrams:")
    for item in [collapse_string_diagram_path, pre_string_diagram_path, pre_circuit_path, pre_state_path, post_path, mid_path]:
        print(f" - {item}")


if __name__ == "__main__":
    main()
