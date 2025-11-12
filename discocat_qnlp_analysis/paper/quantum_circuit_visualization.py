#!/usr/bin/env python3
"""
Quantum Circuit Visualization for DisCoCat QNLP Analysis
Shows the quantum circuit structure used in the analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

# Set font configuration to avoid rendering issues
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'serif'

def create_quantum_circuit_diagram():
    """Create a quantum circuit diagram for DisCoCat analysis"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Title
    ax.text(5, 5.5, 'DisCoCat Quantum Circuit for QNLP Analysis', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Qubit lines (4 qubits for example)
    qubit_positions = [1, 2, 3, 4]
    qubit_labels = ['q0 (Noun)', 'q1 (Verb)', 'q2 (Adj)', 'q3 (Func)']
    
    # Draw qubit lines
    for i, (y, label) in enumerate(zip(qubit_positions, qubit_labels)):
        ax.plot([0.5, 9.5], [y, y], 'k-', linewidth=2, alpha=0.7)
        ax.text(0.2, y, label, fontsize=10, va='center', ha='right')
    
    # Circuit elements
    x_positions = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    element_labels = ['H', 'RY', 'RZ', 'CX', 'CRZ', 'RY', 'RZ', 'Measure']
    
    # Draw circuit elements
    for i, (x, label) in enumerate(zip(x_positions, element_labels)):
        for j, y in enumerate(qubit_positions):
            if label == 'H':  # Hadamard gate
                draw_hadamard_gate(ax, x, y)
            elif label == 'RY':  # RY rotation
                draw_rotation_gate(ax, x, y, 'RY')
            elif label == 'RZ':  # RZ rotation
                draw_rotation_gate(ax, x, y, 'RZ')
            elif label == 'CX':  # CNOT gate
                if j < len(qubit_positions) - 1:
                    draw_cnot_gate(ax, x, y, qubit_positions[j+1])
            elif label == 'CRZ':  # Controlled RZ
                if j < len(qubit_positions) - 1:
                    draw_controlled_rz(ax, x, y, qubit_positions[j+1])
            elif label == 'Measure':  # Measurement
                draw_measurement(ax, x, y)
    
    # Add step labels
    step_labels = ['Initialize', 'Category\nEncoding', 'Phase\nEncoding', 'Entanglement', 
                   'Controlled\nRotation', 'Final\nRotation', 'Phase\nAdjust', 'Measure']
    
    for i, (x, label) in enumerate(zip(x_positions, step_labels)):
        ax.text(x, 0.3, label, fontsize=8, ha='center', va='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    # Add quantum state representation
    ax.text(5, 0.8, 'Quantum State: |ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩', 
            fontsize=12, ha='center', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/junghualiu/case/a2a/qnlp/discocat_qnlp_analysis/paper/quantum_circuit_diagram.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def draw_hadamard_gate(ax, x, y):
    """Draw a Hadamard gate"""
    # H symbol in a box
    box = FancyBboxPatch((x-0.2, y-0.2), 0.4, 0.4, 
                        boxstyle="round,pad=0.05", 
                        facecolor='lightgreen', edgecolor='black')
    ax.add_patch(box)
    ax.text(x, y, 'H', fontsize=12, ha='center', va='center', fontweight='bold')

def draw_rotation_gate(ax, x, y, gate_type):
    """Draw a rotation gate (RY or RZ)"""
    # Rotation symbol in a box
    box = FancyBboxPatch((x-0.25, y-0.2), 0.5, 0.4, 
                        boxstyle="round,pad=0.05", 
                        facecolor='lightcoral', edgecolor='black')
    ax.add_patch(box)
    ax.text(x, y, gate_type, fontsize=10, ha='center', va='center', fontweight='bold')

def draw_cnot_gate(ax, x, y1, y2):
    """Draw a CNOT gate"""
    # Control qubit (circle)
    circle = Circle((x, y1), 0.15, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y1, '•', fontsize=16, ha='center', va='center')
    
    # Target qubit (cross)
    ax.plot([x-0.15, x+0.15], [y2, y2], 'k-', linewidth=3)
    ax.plot([x, x], [y2-0.15, y2+0.15], 'k-', linewidth=3)
    
    # Connection line
    ax.plot([x, x], [y1+0.15, y2-0.15], 'k-', linewidth=1)

def draw_controlled_rz(ax, x, y1, y2):
    """Draw a controlled RZ gate"""
    # Control qubit (circle)
    circle = Circle((x, y1), 0.15, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y1, '•', fontsize=16, ha='center', va='center')
    
    # RZ gate on target
    box = FancyBboxPatch((x-0.2, y2-0.2), 0.4, 0.4, 
                        boxstyle="round,pad=0.05", 
                        facecolor='lightblue', edgecolor='black')
    ax.add_patch(box)
    ax.text(x, y2, 'RZ', fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Connection line
    ax.plot([x, x], [y1+0.15, y2-0.15], 'k-', linewidth=1)

def draw_measurement(ax, x, y):
    """Draw a measurement symbol"""
    # Measurement box
    box = FancyBboxPatch((x-0.2, y-0.2), 0.4, 0.4, 
                        boxstyle="round,pad=0.05", 
                        facecolor='lightgray', edgecolor='black')
    ax.add_patch(box)
    ax.text(x, y, 'M', fontsize=12, ha='center', va='center', fontweight='bold')

def create_circuit_flow_diagram():
    """Create a flow diagram showing the circuit construction process"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Title
    ax.text(5, 7.5, 'DisCoCat Quantum Circuit Construction Process', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Process steps
    steps = [
        (2, 6.5, 'Text Input\n"麥當勞性侵案後改革"', 'lightblue'),
        (4, 6.5, 'POS Tagging\nN-N-N-F-V', 'lightgreen'),
        (6, 6.5, 'Category Mapping\nq0-q1-q2-q3-q4', 'lightyellow'),
        (8, 6.5, 'Qubit Assignment\n5 qubits needed', 'lightcoral'),
        (2, 5, 'Initialize States\n|0⟩⊗n', 'lightblue'),
        (4, 5, 'Hadamard Gates\nSuperposition', 'lightgreen'),
        (6, 5, 'Rotation Gates\nRY(θ), RZ(φ)', 'lightyellow'),
        (8, 5, 'Entanglement\nCX, CRZ gates', 'lightcoral'),
        (2, 3.5, 'Frame Competition\nGates', 'lightblue'),
        (4, 3.5, 'Semantic Ambiguity\nGates', 'lightgreen'),
        (6, 3.5, 'Compositional\nEntanglement', 'lightyellow'),
        (8, 3.5, 'Final State\n|ψ⟩', 'lightcoral'),
        (5, 2, 'Density Matrix\nρ = |ψ⟩⟨ψ|', 'lightgray'),
        (5, 1, 'Quantum Metrics\nEntropy, Interference', 'lightpink')
    ]
    
    # Draw process boxes
    for x, y, text, color in steps:
        box = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6, 
                            boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Draw arrows
    arrows = [
        ((2, 6.2), (4, 6.2)), ((4, 6.2), (6, 6.2)), ((6, 6.2), (8, 6.2)),
        ((2, 5.7), (2, 5.3)), ((4, 5.7), (4, 5.3)), ((6, 5.7), (6, 5.3)), ((8, 5.7), (8, 5.3)),
        ((2, 4.7), (2, 3.8)), ((4, 4.7), (4, 3.8)), ((6, 4.7), (6, 3.8)), ((8, 4.7), (8, 3.8)),
        ((2, 3.2), (5, 2.3)), ((4, 3.2), (5, 2.3)), ((6, 3.2), (5, 2.3)), ((8, 3.2), (5, 2.3)),
        ((5, 1.7), (5, 1.3))
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    plt.tight_layout()
    plt.savefig('/Users/junghualiu/case/a2a/qnlp/discocat_qnlp_analysis/paper/quantum_circuit_flow.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_gate_details_diagram():
    """Create a detailed diagram of quantum gates used"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Title
    ax.text(6, 9.5, 'Quantum Gates Used in DisCoCat Analysis', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Gate types and descriptions
    gates = [
        (2, 8, 'Hadamard (H)', 'Creates superposition\n|0⟩ → (|0⟩ + |1⟩)/√2', 'lightgreen'),
        (5, 8, 'RY Rotation', 'Y-axis rotation\nRY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩', 'lightblue'),
        (8, 8, 'RZ Rotation', 'Z-axis rotation\nRZ(φ)|ψ⟩ = e^(-iφ/2)|ψ⟩', 'lightcoral'),
        (11, 8, 'CNOT Gate', 'Controlled NOT\n|00⟩ → |00⟩, |10⟩ → |11⟩', 'lightyellow'),
        (2, 6, 'CRZ Gate', 'Controlled RZ\nConditional phase', 'lightpink'),
        (5, 6, 'Frame Competition', 'Semantic interference\nbetween frames', 'lightgray'),
        (8, 6, 'Compositional', 'Grammatical\nentanglement', 'lightcyan'),
        (11, 6, 'Measurement', 'State collapse\nto classical bits', 'lightsteelblue'),
        (3.5, 4, 'Gate Parameters', 'θ = word_frequency × π/4\nφ = emotional_weight × π/2', 'white'),
        (8.5, 4, 'Entanglement Rules', 'Similar POS → Strong coupling\nDifferent POS → Weak coupling', 'white')
    ]
    
    # Draw gate boxes
    for x, y, title, description, color in gates:
        box = FancyBboxPatch((x-1, y-0.8), 2, 1.6, 
                            boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y+0.3, title, fontsize=11, ha='center', va='center', fontweight='bold')
        ax.text(x, y-0.3, description, fontsize=9, ha='center', va='center')
    
    # Add mathematical formulas
    formulas = [
        (6, 2.5, 'Quantum State Evolution:', '|ψ⟩ = Uₙ...U₂U₁|0⟩⊗n'),
        (6, 2, 'Density Matrix:', 'ρ = |ψ⟩⟨ψ|'),
        (6, 1.5, 'Von Neumann Entropy:', 'S = -Tr(ρ log₂ ρ)'),
        (6, 1, 'Frame Competition:', 'C = -Tr(ρ log₂ ρ) / log₂ N')
    ]
    
    for x, y, label, formula in formulas:
        ax.text(x, y, f'{label} {formula}', fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/junghualiu/case/a2a/qnlp/discocat_qnlp_analysis/paper/quantum_gates_details.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating quantum circuit visualizations...")
    
    print("1. Creating quantum circuit diagram...")
    create_quantum_circuit_diagram()
    
    print("2. Creating circuit flow diagram...")
    create_circuit_flow_diagram()
    
    print("3. Creating gate details diagram...")
    create_gate_details_diagram()
    
    print("All quantum circuit visualizations generated successfully!")
    print("Files saved:")
    print("- quantum_circuit_diagram.png")
    print("- quantum_circuit_flow.png") 
    print("- quantum_gates_details.png")
