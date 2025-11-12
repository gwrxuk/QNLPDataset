#!/usr/bin/env python3
"""
Density Matrix Visualization for Quantum NLP - Fixed Font Version
Creates charts and images to explain density matrix calculation process
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set font support for Chinese and mathematical symbols
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'serif'

class DensityMatrixVisualizer:
    def __init__(self):
        """Initialize the density matrix visualizer"""
        pass
        
    def create_text_segmentation_diagram(self):
        """Create diagram showing text segmentation process"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Example text
        text = "麥當勞性侵案後改革 董事長發聲承諾改善"
        
        # Segmentation result
        words_with_pos = [
            ('麥當勞', 'nt'), ('性侵', 'n'), ('案', 'n'), ('後', 'f'),
            ('改革', 'v'), ('董事長', 'n'), ('發聲', 'v'), ('承諾', 'v'), ('改善', 'v')
        ]
        
        # Create visualization
        y_pos = 0.8
        ax.text(0.1, y_pos, f"Input Text: {text}", fontsize=14, weight='bold')
        
        y_pos -= 0.1
        ax.text(0.1, y_pos, "Segmentation Result:", fontsize=12, weight='bold')
        
        y_pos -= 0.08
        for i, (word, pos) in enumerate(words_with_pos):
            x_pos = 0.1 + (i % 3) * 0.25
            if i > 0 and i % 3 == 0:
                y_pos -= 0.08
            
            # Color code by POS
            color = {'nt': 'lightblue', 'n': 'lightgreen', 'f': 'lightcoral', 'v': 'lightyellow'}
            rect = Rectangle((x_pos-0.01, y_pos-0.02), 0.2, 0.06, 
                           facecolor=color.get(pos, 'lightgray'), alpha=0.7)
            ax.add_patch(rect)
            
            ax.text(x_pos, y_pos, f"{word}\n({pos})", fontsize=10, ha='center', va='center')
        
        # Category mapping
        y_pos -= 0.15
        ax.text(0.1, y_pos, "Category Mapping:", fontsize=12, weight='bold')
        
        y_pos -= 0.08
        categories = ['N', 'N', 'N', 'F', 'V', 'N', 'V', 'V', 'V']
        for i, cat in enumerate(categories):
            x_pos = 0.1 + (i % 3) * 0.25
            if i > 0 and i % 3 == 0:
                y_pos -= 0.08
            
            color = {'N': 'lightgreen', 'F': 'lightcoral', 'V': 'lightyellow'}
            rect = Rectangle((x_pos-0.01, y_pos-0.02), 0.2, 0.06, 
                           facecolor=color.get(cat, 'lightgray'), alpha=0.7)
            ax.add_patch(rect)
            
            ax.text(x_pos, y_pos, cat, fontsize=10, ha='center', va='center')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Text Segmentation and Category Mapping', fontsize=16, weight='bold')
        
        plt.tight_layout()
        plt.savefig('density_matrix_text_segmentation_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_quantum_circuit_diagram(self):
        """Create quantum circuit visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Circuit parameters
        num_qubits = 8
        qubit_spacing = 0.1
        gate_width = 0.08
        
        # Draw qubit lines
        for i in range(num_qubits):
            y = 0.9 - i * qubit_spacing
            ax.plot([0, 1], [y, y], 'k-', linewidth=2)
            ax.text(-0.05, y, f'q{i}', fontsize=10, ha='right', va='center')
        
        # Draw Hadamard gates
        for i in range(num_qubits):
            y = 0.9 - i * qubit_spacing
            rect = Rectangle((0.1, y-0.02), gate_width, 0.04, 
                           facecolor='lightblue', alpha=0.8)
            ax.add_patch(rect)
            ax.text(0.14, y, 'H', fontsize=8, ha='center', va='center')
        
        # Draw rotation gates
        rotation_gates = [
            (0.25, 0, 'RY(pi/18)', 'lightgreen'),  # Noun
            (0.25, 1, 'RY(2pi/135)', 'lightcoral'),  # Function
            (0.25, 2, 'RY(pi/15)', 'lightyellow')   # Verb
        ]
        
        for x, qubit, label, color in rotation_gates:
            y = 0.9 - qubit * qubit_spacing
            rect = Rectangle((x, y-0.02), gate_width, 0.04, 
                           facecolor=color, alpha=0.8)
            ax.add_patch(rect)
            ax.text(x+0.04, y, label, fontsize=7, ha='center', va='center')
        
        # Draw CNOT gate
        y1 = 0.9 - 0 * qubit_spacing  # Control qubit (Noun)
        y2 = 0.9 - 2 * qubit_spacing  # Target qubit (Verb)
        
        # Control dot
        ax.plot(0.4, y1, 'ko', markersize=8)
        # Target with X
        ax.plot(0.4, y2, 'ko', markersize=8)
        ax.text(0.4, y2-0.03, 'X', fontsize=10, ha='center', va='center', color='white')
        # Connection line
        ax.plot([0.4, 0.4], [y1, y2], 'k-', linewidth=2)
        ax.text(0.45, (y1+y2)/2, 'CNOT', fontsize=8, ha='left', va='center')
        
        # Draw frame competition gates
        for i in range(3):
            y1 = 0.9 - i * qubit_spacing
            y2 = 0.9 - (i+1) * qubit_spacing
            ax.plot(0.6, y1, 'ko', markersize=6)
            ax.plot(0.6, y2, 'ko', markersize=6)
            ax.plot([0.6, 0.6], [y1, y2], 'k-', linewidth=1)
            ax.text(0.65, (y1+y2)/2, f'CRY{i}', fontsize=7, ha='left', va='center')
        
        ax.set_xlim(-0.1, 1)
        ax.set_ylim(0.1, 1)
        ax.axis('off')
        ax.set_title('Quantum Circuit Construction', fontsize=16, weight='bold')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Hadamard (H)'),
            mpatches.Patch(color='lightgreen', label='RY Rotation (Noun)'),
            mpatches.Patch(color='lightcoral', label='RY Rotation (Function)'),
            mpatches.Patch(color='lightyellow', label='RY Rotation (Verb)'),
            mpatches.Patch(color='black', label='CNOT/CRY Gates')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.savefig('density_matrix_quantum_circuit_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_statevector_visualization(self):
        """Create statevector amplitude visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Simulate statevector (simplified for visualization)
        n_states = 16  # Show first 16 states
        amplitudes = np.random.random(n_states) + 1j * np.random.random(n_states)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        # Plot 1: Amplitude magnitudes
        states = [f'|{i:04b}>' for i in range(n_states)]
        mag_amplitudes = np.abs(amplitudes)
        
        bars1 = ax1.bar(range(n_states), mag_amplitudes, color='lightblue', alpha=0.7)
        ax1.set_xlabel('Quantum States')
        ax1.set_ylabel('Amplitude Magnitude')
        ax1.set_title('Statevector Amplitude Magnitudes')
        ax1.set_xticks(range(0, n_states, 2))
        ax1.set_xticklabels([states[i] for i in range(0, n_states, 2)], rotation=45)
        
        # Add values on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Phase visualization
        phases = np.angle(amplitudes)
        colors = plt.cm.hsv((phases + np.pi) / (2 * np.pi))
        
        bars2 = ax2.bar(range(n_states), phases, color=colors, alpha=0.7)
        ax2.set_xlabel('Quantum States')
        ax2.set_ylabel('Phase (radians)')
        ax2.set_title('Statevector Phases')
        ax2.set_xticks(range(0, n_states, 2))
        ax2.set_xticklabels([states[i] for i in range(0, n_states, 2)], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add phase values
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('density_matrix_statevector_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_density_matrix_heatmap(self):
        """Create density matrix heatmap visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Create sample density matrix (8x8 for visualization)
        np.random.seed(42)
        real_part = np.random.random((8, 8))
        imag_part = np.random.random((8, 8))
        density_matrix = real_part + 1j * imag_part
        
        # Make it Hermitian and normalized
        density_matrix = (density_matrix + density_matrix.conj().T) / 2
        density_matrix = density_matrix / np.trace(density_matrix)
        
        # Plot 1: Real part
        im1 = ax1.imshow(density_matrix.real, cmap='RdBu_r', aspect='equal')
        ax1.set_title('Density Matrix - Real Part')
        ax1.set_xlabel('Column Index')
        ax1.set_ylabel('Row Index')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Real Value')
        
        # Add matrix values
        for i in range(8):
            for j in range(8):
                text = ax1.text(j, i, f'{density_matrix.real[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # Plot 2: Imaginary part
        im2 = ax2.imshow(density_matrix.imag, cmap='RdBu_r', aspect='equal')
        ax2.set_title('Density Matrix - Imaginary Part')
        ax2.set_xlabel('Column Index')
        ax2.set_ylabel('Row Index')
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Imaginary Value')
        
        # Add matrix values
        for i in range(8):
            for j in range(8):
                text = ax2.text(j, i, f'{density_matrix.imag[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig('density_matrix_heatmap_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_metrics_calculation_diagram(self):
        """Create diagram showing metrics calculation process"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Title
        ax.text(0.5, 0.95, 'Quantum Metrics Calculation Process', 
                fontsize=16, weight='bold', ha='center')
        
        # Step 1: Density Matrix
        ax.text(0.1, 0.85, '1. Density Matrix rho = |psi><psi|', fontsize=12, weight='bold')
        ax.text(0.1, 0.82, '   • Trace: Tr(rho) = 1.0', fontsize=10)
        ax.text(0.1, 0.79, '   • Hermitian: rho = rho†', fontsize=10)
        ax.text(0.1, 0.76, '   • Positive semidefinite: lambda_i >= 0', fontsize=10)
        
        # Step 2: Von Neumann Entropy
        ax.text(0.1, 0.68, '2. Von Neumann Entropy', fontsize=12, weight='bold')
        ax.text(0.1, 0.65, '   S(rho) = -Tr(rho log rho) = -sum_i lambda_i log2(lambda_i)', fontsize=10)
        ax.text(0.1, 0.62, '   Example: S(rho) = 3.169925', fontsize=10)
        
        # Step 3: Frame Competition
        ax.text(0.1, 0.51, '3. Frame Competition', fontsize=12, weight='bold')
        ax.text(0.1, 0.48, '   Competition = min(1.0, S(rho) / log2(d))', fontsize=10)
        ax.text(0.1, 0.45, '   Example: 0.999999', fontsize=10)
        
        # Step 4: Semantic Interference
        ax.text(0.1, 0.34, '4. Semantic Interference', fontsize=12, weight='bold')
        ax.text(0.1, 0.31, '   Interference = min(1.0, Var(phi) / pi^2)', fontsize=10)
        ax.text(0.1, 0.28, '   Example: 0.000000', fontsize=10)
        
        # Step 5: Multiple Reality
        ax.text(0.1, 0.17, '5. Multiple Reality Strength', fontsize=12, weight='bold')
        ax.text(0.1, 0.14, '   Reality = alpha1*S1 + alpha2*C + alpha3*(1-K) + alpha4*D + alpha5*M', fontsize=10)
        ax.text(0.1, 0.11, '   Example: 0.750000', fontsize=10)
        
        # Add arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='blue')
        
        # Arrow from density matrix to entropy
        ax.annotate('', xy=(0.3, 0.7), xytext=(0.3, 0.8),
                   arrowprops=arrow_props)
        
        # Arrow from entropy to competition
        ax.annotate('', xy=(0.3, 0.6), xytext=(0.3, 0.65),
                   arrowprops=arrow_props)
        
        # Arrow from statevector to interference
        ax.annotate('', xy=(0.3, 0.4), xytext=(0.3, 0.45),
                   arrowprops=arrow_props)
        
        # Arrow from all metrics to reality
        ax.annotate('', xy=(0.3, 0.2), xytext=(0.3, 0.25),
                   arrowprops=arrow_props)
        
        # Add boxes around each step
        boxes = [
            (0.05, 0.72, 0.9, 0.12),  # Density matrix
            (0.05, 0.55, 0.9, 0.12),  # Entropy
            (0.05, 0.38, 0.9, 0.12),  # Competition
            (0.05, 0.21, 0.9, 0.12),  # Interference
            (0.05, 0.04, 0.9, 0.12)   # Reality
        ]
        
        for x, y, w, h in boxes:
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', 
                           facecolor='lightblue', alpha=0.1)
            ax.add_patch(rect)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('density_matrix_metrics_calculation_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_complete_workflow_diagram(self):
        """Create complete workflow diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Workflow steps
        steps = [
            ("Chinese Text\n麥當勞性侵案後改革...", 0.1, 0.9, 'lightcoral'),
            ("Text Segmentation\njieba + POS tagging", 0.1, 0.8, 'lightblue'),
            ("Category Mapping\nN, F, V categories", 0.1, 0.7, 'lightgreen'),
            ("Quantum Circuit\n8 qubits, rotations", 0.1, 0.6, 'lightyellow'),
            ("Statevector\n|psi> = sum alpha_i|i>", 0.1, 0.5, 'lightpink'),
            ("Density Matrix\nrho = |psi><psi|", 0.1, 0.4, 'lightcyan'),
            ("Von Neumann Entropy\nS(rho) = -Tr(rho log rho)", 0.1, 0.3, 'lightgray'),
            ("Frame Competition\nmin(1.0, S(rho)/log2(d))", 0.1, 0.2, 'lightsteelblue'),
            ("Semantic Interference\nVar(phi)/pi^2", 0.1, 0.1, 'lightgoldenrodyellow')
        ]
        
        # Draw workflow boxes
        for text, x, y, color in steps:
            rect = Rectangle((x, y-0.04), 0.15, 0.08, 
                           facecolor=color, alpha=0.7, edgecolor='black')
            ax.add_patch(rect)
            ax.text(x+0.075, y, text, fontsize=9, ha='center', va='center', weight='bold')
        
        # Draw arrows between steps
        arrow_props = dict(arrowstyle='->', lw=3, color='darkblue')
        for i in range(len(steps)-1):
            ax.annotate('', xy=(0.175, steps[i+1][2]+0.04), 
                       xytext=(0.175, steps[i][2]-0.04),
                       arrowprops=arrow_props)
        
        # Add mathematical formulas on the right
        formulas = [
            ("Input: Chinese text", 0.4, 0.9),
            ("Words: ['麥當勞', '性侵', '案', ...]", 0.4, 0.8),
            ("POS: ['nt', 'n', 'n', 'f', 'v', ...]", 0.4, 0.7),
            ("Categories: ['N', 'N', 'N', 'F', 'V', ...]", 0.4, 0.6),
            ("Circuit: H, RY, RZ, CNOT, CRY gates", 0.4, 0.5),
            ("|psi> = alpha0|00000000> + alpha1|00000001> + ...", 0.4, 0.4),
            ("rho = |psi><psi| = sum_ij alpha_i*alpha_j* |i><j|", 0.4, 0.3),
            ("S(rho) = -sum_i lambda_i log2(lambda_i)", 0.4, 0.2),
            ("Competition = min(1.0, S(rho)/log2(256))", 0.4, 0.1)
        ]
        
        for text, x, y in formulas:
            ax.text(x, y, text, fontsize=10, ha='left', va='center')
        
        # Add example values
        values = [
            ("Result: 3.169925", 0.7, 0.3),
            ("Result: 0.999999", 0.7, 0.2),
            ("Result: 0.000000", 0.7, 0.1)
        ]
        
        for text, x, y in values:
            ax.text(x, y, text, fontsize=10, ha='left', va='center', 
                   weight='bold', color='red')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Complete Density Matrix Calculation Workflow', 
                    fontsize=16, weight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('density_matrix_complete_workflow_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_all_visualizations(self):
        """Create all density matrix visualizations"""
        print("Creating density matrix visualizations (fixed font version)...")
        
        self.create_text_segmentation_diagram()
        print("✓ Text segmentation diagram created")
        
        self.create_quantum_circuit_diagram()
        print("✓ Quantum circuit diagram created")
        
        self.create_statevector_visualization()
        print("✓ Statevector visualization created")
        
        self.create_density_matrix_heatmap()
        print("✓ Density matrix heatmap created")
        
        self.create_metrics_calculation_diagram()
        print("✓ Metrics calculation diagram created")
        
        self.create_complete_workflow_diagram()
        print("✓ Complete workflow diagram created")
        
        print("\nAll visualizations saved to paper/ directory:")
        print("- density_matrix_text_segmentation_fixed.png")
        print("- density_matrix_quantum_circuit_fixed.png")
        print("- density_matrix_statevector_fixed.png")
        print("- density_matrix_heatmap_fixed.png")
        print("- density_matrix_metrics_calculation_fixed.png")
        print("- density_matrix_complete_workflow_fixed.png")

def main():
    """Main function to create all visualizations"""
    visualizer = DensityMatrixVisualizer()
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()
