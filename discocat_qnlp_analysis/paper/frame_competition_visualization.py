#!/usr/bin/env python3
"""
Frame Competition Visualization
Illustrates the mathematical definition and quantum components of Frame Competition
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

# Set font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

class FrameCompetitionVisualization:
    def __init__(self):
        """Initialize the frame competition visualization"""
        
        # Sample density matrix for illustration
        np.random.seed(42)
        self.n_qubits = 3
        self.dimension = 2**self.n_qubits
        
        # Create a sample statevector
        statevector = np.random.random(self.dimension) + 1j * np.random.random(self.dimension)
        statevector = statevector / np.linalg.norm(statevector)
        
        # Construct density matrix
        self.density_matrix = np.outer(statevector, np.conj(statevector))
        
        # Calculate von Neumann entropy
        eigenvalues = np.linalg.eigvals(self.density_matrix)
        eigenvalues = np.real(eigenvalues)  # Take real part
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero values
        
        self.von_neumann_entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        self.frame_competition = self.von_neumann_entropy / np.log2(self.dimension)
        
        print(f"Sample calculation:")
        print(f"Dimension (N): {self.dimension}")
        print(f"Von Neumann Entropy: {self.von_neumann_entropy:.4f}")
        print(f"Frame Competition: {self.frame_competition:.4f}")
    
    def create_formula_visualization(self):
        """Create visualization of the Frame Competition formula"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        ax.set_title('Frame Competition: Mathematical Definition and Components', 
                    fontsize=18, weight='bold', pad=20)
        
        # Main formula
        formula_text = r'$C = \frac{-Tr(\rho \log_2 \rho)}{\log_2 N}$'
        ax.text(0.5, 0.9, formula_text, fontsize=24, ha='center', va='center', 
                weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # Component explanations
        components = [
            (0.1, 0.75, 'C', 'Frame Competition\n(Normalized von Neumann entropy)', 'lightgreen'),
            (0.1, 0.65, 'ρ', 'Density Matrix\n(All possible meanings/frames)', 'lightcoral'),
            (0.1, 0.55, 'Tr(ρ log₂ ρ)', 'Von Neumann Entropy\n(Quantum information content)', 'lightyellow'),
            (0.1, 0.45, 'log₂ N', 'Normalization Factor\n(Maximum possible entropy)', 'lightpink'),
            (0.1, 0.35, 'N', 'Dimension\n(2^n for n qubits)', 'lightcyan')
        ]
        
        for x, y, symbol, description, color in components:
            # Symbol
            ax.text(x, y, symbol, fontsize=20, weight='bold', ha='center', va='center')
            
            # Description box
            rect = FancyBboxPatch((x+0.05, y-0.05), 0.4, 0.08, 
                                boxstyle="round,pad=0.01", facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Description text
            ax.text(x+0.25, y, description, fontsize=12, ha='left', va='center', weight='bold')
        
        # Density matrix construction
        construction_text = r'$\rho = |\psi\rangle\langle\psi|$'
        ax.text(0.7, 0.75, construction_text, fontsize=18, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightsteelblue', alpha=0.7))
        
        ax.text(0.7, 0.65, 'Density Matrix Construction:', fontsize=14, weight='bold', ha='center')
        ax.text(0.7, 0.6, '• |ψ⟩ is the quantum statevector', fontsize=12, ha='center')
        ax.text(0.7, 0.55, '• ⟨ψ| is the conjugate transpose', fontsize=12, ha='center')
        ax.text(0.7, 0.5, '• ρ encodes all possible meanings', fontsize=12, ha='center')
        
        # Example calculation
        ax.text(0.7, 0.35, 'Example Calculation:', fontsize=14, weight='bold', ha='center')
        ax.text(0.7, 0.3, f'• N = {self.dimension} (3 qubits)', fontsize=12, ha='center')
        ax.text(0.7, 0.25, f'• S(ρ) = {self.von_neumann_entropy:.4f}', fontsize=12, ha='center')
        ax.text(0.7, 0.2, f'• C = {self.frame_competition:.4f}', fontsize=12, ha='center')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('frame_competition_formula.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_density_matrix_visualization(self):
        """Create visualization of density matrix construction"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Statevector
        ax1.set_title('Quantum Statevector |ψ⟩', fontsize=14, weight='bold')
        
        # Create sample statevector
        n_states = 8
        states = [f'|{i:03b}⟩' for i in range(n_states)]
        amplitudes = np.random.random(n_states) + 1j * np.random.random(n_states)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        bars = ax1.bar(range(n_states), np.abs(amplitudes), color='lightblue', alpha=0.7)
        ax1.set_xlabel('Quantum States')
        ax1.set_ylabel('Amplitude Magnitude')
        ax1.set_xticks(range(n_states))
        ax1.set_xticklabels(states, rotation=45)
        
        # Add amplitude values
        for bar, amp in zip(bars, np.abs(amplitudes)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{amp:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Density Matrix Real Part
        ax2.set_title('Density Matrix ρ - Real Part', fontsize=14, weight='bold')
        
        # Create sample density matrix
        density_matrix = np.outer(amplitudes, np.conj(amplitudes))
        
        im = ax2.imshow(density_matrix.real, cmap='RdBu_r', aspect='equal')
        ax2.set_xlabel('Column Index')
        ax2.set_ylabel('Row Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Real Value')
        
        # Add matrix values
        for i in range(n_states):
            for j in range(n_states):
                text = ax2.text(j, i, f'{density_matrix.real[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # Plot 3: Density Matrix Imaginary Part
        ax3.set_title('Density Matrix ρ - Imaginary Part', fontsize=14, weight='bold')
        
        im = ax3.imshow(density_matrix.imag, cmap='RdBu_r', aspect='equal')
        ax3.set_xlabel('Column Index')
        ax3.set_ylabel('Row Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Imaginary Value')
        
        # Add matrix values
        for i in range(n_states):
            for j in range(n_states):
                text = ax3.text(j, i, f'{density_matrix.imag[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # Plot 4: Eigenvalues and Entropy
        ax4.set_title('Eigenvalues and Von Neumann Entropy', fontsize=14, weight='bold')
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(density_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Sort eigenvalues
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Plot eigenvalues
        bars = ax4.bar(range(len(eigenvalues)), eigenvalues, color='lightgreen', alpha=0.7)
        ax4.set_xlabel('Eigenvalue Index')
        ax4.set_ylabel('Eigenvalue Magnitude')
        
        # Add eigenvalue values
        for bar, eigenval in zip(bars, eigenvalues):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{eigenval:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Calculate and display entropy
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        ax4.text(0.5, 0.8, f'Von Neumann Entropy: {entropy:.4f}', 
                transform=ax4.transAxes, fontsize=12, weight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('frame_competition_density_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_quantum_circuit_visualization(self):
        """Create visualization of quantum circuit for frame competition"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        ax.set_title('Quantum Circuit for Frame Competition Analysis', fontsize=16, weight='bold')
        
        # Draw qubit lines
        n_qubits = 3
        for i in range(n_qubits):
            y = 0.8 - i * 0.2
            ax.plot([0, 1], [y, y], 'k-', linewidth=3)
            ax.text(-0.05, y, f'q{i}', fontsize=12, ha='right', va='center', weight='bold')
        
        # Draw Hadamard gates
        for i in range(n_qubits):
            y = 0.8 - i * 0.2
            rect = Rectangle((0.1, y-0.03), 0.06, 0.06, facecolor='lightblue', alpha=0.8)
            ax.add_patch(rect)
            ax.text(0.13, y, 'H', fontsize=10, ha='center', va='center', weight='bold')
        
        # Draw rotation gates
        for i in range(n_qubits):
            y = 0.8 - i * 0.2
            rect = Rectangle((0.2, y-0.03), 0.06, 0.06, facecolor='lightgreen', alpha=0.8)
            ax.add_patch(rect)
            ax.text(0.23, y, 'RY', fontsize=8, ha='center', va='center', weight='bold')
        
        # Draw CNOT gates
        for i in range(n_qubits-1):
            y1 = 0.8 - i * 0.2
            y2 = 0.8 - (i+1) * 0.2
            
            # Control dot
            ax.plot(0.35, y1, 'ko', markersize=8)
            # Target with X
            ax.plot(0.35, y2, 'ko', markersize=8)
            ax.text(0.35, y2-0.02, 'X', fontsize=10, ha='center', va='center', color='white')
            # Connection line
            ax.plot([0.35, 0.35], [y1, y2], 'k-', linewidth=2)
            ax.text(0.4, (y1+y2)/2, 'CNOT', fontsize=8, ha='left', va='center')
        
        # Draw measurement
        for i in range(n_qubits):
            y = 0.8 - i * 0.2
            rect = Rectangle((0.5, y-0.03), 0.06, 0.06, facecolor='lightcoral', alpha=0.8)
            ax.add_patch(rect)
            ax.text(0.53, y, 'M', fontsize=10, ha='center', va='center', weight='bold')
        
        # Add formula annotations
        ax.text(0.7, 0.9, 'Frame Competition Calculation:', fontsize=14, weight='bold')
        ax.text(0.7, 0.8, r'$C = \frac{-Tr(\rho \log_2 \rho)}{\log_2 N}$', 
                fontsize=12, ha='left')
        ax.text(0.7, 0.7, f'• N = {2**n_qubits} (dimension)', fontsize=10, ha='left')
        ax.text(0.7, 0.65, f'• ρ = |ψ⟩⟨ψ| (density matrix)', fontsize=10, ha='left')
        ax.text(0.7, 0.6, f'• Tr(ρ log₂ ρ) = von Neumann entropy', fontsize=10, ha='left')
        ax.text(0.7, 0.55, f'• C = normalized competition measure', fontsize=10, ha='left')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Hadamard (H) - Superposition'),
            mpatches.Patch(color='lightgreen', label='RY Gate - Rotation'),
            mpatches.Patch(color='black', label='CNOT Gate - Entanglement'),
            mpatches.Patch(color='lightcoral', label='Measurement (M)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        ax.set_xlim(-0.1, 1)
        ax.set_ylim(0.2, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('frame_competition_quantum_circuit.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_visualization(self):
        """Create comprehensive frame competition visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        
        ax.set_title('Frame Competition: Complete Quantum NLP Analysis', 
                    fontsize=20, weight='bold', pad=20)
        
        # Main formula at the top
        formula_text = r'$C = \frac{-\text{Tr}(\rho \log_2 \rho)}{\log_2 N}$'
        ax.text(0.5, 0.95, formula_text, fontsize=28, ha='center', va='center', 
                weight='bold', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Step-by-step process
        steps = [
            (0.1, 0.8, '1. Text Input', 'Chinese text segmentation\nand POS tagging', 'lightgreen'),
            (0.1, 0.7, '2. Quantum Circuit', 'Construct quantum circuit\nwith qubits and gates', 'lightblue'),
            (0.1, 0.6, '3. Statevector |ψ⟩', 'Calculate quantum statevector\nfrom circuit execution', 'lightyellow'),
            (0.1, 0.5, '4. Density Matrix ρ', 'Construct ρ = |ψ⟩⟨ψ|\nouter product', 'lightcoral'),
            (0.1, 0.4, '5. Von Neumann Entropy', 'Calculate S(ρ) = -Tr(ρ log₂ ρ)\nquantum information', 'lightpink'),
            (0.1, 0.3, '6. Frame Competition', 'Normalize: C = S(ρ) / log₂ N\ncompetition measure', 'lightcyan')
        ]
        
        for x, y, title, description, color in steps:
            # Step box
            rect = FancyBboxPatch((x, y-0.08), 0.25, 0.1, 
                                boxstyle="round,pad=0.01", facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Step title
            ax.text(x+0.125, y+0.02, title, fontsize=12, weight='bold', ha='center', va='center')
            
            # Step description
            ax.text(x+0.125, y-0.02, description, fontsize=10, ha='center', va='center')
        
        # Mathematical details
        ax.text(0.5, 0.8, 'Mathematical Components:', fontsize=16, weight='bold', ha='center')
        
        components = [
            (0.4, 0.7, 'ρ (Density Matrix)', 'Encodes all possible meanings\nand semantic frames'),
            (0.4, 0.6, '|ψ⟩ (Statevector)', 'Quantum superposition of\nall possible interpretations'),
            (0.4, 0.5, 'Tr(ρ log₂ ρ)', 'Von Neumann entropy\nquantum information content'),
            (0.4, 0.4, 'log₂ N', 'Normalization factor\nmaximum possible entropy'),
            (0.4, 0.3, 'C (Frame Competition)', 'Normalized measure of\nsemantic frame competition')
        ]
        
        for x, y, title, description in components:
            # Component box
            rect = FancyBboxPatch((x, y-0.06), 0.25, 0.08, 
                                boxstyle="round,pad=0.01", facecolor='lightsteelblue', alpha=0.7)
            ax.add_patch(rect)
            
            # Component title
            ax.text(x+0.125, y+0.01, title, fontsize=11, weight='bold', ha='center', va='center')
            
            # Component description
            ax.text(x+0.125, y-0.02, description, fontsize=9, ha='center', va='center')
        
        # Example calculation
        ax.text(0.8, 0.8, 'Example Calculation:', fontsize=16, weight='bold', ha='center')
        ax.text(0.8, 0.75, f'• N = {self.dimension} (3 qubits)', fontsize=12, ha='center')
        ax.text(0.8, 0.7, f'• S(ρ) = {self.von_neumann_entropy:.4f}', fontsize=12, ha='center')
        ax.text(0.8, 0.65, f'• C = {self.frame_competition:.4f}', fontsize=12, ha='center')
        ax.text(0.8, 0.6, f'• Interpretation: {self._interpret_competition()}', fontsize=10, ha='center')
        
        # Arrows showing flow
        arrow_props = dict(arrowstyle='->', lw=3, color='darkblue')
        
        # Vertical flow arrows
        for i in range(len(steps)-1):
            ax.annotate('', xy=(0.225, steps[i+1][1]+0.08), 
                       xytext=(0.225, steps[i][1]-0.08),
                       arrowprops=arrow_props)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('frame_competition_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _interpret_competition(self):
        """Interpret the frame competition value"""
        if self.frame_competition > 0.8:
            return "High frame competition - multiple competing interpretations"
        elif self.frame_competition > 0.5:
            return "Moderate frame competition - some semantic ambiguity"
        else:
            return "Low frame competition - clear, unambiguous meaning"
    
    def create_all_visualizations(self):
        """Create all frame competition visualizations"""
        print("Creating Frame Competition visualizations...")
        
        self.create_formula_visualization()
        print("✓ Formula visualization created")
        
        self.create_density_matrix_visualization()
        print("✓ Density matrix visualization created")
        
        self.create_quantum_circuit_visualization()
        print("✓ Quantum circuit visualization created")
        
        self.create_comprehensive_visualization()
        print("✓ Comprehensive visualization created")
        
        print("\nAll visualizations saved:")
        print("- frame_competition_formula.png")
        print("- frame_competition_density_matrix.png")
        print("- frame_competition_quantum_circuit.png")
        print("- frame_competition_comprehensive.png")

def main():
    """Main function to create frame competition visualizations"""
    visualizer = FrameCompetitionVisualization()
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()
