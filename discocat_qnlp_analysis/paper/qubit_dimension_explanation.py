#!/usr/bin/env python3
"""
Qubit Dimension Explanation
Explains why N = 8 for 3 qubits and how quantum dimensions work
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class QubitDimensionExplanation:
    def __init__(self):
        """Initialize the qubit dimension explanation"""
        self.n_qubits = 3
        self.dimension = 2**self.n_qubits  # 2^3 = 8
        
        print("=" * 60)
        print("QUBIT DIMENSION EXPLANATION")
        print("=" * 60)
        print(f"Number of qubits: {self.n_qubits}")
        print(f"Dimension N = 2^{self.n_qubits} = {self.dimension}")
        print()
    
    def explain_quantum_dimensions(self):
        """Explain how quantum dimensions work"""
        print("QUANTUM DIMENSION CALCULATION:")
        print("-" * 40)
        print("Each qubit can be in 2 states: |0> or |1>")
        print("For n qubits, the total number of possible states is 2^n")
        print()
        
        for n in range(1, 6):
            dimension = 2**n
            print(f"n = {n} qubits → N = 2^{n} = {dimension} possible states")
        
        print()
        print("WHY N = 8 FOR 3 QUBITS:")
        print("-" * 40)
        print("3 qubits can be in 8 different combinations:")
        
        # Show all possible states for 3 qubits
        states = []
        for i in range(2**3):
            binary = format(i, '03b')  # 3-bit binary representation
            ket_state = f"|{binary}>"
            states.append(ket_state)
            print(f"  State {i}: {ket_state}")
        
        print()
        print("QUANTUM SUPERPOSITION:")
        print("-" * 40)
        print("A 3-qubit system can be in a superposition of all 8 states:")
        print("|ψ> = α₀|000> + α₁|001> + α₂|010> + α₃|011> +")
        print("      α₄|100> + α₅|101> + α₆|110> + α₇|111>")
        print()
        print("The density matrix ρ is 8×8 because it represents")
        print("all possible combinations of these 8 states.")
    
    def create_dimension_visualization(self):
        """Create visualization of qubit dimensions"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Single Qubit States
        ax1.set_title('Single Qubit: 2 States', fontsize=14, weight='bold')
        
        states_1 = ['|0>', '|1>']
        amplitudes_1 = [0.7, 0.7]  # Equal superposition
        
        bars = ax1.bar(states_1, amplitudes_1, color=['lightblue', 'lightcoral'], alpha=0.7)
        ax1.set_ylabel('Amplitude')
        ax1.set_xlabel('Quantum States')
        
        # Add amplitude values
        for bar, amp in zip(bars, amplitudes_1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{amp:.1f}', ha='center', va='bottom', fontsize=12, weight='bold')
        
        ax1.text(0.5, 0.8, 'N = 2¹ = 2', transform=ax1.transAxes, 
                fontsize=16, weight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # Plot 2: Two Qubit States
        ax2.set_title('Two Qubits: 4 States', fontsize=14, weight='bold')
        
        states_2 = ['|00>', '|01>', '|10>', '|11>']
        amplitudes_2 = [0.5, 0.5, 0.5, 0.5]  # Equal superposition
        
        bars = ax2.bar(states_2, amplitudes_2, color=['lightgreen', 'lightblue', 'lightcoral', 'lightyellow'], alpha=0.7)
        ax2.set_ylabel('Amplitude')
        ax2.set_xlabel('Quantum States')
        
        # Add amplitude values
        for bar, amp in zip(bars, amplitudes_2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{amp:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        ax2.text(0.5, 0.8, 'N = 2² = 4', transform=ax2.transAxes, 
                fontsize=16, weight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # Plot 3: Three Qubit States
        ax3.set_title('Three Qubits: 8 States', fontsize=14, weight='bold')
        
        states_3 = ['|000>', '|001>', '|010>', '|011>', '|100>', '|101>', '|110>', '|111>']
        amplitudes_3 = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35]  # Equal superposition
        
        bars = ax3.bar(range(len(states_3)), amplitudes_3, 
                      color=['lightgreen', 'lightblue', 'lightcoral', 'lightyellow',
                             'lightpink', 'lightcyan', 'lightgray', 'lightsteelblue'], alpha=0.7)
        ax3.set_ylabel('Amplitude')
        ax3.set_xlabel('Quantum States')
        ax3.set_xticks(range(len(states_3)))
        ax3.set_xticklabels(states_3, rotation=45)
        
        # Add amplitude values
        for bar, amp in zip(bars, amplitudes_3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{amp:.2f}', ha='center', va='bottom', fontsize=8, weight='bold')
        
        ax3.text(0.5, 0.8, 'N = 2³ = 8', transform=ax3.transAxes, 
                fontsize=16, weight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # Plot 4: Dimension Growth
        ax4.set_title('Exponential Growth of Quantum Dimensions', fontsize=14, weight='bold')
        
        n_qubits = np.arange(1, 9)
        dimensions = 2**n_qubits
        
        ax4.plot(n_qubits, dimensions, 'bo-', linewidth=3, markersize=8, label='N = 2^n')
        ax4.set_xlabel('Number of Qubits (n)')
        ax4.set_ylabel('Dimension (N)')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Highlight 3 qubits
        ax4.plot(3, 8, 'ro', markersize=12, label='3 qubits = 8 dimensions')
        ax4.annotate('3 qubits\nN = 8', xy=(3, 8), xytext=(4, 20),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, weight='bold', color='red')
        
        # Add dimension values
        for n, dim in zip(n_qubits, dimensions):
            ax4.text(n, dim*1.2, f'{dim}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        plt.tight_layout()
        plt.savefig('qubit_dimension_explanation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_density_matrix_explanation(self):
        """Create explanation of why density matrix is 8×8"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        ax.set_title('Why Density Matrix is 8×8 for 3 Qubits', fontsize=18, weight='bold', pad=20)
        
        # Explain the concept
        ax.text(0.5, 0.95, 'Density Matrix Dimension = N × N = 8 × 8', 
                fontsize=20, weight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Step-by-step explanation
        steps = [
            (0.1, 0.85, '1. Statevector |ψ>', '8-dimensional vector\n|ψ> = [α₀, α₁, α₂, α₃, α₄, α₅, α₆, α₇]ᵀ', 'lightgreen'),
            (0.1, 0.75, '2. Outer Product', '|ψ><ψ| creates 8×8 matrix\nEach element: ρᵢⱼ = αᵢαⱼ*', 'lightblue'),
            (0.1, 0.65, '3. Density Matrix ρ', '8×8 Hermitian matrix\nRepresents all quantum correlations', 'lightcoral'),
            (0.1, 0.55, '4. Von Neumann Entropy', 'S(ρ) = -Tr(ρ log₂ ρ)\nUses all 8×8 matrix elements', 'lightyellow'),
            (0.1, 0.45, '5. Frame Competition', 'C = S(ρ) / log₂(8)\nNormalized by log₂(8) = 3', 'lightpink')
        ]
        
        for x, y, title, description, color in steps:
            # Step box
            rect = FancyBboxPatch((x, y-0.08), 0.35, 0.1, 
                                boxstyle="round,pad=0.01", facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Step title
            ax.text(x+0.175, y+0.02, title, fontsize=14, weight='bold', ha='center', va='center')
            
            # Step description
            ax.text(x+0.175, y-0.02, description, fontsize=11, ha='center', va='center')
        
        # Mathematical details
        ax.text(0.6, 0.85, 'Mathematical Details:', fontsize=16, weight='bold', ha='center')
        
        details = [
            (0.5, 0.75, 'Statevector Components:', 'α₀|000> + α₁|001> + α₂|010> + α₃|011> +'),
            (0.5, 0.7, '', 'α₄|100> + α₅|101> + α₆|110> + α₇|111>'),
            (0.5, 0.65, 'Density Matrix Elements:', 'ρᵢⱼ = αᵢαⱼ* for i,j = 0,1,2,3,4,5,6,7'),
            (0.5, 0.6, 'Matrix Size:', '8×8 = 64 total elements'),
            (0.5, 0.55, 'Trace Operation:', 'Tr(ρ) = Σᵢ₌₀⁷ ρᵢᵢ = 1 (normalization)'),
            (0.5, 0.5, 'Log₂ Operation:', 'log₂(8) = 3 (normalization factor)')
        ]
        
        for x, y, title, description in details:
            if title:
                ax.text(x, y, title, fontsize=12, weight='bold', ha='left')
            ax.text(x, y-0.03, description, fontsize=11, ha='left', va='top')
        
        # Example calculation
        ax.text(0.6, 0.35, 'Example Calculation:', fontsize=16, weight='bold', ha='center')
        ax.text(0.6, 0.3, f'• 3 qubits → N = 2³ = {self.dimension}', fontsize=12, ha='center')
        ax.text(0.6, 0.25, f'• Density matrix: {self.dimension}×{self.dimension} = {self.dimension**2} elements', fontsize=12, ha='center')
        ax.text(0.6, 0.2, f'• Normalization: log₂({self.dimension}) = {np.log2(self.dimension):.1f}', fontsize=12, ha='center')
        ax.text(0.6, 0.15, f'• Frame Competition: C = S(ρ) / {np.log2(self.dimension):.1f}', fontsize=12, ha='center')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('density_matrix_dimension_explanation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_quantum_circuit_visualization(self):
        """Create visualization showing 3-qubit circuit"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        ax.set_title('3-Qubit Quantum Circuit → 8-Dimensional State Space', fontsize=18, weight='bold')
        
        # Draw 3 qubit lines
        for i in range(3):
            y = 0.8 - i * 0.2
            ax.plot([0, 1], [y, y], 'k-', linewidth=4)
            ax.text(-0.05, y, f'q{i}', fontsize=16, ha='right', va='center', weight='bold')
        
        # Draw Hadamard gates
        for i in range(3):
            y = 0.8 - i * 0.2
            rect = Rectangle((0.1, y-0.03), 0.06, 0.06, facecolor='lightblue', alpha=0.8)
            ax.add_patch(rect)
            ax.text(0.13, y, 'H', fontsize=14, ha='center', va='center', weight='bold')
        
        # Draw CNOT gates
        for i in range(2):
            y1 = 0.8 - i * 0.2
            y2 = 0.8 - (i+1) * 0.2
            
            # Control dot
            ax.plot(0.25, y1, 'ko', markersize=12)
            # Target with X
            ax.plot(0.25, y2, 'ko', markersize=12)
            ax.text(0.25, y2-0.02, 'X', fontsize=14, ha='center', va='center', color='white')
            # Connection line
            ax.plot([0.25, 0.25], [y1, y2], 'k-', linewidth=4)
            ax.text(0.3, (y1+y2)/2, 'CNOT', fontsize=12, ha='left', va='center')
        
        # Draw measurement
        for i in range(3):
            y = 0.8 - i * 0.2
            rect = Rectangle((0.4, y-0.03), 0.06, 0.06, facecolor='lightcoral', alpha=0.8)
            ax.add_patch(rect)
            ax.text(0.43, y, 'M', fontsize=14, ha='center', va='center', weight='bold')
        
        # Add dimension explanation
        ax.text(0.6, 0.9, '3 Qubits → 8 Dimensions', fontsize=18, weight='bold', ha='center')
        ax.text(0.6, 0.8, 'Each qubit: |0> or |1>', fontsize=14, ha='center')
        ax.text(0.6, 0.75, '3 qubits: 2³ = 8 combinations', fontsize=14, ha='center')
        ax.text(0.6, 0.7, 'States: |000>, |001>, |010>, |011>', fontsize=12, ha='center')
        ax.text(0.6, 0.65, '        |100>, |101>, |110>, |111>', fontsize=12, ha='center')
        
        ax.text(0.6, 0.55, 'Density Matrix:', fontsize=16, weight='bold', ha='center')
        ax.text(0.6, 0.5, 'ρ = |ψ><ψ| → 8×8 matrix', fontsize=14, ha='center')
        ax.text(0.6, 0.45, '64 total elements', fontsize=12, ha='center')
        ax.text(0.6, 0.4, 'Represents all quantum correlations', fontsize=12, ha='center')
        
        ax.text(0.6, 0.3, 'Frame Competition:', fontsize=16, weight='bold', ha='center')
        ax.text(0.6, 0.25, 'C = S(ρ) / log₂(8)', fontsize=14, ha='center')
        ax.text(0.6, 0.2, 'C = S(ρ) / 3', fontsize=14, ha='center')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Hadamard (H) - Superposition'),
            mpatches.Patch(color='black', label='CNOT Gate - Entanglement'),
            mpatches.Patch(color='lightcoral', label='Measurement (M)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        ax.set_xlim(-0.1, 1)
        ax.set_ylim(0.2, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('quantum_circuit_dimension.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_explanation(self):
        """Run the complete qubit dimension explanation"""
        self.explain_quantum_dimensions()
        
        self.create_dimension_visualization()
        print("✓ Dimension visualization created")
        
        self.create_density_matrix_explanation()
        print("✓ Density matrix explanation created")
        
        self.create_quantum_circuit_visualization()
        print("✓ Quantum circuit visualization created")
        
        print("\nAll visualizations saved:")
        print("- qubit_dimension_explanation.png")
        print("- density_matrix_dimension_explanation.png")
        print("- quantum_circuit_dimension.png")

def main():
    """Main function to explain qubit dimensions"""
    explanation = QubitDimensionExplanation()
    explanation.run_complete_explanation()

if __name__ == "__main__":
    main()
