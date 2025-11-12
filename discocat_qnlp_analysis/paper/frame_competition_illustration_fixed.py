#!/usr/bin/env python3
"""
Frame Competition Illustration - Fixed Font Version
Creates comprehensive visualization without font rendering issues
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

class FrameCompetitionIllustrationFixed:
    def __init__(self):
        """Initialize the frame competition illustration"""
        
        # Sample data from technical methodology document
        self.sample_text = "麥當勞性侵案後改革 董事長發聲承諾改善"
        self.von_neumann_entropy = 1.7782
        self.frame_competition = 0.8891
        self.semantic_interference = 0.0121
        self.multiple_reality_strength = 1.7056
        
        print("=" * 80)
        print("FRAME COMPETITION ILLUSTRATION - FIXED VERSION")
        print("=" * 80)
        print(f"Sample Text: {self.sample_text}")
        print(f"Von Neumann Entropy: {self.von_neumann_entropy}")
        print(f"Frame Competition: {self.frame_competition}")
        print()
    
    def create_main_illustration(self):
        """Create the main frame competition illustration"""
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        
        ax.set_title('Frame Competition: Normalized von Neumann Entropy of Density Matrix', 
                    fontsize=20, weight='bold', pad=20)
        
        # Main formula (using simple text to avoid font issues)
        formula_text = 'Frame Competition = min(1.0, S(rho) × 0.5)'
        ax.text(0.5, 0.95, formula_text, fontsize=24, ha='center', va='center', 
                weight='bold', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Mathematical components
        components = [
            (0.1, 0.85, 'S(rho)', 'Von Neumann Entropy\nS(rho) = -Tr(rho log rho)', 'lightgreen'),
            (0.1, 0.75, 'rho', 'Density Matrix\nrho = |psi><psi|', 'lightcoral'),
            (0.1, 0.65, 'Tr(rho log rho)', 'Trace Operation\nQuantum information content', 'lightyellow'),
            (0.1, 0.55, '0.5', 'Normalization Factor\nScaling coefficient', 'lightpink'),
            (0.1, 0.45, 'min(1.0, ...)', 'Capping Function\nMaximum value constraint', 'lightcyan')
        ]
        
        for x, y, symbol, description, color in components:
            # Symbol
            ax.text(x, y, symbol, fontsize=18, weight='bold', ha='center', va='center')
            
            # Description box
            rect = FancyBboxPatch((x+0.05, y-0.06), 0.4, 0.1, 
                                boxstyle="round,pad=0.01", facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Description text
            ax.text(x+0.25, y, description, fontsize=12, ha='left', va='center', weight='bold')
        
        # Sample calculation
        ax.text(0.7, 0.85, 'Sample Calculation:', fontsize=16, weight='bold', ha='center')
        ax.text(0.7, 0.8, f'Text: "{self.sample_text}"', fontsize=12, ha='center')
        ax.text(0.7, 0.75, f'S(rho) = {self.von_neumann_entropy:.4f}', fontsize=12, ha='center')
        ax.text(0.7, 0.7, f'Frame Competition = min(1.0, {self.von_neumann_entropy:.4f} × 0.5)', fontsize=12, ha='center')
        ax.text(0.7, 0.65, f'Frame Competition = min(1.0, {self.von_neumann_entropy * 0.5:.4f})', fontsize=12, ha='center')
        ax.text(0.7, 0.6, f'Frame Competition = {self.frame_competition:.4f}', fontsize=12, ha='center', 
                weight='bold', color='red')
        
        # Interpretation
        ax.text(0.7, 0.5, 'Interpretation:', fontsize=16, weight='bold', ha='center')
        ax.text(0.7, 0.45, f'• High frame competition ({self.frame_competition:.4f})', fontsize=12, ha='center')
        ax.text(0.7, 0.4, '• Multiple competing semantic frames', fontsize=12, ha='center')
        ax.text(0.7, 0.35, '• High quantum superposition', fontsize=12, ha='center')
        ax.text(0.7, 0.3, '• Strong semantic ambiguity', fontsize=12, ha='center')
        
        # Density matrix visualization
        ax.text(0.1, 0.35, 'Density Matrix Construction:', fontsize=16, weight='bold', ha='center')
        ax.text(0.1, 0.3, 'rho = |psi><psi|', fontsize=18, ha='center', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightsteelblue', alpha=0.7))
        ax.text(0.1, 0.25, '• |psi>: Quantum statevector', fontsize=12, ha='center')
        ax.text(0.1, 0.2, '• <psi|: Conjugate transpose', fontsize=12, ha='center')
        ax.text(0.1, 0.15, '• rho: All possible meanings/frames', fontsize=12, ha='center')
        
        # Arrows showing flow
        arrow_props = dict(arrowstyle='->', lw=3, color='darkblue')
        
        # Arrow from density matrix to entropy
        ax.annotate('', xy=(0.3, 0.7), xytext=(0.3, 0.8),
                   arrowprops=arrow_props)
        
        # Arrow from entropy to competition
        ax.annotate('', xy=(0.3, 0.6), xytext=(0.3, 0.65),
                   arrow_props=arrow_props)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('frame_competition_main_illustration_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_density_matrix_visualization(self):
        """Create density matrix visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Create sample density matrix
        np.random.seed(42)
        n_qubits = 3
        dimension = 2**n_qubits
        
        # Create statevector
        statevector = np.random.random(dimension) + 1j * np.random.random(dimension)
        statevector = statevector / np.linalg.norm(statevector)
        
        # Construct density matrix
        density_matrix = np.outer(statevector, np.conj(statevector))
        
        # Plot 1: Statevector
        ax1.set_title('Quantum Statevector |psi>', fontsize=14, weight='bold')
        
        states = [f'|{i:03b}>' for i in range(dimension)]
        amplitudes = np.abs(statevector)
        
        bars = ax1.bar(range(dimension), amplitudes, color='lightblue', alpha=0.7)
        ax1.set_xlabel('Quantum States')
        ax1.set_ylabel('Amplitude Magnitude')
        ax1.set_xticks(range(dimension))
        ax1.set_xticklabels(states, rotation=45)
        
        # Add amplitude values
        for bar, amp in zip(bars, amplitudes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{amp:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Density Matrix Real Part
        ax2.set_title('Density Matrix rho - Real Part', fontsize=14, weight='bold')
        
        im = ax2.imshow(density_matrix.real, cmap='RdBu_r', aspect='equal')
        ax2.set_xlabel('Column Index')
        ax2.set_ylabel('Row Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Real Value')
        
        # Add matrix values
        for i in range(dimension):
            for j in range(dimension):
                text = ax2.text(j, i, f'{density_matrix.real[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # Plot 3: Density Matrix Imaginary Part
        ax3.set_title('Density Matrix rho - Imaginary Part', fontsize=14, weight='bold')
        
        im = ax3.imshow(density_matrix.imag, cmap='RdBu_r', aspect='equal')
        ax3.set_xlabel('Column Index')
        ax3.set_ylabel('Row Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Imaginary Value')
        
        # Add matrix values
        for i in range(dimension):
            for j in range(dimension):
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
                    f'{eigenval:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Calculate and display entropy
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        ax4.text(0.5, 0.8, f'Von Neumann Entropy: {entropy:.4f}', 
                transform=ax4.transAxes, fontsize=12, weight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # Add frame competition calculation
        frame_comp = min(1.0, entropy * 0.5)
        ax4.text(0.5, 0.7, f'Frame Competition: {frame_comp:.4f}', 
                transform=ax4.transAxes, fontsize=12, weight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('frame_competition_density_matrix_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparison_visualization(self):
        """Create comparison visualization with AI vs CNA data"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        ax.set_title('Frame Competition: AI Generated vs CNA News Comparison', 
                    fontsize=18, weight='bold', pad=20)
        
        # Data from technical methodology document
        datasets = [
            ('AI Generated\n新聞標題', 1.0000, 3.9966),
            ('AI Generated\n影片對話', 1.0000, 4.0000),
            ('AI Generated\n影片描述', 1.0000, 3.9966),
            ('CNA News\n新聞標題', 0.9985, 3.4378),
            ('CNA News\n新聞內容', 0.9173, 7.3508)
        ]
        
        # Extract data
        labels = [item[0] for item in datasets]
        frame_competition = [item[1] for item in datasets]
        von_neumann_entropy = [item[2] for item in datasets]
        
        # Create subplot
        x = np.arange(len(labels))
        width = 0.35
        
        # Plot Frame Competition
        bars1 = ax.bar(x - width/2, frame_competition, width, label='Frame Competition', 
                      color='lightblue', alpha=0.7)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Frame Competition Value')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels for Frame Competition
        for bar, value in zip(bars1, frame_competition):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # Create second y-axis for Von Neumann Entropy
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, von_neumann_entropy, width, label='Von Neumann Entropy', 
                       color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Von Neumann Entropy')
        
        # Add value labels for Von Neumann Entropy
        for bar, value in zip(bars2, von_neumann_entropy):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Add interpretation
        ax.text(0.5, 0.95, 'Key Insights:', transform=ax.transAxes, fontsize=14, weight='bold', ha='center')
        ax.text(0.5, 0.9, '• AI Generated: Perfect frame competition (1.0000)', transform=ax.transAxes, fontsize=12, ha='center')
        ax.text(0.5, 0.85, '• CNA News: High but not perfect competition', transform=ax.transAxes, fontsize=12, ha='center')
        ax.text(0.5, 0.8, '• AI maintains complete frame equality', transform=ax.transAxes, fontsize=12, ha='center')
        ax.text(0.5, 0.75, '• CNA shows slight frame dominance patterns', transform=ax.transAxes, fontsize=12, ha='center')
        
        plt.tight_layout()
        plt.savefig('frame_competition_comparison_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_process_flow(self):
        """Create process flow visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))
        
        ax.set_title('Frame Competition Calculation Process Flow', fontsize=18, weight='bold', pad=20)
        
        # Process steps
        steps = [
            (0.1, 0.9, '1. Text Input', 'Chinese text segmentation\nand POS tagging', 'lightgreen'),
            (0.1, 0.8, '2. Category Mapping', 'Map POS tags to\nDisCoCat categories', 'lightblue'),
            (0.1, 0.7, '3. Quantum Circuit', 'Construct quantum circuit\nwith qubits and gates', 'lightyellow'),
            (0.1, 0.6, '4. Statevector |psi>', 'Calculate quantum statevector\nfrom circuit execution', 'lightcoral'),
            (0.1, 0.5, '5. Density Matrix rho', 'Construct rho = |psi><psi|\nouter product', 'lightpink'),
            (0.1, 0.4, '6. Von Neumann Entropy', 'Calculate S(rho) = -Tr(rho log rho)\nquantum information', 'lightcyan'),
            (0.1, 0.3, '7. Frame Competition', 'Normalize: min(1.0, S(rho) × 0.5)\ncompetition measure', 'lightsteelblue')
        ]
        
        for x, y, title, description, color in steps:
            # Step box
            rect = FancyBboxPatch((x, y-0.08), 0.25, 0.1, 
                                boxstyle="round,pad=0.01", facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Step title
            ax.text(x+0.125, y+0.02, title, fontsize=14, weight='bold', ha='center', va='center')
            
            # Step description
            ax.text(x+0.125, y-0.02, description, fontsize=11, ha='center', va='center')
        
        # Mathematical details
        ax.text(0.5, 0.9, 'Mathematical Components:', fontsize=16, weight='bold', ha='center')
        
        components = [
            (0.4, 0.8, 'S(rho) = -Tr(rho log rho)', 'Von Neumann entropy\nquantum information content'),
            (0.4, 0.7, 'rho = |psi><psi|', 'Density matrix\nall possible meanings'),
            (0.4, 0.6, 'Tr(rho log rho)', 'Trace operation\nquantum correlations'),
            (0.4, 0.5, '0.5', 'Normalization factor\nscaling coefficient'),
            (0.4, 0.4, 'min(1.0, ...)', 'Capping function\nmaximum constraint')
        ]
        
        for x, y, title, description in components:
            # Component box
            rect = FancyBboxPatch((x, y-0.06), 0.25, 0.08, 
                                boxstyle="round,pad=0.01", facecolor='lightsteelblue', alpha=0.7)
            ax.add_patch(rect)
            
            # Component title
            ax.text(x+0.125, y+0.01, title, fontsize=12, weight='bold', ha='center', va='center')
            
            # Component description
            ax.text(x+0.125, y-0.02, description, fontsize=10, ha='center', va='center')
        
        # Example calculation
        ax.text(0.8, 0.9, 'Example Calculation:', fontsize=16, weight='bold', ha='center')
        ax.text(0.8, 0.85, f'Text: "{self.sample_text}"', fontsize=12, ha='center')
        ax.text(0.8, 0.8, f'S(rho) = {self.von_neumann_entropy:.4f}', fontsize=12, ha='center')
        ax.text(0.8, 0.75, f'Frame Competition = min(1.0, {self.von_neumann_entropy:.4f} × 0.5)', fontsize=12, ha='center')
        ax.text(0.8, 0.7, f'Frame Competition = {self.frame_competition:.4f}', fontsize=12, ha='center', 
                weight='bold', color='red')
        
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
        plt.savefig('frame_competition_process_flow_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_all_visualizations(self):
        """Create all frame competition visualizations"""
        print("Creating Frame Competition visualizations (fixed version)...")
        
        self.create_main_illustration()
        print("✓ Main illustration created")
        
        self.create_density_matrix_visualization()
        print("✓ Density matrix visualization created")
        
        self.create_comparison_visualization()
        print("✓ Comparison visualization created")
        
        self.create_process_flow()
        print("✓ Process flow visualization created")
        
        print("\nAll visualizations saved:")
        print("- frame_competition_main_illustration_fixed.png")
        print("- frame_competition_density_matrix_fixed.png")
        print("- frame_competition_comparison_fixed.png")
        print("- frame_competition_process_flow_fixed.png")

def main():
    """Main function to create frame competition illustrations"""
    illustration = FrameCompetitionIllustrationFixed()
    illustration.create_all_visualizations()

if __name__ == "__main__":
    main()
