#!/usr/bin/env python3
"""
Qubit Word Count Representation
Shows how different word counts per qubit are encoded in quantum circuits
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class QubitWordCountRepresentation:
    def __init__(self):
        """Initialize the qubit word count representation"""
        
        # Real data from the AI description example
        self.qubit_word_counts = {
            0: 12,  # Nouns (N)
            1: 17,  # Verbs (V) 
            2: 3,   # Adjectives (A)
            3: 2,   # Adverbs (D)
            4: 1,   # Prepositions (P)
            5: 1,   # Pronouns (R)
            6: 1,   # Conjunctions (C)
            7: 23   # Other/Unknown (X)
        }
        
        # Base quantum parameters for each category
        self.base_parameters = {
            0: {'rotation': np.pi/4, 'phase': 0, 'name': 'Nouns'},
            1: {'rotation': np.pi/3, 'phase': np.pi/2, 'name': 'Verbs'},
            2: {'rotation': np.pi/6, 'phase': np.pi/4, 'name': 'Adjectives'},
            3: {'rotation': np.pi/5, 'phase': np.pi/3, 'name': 'Adverbs'},
            4: {'rotation': np.pi/8, 'phase': np.pi/6, 'name': 'Prepositions'},
            5: {'rotation': np.pi/7, 'phase': np.pi/8, 'name': 'Pronouns'},
            6: {'rotation': np.pi/2, 'phase': np.pi, 'name': 'Conjunctions'},
            7: {'rotation': np.pi/12, 'phase': 0, 'name': 'Other'}
        }
    
    def calculate_word_count_parameters(self):
        """Calculate how word counts affect quantum parameters"""
        print("=" * 80)
        print("QUBIT WORD COUNT REPRESENTATION ANALYSIS")
        print("=" * 80)
        
        print("Qubit | Category | Word Count | Base Rotation | Word Weight | Final Rotation | Final Phase")
        print("-" * 80)
        
        word_count_parameters = {}
        
        for qubit, word_count in self.qubit_word_counts.items():
            base_params = self.base_parameters[qubit]
            
            # Method 1: Word count as amplitude scaling
            word_weight = min(1.0, word_count / 20.0)  # Normalize to max 20 words
            
            # Method 2: Word count affects rotation angle
            rotation_scale = 1.0 + (word_count * 0.05)  # 5% increase per word
            final_rotation = base_params['rotation'] * rotation_scale
            
            # Method 3: Word count affects phase
            phase_scale = 1.0 + (word_count * 0.02)  # 2% increase per word
            final_phase = base_params['phase'] * phase_scale
            
            word_count_parameters[qubit] = {
                'word_count': word_count,
                'word_weight': word_weight,
                'base_rotation': base_params['rotation'],
                'final_rotation': final_rotation,
                'base_phase': base_params['phase'],
                'final_phase': final_phase,
                'rotation_scale': rotation_scale,
                'phase_scale': phase_scale
            }
            
            print(f"Q{qubit:2} | {base_params['name']:8} | {word_count:10} | {base_params['rotation']:12.4f} | {word_weight:10.4f} | {final_rotation:13.4f} | {final_phase:10.4f}")
        
        return word_count_parameters
    
    def create_word_count_visualization(self, word_count_parameters):
        """Create visualization of word count representation"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Word Count Distribution
        ax1.set_title('Word Count per Qubit', fontsize=14, weight='bold')
        
        qubits = list(self.qubit_word_counts.keys())
        counts = list(self.qubit_word_counts.values())
        colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral', 
                 'lightpink', 'lightcyan', 'lightgray', 'lightsteelblue']
        
        bars = ax1.bar([f'Q{qubit}' for qubit in qubits], counts, color=colors, alpha=0.7)
        ax1.set_xlabel('Qubit Number')
        ax1.set_ylabel('Word Count')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # Plot 2: Word Weight vs Word Count
        ax2.set_title('Word Weight Scaling', fontsize=14, weight='bold')
        
        word_counts = [params['word_count'] for params in word_count_parameters.values()]
        word_weights = [params['word_weight'] for params in word_count_parameters.values()]
        
        ax2.scatter(word_counts, word_weights, s=100, c=colors, alpha=0.7)
        ax2.set_xlabel('Word Count')
        ax2.set_ylabel('Word Weight (Normalized)')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(word_counts, word_weights, 1)
        p = np.poly1d(z)
        ax2.plot(word_counts, p(word_counts), "r--", alpha=0.8, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        ax2.legend()
        
        # Add labels for each point
        for i, (count, weight) in enumerate(zip(word_counts, word_weights)):
            ax2.annotate(f'Q{i}', (count, weight), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 3: Rotation Scaling
        ax3.set_title('Rotation Angle Scaling by Word Count', fontsize=14, weight='bold')
        
        base_rotations = [params['base_rotation'] for params in word_count_parameters.values()]
        final_rotations = [params['final_rotation'] for params in word_count_parameters.values()]
        
        x = np.arange(len(qubits))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, base_rotations, width, label='Base Rotation', color=colors, alpha=0.7)
        bars2 = ax3.bar(x + width/2, final_rotations, width, label='Final Rotation', color=colors, alpha=0.5)
        
        ax3.set_xlabel('Qubit Number')
        ax3.set_ylabel('Rotation Angle (radians)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Q{qubit}' for qubit in qubits])
        ax3.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Quantum State Amplitude Distribution
        ax4.set_title('Quantum State Amplitude Distribution', fontsize=14, weight='bold')
        
        # Simulate quantum state amplitudes based on word counts
        n_qubits = len(qubits)
        total_words = sum(self.qubit_word_counts.values())
        
        # Amplitude proportional to word count
        amplitudes = []
        for qubit in qubits:
            word_count = self.qubit_word_counts[qubit]
            # Amplitude is proportional to word count
            amplitude = np.sqrt(word_count / total_words)
            amplitudes.append(amplitude)
        
        # Normalize amplitudes
        amplitudes = np.array(amplitudes)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        bars = ax4.bar([f'Q{qubit}' for qubit in qubits], amplitudes, color=colors, alpha=0.7)
        ax4.set_xlabel('Qubit Number')
        ax4.set_ylabel('Amplitude Magnitude')
        
        # Add amplitude values
        for bar, amp in zip(bars, amplitudes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{amp:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('qubit_word_count_representation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_quantum_circuit_representation(self, word_count_parameters):
        """Create quantum circuit representation showing word count effects"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        ax.set_title('Quantum Circuit with Word Count Representation', fontsize=16, weight='bold')
        
        # Draw qubit lines
        qubits = list(self.qubit_word_counts.keys())
        for i, qubit in enumerate(qubits):
            y = 0.9 - i * 0.1
            ax.plot([0, 1], [y, y], 'k-', linewidth=3)
            ax.text(-0.05, y, f'Q{qubit}', fontsize=12, ha='right', va='center', weight='bold')
            
            # Add word count label
            word_count = self.qubit_word_counts[qubit]
            ax.text(-0.15, y, f'({word_count} words)', fontsize=8, ha='right', va='center', color='gray')
        
        # Draw Hadamard gates (initialization)
        for i, qubit in enumerate(qubits):
            y = 0.9 - i * 0.1
            rect = Rectangle((0.1, y-0.02), 0.08, 0.04, facecolor='lightblue', alpha=0.8)
            ax.add_patch(rect)
            ax.text(0.14, y, 'H', fontsize=10, ha='center', va='center', weight='bold')
        
        # Draw rotation gates with word count scaling
        for i, qubit in enumerate(qubits):
            y = 0.9 - i * 0.1
            params = word_count_parameters[qubit]
            
            # Gate size proportional to word count
            gate_width = 0.08 + (params['word_count'] * 0.01)
            gate_height = 0.04 + (params['word_count'] * 0.005)
            
            rect = Rectangle((0.25, y-gate_height/2), gate_width, gate_height, 
                           facecolor='lightgreen', alpha=0.8)
            ax.add_patch(rect)
            
            # Label with rotation angle
            ax.text(0.25 + gate_width/2, y, f'RY({params["final_rotation"]:.3f})', 
                   fontsize=8, ha='center', va='center', weight='bold')
        
        # Draw phase gates
        for i, qubit in enumerate(qubits):
            y = 0.9 - i * 0.1
            params = word_count_parameters[qubit]
            
            # Gate size proportional to word count
            gate_width = 0.08 + (params['word_count'] * 0.01)
            gate_height = 0.04 + (params['word_count'] * 0.005)
            
            rect = Rectangle((0.4, y-gate_height/2), gate_width, gate_height, 
                           facecolor='lightcoral', alpha=0.8)
            ax.add_patch(rect)
            
            # Label with phase angle
            ax.text(0.4 + gate_width/2, y, f'RZ({params["final_phase"]:.3f})', 
                   fontsize=8, ha='center', va='center', weight='bold')
        
        # Draw entanglement gates (CNOT) based on word count relationships
        for i in range(len(qubits)-1):
            y1 = 0.9 - i * 0.1
            y2 = 0.9 - (i+1) * 0.1
            
            # Entanglement strength based on word count similarity
            count1 = self.qubit_word_counts[qubits[i]]
            count2 = self.qubit_word_counts[qubits[i+1]]
            similarity = 1.0 - abs(count1 - count2) / max(count1, count2)
            
            if similarity > 0.3:  # Only draw if similar word counts
                ax.plot(0.6, y1, 'ko', markersize=8)
                ax.plot(0.6, y2, 'ko', markersize=8)
                ax.plot([0.6, 0.6], [y1, y2], 'k-', linewidth=2)
                ax.text(0.65, (y1+y2)/2, f'CNOT\n(sim={similarity:.2f})', 
                       fontsize=8, ha='left', va='center')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Hadamard (H) - Initialization'),
            mpatches.Patch(color='lightgreen', label='RY Gate - Rotation (scaled by word count)'),
            mpatches.Patch(color='lightcoral', label='RZ Gate - Phase (scaled by word count)'),
            mpatches.Patch(color='black', label='CNOT Gate - Entanglement (based on word count similarity)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        ax.set_xlim(-0.2, 1)
        ax.set_ylim(0.1, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('qubit_word_count_quantum_circuit.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def explain_word_count_encoding(self, word_count_parameters):
        """Explain how word counts are encoded in quantum circuits"""
        print("\n" + "=" * 80)
        print("HOW WORD COUNTS ARE REPRESENTED IN QUANTUM CIRCUITS")
        print("=" * 80)
        
        print("\n1. AMPLITUDE ENCODING:")
        print("   - Each qubit's amplitude is proportional to its word count")
        print("   - Higher word count = higher amplitude = stronger quantum state")
        print("   - Formula: amplitude = sqrt(word_count / total_words)")
        
        total_words = sum(self.qubit_word_counts.values())
        print(f"   - Total words: {total_words}")
        
        for qubit, params in word_count_parameters.items():
            amplitude = np.sqrt(params['word_count'] / total_words)
            print(f"   - Q{qubit}: {params['word_count']} words → amplitude = {amplitude:.4f}")
        
        print("\n2. ROTATION ANGLE SCALING:")
        print("   - Base rotation angle is scaled by word count")
        print("   - Formula: final_rotation = base_rotation * (1 + word_count * 0.05)")
        print("   - More words = larger rotation = more quantum state change")
        
        for qubit, params in word_count_parameters.items():
            print(f"   - Q{qubit}: {params['word_count']} words → rotation scale = {params['rotation_scale']:.3f}")
        
        print("\n3. PHASE SCALING:")
        print("   - Base phase is scaled by word count")
        print("   - Formula: final_phase = base_phase * (1 + word_count * 0.02)")
        print("   - More words = larger phase = more quantum interference")
        
        for qubit, params in word_count_parameters.items():
            print(f"   - Q{qubit}: {params['word_count']} words → phase scale = {params['phase_scale']:.3f}")
        
        print("\n4. ENTANGLEMENT BASED ON WORD COUNT SIMILARITY:")
        print("   - Qubits with similar word counts are more likely to be entangled")
        print("   - Formula: similarity = 1 - |count1 - count2| / max(count1, count2)")
        
        for i in range(len(self.qubit_word_counts)-1):
            qubit1 = list(self.qubit_word_counts.keys())[i]
            qubit2 = list(self.qubit_word_counts.keys())[i+1]
            count1 = self.qubit_word_counts[qubit1]
            count2 = self.qubit_word_counts[qubit2]
            similarity = 1.0 - abs(count1 - count2) / max(count1, count2)
            print(f"   - Q{qubit1} ({count1} words) ↔ Q{qubit2} ({count2} words): similarity = {similarity:.3f}")
        
        print("\n5. QUANTUM STATE REPRESENTATION:")
        print("   - The final quantum state |ψ⟩ is a superposition of all qubit states")
        print("   - Each qubit's contribution is weighted by its word count")
        print("   - Formula: |ψ⟩ = Σᵢ αᵢ|i⟩ where αᵢ ∝ sqrt(word_count_i)")
        
        # Calculate final quantum state
        amplitudes = []
        for qubit in self.qubit_word_counts.keys():
            word_count = self.qubit_word_counts[qubit]
            amplitude = np.sqrt(word_count / total_words)
            amplitudes.append(amplitude)
        
        amplitudes = np.array(amplitudes)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        print("   - Final normalized amplitudes:")
        for i, (qubit, amp) in enumerate(zip(self.qubit_word_counts.keys(), amplitudes)):
            print(f"     α_{qubit} = {amp:.4f} (Q{qubit} with {self.qubit_word_counts[qubit]} words)")
    
    def run_complete_analysis(self):
        """Run the complete word count representation analysis"""
        # Calculate parameters
        word_count_parameters = self.calculate_word_count_parameters()
        
        # Create visualizations
        self.create_word_count_visualization(word_count_parameters)
        self.create_quantum_circuit_representation(word_count_parameters)
        
        # Explain encoding
        self.explain_word_count_encoding(word_count_parameters)
        
        print(f"\nVisualizations saved:")
        print(f"- qubit_word_count_representation.png")
        print(f"- qubit_word_count_quantum_circuit.png")

def main():
    """Main function to run the word count representation analysis"""
    analysis = QubitWordCountRepresentation()
    analysis.run_complete_analysis()

if __name__ == "__main__":
    main()
