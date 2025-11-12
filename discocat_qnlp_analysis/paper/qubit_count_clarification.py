#!/usr/bin/env python3
"""
Qubit Count Clarification
Determines the exact number of qubits needed for the specific text case
"""

import jieba.posseg as pseg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class QubitCountClarification:
    def __init__(self):
        """Initialize the qubit count clarification"""
        
        # The specific text
        self.text = "麥當勞性侵案後改革 董事長發聲承諾改善"
        
        print("=" * 80)
        print("QUBIT COUNT CLARIFICATION")
        print("=" * 80)
        print(f"Text: {self.text}")
        print()
    
    def analyze_qubit_requirements(self):
        """Analyze the exact qubit requirements"""
        print("DETAILED QUBIT REQUIREMENT ANALYSIS:")
        print("-" * 60)
        
        # Segment the text
        words_with_pos = list(pseg.cut(self.text))
        
        print("Token Analysis:")
        print("Token | POS | Category | Qubit | Reason")
        print("-" * 60)
        
        categories = []
        qubits = []
        
        for word, pos in words_with_pos:
            category = self._pos_to_category(pos)
            qubit = self._category_to_qubit(category)
            
            categories.append(category)
            qubits.append(qubit)
            
            reason = self._get_qubit_reason(category, qubit)
            print(f"{word:8} | {pos:4} | {category:8} | Q{qubit:5} | {reason}")
        
        print()
        
        # Analyze qubit usage
        unique_categories = set(categories)
        unique_qubits = set(qubits)
        max_qubit = max(qubits)
        
        print("QUBIT USAGE ANALYSIS:")
        print("-" * 40)
        print(f"Unique categories: {unique_categories}")
        print(f"Unique qubits used: {sorted(unique_qubits)}")
        print(f"Maximum qubit number: {max_qubit}")
        print(f"Total qubits needed: {max_qubit + 1}")
        
        # Count qubit usage
        qubit_counts = {}
        for qubit in qubits:
            qubit_counts[qubit] = qubit_counts.get(qubit, 0) + 1
        
        print(f"\nQubit usage distribution:")
        for qubit in sorted(qubit_counts.keys()):
            count = qubit_counts[qubit]
            category = [k for k, v in self._get_category_map().items() if v == qubit][0]
            print(f"  Q{qubit} ({category}): {count} tokens")
        
        return {
            'unique_categories': unique_categories,
            'unique_qubits': unique_qubits,
            'max_qubit': max_qubit,
            'total_qubits': max_qubit + 1,
            'qubit_counts': qubit_counts
        }
    
    def _pos_to_category(self, pos):
        """Map POS tag to grammatical category"""
        pos_mapping = {
            'n': 'N', 'nr': 'N', 'ns': 'N', 'nt': 'N', 'nz': 'N',
            'v': 'V', 'vd': 'V', 'vn': 'V', 'vg': 'V',
            'a': 'A', 'ad': 'A', 'an': 'A',
            'd': 'D', 'p': 'P', 'r': 'R', 'c': 'C', 'f': 'F',
            'm': 'X', 'q': 'X', 'u': 'X', 'xc': 'X'
        }
        return pos_mapping.get(pos, 'X')
    
    def _category_to_qubit(self, category):
        """Map category to qubit number"""
        category_map = self._get_category_map()
        return category_map.get(category, 8)
    
    def _get_category_map(self):
        """Get the category to qubit mapping"""
        return {
            'N': 0,    # Nouns
            'V': 1,    # Verbs  
            'A': 2,    # Adjectives
            'D': 3,    # Adverbs
            'P': 4,    # Prepositions
            'R': 5,    # Pronouns
            'C': 6,    # Conjunctions
            'F': 7,    # Function words
            'X': 8     # Other/Unknown
        }
    
    def _get_qubit_reason(self, category, qubit):
        """Get reason for qubit assignment"""
        reasons = {
            'N': 'Noun category',
            'V': 'Verb category',
            'A': 'Adjective category',
            'D': 'Adverb category',
            'P': 'Preposition category',
            'R': 'Pronoun category',
            'C': 'Conjunction category',
            'F': 'Function word category',
            'X': 'Other/Unknown category'
        }
        return reasons.get(category, 'Unknown category')
    
    def create_qubit_count_visualization(self, qubit_info):
        """Create visualization of qubit count analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Qubit Usage Distribution
        ax1.set_title('Qubit Usage Distribution', fontsize=14, weight='bold')
        
        qubits = sorted(qubit_info['qubit_counts'].keys())
        counts = [qubit_info['qubit_counts'][q] for q in qubits]
        colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral', 
                 'lightpink', 'lightcyan', 'lightgray', 'lightsteelblue', 'lightgoldenrodyellow']
        
        bars = ax1.bar([f'Q{q}' for q in qubits], counts, 
                      color=[colors[q % len(colors)] for q in qubits], alpha=0.7)
        ax1.set_xlabel('Qubit Number')
        ax1.set_ylabel('Token Count')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=12, weight='bold')
        
        # Plot 2: Category Distribution
        ax2.set_title('Category Distribution', fontsize=14, weight='bold')
        
        category_counts = {}
        for category in qubit_info['unique_categories']:
            category_counts[category] = sum(1 for c in qubit_info['unique_categories'] if c == category)
        
        categories = list(qubit_info['unique_categories'])
        counts = [1] * len(categories)  # Each category appears once
        colors_cat = [colors[i % len(colors)] for i in range(len(categories))]
        
        bars = ax2.bar(categories, counts, color=colors_cat, alpha=0.7)
        ax2.set_xlabel('Grammatical Category')
        ax2.set_ylabel('Count')
        
        # Add category labels
        for bar, category in zip(bars, categories):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    category, ha='center', va='bottom', fontsize=12, weight='bold')
        
        # Plot 3: Qubit Range Analysis
        ax3.set_title('Qubit Range Analysis', fontsize=14, weight='bold')
        
        max_qubit = qubit_info['max_qubit']
        total_qubits = qubit_info['total_qubits']
        
        # Show qubit range
        qubit_range = list(range(total_qubits))
        used_qubits = [1 if q in qubit_info['unique_qubits'] else 0 for q in qubit_range]
        
        bars = ax3.bar([f'Q{q}' for q in qubit_range], used_qubits, 
                      color=['lightgreen' if used else 'lightgray' for used in used_qubits], alpha=0.7)
        ax3.set_xlabel('Qubit Number')
        ax3.set_ylabel('Used (1) / Unused (0)')
        ax3.set_ylim(0, 1.2)
        
        # Add labels
        for bar, used in zip(bars, used_qubits):
            height = bar.get_height()
            if used:
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        'USED', ha='center', va='bottom', fontsize=10, weight='bold')
            else:
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        'UNUSED', ha='center', va='bottom', fontsize=8, color='gray')
        
        # Plot 4: Summary
        ax4.set_title('Qubit Count Summary', fontsize=14, weight='bold')
        
        summary_data = [
            ('Unique Categories', len(qubit_info['unique_categories'])),
            ('Used Qubits', len(qubit_info['unique_qubits'])),
            ('Max Qubit Number', qubit_info['max_qubit']),
            ('Total Qubits Needed', qubit_info['total_qubits'])
        ]
        
        labels, values = zip(*summary_data)
        colors_summary = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral']
        
        bars = ax4.bar(labels, values, color=colors_summary, alpha=0.7)
        ax4.set_ylabel('Count')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value}', ha='center', va='bottom', fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.savefig('qubit_count_clarification.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_quantum_circuit_specification(self, qubit_info):
        """Create quantum circuit specification"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        total_qubits = qubit_info['total_qubits']
        
        ax.set_title(f'Quantum Circuit Specification: {total_qubits} Qubits', fontsize=18, weight='bold')
        
        # Draw qubit lines
        for i in range(total_qubits):
            y = 0.9 - i * 0.1
            used = i in qubit_info['unique_qubits']
            color = 'k' if used else 'gray'
            alpha = 1.0 if used else 0.5
            ax.plot([0, 1], [y, y], color=color, linewidth=3, alpha=alpha)
            ax.text(-0.05, y, f'q{i}', fontsize=14, ha='right', va='center', 
                   weight='bold', color=color, alpha=alpha)
            
            # Add usage indicator
            if used:
                ax.text(-0.15, y, '✓', fontsize=12, ha='right', va='center', color='green', weight='bold')
            else:
                ax.text(-0.15, y, '○', fontsize=12, ha='right', va='center', color='gray')
        
        # Draw gates for used qubits only
        for i in range(total_qubits):
            if i in qubit_info['unique_qubits']:
                y = 0.9 - i * 0.1
                
                # Hadamard gate
                rect = Rectangle((0.1, y-0.02), 0.06, 0.04, facecolor='lightblue', alpha=0.8)
                ax.add_patch(rect)
                ax.text(0.13, y, 'H', fontsize=12, ha='center', va='center', weight='bold')
                
                # Rotation gate
                rect = Rectangle((0.2, y-0.02), 0.06, 0.04, facecolor='lightgreen', alpha=0.8)
                ax.add_patch(rect)
                ax.text(0.23, y, 'RY', fontsize=10, ha='center', va='center', weight='bold')
                
                # Measurement
                rect = Rectangle((0.3, y-0.02), 0.06, 0.04, facecolor='lightcoral', alpha=0.8)
                ax.add_patch(rect)
                ax.text(0.33, y, 'M', fontsize=12, ha='center', va='center', weight='bold')
        
        # Add specifications
        ax.text(0.5, 0.9, 'Circuit Specifications:', fontsize=16, weight='bold')
        ax.text(0.5, 0.85, f'• Total qubits: {total_qubits}', fontsize=14, ha='left')
        ax.text(0.5, 0.8, f'• Used qubits: {len(qubit_info["unique_qubits"])}', fontsize=14, ha='left')
        ax.text(0.5, 0.75, f'• Unused qubits: {total_qubits - len(qubit_info["unique_qubits"])}', fontsize=14, ha='left')
        ax.text(0.5, 0.7, f'• Dimension: 2^{total_qubits} = {2**total_qubits}', fontsize=14, ha='left')
        ax.text(0.5, 0.65, f'• Density matrix: {2**total_qubits}×{2**total_qubits}', fontsize=14, ha='left')
        
        ax.text(0.5, 0.55, 'Frame Competition:', fontsize=16, weight='bold')
        ax.text(0.5, 0.5, f'C = S(ρ) / log₂({2**total_qubits})', fontsize=14, ha='left')
        ax.text(0.5, 0.45, f'C = S(ρ) / {np.log2(2**total_qubits):.1f}', fontsize=14, ha='left')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Hadamard (H)'),
            mpatches.Patch(color='lightgreen', label='RY Gate'),
            mpatches.Patch(color='lightcoral', label='Measurement (M)'),
            mpatches.Patch(color='black', label='Used Qubit'),
            mpatches.Patch(color='gray', label='Unused Qubit')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        ax.set_xlim(-0.2, 1)
        ax.set_ylim(0.1, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('quantum_circuit_specification.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_analysis(self):
        """Run the complete qubit count analysis"""
        # Analyze qubit requirements
        qubit_info = self.analyze_qubit_requirements()
        
        # Create visualizations
        self.create_qubit_count_visualization(qubit_info)
        print("✓ Qubit count visualization created")
        
        self.create_quantum_circuit_specification(qubit_info)
        print("✓ Quantum circuit specification created")
        
        # Final answer
        print(f"\n" + "=" * 80)
        print("FINAL ANSWER:")
        print("=" * 80)
        print(f"Text: {self.text}")
        print(f"Total qubits needed: {qubit_info['total_qubits']}")
        print(f"Used qubits: {sorted(qubit_info['unique_qubits'])}")
        print(f"Unused qubits: {[i for i in range(qubit_info['total_qubits']) if i not in qubit_info['unique_qubits']]}")
        print(f"Circuit dimension: 2^{qubit_info['total_qubits']} = {2**qubit_info['total_qubits']}")
        
        print(f"\nVisualizations saved:")
        print(f"- qubit_count_clarification.png")
        print(f"- quantum_circuit_specification.png")

def main():
    """Main function to run qubit count clarification"""
    clarification = QubitCountClarification()
    clarification.run_complete_analysis()

if __name__ == "__main__":
    main()
