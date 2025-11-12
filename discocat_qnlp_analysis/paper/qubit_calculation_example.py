#!/usr/bin/env python3
"""
Qubit Calculation for Specific Text Example
Analyzes "麥當勞性侵案後改革 董事長發聲承諾改善" to determine qubit requirements
"""

import jieba.posseg as pseg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class QubitCalculationExample:
    def __init__(self):
        """Initialize the qubit calculation example"""
        
        # The specific text example
        self.text = "麥當勞性侵案後改革 董事長發聲承諾改善"
        
        print("=" * 80)
        print("QUBIT CALCULATION FOR SPECIFIC TEXT EXAMPLE")
        print("=" * 80)
        print(f"Input Text: {self.text}")
        print()
    
    def analyze_text_structure(self):
        """Analyze the text structure and POS categories"""
        print("TEXT STRUCTURE ANALYSIS:")
        print("-" * 50)
        
        # Segment the text
        words_with_pos = list(pseg.cut(self.text))
        
        print("Token Analysis:")
        print("Token | POS Tag | Category | Qubit")
        print("-" * 50)
        
        tokens = []
        categories = []
        qubit_assignments = []
        
        for i, (word, pos) in enumerate(words_with_pos):
            # Map POS to category
            category = self._pos_to_category(pos)
            qubit = self._category_to_qubit(category)
            
            tokens.append(word)
            categories.append(category)
            qubit_assignments.append(qubit)
            
            print(f"{word:8} | {pos:6} | {category:8} | {qubit}")
        
        print()
        print(f"Total tokens: {len(tokens)}")
        print(f"Unique categories: {len(set(categories))}")
        print(f"Category sequence: {' → '.join(categories)}")
        
        return tokens, categories, qubit_assignments
    
    def _pos_to_category(self, pos):
        """Map POS tag to grammatical category"""
        pos_mapping = {
            # Nouns
            'n': 'N', 'nr': 'N', 'ns': 'N', 'nt': 'N', 'nz': 'N',
            # Verbs  
            'v': 'V', 'vd': 'V', 'vn': 'V', 'vg': 'V',
            # Adjectives
            'a': 'A', 'ad': 'A', 'an': 'A',
            # Adverbs
            'd': 'D',
            # Prepositions
            'p': 'P',
            # Pronouns
            'r': 'R',
            # Conjunctions
            'c': 'C', 'cc': 'C',
            # Function words
            'f': 'F',
            # Others
            'm': 'X', 'q': 'X', 'u': 'X', 'xc': 'X'
        }
        return pos_mapping.get(pos, 'X')
    
    def _category_to_qubit(self, category):
        """Map category to qubit number"""
        category_qubit_map = {
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
        return category_qubit_map.get(category, 8)
    
    def calculate_qubit_requirements(self, categories, qubit_assignments):
        """Calculate the number of qubits needed"""
        print("\nQUBIT REQUIREMENT CALCULATION:")
        print("-" * 50)
        
        # Method 1: Based on unique categories
        unique_categories = set(categories)
        n_unique_categories = len(unique_categories)
        
        print(f"Method 1 - Unique Categories:")
        print(f"  Unique categories: {unique_categories}")
        print(f"  Number of unique categories: {n_unique_categories}")
        print(f"  Required qubits: {n_unique_categories}")
        
        # Method 2: Based on qubit range
        used_qubits = set(qubit_assignments)
        n_used_qubits = len(used_qubits)
        max_qubit = max(used_qubits)
        
        print(f"\nMethod 2 - Qubit Range:")
        print(f"  Used qubits: {sorted(used_qubits)}")
        print(f"  Number of used qubits: {n_used_qubits}")
        print(f"  Maximum qubit number: {max_qubit}")
        print(f"  Required qubits: {max_qubit + 1}")
        
        # Method 3: Based on quantum circuit requirements
        # For quantum NLP, we typically need at least 2 qubits for entanglement
        min_qubits = max(2, n_unique_categories)
        
        print(f"\nMethod 3 - Quantum Circuit Requirements:")
        print(f"  Minimum for entanglement: 2 qubits")
        print(f"  Categories to represent: {n_unique_categories}")
        print(f"  Recommended qubits: {min_qubits}")
        
        # Method 4: Based on text complexity
        # More complex texts might need more qubits for better representation
        complexity_qubits = min(8, max(3, n_unique_categories + 1))
        
        print(f"\nMethod 4 - Text Complexity:")
        print(f"  Text complexity factor: +1")
        print(f"  Recommended qubits: {complexity_qubits}")
        
        return {
            'unique_categories': n_unique_categories,
            'used_qubits': n_used_qubits,
            'max_qubit': max_qubit,
            'min_qubits': min_qubits,
            'complexity_qubits': complexity_qubits
        }
    
    def create_qubit_visualization(self, tokens, categories, qubit_assignments, qubit_info):
        """Create visualization of qubit assignment"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Token-to-Qubit Mapping
        ax1.set_title('Token-to-Qubit Mapping', fontsize=14, weight='bold')
        
        n_tokens = len(tokens)
        colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral', 
                 'lightpink', 'lightcyan', 'lightgray', 'lightsteelblue', 'lightgoldenrodyellow']
        
        for i, (token, category, qubit) in enumerate(zip(tokens, categories, qubit_assignments)):
            x = i
            y = qubit
            color = colors[qubit % len(colors)]
            
            # Draw rectangle for each token
            rect = Rectangle((x-0.4, y-0.3), 0.8, 0.6, facecolor=color, alpha=0.7)
            ax1.add_patch(rect)
            
            # Add token text
            ax1.text(x, y, token, fontsize=10, ha='center', va='center', weight='bold')
            
            # Add qubit number
            ax1.text(x, y-0.4, f'Q{qubit}', fontsize=8, ha='center', va='center')
        
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Qubit Number')
        ax1.set_xticks(range(n_tokens))
        ax1.set_xticklabels([f'{i+1}' for i in range(n_tokens)])
        ax1.set_yticks(range(max(qubit_assignments) + 1))
        ax1.set_yticklabels([f'Q{i}' for i in range(max(qubit_assignments) + 1)])
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Category Distribution
        ax2.set_title('Category Distribution', fontsize=14, weight='bold')
        
        category_counts = {}
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        categories_list = list(category_counts.keys())
        counts = list(category_counts.values())
        colors_cat = [colors[i % len(colors)] for i in range(len(categories_list))]
        
        bars = ax2.bar(categories_list, counts, color=colors_cat, alpha=0.7)
        ax2.set_xlabel('Grammatical Category')
        ax2.set_ylabel('Token Count')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=12, weight='bold')
        
        # Plot 3: Qubit Usage
        ax3.set_title('Qubit Usage Pattern', fontsize=14, weight='bold')
        
        used_qubits = sorted(set(qubit_assignments))
        qubit_counts = [qubit_assignments.count(q) for q in used_qubits]
        
        bars = ax3.bar([f'Q{q}' for q in used_qubits], qubit_counts, 
                      color=[colors[q % len(colors)] for q in used_qubits], alpha=0.7)
        ax3.set_xlabel('Qubit Number')
        ax3.set_ylabel('Usage Count')
        
        # Add count labels
        for bar, count in zip(bars, qubit_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=12, weight='bold')
        
        # Plot 4: Qubit Requirements Summary
        ax4.set_title('Qubit Requirements Summary', fontsize=14, weight='bold')
        
        methods = ['Unique\nCategories', 'Used\nQubits', 'Min\nQubits', 'Complexity\nQubits']
        values = [qubit_info['unique_categories'], qubit_info['used_qubits'], 
                 qubit_info['min_qubits'], qubit_info['complexity_qubits']]
        colors_methods = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral']
        
        bars = ax4.bar(methods, values, color=colors_methods, alpha=0.7)
        ax4.set_ylabel('Number of Qubits')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value}', ha='center', va='bottom', fontsize=12, weight='bold')
        
        # Add recommended qubits
        recommended = qubit_info['complexity_qubits']
        ax4.axhline(y=recommended, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax4.text(0.5, recommended + 0.2, f'Recommended: {recommended} qubits', 
                ha='center', va='bottom', fontsize=12, weight='bold', color='red')
        
        plt.tight_layout()
        plt.savefig('qubit_calculation_example.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_quantum_circuit_design(self, qubit_info):
        """Create quantum circuit design for the calculated qubits"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        n_qubits = qubit_info['complexity_qubits']
        
        ax.set_title(f'Quantum Circuit Design for {n_qubits} Qubits', fontsize=18, weight='bold')
        
        # Draw qubit lines
        for i in range(n_qubits):
            y = 0.9 - i * 0.1
            ax.plot([0, 1], [y, y], 'k-', linewidth=3)
            ax.text(-0.05, y, f'q{i}', fontsize=14, ha='right', va='center', weight='bold')
        
        # Draw Hadamard gates
        for i in range(n_qubits):
            y = 0.9 - i * 0.1
            rect = Rectangle((0.1, y-0.02), 0.06, 0.04, facecolor='lightblue', alpha=0.8)
            ax.add_patch(rect)
            ax.text(0.13, y, 'H', fontsize=12, ha='center', va='center', weight='bold')
        
        # Draw rotation gates
        for i in range(n_qubits):
            y = 0.9 - i * 0.1
            rect = Rectangle((0.2, y-0.02), 0.06, 0.04, facecolor='lightgreen', alpha=0.8)
            ax.add_patch(rect)
            ax.text(0.23, y, 'RY', fontsize=10, ha='center', va='center', weight='bold')
        
        # Draw CNOT gates (entanglement)
        for i in range(n_qubits-1):
            y1 = 0.9 - i * 0.1
            y2 = 0.9 - (i+1) * 0.1
            
            # Control dot
            ax.plot(0.35, y1, 'ko', markersize=8)
            # Target with X
            ax.plot(0.35, y2, 'ko', markersize=8)
            ax.text(0.35, y2-0.01, 'X', fontsize=10, ha='center', va='center', color='white')
            # Connection line
            ax.plot([0.35, 0.35], [y1, y2], 'k-', linewidth=2)
            ax.text(0.4, (y1+y2)/2, 'CNOT', fontsize=8, ha='left', va='center')
        
        # Draw measurement
        for i in range(n_qubits):
            y = 0.9 - i * 0.1
            rect = Rectangle((0.5, y-0.02), 0.06, 0.04, facecolor='lightcoral', alpha=0.8)
            ax.add_patch(rect)
            ax.text(0.53, y, 'M', fontsize=12, ha='center', va='center', weight='bold')
        
        # Add circuit information
        ax.text(0.7, 0.9, f'Circuit Specifications:', fontsize=16, weight='bold')
        ax.text(0.7, 0.85, f'• Number of qubits: {n_qubits}', fontsize=12, ha='left')
        ax.text(0.7, 0.8, f'• Dimension: 2^{n_qubits} = {2**n_qubits}', fontsize=12, ha='left')
        ax.text(0.7, 0.75, f'• Density matrix: {2**n_qubits}×{2**n_qubits}', fontsize=12, ha='left')
        ax.text(0.7, 0.7, f'• Max entropy: log₂({2**n_qubits}) = {np.log2(2**n_qubits):.1f}', fontsize=12, ha='left')
        
        ax.text(0.7, 0.6, f'Frame Competition:', fontsize=14, weight='bold')
        ax.text(0.7, 0.55, f'C = S(ρ) / log₂({2**n_qubits})', fontsize=12, ha='left')
        ax.text(0.7, 0.5, f'C = S(ρ) / {np.log2(2**n_qubits):.1f}', fontsize=12, ha='left')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Hadamard (H) - Superposition'),
            mpatches.Patch(color='lightgreen', label='RY Gate - Rotation'),
            mpatches.Patch(color='black', label='CNOT Gate - Entanglement'),
            mpatches.Patch(color='lightcoral', label='Measurement (M)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        ax.set_xlim(-0.1, 1)
        ax.set_ylim(0.1, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('quantum_circuit_design.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_analysis(self):
        """Run the complete qubit calculation analysis"""
        # Analyze text structure
        tokens, categories, qubit_assignments = self.analyze_text_structure()
        
        # Calculate qubit requirements
        qubit_info = self.calculate_qubit_requirements(categories, qubit_assignments)
        
        # Create visualizations
        self.create_qubit_visualization(tokens, categories, qubit_assignments, qubit_info)
        print("✓ Qubit visualization created")
        
        self.create_quantum_circuit_design(qubit_info)
        print("✓ Quantum circuit design created")
        
        # Final recommendation
        print(f"\n" + "=" * 80)
        print("FINAL RECOMMENDATION:")
        print("=" * 80)
        print(f"Text: {self.text}")
        print(f"Tokens: {len(tokens)}")
        print(f"Categories: {set(categories)}")
        print(f"Recommended qubits: {qubit_info['complexity_qubits']}")
        print(f"Circuit dimension: 2^{qubit_info['complexity_qubits']} = {2**qubit_info['complexity_qubits']}")
        print(f"Density matrix size: {2**qubit_info['complexity_qubits']}×{2**qubit_info['complexity_qubits']}")
        
        print(f"\nVisualizations saved:")
        print(f"- qubit_calculation_example.png")
        print(f"- quantum_circuit_design.png")

def main():
    """Main function to run qubit calculation analysis"""
    analysis = QubitCalculationExample()
    analysis.run_complete_analysis()

if __name__ == "__main__":
    main()
