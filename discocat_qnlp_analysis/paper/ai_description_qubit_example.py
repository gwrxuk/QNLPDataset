#!/usr/bin/env python3
"""
AI Description Qubit Representation Example
Shows step-by-step conversion of an AI description to quantum qubits
"""

import jieba.posseg as pseg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AIDescriptionQubitExample:
    def __init__(self):
        """Initialize the AI description qubit example"""
        
        # Real AI description from the dataset
        self.ai_description = "台灣麥當勞於今（19日）發表最新聲明，針對去年底發生的職場性侵案，首次公布「3大安心作為」，並承諾加強員工保護機制，提升企業責任。董事長李昌霖發表公開信，強調「沒有任何藉口，企業必須檢討！」，承諾確保"
        
        # Qubit mapping for grammatical categories
        self.category_qubit_map = {
            'N': 0,    # Nouns (名词)
            'V': 1,    # Verbs (动词)  
            'A': 2,    # Adjectives (形容词)
            'D': 3,    # Adverbs (副词)
            'P': 4,    # Prepositions (介词)
            'R': 5,    # Pronouns (代词)
            'C': 6,    # Conjunctions (连词)
            'X': 7     # Other/Unknown (其他)
        }
        
        # Quantum gate parameters for each category
        self.category_gates = {
            'N': {'rotation': np.pi/4, 'phase': 0, 'color': 'lightgreen'},
            'V': {'rotation': np.pi/3, 'phase': np.pi/2, 'color': 'lightblue'},
            'A': {'rotation': np.pi/6, 'phase': np.pi/4, 'color': 'lightyellow'},
            'D': {'rotation': np.pi/5, 'phase': np.pi/3, 'color': 'lightcoral'},
            'P': {'rotation': np.pi/8, 'phase': np.pi/6, 'color': 'lightpink'},
            'R': {'rotation': np.pi/7, 'phase': np.pi/8, 'color': 'lightcyan'},
            'C': {'rotation': np.pi/2, 'phase': np.pi, 'color': 'lightgray'},
            'X': {'rotation': np.pi/12, 'phase': 0, 'color': 'lightsteelblue'}
        }
    
    def process_text_to_qubits(self):
        """Process the AI description text into qubit representation"""
        print("=" * 80)
        print("AI DESCRIPTION QUBIT REPRESENTATION EXAMPLE")
        print("=" * 80)
        print(f"Input Text: {self.ai_description}")
        print()
        
        # Step 1: Text segmentation
        print("STEP 1: TEXT SEGMENTATION")
        print("-" * 40)
        words_with_pos = list(pseg.cut(self.ai_description))
        
        print("Word | POS Tag | Category | Qubit")
        print("-" * 40)
        
        qubit_assignments = []
        for word, pos in words_with_pos:
            # Map POS to category
            category = self._pos_to_category(pos)
            qubit = self.category_qubit_map[category]
            qubit_assignments.append((word, pos, category, qubit))
            print(f"{word:8} | {pos:6} | {category:8} | {qubit}")
        
        print()
        
        # Step 2: Qubit frequency analysis
        print("STEP 2: QUBIT FREQUENCY ANALYSIS")
        print("-" * 40)
        qubit_counts = {}
        for word, pos, category, qubit in qubit_assignments:
            qubit_counts[qubit] = qubit_counts.get(qubit, 0) + 1
        
        for qubit, count in sorted(qubit_counts.items()):
            category = [k for k, v in self.category_qubit_map.items() if v == qubit][0]
            print(f"Qubit {qubit} ({category}): {count} words")
        
        print()
        
        # Step 3: Quantum circuit parameters
        print("STEP 3: QUANTUM CIRCUIT PARAMETERS")
        print("-" * 40)
        print("Category | Qubit | Rotation Angle | Phase | Color")
        print("-" * 40)
        
        for category, qubit in self.category_qubit_map.items():
            if qubit in qubit_counts:
                params = self.category_gates[category]
                print(f"{category:8} | {qubit:5} | {params['rotation']:13.4f} | {params['phase']:5.4f} | {params['color']}")
        
        return qubit_assignments, qubit_counts
    
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
            # Others
            'm': 'X', 'q': 'X', 'f': 'X', 'u': 'X', 'xc': 'X'
        }
        return pos_mapping.get(pos, 'X')
    
    def create_qubit_visualization(self, qubit_assignments, qubit_counts):
        """Create visualization of qubit representation"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Word-to-Qubit Mapping
        ax1.set_title('AI Description: Word-to-Qubit Mapping', fontsize=14, weight='bold')
        
        y_pos = 0.9
        for i, (word, pos, category, qubit) in enumerate(qubit_assignments[:15]):  # Show first 15 words
            if i > 0 and i % 5 == 0:
                y_pos -= 0.15
            
            x_pos = 0.1 + (i % 5) * 0.15
            
            # Color by category
            color = self.category_gates[category]['color']
            rect = Rectangle((x_pos-0.06, y_pos-0.04), 0.12, 0.08, 
                           facecolor=color, alpha=0.7, edgecolor='black')
            ax1.add_patch(rect)
            
            ax1.text(x_pos, y_pos, f"{word}\nQ{qubit}", fontsize=8, ha='center', va='center', weight='bold')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Plot 2: Qubit Frequency Distribution
        ax2.set_title('Qubit Frequency Distribution', fontsize=14, weight='bold')
        
        qubits = list(qubit_counts.keys())
        counts = list(qubit_counts.values())
        colors = [self.category_gates[[k for k, v in self.category_qubit_map.items() if v == q][0]]['color'] 
                 for q in qubits]
        
        bars = ax2.bar([f'Q{qubit}' for qubit in qubits], counts, color=colors, alpha=0.7)
        ax2.set_xlabel('Qubit Number')
        ax2.set_ylabel('Word Count')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # Plot 3: Quantum Circuit Parameters
        ax3.set_title('Quantum Gate Parameters by Category', fontsize=14, weight='bold')
        
        categories = list(self.category_qubit_map.keys())
        rotations = [self.category_gates[cat]['rotation'] for cat in categories]
        phases = [self.category_gates[cat]['phase'] for cat in categories]
        colors = [self.category_gates[cat]['color'] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, rotations, width, label='Rotation Angle', color=colors, alpha=0.7)
        bars2 = ax3.bar(x + width/2, phases, width, label='Phase', color=colors, alpha=0.5)
        
        ax3.set_xlabel('Grammatical Category')
        ax3.set_ylabel('Angle (radians)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Quantum State Representation
        ax4.set_title('Quantum State Superposition', fontsize=14, weight='bold')
        
        # Simulate quantum state amplitudes
        n_qubits = len(qubit_counts)
        state_amplitudes = np.random.random(2**n_qubits) + 1j * np.random.random(2**n_qubits)
        state_amplitudes = state_amplitudes / np.linalg.norm(state_amplitudes)
        
        # Show first 8 states
        n_states = min(8, 2**n_qubits)
        states = [f'|{i:0{n_qubits}b}>' for i in range(n_states)]
        amplitudes = np.abs(state_amplitudes[:n_states])
        
        bars = ax4.bar(range(n_states), amplitudes, color='lightblue', alpha=0.7)
        ax4.set_xlabel('Quantum States')
        ax4.set_ylabel('Amplitude Magnitude')
        ax4.set_xticks(range(n_states))
        ax4.set_xticklabels(states, rotation=45)
        
        # Add amplitude values
        for bar, amp in zip(bars, amplitudes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{amp:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('ai_description_qubit_example.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_detailed_qubit_table(self, qubit_assignments):
        """Create detailed qubit assignment table"""
        print("\n" + "=" * 80)
        print("DETAILED QUBIT ASSIGNMENT TABLE")
        print("=" * 80)
        
        # Group by qubit
        qubit_groups = {}
        for word, pos, category, qubit in qubit_assignments:
            if qubit not in qubit_groups:
                qubit_groups[qubit] = []
            qubit_groups[qubit].append((word, pos, category))
        
        for qubit in sorted(qubit_groups.keys()):
            category = [k for k, v in self.category_qubit_map.items() if v == qubit][0]
            params = self.category_gates[category]
            
            print(f"\nQUBIT {qubit} ({category} - {self._get_category_name(category)})")
            print(f"Rotation Angle: {params['rotation']:.4f} radians")
            print(f"Phase: {params['phase']:.4f} radians")
            print(f"Words assigned: {len(qubit_groups[qubit])}")
            print("-" * 50)
            
            for word, pos, cat in qubit_groups[qubit]:
                print(f"  {word:12} | {pos:6} | {cat}")
    
    def _get_category_name(self, category):
        """Get full category name"""
        names = {
            'N': 'Nouns (名词)',
            'V': 'Verbs (动词)',
            'A': 'Adjectives (形容词)',
            'D': 'Adverbs (副词)',
            'P': 'Prepositions (介词)',
            'R': 'Pronouns (代词)',
            'C': 'Conjunctions (连词)',
            'X': 'Other/Unknown (其他)'
        }
        return names.get(category, 'Unknown')
    
    def run_complete_example(self):
        """Run the complete AI description qubit example"""
        # Process text to qubits
        qubit_assignments, qubit_counts = self.process_text_to_qubits()
        
        # Create visualizations
        self.create_qubit_visualization(qubit_assignments, qubit_counts)
        
        # Create detailed table
        self.create_detailed_qubit_table(qubit_assignments)
        
        print("\n" + "=" * 80)
        print("QUANTUM CIRCUIT CONSTRUCTION SUMMARY")
        print("=" * 80)
        print(f"Total words processed: {len(qubit_assignments)}")
        print(f"Unique qubits used: {len(qubit_counts)}")
        print(f"Qubit range: {min(qubit_counts.keys())} to {max(qubit_counts.keys())}")
        print(f"Most frequent qubit: Q{max(qubit_counts, key=qubit_counts.get)} "
              f"({qubit_counts[max(qubit_counts, key=qubit_counts.get)]} words)")
        
        print("\nQuantum Circuit Parameters:")
        for qubit in sorted(qubit_counts.keys()):
            category = [k for k, v in self.category_qubit_map.items() if v == qubit][0]
            params = self.category_gates[category]
            print(f"  Q{qubit} ({category}): RY({params['rotation']:.4f}), RZ({params['phase']:.4f})")
        
        print(f"\nVisualization saved as: ai_description_qubit_example.png")

def main():
    """Main function to run the AI description qubit example"""
    example = AIDescriptionQubitExample()
    example.run_complete_example()

if __name__ == "__main__":
    main()
