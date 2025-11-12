#!/usr/bin/env python3
"""
Qubit Assignment Validation
Validates the proposed qubit assignment against actual text analysis
"""

import jieba.posseg as pseg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class QubitAssignmentValidation:
    def __init__(self):
        """Initialize the qubit assignment validation"""
        
        # The text we're analyzing
        self.text = "麥當勞性侵案後改革 董事長發聲承諾改善"
        
        # Proposed qubit assignment
        self.proposed_assignment = {
            0: {'role': 'Noun (Subject)', 'example': '麥當勞 (McDonald\'s)'},
            1: {'role': 'Verb (Action)', 'example': '改革 (reform)'},
            2: {'role': 'Function / Modifier', 'example': '後 (after)'},
            3: {'role': 'Complement Noun', 'example': '董事長 (chairman)'},
            4: {'role': 'Adjective / Evaluation', 'example': '良好 (good)'},
            5: {'role': 'Adverb / Tone', 'example': '積極 (actively)'},
            6: {'role': 'Contextual Frame', 'example': '政治 / 經濟 (political/economic)'},
            7: {'role': 'Rhetorical Mode', 'example': 'Hopeful / Critical tone'}
        }
        
        print("=" * 80)
        print("QUBIT ASSIGNMENT VALIDATION")
        print("=" * 80)
        print(f"Text: {self.text}")
        print()
    
    def analyze_actual_text(self):
        """Analyze the actual text to see what categories are present"""
        print("ACTUAL TEXT ANALYSIS:")
        print("-" * 50)
        
        # Segment the text
        words_with_pos = list(pseg.cut(self.text))
        
        print("Token | POS | Category | Proposed Qubit | Actual Qubit")
        print("-" * 60)
        
        actual_categories = []
        proposed_assignments = []
        actual_assignments = []
        
        for word, pos in words_with_pos:
            # Map POS to category
            category = self._pos_to_category(pos)
            actual_categories.append(category)
            
            # Determine proposed qubit based on linguistic role
            proposed_qubit = self._get_proposed_qubit(word, category)
            proposed_assignments.append(proposed_qubit)
            
            # Determine actual qubit based on grammatical category
            actual_qubit = self._get_actual_qubit(category)
            actual_assignments.append(actual_qubit)
            
            print(f"{word:8} | {pos:4} | {category:8} | Q{proposed_qubit:8} | Q{actual_qubit}")
        
        return actual_categories, proposed_assignments, actual_assignments
    
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
    
    def _get_proposed_qubit(self, word, category):
        """Get proposed qubit based on linguistic role"""
        # This is a simplified mapping based on the proposed assignment
        if category == 'N':
            if '麥當勞' in word or '董事' in word:
                return 0  # Subject noun
            else:
                return 3  # Complement noun
        elif category == 'V':
            return 1  # Action verb
        elif category == 'P' or category == 'F':
            return 2  # Function/modifier
        elif category == 'A':
            return 4  # Adjective/evaluation
        elif category == 'D':
            return 5  # Adverb/tone
        else:
            return 6  # Contextual frame
    
    def _get_actual_qubit(self, category):
        """Get actual qubit based on grammatical category"""
        category_qubit_map = {
            'N': 0, 'V': 1, 'A': 2, 'D': 3, 'P': 4, 'R': 5, 'C': 6, 'F': 7, 'X': 8
        }
        return category_qubit_map.get(category, 8)
    
    def validate_assignment(self, actual_categories, proposed_assignments, actual_assignments):
        """Validate the proposed qubit assignment"""
        print("\nVALIDATION ANALYSIS:")
        print("-" * 50)
        
        # Check if proposed assignment covers all categories
        unique_categories = set(actual_categories)
        unique_proposed = set(proposed_assignments)
        unique_actual = set(actual_assignments)
        
        print(f"Unique categories in text: {unique_categories}")
        print(f"Proposed qubits used: {sorted(unique_proposed)}")
        print(f"Actual qubits used: {sorted(unique_actual)}")
        
        # Check coverage
        coverage_proposed = len(unique_proposed) / len(unique_categories) if unique_categories else 0
        coverage_actual = len(unique_actual) / len(unique_categories) if unique_categories else 0
        
        print(f"\nCoverage Analysis:")
        print(f"Proposed assignment coverage: {coverage_proposed:.2%}")
        print(f"Actual assignment coverage: {coverage_actual:.2%}")
        
        # Check for conflicts
        conflicts = []
        for i, (proposed, actual) in enumerate(zip(proposed_assignments, actual_assignments)):
            if proposed != actual:
                conflicts.append((i, proposed, actual))
        
        print(f"\nConflicts found: {len(conflicts)}")
        for conflict in conflicts:
            i, proposed, actual = conflict
            print(f"  Token {i+1}: Proposed Q{proposed} vs Actual Q{actual}")
        
        return {
            'unique_categories': unique_categories,
            'unique_proposed': unique_proposed,
            'unique_actual': unique_actual,
            'coverage_proposed': coverage_proposed,
            'coverage_actual': coverage_actual,
            'conflicts': conflicts
        }
    
    def create_comparison_visualization(self, actual_categories, proposed_assignments, actual_assignments, validation_results):
        """Create visualization comparing proposed vs actual assignments"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Token-by-token comparison
        ax1.set_title('Token-by-Token Qubit Assignment Comparison', fontsize=14, weight='bold')
        
        n_tokens = len(actual_categories)
        x = np.arange(n_tokens)
        
        # Plot proposed vs actual assignments
        ax1.plot(x, proposed_assignments, 'bo-', label='Proposed', linewidth=2, markersize=8)
        ax1.plot(x, actual_assignments, 'ro-', label='Actual', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Qubit Number')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{i+1}' for i in range(n_tokens)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Highlight conflicts
        conflicts = validation_results['conflicts']
        for conflict in conflicts:
            i, proposed, actual = conflict
            ax1.axvline(x=i, color='yellow', alpha=0.5, linestyle='--')
        
        # Plot 2: Category distribution
        ax2.set_title('Category Distribution', fontsize=14, weight='bold')
        
        category_counts = {}
        for category in actual_categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral', 'lightpink']
        
        bars = ax2.bar(categories, counts, color=colors[:len(categories)], alpha=0.7)
        ax2.set_xlabel('Category')
        ax2.set_ylabel('Count')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=12, weight='bold')
        
        # Plot 3: Qubit usage comparison
        ax3.set_title('Qubit Usage Comparison', fontsize=14, weight='bold')
        
        proposed_counts = [proposed_assignments.count(q) for q in range(8)]
        actual_counts = [actual_assignments.count(q) for q in range(9)]
        
        x = np.arange(8)
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, proposed_counts, width, label='Proposed', color='lightblue', alpha=0.7)
        bars2 = ax3.bar(x + width/2, actual_counts[:8], width, label='Actual', color='lightcoral', alpha=0.7)
        
        ax3.set_xlabel('Qubit Number')
        ax3.set_ylabel('Usage Count')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Q{i}' for i in range(8)])
        ax3.legend()
        
        # Add count labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                            f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # Plot 4: Validation summary
        ax4.set_title('Validation Summary', fontsize=14, weight='bold')
        
        metrics = ['Coverage\nProposed', 'Coverage\nActual', 'Conflicts\nFound', 'Categories\nCovered']
        values = [
            validation_results['coverage_proposed'] * 100,
            validation_results['coverage_actual'] * 100,
            len(validation_results['conflicts']),
            len(validation_results['unique_categories'])
        ]
        colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}' if value < 10 else f'{int(value)}', 
                    ha='center', va='bottom', fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.savefig('qubit_assignment_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_proposed_assignment_visualization(self):
        """Create visualization of the proposed qubit assignment"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        ax.set_title('Proposed Qubit Assignment Framework', fontsize=18, weight='bold', pad=20)
        
        # Create qubit assignment table
        y_pos = 0.9
        for qubit, info in self.proposed_assignment.items():
            # Qubit number
            ax.text(0.1, y_pos, f'q{qubit}', fontsize=16, weight='bold', ha='left')
            
            # Role
            ax.text(0.2, y_pos, info['role'], fontsize=14, ha='left')
            
            # Example
            ax.text(0.6, y_pos, info['example'], fontsize=12, ha='left', style='italic')
            
            # Color-coded box
            colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightcoral', 
                     'lightpink', 'lightcyan', 'lightgray', 'lightsteelblue']
            rect = FancyBboxPatch((0.05, y_pos-0.03), 0.9, 0.06, 
                                boxstyle="round,pad=0.01", facecolor=colors[qubit], alpha=0.7)
            ax.add_patch(rect)
            
            y_pos -= 0.1
        
        # Add explanation
        ax.text(0.5, 0.15, 'Linguistic Role-Based Qubit Assignment', 
                fontsize=16, weight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        ax.text(0.5, 0.1, 'This framework assigns qubits based on linguistic roles rather than grammatical categories', 
                fontsize=12, ha='center')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('proposed_qubit_assignment.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_validation(self):
        """Run the complete qubit assignment validation"""
        # Analyze actual text
        actual_categories, proposed_assignments, actual_assignments = self.analyze_actual_text()
        
        # Validate assignment
        validation_results = self.validate_assignment(actual_categories, proposed_assignments, actual_assignments)
        
        # Create visualizations
        self.create_comparison_visualization(actual_categories, proposed_assignments, actual_assignments, validation_results)
        print("✓ Comparison visualization created")
        
        self.create_proposed_assignment_visualization()
        print("✓ Proposed assignment visualization created")
        
        # Final assessment
        print(f"\n" + "=" * 80)
        print("FINAL ASSESSMENT:")
        print("=" * 80)
        
        if validation_results['coverage_proposed'] >= 0.8:
            print("✅ PROPOSED ASSIGNMENT IS LARGELY CORRECT")
        elif validation_results['coverage_proposed'] >= 0.6:
            print("⚠️  PROPOSED ASSIGNMENT IS PARTIALLY CORRECT")
        else:
            print("❌ PROPOSED ASSIGNMENT NEEDS REVISION")
        
        print(f"\nKey Findings:")
        print(f"• Coverage: {validation_results['coverage_proposed']:.1%}")
        print(f"• Conflicts: {len(validation_results['conflicts'])}")
        print(f"• Categories covered: {len(validation_results['unique_categories'])}")
        
        print(f"\nVisualizations saved:")
        print(f"- qubit_assignment_validation.png")
        print(f"- proposed_qubit_assignment.png")

def main():
    """Main function to run qubit assignment validation"""
    validation = QubitAssignmentValidation()
    validation.run_complete_validation()

if __name__ == "__main__":
    main()
