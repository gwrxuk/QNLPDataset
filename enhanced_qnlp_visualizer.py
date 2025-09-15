#!/usr/bin/env python3
"""
Enhanced QNLP Analysis with Improved Quantum Measurements and Visualizations
Focuses on detecting multiple reality phenomena in AI-generated news narratives
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# Configure matplotlib to display Chinese characters correctly
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Heiti TC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import Statevector, partial_trace, entropy
import jieba
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class EnhancedQNLPAnalyzer:
    """Enhanced QNLP Analyzer with improved quantum measurements"""
    
    def __init__(self, dataset_path='dataseet.xlsx'):
        self.dataset_path = dataset_path
        self.df = None
        self.backend = Aer.get_backend('qasm_simulator')
        self.statevector_backend = Aer.get_backend('statevector_simulator')
        self.results = {}
        
    def load_data(self):
        """Load the dataset"""
        self.df = pd.read_excel(self.dataset_path)
        return self.df
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if pd.isna(text):
            return []
        
        text = str(text)
        # Clean text but preserve more Chinese characters
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9\s，。！？；：]', '', text)
        
        # Segment Chinese text
        words = list(jieba.cut(text))
        words = [w.strip() for w in words if len(w.strip()) > 0 and w not in ['的', '了', '是', '在', '有', '和', '與']]
        
        return words
    
    def create_enhanced_quantum_state(self, text_segments, max_qubits=6):
        """Create enhanced quantum state with better encoding"""
        if len(text_segments) == 0:
            return None, 0
        
        n_qubits = min(len(text_segments), max_qubits)
        qc = QuantumCircuit(n_qubits)
        
        # Initialize superposition
        for i in range(n_qubits):
            qc.h(i)
        
        # Add semantic-weighted rotations
        try:
            # Use TF-IDF to weight semantic importance
            vectorizer = TfidfVectorizer(max_features=n_qubits)
            if len(text_segments) > 1:
                tfidf_matrix = vectorizer.fit_transform([' '.join(text_segments[:n_qubits])])
                weights = tfidf_matrix.toarray()[0]
            else:
                weights = np.ones(n_qubits) / n_qubits
        except:
            weights = np.ones(n_qubits) / n_qubits
        
        # Apply weighted rotations
        for i in range(n_qubits):
            if i < len(weights):
                angle = weights[i] * np.pi + np.pi/4  # Add baseline rotation
                qc.ry(angle, i)
        
        # Create entanglement patterns based on semantic similarity
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        
        # Add controlled phase gates for interference
        for i in range(n_qubits - 1):
            qc.cp(np.pi/4, i, (i + 1) % n_qubits)
        
        return qc, n_qubits
    
    def measure_quantum_properties(self, quantum_circuit, n_qubits):
        """Measure various quantum properties"""
        if quantum_circuit is None:
            return {'entropy': 0, 'coherence': 0, 'interference': 0}
        
        # Get statevector
        job = execute(quantum_circuit, self.statevector_backend)
        result = job.result()
        statevector = result.get_statevector()
        
        # Calculate entropy (narrative complexity)
        entropy_val = entropy(statevector)
        
        # Calculate coherence (narrative consistency)
        amplitudes = np.abs(statevector.data)
        coherence = 1 - np.sum(amplitudes**4)  # Participation ratio
        
        # Calculate quantum interference
        phases = np.angle(statevector.data)
        phase_variance = np.var(phases)
        interference = 1 - (phase_variance / (np.pi**2))
        
        # Measure in computational basis for probability distribution
        qc_measure = quantum_circuit.copy()
        qc_measure.measure_all()
        
        job = execute(qc_measure, self.backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate probability distribution entropy
        total_shots = sum(counts.values())
        probs = [counts.get(format(i, f'0{n_qubits}b'), 0) / total_shots for i in range(2**n_qubits)]
        prob_entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
        
        return {
            'entropy': entropy_val,
            'coherence': coherence,
            'interference': interference,
            'prob_entropy': prob_entropy,
            'counts': counts
        }
    
    def detect_narrative_superposition(self, text_list):
        """Detect superposition of multiple narratives"""
        if len(text_list) < 2:
            return 0
        
        # Create quantum circuit for narrative superposition
        n_narratives = min(len(text_list), 4)
        qc = QuantumCircuit(n_narratives + 1)  # +1 for superposition detector
        
        # Initialize narratives in superposition
        for i in range(n_narratives):
            qc.h(i)
        
        # Add narrative-specific rotations
        for i, text in enumerate(text_list[:n_narratives]):
            words = self.preprocess_text(text)
            if words:
                # Rotation based on text length and complexity
                complexity = len(set(words)) / max(1, len(words))
                angle = complexity * np.pi
                qc.ry(angle, i)
        
        # Detect superposition using controlled operations
        for i in range(n_narratives):
            qc.cx(i, n_narratives)
        
        # Measure superposition
        qc.measure_all()
        job = execute(qc, self.backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate superposition strength
        superposition_states = [k for k in counts.keys() if k[-1] == '1']
        superposition_prob = sum(counts[k] for k in superposition_states) / 1000
        
        return superposition_prob
    
    def analyze_semantic_entanglement_network(self, texts):
        """Analyze entanglement between multiple texts"""
        if len(texts) < 2:
            return np.zeros((1, 1))
        
        n_texts = min(len(texts), 8)
        entanglement_matrix = np.zeros((n_texts, n_texts))
        
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                # Create entangled quantum state
                words_i = self.preprocess_text(texts[i])
                words_j = self.preprocess_text(texts[j])
                
                if len(words_i) == 0 or len(words_j) == 0:
                    continue
                
                # Simple entanglement measure based on semantic similarity
                try:
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform([' '.join(words_i), ' '.join(words_j)])
                    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                    
                    # Quantum entanglement simulation
                    qc = QuantumCircuit(2)
                    qc.h(0)
                    qc.cx(0, 1)
                    
                    # Apply similarity-based rotation
                    qc.ry(similarity * np.pi, 0)
                    qc.ry(similarity * np.pi, 1)
                    
                    # Measure entanglement
                    job = execute(qc, self.statevector_backend)
                    result = job.result()
                    statevector = result.get_statevector()
                    
                    # Calculate mutual information
                    rho_a = partial_trace(statevector, [1])
                    rho_b = partial_trace(statevector, [0])
                    
                    entropy_a = entropy(rho_a)
                    entropy_b = entropy(rho_b)
                    entropy_ab = entropy(statevector)
                    
                    mutual_info = entropy_a + entropy_b - entropy_ab
                    entanglement_matrix[i][j] = mutual_info
                    entanglement_matrix[j][i] = mutual_info
                    
                except:
                    entanglement_matrix[i][j] = 0
                    entanglement_matrix[j][i] = 0
        
        return entanglement_matrix
    
    def analyze_field_comprehensive(self, field_name):
        """Comprehensive analysis of a field"""
        print(f"\n{'='*60}")
        print(f"Enhanced QNLP Analysis: {field_name}")
        print(f"{'='*60}")
        
        field_data = self.df[field_name].dropna()
        results = {
            'quantum_properties': [],
            'superposition_scores': [],
            'entanglement_matrix': None,
            'semantic_clusters': [],
            'text_lengths': [],
            'complexity_scores': []
        }
        
        print(f"Processing {len(field_data)} records...")
        
        # Analyze individual texts
        for idx, text in enumerate(field_data):
            if idx % 50 == 0:
                print(f"Progress: {idx}/{len(field_data)}")
            
            words = self.preprocess_text(text)
            if len(words) == 0:
                continue
            
            # Create quantum state and measure properties
            qc, n_qubits = self.create_enhanced_quantum_state(words)
            quantum_props = self.measure_quantum_properties(qc, n_qubits)
            results['quantum_properties'].append(quantum_props)
            
            # Calculate complexity metrics
            unique_words = len(set(words))
            total_words = len(words)
            complexity = unique_words / max(1, total_words)
            
            results['text_lengths'].append(total_words)
            results['complexity_scores'].append(complexity)
        
        # Analyze narrative superposition across all texts
        print("Analyzing narrative superposition...")
        sample_texts = field_data.sample(min(20, len(field_data))).tolist()
        for i in range(0, len(sample_texts), 4):
            batch = sample_texts[i:i+4]
            superposition_score = self.detect_narrative_superposition(batch)
            results['superposition_scores'].append(superposition_score)
        
        # Analyze semantic entanglement network
        print("Analyzing semantic entanglement network...")
        sample_for_entanglement = field_data.sample(min(8, len(field_data))).tolist()
        entanglement_matrix = self.analyze_semantic_entanglement_network(sample_for_entanglement)
        results['entanglement_matrix'] = entanglement_matrix
        
        # Semantic clustering
        print("Performing semantic clustering...")
        try:
            all_words = []
            for text in field_data.sample(min(50, len(field_data))):
                words = self.preprocess_text(text)
                all_words.append(' '.join(words))
            
            if len(all_words) > 5:
                vectorizer = TfidfVectorizer(max_features=100)
                tfidf_matrix = vectorizer.fit_transform(all_words)
                
                # Perform clustering
                n_clusters = min(5, len(all_words) // 3)
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(tfidf_matrix.toarray())
                    results['semantic_clusters'] = clusters
        except:
            results['semantic_clusters'] = []
        
        self.results[field_name] = results
        self.print_enhanced_summary(field_name, results)
        
        return results
    
    def print_enhanced_summary(self, field_name, results):
        """Print enhanced summary statistics"""
        print(f"\n{field_name} - Enhanced Quantum Analysis Summary:")
        print("-" * 50)
        
        if results['quantum_properties']:
            entropies = [r['entropy'] for r in results['quantum_properties']]
            coherences = [r['coherence'] for r in results['quantum_properties']]
            interferences = [r['interference'] for r in results['quantum_properties']]
            
            print(f"Quantum Narrative Complexity (Entropy): {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")
            print(f"Quantum Coherence: {np.mean(coherences):.4f} ± {np.std(coherences):.4f}")
            print(f"Quantum Interference: {np.mean(interferences):.4f} ± {np.std(interferences):.4f}")
        
        if results['superposition_scores']:
            avg_superposition = np.mean(results['superposition_scores'])
            print(f"Narrative Superposition Strength: {avg_superposition:.4f}")
        
        if results['entanglement_matrix'] is not None:
            avg_entanglement = np.mean(results['entanglement_matrix'][results['entanglement_matrix'] > 0])
            print(f"Average Semantic Entanglement: {avg_entanglement:.4f}")
        
        if results['complexity_scores']:
            avg_complexity = np.mean(results['complexity_scores'])
            print(f"Semantic Complexity Score: {avg_complexity:.4f}")
        
        if results['text_lengths']:
            avg_length = np.mean(results['text_lengths'])
            print(f"Average Text Length: {avg_length:.1f} words")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quantum Natural Language Processing Analysis Results', fontsize=16)
        
        fields = ['新聞標題', '影片對話', '影片描述']
        colors = ['red', 'blue', 'green']
        
        # Plot 1: Quantum Entropy Comparison
        ax1 = axes[0, 0]
        for i, field in enumerate(fields):
            if field in self.results and self.results[field]['quantum_properties']:
                entropies = [r['entropy'] for r in self.results[field]['quantum_properties']]
                ax1.hist(entropies, alpha=0.6, label=field, color=colors[i], bins=20)
        ax1.set_xlabel('Quantum Entropy')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Narrative Complexity Distribution')
        ax1.legend()
        
        # Plot 2: Coherence vs Interference
        ax2 = axes[0, 1]
        for i, field in enumerate(fields):
            if field in self.results and self.results[field]['quantum_properties']:
                coherences = [r['coherence'] for r in self.results[field]['quantum_properties']]
                interferences = [r['interference'] for r in self.results[field]['quantum_properties']]
                ax2.scatter(coherences, interferences, alpha=0.6, label=field, color=colors[i])
        ax2.set_xlabel('Quantum Coherence')
        ax2.set_ylabel('Quantum Interference')
        ax2.set_title('Coherence vs Interference')
        ax2.legend()
        
        # Plot 3: Superposition Strength
        ax3 = axes[0, 2]
        superposition_data = []
        labels = []
        for field in fields:
            if field in self.results and self.results[field]['superposition_scores']:
                superposition_data.append(self.results[field]['superposition_scores'])
                labels.append(field)
        if superposition_data:
            ax3.boxplot(superposition_data, labels=labels)
            ax3.set_ylabel('Superposition Strength')
            ax3.set_title('Narrative Superposition by Field')
            plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Plot 4: Entanglement Heatmap
        ax4 = axes[1, 0]
        for i, field in enumerate(fields):
            if field in self.results and self.results[field]['entanglement_matrix'] is not None:
                matrix = self.results[field]['entanglement_matrix']
                if matrix.size > 1:
                    im = ax4.imshow(matrix, cmap='viridis', aspect='auto')
                    ax4.set_title(f'{field} - Semantic Entanglement')
                    plt.colorbar(im, ax=ax4)
                break
        
        # Plot 5: Complexity Distribution
        ax5 = axes[1, 1]
        for i, field in enumerate(fields):
            if field in self.results and self.results[field]['complexity_scores']:
                complexity_scores = self.results[field]['complexity_scores']
                ax5.hist(complexity_scores, alpha=0.6, label=field, color=colors[i], bins=20)
        ax5.set_xlabel('Semantic Complexity')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Semantic Complexity Distribution')
        ax5.legend()
        
        # Plot 6: Text Length vs Quantum Properties
        ax6 = axes[1, 2]
        for i, field in enumerate(fields):
            if field in self.results and self.results[field]['text_lengths'] and self.results[field]['quantum_properties']:
                lengths = self.results[field]['text_lengths']
                entropies = [r['entropy'] for r in self.results[field]['quantum_properties']]
                if len(lengths) == len(entropies):
                    ax6.scatter(lengths, entropies, alpha=0.6, label=field, color=colors[i])
        ax6.set_xlabel('Text Length (words)')
        ax6.set_ylabel('Quantum Entropy')
        ax6.set_title('Text Length vs Narrative Complexity')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig('qnlp_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'qnlp_analysis_results.png'")
    
    def generate_academic_report(self):
        """Generate academic report summary"""
        report = f"""
# Quantum Natural Language Processing Analysis Report

## Abstract
This analysis applies quantum computing principles to analyze multiple reality phenomena in AI-generated news narratives using IBM Qiskit. The study examines quantum superposition, entanglement, and interference patterns in three text fields: news titles (新聞標題), video dialogue (影片對話), and video descriptions (影片描述).

## Methodology
- **Quantum Superposition**: Encoded narrative elements as quantum states in superposition
- **Semantic Entanglement**: Measured quantum entanglement between semantic elements
- **Narrative Complexity**: Calculated von Neumann entropy of quantum states
- **Interference Patterns**: Detected narrative conflicts through quantum interference

## Results Summary

"""
        
        for field in ['新聞標題', '影片對話', '影片描述']:
            if field in self.results:
                results = self.results[field]
                report += f"\n### {field}\n"
                
                if results['quantum_properties']:
                    entropies = [r['entropy'] for r in results['quantum_properties']]
                    coherences = [r['coherence'] for r in results['quantum_properties']]
                    
                    report += f"- Average Quantum Entropy: {np.mean(entropies):.4f} ± {np.std(entropies):.4f}\n"
                    report += f"- Average Coherence: {np.mean(coherences):.4f} ± {np.std(coherences):.4f}\n"
                
                if results['superposition_scores']:
                    avg_superposition = np.mean(results['superposition_scores'])
                    report += f"- Narrative Superposition: {avg_superposition:.4f}\n"
                
                if results['entanglement_matrix'] is not None:
                    avg_entanglement = np.mean(results['entanglement_matrix'][results['entanglement_matrix'] > 0])
                    report += f"- Semantic Entanglement: {avg_entanglement:.4f}\n"
        
        report += """
## Conclusions
The quantum analysis reveals patterns of narrative complexity and semantic entanglement that suggest multiple reality phenomena in AI-generated content. The quantum superposition measurements indicate varying degrees of narrative multiplicity across different content types.

## Academic Implications
This quantum approach to NLP provides a novel framework for understanding how AI-generated content can simultaneously encode multiple, potentially conflicting narratives - a phenomenon that mirrors quantum superposition in physics.
"""
        
        with open('qnlp_academic_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("Academic report saved as 'qnlp_academic_report.md'")
        return report
    
    def run_enhanced_analysis(self):
        """Run the complete enhanced analysis"""
        print("Starting Enhanced Quantum Natural Language Processing Analysis")
        print("=" * 70)
        
        self.load_data()
        
        # Analyze each field
        fields = ['新聞標題', '影片對話', '影片描述']
        for field in fields:
            try:
                self.analyze_field_comprehensive(field)
            except Exception as e:
                print(f"Error analyzing {field}: {e}")
                continue
        
        # Create visualizations
        try:
            self.create_visualizations()
        except Exception as e:
            print(f"Error creating visualizations: {e}")
        
        # Generate academic report
        self.generate_academic_report()
        
        print(f"\n{'='*70}")
        print("Enhanced QNLP Analysis Complete!")
        print(f"{'='*70}")

if __name__ == "__main__":
    analyzer = EnhancedQNLPAnalyzer()
    analyzer.run_enhanced_analysis()
