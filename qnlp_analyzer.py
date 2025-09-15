#!/usr/bin/env python3
"""
Quantum Natural Language Processing (QNLP) Analysis using Qiskit
For analyzing multiple reality phenomena in AI-generated news narratives

This implementation focuses on:
1. Quantum superposition of conflicting narratives
2. Entanglement between semantic frames
3. Detection of narrative divergence and framing conflicts
4. Post-classical media meaning construction modeling
"""

import pandas as pd
import numpy as np
from qiskit import QuantumCircuit, transpile, Aer, execute
from qiskit.quantum_info import Statevector, partial_trace, entropy, DensityMatrix
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import jieba
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class QNLPAnalyzer:
    """
    Quantum Natural Language Processing Analyzer for detecting multiple realities
    in AI-generated news narratives using quantum superposition and entanglement
    """
    
    def __init__(self, dataset_path='dataseet.xlsx'):
        """Initialize the QNLP analyzer"""
        self.dataset_path = dataset_path
        self.df = None
        self.backend = Aer.get_backend('qasm_simulator')
        self.statevector_backend = Aer.get_backend('statevector_simulator')
        
        # Initialize analysis results storage
        self.results = {
            '新聞標題': {},
            '影片對話': {},
            '影片描述': {}
        }
        
    def load_data(self):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        self.df = pd.read_excel(self.dataset_path)
        print(f"Loaded {len(self.df)} records with {len(self.df.columns)} fields")
        return self.df
    
    def preprocess_text(self, text):
        """Preprocess Chinese text for quantum encoding"""
        if pd.isna(text):
            return ""
        
        # Clean text
        text = str(text)
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9\s]', '', text)
        
        # Segment Chinese text
        words = list(jieba.cut(text))
        words = [w.strip() for w in words if len(w.strip()) > 1]
        
        return words
    
    def create_semantic_quantum_state(self, text_segments, max_qubits=8):
        """
        Create quantum state representation of text segments
        Uses superposition to represent multiple semantic interpretations
        """
        # Limit to max_qubits to keep computation tractable
        n_segments = min(len(text_segments), max_qubits)
        if n_segments == 0:
            return None, None
            
        # Create quantum circuit
        qc = QuantumCircuit(n_segments)
        
        # Initialize superposition states
        for i in range(n_segments):
            qc.h(i)  # Hadamard gate for superposition
        
        # Add entanglement between semantic elements
        for i in range(n_segments - 1):
            qc.cx(i, i + 1)  # CNOT gates for entanglement
        
        # Add rotation gates based on semantic weights
        vectorizer = TfidfVectorizer(max_features=n_segments)
        try:
            tfidf_matrix = vectorizer.fit_transform([' '.join(text_segments)])
            weights = tfidf_matrix.toarray()[0]
            
            for i, weight in enumerate(weights):
                if i < n_segments:
                    # Rotation angle based on TF-IDF weight
                    angle = weight * np.pi
                    qc.ry(angle, i)
        except:
            # Fallback to uniform weights
            for i in range(n_segments):
                qc.ry(np.pi/4, i)
        
        return qc, n_segments
    
    def measure_narrative_superposition(self, quantum_circuit, n_qubits):
        """
        Measure quantum superposition to detect narrative multiplicity
        Returns entropy and coherence measures
        """
        if quantum_circuit is None:
            return 0, 0, []
        
        # Get statevector
        job = execute(quantum_circuit, self.statevector_backend)
        result = job.result()
        statevector = result.get_statevector()
        
        # Calculate von Neumann entropy (measure of narrative complexity)
        entropy_val = entropy(statevector)
        
        # Calculate coherence (measure of narrative consistency)
        amplitudes = np.abs(statevector.data)
        coherence = 1 - np.sum(amplitudes**4)  # Participation ratio
        
        # Measure in computational basis
        qc_measure = quantum_circuit.copy()
        qc_measure.measure_all()
        
        job = execute(qc_measure, self.backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        return entropy_val, coherence, counts
    
    def detect_semantic_entanglement(self, text1_segments, text2_segments, max_qubits=6):
        """
        Detect semantic entanglement between different text segments
        Uses quantum entanglement to model narrative interdependence
        """
        n1 = min(len(text1_segments), max_qubits//2)
        n2 = min(len(text2_segments), max_qubits//2)
        
        if n1 == 0 or n2 == 0:
            return 0, 0
        
        total_qubits = n1 + n2
        qc = QuantumCircuit(total_qubits)
        
        # Initialize both subsystems in superposition
        for i in range(total_qubits):
            qc.h(i)
        
        # Create entanglement between the two text segments
        for i in range(n1):
            for j in range(n2):
                qc.cx(i, n1 + j)
        
        # Apply semantic rotations
        for i in range(n1):
            qc.ry(np.pi/3, i)
        for i in range(n2):
            qc.ry(np.pi/4, n1 + i)
        
        # Get statevector and calculate entanglement
        job = execute(qc, self.statevector_backend)
        result = job.result()
        statevector = result.get_statevector()
        
        # Calculate mutual information (entanglement measure)
        # Trace out subsystem B to get reduced density matrix of A
        rho_a = partial_trace(statevector, list(range(n1, total_qubits)))
        rho_b = partial_trace(statevector, list(range(n1)))
        
        entropy_a = entropy(rho_a)
        entropy_b = entropy(rho_b)
        entropy_ab = entropy(statevector)
        
        mutual_information = entropy_a + entropy_b - entropy_ab
        entanglement_strength = min(mutual_information, min(entropy_a, entropy_b))
        
        return mutual_information, entanglement_strength
    
    def analyze_framing_conflicts(self, text_segments):
        """
        Analyze framing conflicts using quantum interference patterns
        Models competing narrative frames as interfering quantum states
        """
        if len(text_segments) < 2:
            return 0, []
        
        n_frames = min(len(text_segments), 6)
        qc = QuantumCircuit(n_frames + 1)  # +1 for interference detection
        
        # Create superposition of different frames
        for i in range(n_frames):
            qc.h(i)
        
        # Add controlled rotations for frame conflicts
        for i in range(n_frames):
            qc.cry(np.pi/4, i, n_frames)  # Controlled rotation on interference qubit
        
        # Apply Grover-like operator for conflict amplification
        for i in range(n_frames):
            qc.x(i)
        qc.mcx(list(range(n_frames)), n_frames)
        for i in range(n_frames):
            qc.x(i)
        
        # Measure interference patterns
        qc.measure_all()
        job = execute(qc, self.backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate conflict intensity
        conflict_states = [k for k in counts.keys() if k[-1] == '1']  # States with interference
        conflict_probability = sum(counts[k] for k in conflict_states) / 1000
        
        return conflict_probability, counts
    
    def quantum_sentiment_analysis(self, text_segments):
        """
        Quantum-enhanced sentiment analysis using amplitude encoding
        Models sentiment superposition and emotional entanglement
        """
        if len(text_segments) == 0:
            return 0, 0, 0
        
        # Simple sentiment scoring (can be enhanced with pre-trained models)
        positive_words = ['好', '優秀', '成功', '進步', '改善', '正面', '積極', '讚', '棒']
        negative_words = ['壞', '失敗', '問題', '危機', '錯誤', '負面', '批評', '擔心', '困難']
        
        sentiment_scores = []
        for segment in text_segments:
            pos_count = sum(1 for word in positive_words if word in segment)
            neg_count = sum(1 for word in negative_words if word in segment)
            score = (pos_count - neg_count) / max(1, len(segment.split()))
            sentiment_scores.append(score)
        
        if len(sentiment_scores) == 0:
            return 0, 0, 0
        
        # Quantum encoding of sentiment
        n_qubits = min(4, len(sentiment_scores))
        qc = QuantumCircuit(n_qubits)
        
        # Encode sentiment as rotation angles
        for i, score in enumerate(sentiment_scores[:n_qubits]):
            angle = (score + 1) * np.pi / 2  # Map [-1,1] to [0,π]
            qc.ry(angle, i)
        
        # Add entanglement for sentiment correlation
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        
        # Measure sentiment superposition
        job = execute(qc, self.statevector_backend)
        result = job.result()
        statevector = result.get_statevector()
        
        sentiment_entropy = entropy(statevector)
        sentiment_coherence = np.abs(np.sum(statevector.data * np.conj(statevector.data)))
        avg_sentiment = np.mean(sentiment_scores)
        
        return sentiment_entropy, sentiment_coherence, avg_sentiment
    
    def analyze_field(self, field_name):
        """
        Comprehensive QNLP analysis of a specific field
        """
        print(f"\n{'='*50}")
        print(f"QNLP Analysis: {field_name}")
        print(f"{'='*50}")
        
        field_data = self.df[field_name].dropna()
        results = {
            'narrative_complexity': [],
            'narrative_coherence': [],
            'semantic_entanglement': [],
            'framing_conflicts': [],
            'sentiment_analysis': [],
            'quantum_measurements': []
        }
        
        print(f"Processing {len(field_data)} records...")
        
        for idx, text in enumerate(field_data):
            if idx % 50 == 0:
                print(f"Progress: {idx}/{len(field_data)}")
            
            # Preprocess text
            segments = self.preprocess_text(text)
            if len(segments) == 0:
                continue
            
            # 1. Narrative Superposition Analysis
            qc, n_qubits = self.create_semantic_quantum_state(segments)
            entropy_val, coherence, counts = self.measure_narrative_superposition(qc, n_qubits)
            
            results['narrative_complexity'].append(entropy_val)
            results['narrative_coherence'].append(coherence)
            results['quantum_measurements'].append(counts)
            
            # 2. Semantic Entanglement (compare with previous text if available)
            if idx > 0:
                prev_segments = self.preprocess_text(field_data.iloc[idx-1])
                mutual_info, entanglement = self.detect_semantic_entanglement(segments, prev_segments)
                results['semantic_entanglement'].append((mutual_info, entanglement))
            
            # 3. Framing Conflicts
            conflict_prob, conflict_counts = self.analyze_framing_conflicts(segments)
            results['framing_conflicts'].append(conflict_prob)
            
            # 4. Quantum Sentiment Analysis
            sent_entropy, sent_coherence, avg_sent = self.quantum_sentiment_analysis(segments)
            results['sentiment_analysis'].append((sent_entropy, sent_coherence, avg_sent))
        
        # Store results
        self.results[field_name] = results
        
        # Calculate summary statistics
        self.print_field_summary(field_name, results)
        
        return results
    
    def print_field_summary(self, field_name, results):
        """Print summary statistics for a field analysis"""
        print(f"\n{field_name} - Quantum Analysis Summary:")
        print("-" * 40)
        
        if results['narrative_complexity']:
            avg_complexity = np.mean(results['narrative_complexity'])
            avg_coherence = np.mean(results['narrative_coherence'])
            print(f"Average Narrative Complexity (Entropy): {avg_complexity:.4f}")
            print(f"Average Narrative Coherence: {avg_coherence:.4f}")
        
        if results['semantic_entanglement']:
            avg_mutual_info = np.mean([x[0] for x in results['semantic_entanglement']])
            avg_entanglement = np.mean([x[1] for x in results['semantic_entanglement']])
            print(f"Average Semantic Mutual Information: {avg_mutual_info:.4f}")
            print(f"Average Semantic Entanglement: {avg_entanglement:.4f}")
        
        if results['framing_conflicts']:
            avg_conflict = np.mean(results['framing_conflicts'])
            print(f"Average Framing Conflict Probability: {avg_conflict:.4f}")
        
        if results['sentiment_analysis']:
            avg_sent_entropy = np.mean([x[0] for x in results['sentiment_analysis']])
            avg_sent_coherence = np.mean([x[1] for x in results['sentiment_analysis']])
            avg_sentiment = np.mean([x[2] for x in results['sentiment_analysis']])
            print(f"Average Sentiment Entropy: {avg_sent_entropy:.4f}")
            print(f"Average Sentiment Coherence: {avg_sent_coherence:.4f}")
            print(f"Average Sentiment Score: {avg_sentiment:.4f}")
    
    def generate_comparative_analysis(self):
        """Generate comparative analysis across all fields"""
        print(f"\n{'='*60}")
        print("COMPARATIVE QUANTUM ANALYSIS ACROSS FIELDS")
        print(f"{'='*60}")
        
        fields = ['新聞標題', '影片對話', '影片描述']
        
        # Compare narrative complexity
        print("\n1. Narrative Complexity Comparison:")
        for field in fields:
            if field in self.results and self.results[field]['narrative_complexity']:
                avg_complexity = np.mean(self.results[field]['narrative_complexity'])
                std_complexity = np.std(self.results[field]['narrative_complexity'])
                print(f"   {field}: {avg_complexity:.4f} ± {std_complexity:.4f}")
        
        # Compare semantic entanglement
        print("\n2. Semantic Entanglement Comparison:")
        for field in fields:
            if field in self.results and self.results[field]['semantic_entanglement']:
                entanglements = [x[1] for x in self.results[field]['semantic_entanglement']]
                avg_entanglement = np.mean(entanglements)
                std_entanglement = np.std(entanglements)
                print(f"   {field}: {avg_entanglement:.4f} ± {std_entanglement:.4f}")
        
        # Compare framing conflicts
        print("\n3. Framing Conflicts Comparison:")
        for field in fields:
            if field in self.results and self.results[field]['framing_conflicts']:
                avg_conflict = np.mean(self.results[field]['framing_conflicts'])
                std_conflict = np.std(self.results[field]['framing_conflicts'])
                print(f"   {field}: {avg_conflict:.4f} ± {std_conflict:.4f}")
    
    def save_results(self, filename='qnlp_results.csv'):
        """Save analysis results to CSV file"""
        all_results = []
        
        for field_name, field_results in self.results.items():
            if not field_results['narrative_complexity']:
                continue
                
            for i in range(len(field_results['narrative_complexity'])):
                row = {
                    'field': field_name,
                    'record_index': i,
                    'narrative_complexity': field_results['narrative_complexity'][i],
                    'narrative_coherence': field_results['narrative_coherence'][i],
                    'framing_conflict': field_results['framing_conflicts'][i] if i < len(field_results['framing_conflicts']) else 0,
                }
                
                if i < len(field_results['semantic_entanglement']):
                    row['mutual_information'] = field_results['semantic_entanglement'][i][0]
                    row['entanglement_strength'] = field_results['semantic_entanglement'][i][1]
                
                if i < len(field_results['sentiment_analysis']):
                    row['sentiment_entropy'] = field_results['sentiment_analysis'][i][0]
                    row['sentiment_coherence'] = field_results['sentiment_analysis'][i][1]
                    row['average_sentiment'] = field_results['sentiment_analysis'][i][2]
                
                all_results.append(row)
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to {filename}")
        
        return results_df
    
    def run_complete_analysis(self):
        """Run complete QNLP analysis on all fields"""
        print("Starting Quantum Natural Language Processing Analysis")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Analyze each field separately
        fields = ['新聞標題', '影片對話', '影片描述']
        
        for field in fields:
            try:
                self.analyze_field(field)
            except Exception as e:
                print(f"Error analyzing {field}: {e}")
                continue
        
        # Generate comparative analysis
        self.generate_comparative_analysis()
        
        # Save results
        results_df = self.save_results()
        
        print(f"\n{'='*60}")
        print("QNLP Analysis Complete!")
        print(f"{'='*60}")
        
        return results_df

# Example usage
if __name__ == "__main__":
    analyzer = QNLPAnalyzer()
    results = analyzer.run_complete_analysis()
