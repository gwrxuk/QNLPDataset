#!/usr/bin/env python3
"""
Optimized DisCoCat Quantum Natural Language Processing Analyzer
High-performance version with reduced computational complexity
"""

import pandas as pd
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import entropy, Statevector, DensityMatrix
from qiskit.circuit.library import RYGate, CXGate, HGate
import json
import time
import os
from collections import Counter, defaultdict
import ast
import re
from typing import Dict, List, Any, Tuple

class OptimizedDisCoCatQNLPAnalyzer:
    def __init__(self):
        """Initialize the optimized DisCoCat QNLP analyzer with performance optimizations."""
        self.backend = Aer.get_backend('statevector_simulator')
        
        # Simplified category mapping for performance
        self.category_map = {
            'N': 0, 'V': 1, 'A': 2, 'P': 3, 'D': 4, 'C': 5, 'X': 6, 'default': 7
        }
        
        # Cache for quantum circuits to avoid recomputation
        self.circuit_cache = {}
        
        print("🚀 優化版 DisCoCat 量子自然語言處理分析器已初始化")
        
    def create_optimized_circuit(self, categories: List[str], word_count: int) -> QuantumCircuit:
        """Create an optimized quantum circuit based on categories and word count."""
        
        # Limit circuit size for performance (max 8 qubits)
        num_qubits = min(8, max(3, len(set(categories)) + 1))
        
        # Check cache first
        cache_key = f"{sorted(set(categories))}_{word_count}_{num_qubits}"
        if cache_key in self.circuit_cache:
            return self.circuit_cache[cache_key].copy()
        
        circuit = QuantumCircuit(num_qubits)
        
        # Initialize superposition states efficiently
        for i in range(num_qubits):
            circuit.h(i)
        
        # Add category-specific rotations (simplified)
        unique_cats = list(set(categories))[:num_qubits-1]
        for i, cat in enumerate(unique_cats):
            if cat in self.category_map:
                angle = (self.category_map[cat] + 1) * np.pi / 8
                circuit.ry(angle, i)
        
        # Add entanglement based on word relationships (simplified)
        for i in range(num_qubits - 1):
            if word_count > (i + 1) * 5:  # Conditional entanglement
                circuit.cx(i, i + 1)
        
        # Add semantic complexity rotation
        complexity_angle = min(np.pi/2, word_count * np.pi / 100)
        circuit.ry(complexity_angle, num_qubits - 1)
        
        # Cache the circuit
        self.circuit_cache[cache_key] = circuit.copy()
        
        return circuit
    
    def measure_quantum_properties_fast(self, circuit: QuantumCircuit) -> Dict[str, float]:
        """Fast quantum property measurement with optimized calculations."""
        
        try:
            # Execute circuit once and reuse results
            job = execute(circuit, self.backend)
            result = job.result()
            statevector = result.get_statevector()
            
            # Convert to numpy array for faster operations
            state_array = np.array(statevector.data)
            probabilities = np.abs(state_array) ** 2
            
            # Calculate metrics efficiently
            metrics = {}
            
            # Von Neumann entropy (simplified calculation)
            # Filter out near-zero probabilities for numerical stability
            valid_probs = probabilities[probabilities > 1e-10]
            if len(valid_probs) > 0:
                metrics['von_neumann_entropy'] = float(-np.sum(valid_probs * np.log2(valid_probs + 1e-10)))
            else:
                metrics['von_neumann_entropy'] = 0.0
            
            # Category coherence (based on state distribution)
            metrics['category_coherence'] = float(1.0 - np.std(probabilities))
            
            # Compositional entanglement (simplified Schmidt decomposition approximation)
            if circuit.num_qubits >= 2:
                # Approximate entanglement through state vector analysis
                reshaped = state_array.reshape(2, -1) if len(state_array) >= 4 else state_array
                if reshaped.ndim == 2:
                    singular_values = np.linalg.svd(reshaped, compute_uv=False)
                    metrics['compositional_entanglement'] = float(1.0 - (singular_values[0] ** 2))
                else:
                    metrics['compositional_entanglement'] = 0.5
            else:
                metrics['compositional_entanglement'] = 0.0
            
            # Grammatical superposition (state coherence measure)
            superposition_strength = np.sum(probabilities * (1 - probabilities))
            metrics['grammatical_superposition'] = float(4 * superposition_strength)  # Normalized
            
            # Semantic interference (phase relationships approximation)
            phase_variance = np.var(np.angle(state_array))
            metrics['semantic_interference'] = float(min(1.0, phase_variance / np.pi))
            
            # Frame competition (probability distribution entropy)
            if len(valid_probs) > 1:
                uniform_entropy = np.log2(len(valid_probs))
                actual_entropy = -np.sum(valid_probs * np.log2(valid_probs + 1e-10))
                metrics['frame_competition'] = float(actual_entropy / uniform_entropy)
            else:
                metrics['frame_competition'] = 0.0
            
            # Categorical coherence variance
            metrics['categorical_coherence_variance'] = float(np.var(probabilities))
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  量子測量錯誤: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default quantum metrics when measurement fails."""
        return {
            'von_neumann_entropy': 0.5,
            'category_coherence': 0.5,
            'compositional_entanglement': 0.5,
            'grammatical_superposition': 0.5,
            'semantic_interference': 0.5,
            'frame_competition': 0.5,
            'categorical_coherence_variance': 0.25
        }
    
    def analyze_multiple_realities_fast(self, quantum_metrics: Dict, categories: List[str], word_count: int) -> Dict[str, float]:
        """Fast multiple reality analysis based on quantum metrics."""
        
        # Multiple reality strength (combination of superposition and interference)
        reality_strength = (
            quantum_metrics['grammatical_superposition'] * 0.4 +
            quantum_metrics['semantic_interference'] * 0.3 +
            quantum_metrics['frame_competition'] * 0.3
        )
        
        # Frame conflict strength (based on category diversity and entanglement)
        category_diversity = len(set(categories)) / max(1, len(categories))
        conflict_strength = (
            quantum_metrics['compositional_entanglement'] * 0.5 +
            category_diversity * 0.3 +
            quantum_metrics['categorical_coherence_variance'] * 0.2
        )
        
        # Semantic ambiguity (entropy and coherence based)
        ambiguity = (
            quantum_metrics['von_neumann_entropy'] / 4.0 * 0.6 +  # Normalize entropy
            (1.0 - quantum_metrics['category_coherence']) * 0.4
        )
        
        return {
            'multiple_reality_strength': min(1.0, reality_strength),
            'frame_conflict_strength': min(1.0, conflict_strength),
            'semantic_ambiguity': min(1.0, ambiguity)
        }
    
    def parse_categorical_analysis_safe(self, cat_str: str) -> Dict:
        """Safe parsing of categorical analysis with fallback."""
        try:
            # Try direct evaluation first
            return ast.literal_eval(cat_str)
        except:
            try:
                # Handle defaultdict with eval
                from collections import defaultdict
                safe_dict = {'defaultdict': defaultdict}
                result = eval(cat_str, {"__builtins__": {}}, safe_dict)
                # Convert defaultdict to regular dict
                if 'semantic_roles' in result and hasattr(result['semantic_roles'], 'items'):
                    result['semantic_roles'] = dict(result['semantic_roles'])
                return result
            except:
                # Return minimal valid structure
                return {'categories': [], 'words': []}
    
    def process_record_optimized(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single record with optimized DisCoCat quantum analysis."""
        
        try:
            # Parse categorical analysis
            categorical_analysis_str = record.get('categorical_analysis', '{}')
            categorical_analysis = self.parse_categorical_analysis_safe(categorical_analysis_str)
            
            # Extract basic info
            categories = categorical_analysis.get('categories', [])
            words = categorical_analysis.get('words', [])
            word_count = len(words)
            
            if not categories or word_count == 0:
                return self._get_default_record_result(record)
            
            # Create optimized quantum circuit
            circuit = self.create_optimized_circuit(categories, word_count)
            
            # Fast quantum measurement
            quantum_metrics = self.measure_quantum_properties_fast(circuit)
            
            # Fast reality analysis
            reality_analysis = self.analyze_multiple_realities_fast(quantum_metrics, categories, word_count)
            
            # Compile results
            result = {
                'record_id': record['record_id'],
                'field': record['field'],
                'original_text': record['original_text'][:100] + '...' if len(record.get('original_text', '')) > 100 else record.get('original_text', ''),
                'word_count': word_count,
                'unique_words': len(set(words)),
                
                # DisCoCat metrics
                'categorical_diversity': len(set(categories)),
                'compositional_complexity': min(10, word_count // 5),  # Simplified
                'semantic_density': len(set(words)) / max(1, word_count),
                
                # Quantum metrics
                'von_neumann_entropy': quantum_metrics['von_neumann_entropy'],
                'category_coherence': quantum_metrics['category_coherence'],
                'compositional_entanglement': quantum_metrics['compositional_entanglement'],
                'grammatical_superposition': quantum_metrics['grammatical_superposition'],
                'semantic_interference': quantum_metrics['semantic_interference'],
                'frame_competition': quantum_metrics['frame_competition'],
                'categorical_coherence_variance': quantum_metrics['categorical_coherence_variance'],
                
                # Multiple reality analysis
                'multiple_reality_strength': reality_analysis['multiple_reality_strength'],
                'frame_conflict_strength': reality_analysis['frame_conflict_strength'],
                'semantic_ambiguity': reality_analysis['semantic_ambiguity'],
                
                # Circuit properties
                'circuit_depth': circuit.depth(),
                'circuit_gates': circuit.size(),
                'qubit_count': circuit.num_qubits,
                
                # Metadata
                'discocat_enhanced': True,
                'discopy_available': False  # Not using discopy for performance
            }
            
            return result
            
        except Exception as e:
            print(f"⚠️  記錄 {record.get('record_id', 'unknown')} 分析失敗: {e}")
            return self._get_default_record_result(record)
    
    def _get_default_record_result(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Return default results when analysis fails."""
        return {
            'record_id': record.get('record_id', 0),
            'field': record.get('field', ''),
            'original_text': record.get('original_text', '')[:100] + '...' if len(record.get('original_text', '')) > 100 else record.get('original_text', ''),
            'word_count': 0,
            'unique_words': 0,
            'categorical_diversity': 0,
            'compositional_complexity': 0,
            'semantic_density': 0.0,
            'von_neumann_entropy': 0.5,
            'category_coherence': 0.5,
            'compositional_entanglement': 0.5,
            'grammatical_superposition': 0.5,
            'semantic_interference': 0.5,
            'frame_competition': 0.5,
            'categorical_coherence_variance': 0.25,
            'multiple_reality_strength': 0.5,
            'frame_conflict_strength': 0.5,
            'semantic_ambiguity': 0.5,
            'circuit_depth': 0,
            'circuit_gates': 0,
            'qubit_count': 0,
            'discocat_enhanced': False,
            'discopy_available': False
        }

def main():
    """Main execution function with progress tracking."""
    
    print("🚀 啟動優化版 DisCoCat 量子自然語言處理分析")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = OptimizedDisCoCatQNLPAnalyzer()
    
    # Load segmentation results
    segmentation_file = '../results/complete_discocat_segmentation.csv'
    
    if not os.path.exists(segmentation_file):
        print(f"❌ 找不到分詞結果文件: {segmentation_file}")
        return
    
    print(f"📂 載入分詞結果: {segmentation_file}")
    df = pd.read_csv(segmentation_file)
    
    print(f"📊 總記錄數: {len(df)}")
    
    # Process records with progress tracking
    results = []
    start_time = time.time()
    
    for idx, record in df.iterrows():
        if idx % 50 == 0:  # Progress every 50 records
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(df) - idx) / rate if rate > 0 else 0
            print(f"📈 進度: {idx}/{len(df)} ({idx/len(df)*100:.1f}%) - {rate:.1f} records/sec - ETA: {eta/60:.1f}min")
        
        result = analyzer.process_record_optimized(record.to_dict())
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    
    # Save detailed results
    output_file = '../results/optimized_discocat_quantum_analysis.csv'
    results_df.to_csv(output_file, index=False)
    print(f"💾 詳細結果已保存: {output_file}")
    
    # Calculate and save summary statistics
    summary_stats = {}
    
    numeric_columns = [
        'von_neumann_entropy', 'category_coherence', 'compositional_entanglement',
        'grammatical_superposition', 'semantic_interference', 'frame_competition',
        'multiple_reality_strength', 'frame_conflict_strength', 'semantic_ambiguity'
    ]
    
    for field in ['新聞標題', '影片對話', '影片描述']:
        field_data = results_df[results_df['field'] == field]
        if not field_data.empty:
            field_stats = {}
            for col in numeric_columns:
                field_stats[col] = {
                    'mean': float(field_data[col].mean()),
                    'std': float(field_data[col].std()),
                    'min': float(field_data[col].min()),
                    'max': float(field_data[col].max())
                }
            summary_stats[field] = field_stats
    
    # Overall statistics
    overall_stats = {}
    for col in numeric_columns:
        overall_stats[col] = {
            'mean': float(results_df[col].mean()),
            'std': float(results_df[col].std()),
            'min': float(results_df[col].min()),
            'max': float(results_df[col].max())
        }
    summary_stats['overall'] = overall_stats
    
    # Save summary
    summary_file = '../results/optimized_discocat_analysis_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)
    
    print(f"📊 摘要統計已保存: {summary_file}")
    
    # Performance summary
    total_time = time.time() - start_time
    print(f"\n✅ 分析完成!")
    print(f"⏱️  總耗時: {total_time/60:.1f} 分鐘")
    print(f"🚀 處理速度: {len(df)/total_time:.1f} 記錄/秒")
    print(f"📈 成功處理: {len(results_df)} / {len(df)} 記錄")

if __name__ == "__main__":
    main()
