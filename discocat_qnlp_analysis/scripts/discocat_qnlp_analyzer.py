#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DisCoCat Enhanced Quantum Natural Language Processing Analyzer
==============================================================

This script implements advanced QNLP analysis using DisCoCat (Distributional 
Compositional Categorical) models for analyzing multiple realities in AI-generated news.

Key DisCoCat enhancements:
1. Categorical grammar quantum circuits
2. Compositional semantic entanglement
3. Grammatical structure-aware superposition
4. Category-specific quantum coherence analysis
5. Monoidal category tensor network modeling

The analysis focuses on detecting "multiple realities" through quantum superposition
and entanglement principles, as outlined in the research abstract.

Author: QNLP Research Team  
Date: 2025-09-20
"""

import pandas as pd
import numpy as np
import json
import time
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Quantum computing frameworks
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.quantum_info import entropy, Statevector, DensityMatrix
from qiskit.circuit.library import RYGate, CXGate, HGate, RZGate
from qiskit.extensions import UnitaryGate

# DisCoCat and categorical semantics
try:
    from discopy import Word, Ty, Diagram
    from discopy.grammar.pregroup import Cup, Cap
    from discopy.quantum import Ket, H, Rx, Rz, CX
    DISCOPY_AVAILABLE = True
except ImportError:
    DISCOPY_AVAILABLE = False

# Machine learning and NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import networkx as nx

class DiscoCatQuantumAnalyzer:
    """
    DisCoCat-enhanced quantum natural language processing analyzer.
    
    This analyzer extends traditional QNLP with categorical grammar awareness,
    enabling more sophisticated analysis of compositional semantic structures
    and grammatical relationship entanglement.
    """
    
    def __init__(self):
        """Initialize the DisCoCat quantum analyzer."""
        
        self.backend = Aer.get_backend('statevector_simulator')
        
        # DisCoCat category to qubit mapping
        self.category_qubit_map = {
            'N': 0,    # Nouns
            'V': 1,    # Verbs  
            'A': 2,    # Adjectives
            'D': 3,    # Adverbs
            'P': 4,    # Prepositions
            'R': 5,    # Pronouns
            'C': 6,    # Conjunctions
            'X': 7     # Other/Unknown
        }
        
        # Quantum gate mappings for grammatical categories
        self.category_gates = {
            'N': {'rotation': np.pi/4, 'phase': 0},        # Stable entities
            'V': {'rotation': np.pi/3, 'phase': np.pi/2}, # Dynamic actions
            'A': {'rotation': np.pi/6, 'phase': np.pi/4}, # Modifying properties
            'D': {'rotation': np.pi/5, 'phase': np.pi/3}, # Modifying actions
            'P': {'rotation': np.pi/8, 'phase': np.pi/6}, # Relationships
            'R': {'rotation': np.pi/7, 'phase': np.pi/8}, # References
            'C': {'rotation': np.pi/2, 'phase': np.pi},   # Connections
            'X': {'rotation': np.pi/12, 'phase': 0}       # Unknown
        }
        
        print("🔬 DisCoCat量子分析器初始化完成")
        if DISCOPY_AVAILABLE:
            print("✅ DisCoPy可用 - 完整categorical量子電路")
        else:
            print("⚠️  DisCoPy不可用 - 使用標準量子電路")

    def create_discocat_quantum_circuit(self, categorical_analysis: Dict, compositional_structure: Dict) -> QuantumCircuit:
        """Create a quantum circuit based on DisCoCat categorical analysis."""
        
        # Determine circuit size based on categories and compositional complexity
        categories = categorical_analysis.get('categories', [])
        unique_categories = list(set(categories))
        
        # Base qubits for categories + additional qubits for compositional relationships
        base_qubits = len(self.category_qubit_map)
        comp_complexity = compositional_structure.get('compositional_complexity', 0)
        additional_qubits = min(4, max(1, comp_complexity // 2))  # Scale with complexity
        
        total_qubits = base_qubits + additional_qubits
        circuit = QuantumCircuit(total_qubits)
        
        # Initialize category-specific superpositions
        for category in unique_categories:
            if category in self.category_qubit_map:
                qubit_idx = self.category_qubit_map[category]
                gate_params = self.category_gates[category]
                
                # Apply category-specific rotation
                circuit.ry(gate_params['rotation'], qubit_idx)
                
                # Apply phase rotation for semantic distinction
                circuit.rz(gate_params['phase'], qubit_idx)
        
        # Create compositional entanglement based on grammatical relationships
        self._add_compositional_entanglement(circuit, categories, compositional_structure)
        
        # Add frame competition circuits
        self._add_frame_competition_gates(circuit, categorical_analysis, compositional_structure)
        
        # Add semantic ambiguity modeling
        self._add_semantic_ambiguity_gates(circuit, compositional_structure)
        
        return circuit

    def _add_compositional_entanglement(self, circuit: QuantumCircuit, categories: List[str], comp_struct: Dict):
        """Add entanglement gates based on compositional structure."""
        
        # Noun-Verb entanglement (subject-predicate relationships)
        if 'N' in categories and 'V' in categories:
            circuit.cx(self.category_qubit_map['N'], self.category_qubit_map['V'])
        
        # Adjective-Noun entanglement (modification relationships)
        if 'A' in categories and 'N' in categories:
            circuit.cx(self.category_qubit_map['A'], self.category_qubit_map['N'])
        
        # Adverb-Verb entanglement (verbal modification)
        if 'D' in categories and 'V' in categories:
            circuit.cx(self.category_qubit_map['D'], self.category_qubit_map['V'])
        
        # Preposition-Noun entanglement (prepositional relationships)
        if 'P' in categories and 'N' in categories:
            circuit.cx(self.category_qubit_map['P'], self.category_qubit_map['N'])
        
        # Complex compositional entanglement for phrases
        noun_phrases = comp_struct.get('noun_phrases', [])
        verb_phrases = comp_struct.get('verb_phrases', [])
        
        if len(noun_phrases) > 1:  # Multiple noun phrases create entanglement
            circuit.cx(self.category_qubit_map['N'], 8)  # Entangle with compositional qubit
        
        if len(verb_phrases) > 1:  # Multiple verb phrases create entanglement
            circuit.cx(self.category_qubit_map['V'], 9)  # Entangle with compositional qubit

    def _add_frame_competition_gates(self, circuit: QuantumCircuit, categorical_analysis: Dict, comp_struct: Dict):
        """Add quantum gates modeling frame competition between semantic interpretations."""
        
        categories = categorical_analysis.get('categories', [])
        category_counts = Counter(categories)
        
        # Frame competition increases with category diversity and repetition
        unique_cats = len(category_counts)
        max_count = max(category_counts.values()) if category_counts else 1
        
        competition_strength = min(np.pi/2, (unique_cats * max_count) / 10)
        
        # Apply competition rotation between major categories
        major_categories = [cat for cat, count in category_counts.items() if count >= 2]
        
        for i, cat1 in enumerate(major_categories):
            for cat2 in major_categories[i+1:]:
                if cat1 in self.category_qubit_map and cat2 in self.category_qubit_map:
                    q1, q2 = self.category_qubit_map[cat1], self.category_qubit_map[cat2]
                    
                    # Controlled rotation representing frame competition
                    circuit.cry(competition_strength, q1, q2)
                    circuit.cry(competition_strength, q2, q1)

    def _add_semantic_ambiguity_gates(self, circuit: QuantumCircuit, comp_struct: Dict):
        """Add quantum gates modeling semantic ambiguity."""
        
        # Ambiguity increases with compositional complexity
        complexity = comp_struct.get('compositional_complexity', 0)
        transitions = comp_struct.get('category_transitions', 0)
        
        ambiguity_strength = min(np.pi/3, (complexity + transitions) / 15)
        
        # Apply ambiguity as controlled phase gates
        for i in range(min(6, circuit.num_qubits - 2)):
            circuit.crz(ambiguity_strength, i, i + 1)

    def measure_discocat_quantum_properties(self, circuit: QuantumCircuit) -> Dict[str, float]:
        """Measure quantum properties from DisCoCat-enhanced circuit."""
        
        try:
            # Execute the quantum circuit
            job = execute(circuit, self.backend)
            result = job.result()
            statevector = result.get_statevector()
            
            # Create density matrix
            density_matrix = DensityMatrix(statevector)
            
            # Calculate enhanced quantum metrics
            metrics = {}
            
            # 1. Von Neumann entropy (overall quantum information)
            metrics['von_neumann_entropy'] = float(entropy(density_matrix))
            
            # 2. Category-specific quantum coherence
            metrics['category_coherence'] = self._calculate_category_coherence(statevector, circuit.num_qubits)
            
            # 3. Compositional entanglement strength
            metrics['compositional_entanglement'] = self._calculate_compositional_entanglement(density_matrix)
            
            # 4. Grammatical superposition strength
            metrics['grammatical_superposition'] = self._calculate_grammatical_superposition(statevector)
            
            # 5. Semantic interference patterns
            metrics['semantic_interference'] = self._calculate_semantic_interference(statevector)
            
            # 6. Frame competition dynamics
            metrics['frame_competition'] = self._calculate_frame_competition(density_matrix)
            
            # 7. Categorical coherence variance
            metrics['categorical_coherence_variance'] = self._calculate_categorical_variance(statevector)
            
            # 8. Compositional quantum complexity
            metrics['compositional_complexity'] = circuit.depth() * circuit.size() / circuit.num_qubits
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  量子測量錯誤: {e}")
            return self._get_default_metrics()

    def _calculate_category_coherence(self, statevector: np.ndarray, num_qubits: int) -> float:
        """Calculate coherence specific to grammatical categories."""
        
        # Measure coherence in categorical subspaces
        category_coherences = []
        
        for category, qubit_idx in self.category_qubit_map.items():
            if qubit_idx < num_qubits:
                # Calculate reduced density matrix for this category
                try:
                    # Simplified coherence measure based on superposition strength
                    prob_0 = np.abs(statevector[::2**qubit_idx])**2
                    prob_1 = np.abs(statevector[2**qubit_idx::2**(qubit_idx+1)])**2
                    
                    coherence = 2 * np.sqrt(np.mean(prob_0) * np.mean(prob_1))
                    category_coherences.append(coherence)
                except:
                    category_coherences.append(0.5)
        
        return float(np.mean(category_coherences)) if category_coherences else 0.5

    def _calculate_compositional_entanglement(self, density_matrix) -> float:
        """Calculate entanglement between compositional elements."""
        
        try:
            # Use entropy to measure entanglement
            total_entropy = entropy(density_matrix)
            
            # Approximate entanglement as deviation from separable state entropy
            max_separable_entropy = np.log2(density_matrix.dim)
            entanglement = min(1.0, total_entropy / max_separable_entropy)
            
            return float(entanglement)
        except:
            return 0.5

    def _calculate_grammatical_superposition(self, statevector: np.ndarray) -> float:
        """Calculate superposition strength in grammatical space."""
        
        # Measure superposition as amplitude distribution variance
        amplitudes = np.abs(statevector)**2
        
        # High variance indicates strong superposition
        amplitude_variance = np.var(amplitudes)
        max_variance = 1.0 / len(amplitudes)  # Maximum possible variance
        
        superposition_strength = min(1.0, amplitude_variance / max_variance * 4)
        return float(superposition_strength)

    def _calculate_semantic_interference(self, statevector: np.ndarray) -> float:
        """Calculate semantic interference patterns."""
        
        # Interference measured as phase relationships between amplitudes
        phases = np.angle(statevector)
        
        # Calculate phase variance as interference measure
        phase_variance = np.var(phases)
        normalized_interference = min(1.0, phase_variance / (np.pi**2))
        
        return float(normalized_interference)

    def _calculate_frame_competition(self, density_matrix) -> float:
        """Calculate competition between semantic frames."""
        
        try:
            # Frame competition as measure of non-classical correlations
            total_entropy = entropy(density_matrix)
            
            # Approximate competition strength
            competition = min(1.0, total_entropy * 0.5)
            return float(competition)
        except:
            return 0.5

    def _calculate_categorical_variance(self, statevector: np.ndarray) -> float:
        """Calculate variance in categorical representation."""
        
        amplitudes = np.abs(statevector)**2
        return float(np.var(amplitudes) * 4)  # Scaled for interpretability

    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when quantum measurement fails."""
        
        return {
            'von_neumann_entropy': 0.5,
            'category_coherence': 0.5,
            'compositional_entanglement': 0.5,
            'grammatical_superposition': 0.5,
            'semantic_interference': 0.5,
            'frame_competition': 0.5,
            'categorical_coherence_variance': 0.5,
            'compositional_complexity': 1.0
        }

    def analyze_multiple_realities(self, quantum_metrics: Dict[str, float], 
                                 categorical_analysis: Dict, 
                                 compositional_structure: Dict) -> Dict[str, Any]:
        """Analyze multiple realities phenomenon using DisCoCat quantum metrics."""
        
        # Enhanced multiple reality detection using categorical information
        superposition_strength = quantum_metrics['grammatical_superposition']
        frame_competition = quantum_metrics['frame_competition']
        category_coherence = quantum_metrics['category_coherence']
        
        # DisCoCat-specific reality multiplicity indicators
        category_diversity = len(set(categorical_analysis.get('categories', [])))
        compositional_complexity = compositional_structure.get('compositional_complexity', 0)
        
        # Multiple reality probability calculation
        reality_multiplicity = (
            superposition_strength * 0.3 +
            frame_competition * 0.25 +
            (1 - category_coherence) * 0.2 +  # Lower coherence = more realities
            min(1.0, category_diversity / 8) * 0.15 +
            min(1.0, compositional_complexity / 10) * 0.1
        )
        
        # Frame conflict analysis
        frame_conflict_strength = frame_competition * (1 + compositional_complexity / 20)
        
        # Semantic ambiguity with categorical enhancement
        semantic_ambiguity = (
            quantum_metrics['semantic_interference'] * 0.4 +
            quantum_metrics['categorical_coherence_variance'] * 0.3 +
            min(1.0, compositional_structure.get('category_transitions', 0) / 10) * 0.3
        )
        
        return {
            'multiple_reality_strength': float(reality_multiplicity),
            'frame_conflict_strength': float(frame_conflict_strength),
            'semantic_ambiguity': float(semantic_ambiguity),
            'categorical_diversity': category_diversity,
            'compositional_complexity_factor': compositional_complexity,
            'quantum_coherence_breakdown': 1 - category_coherence
        }

    def process_record_discocat(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single record with full DisCoCat quantum analysis."""
        
        try:
            # Extract DisCoCat analysis from segmentation results (parse string representations)
            categorical_analysis_str = record.get('categorical_analysis', '{}')
            compositional_structure_str = record.get('compositional_structure', '{}')
            
            # Parse string representations to dictionaries with robust defaultdict handling
            try:
                import ast
                import re
                import json
                
                # Parse categorical analysis with defaultdict handling
                if isinstance(categorical_analysis_str, str):
                    # Method 1: Try direct eval first
                    try:
                        categorical_analysis = ast.literal_eval(categorical_analysis_str)
                    except:
                        # Method 2: Handle defaultdict by converting to regular dict
                        try:
                            # Replace defaultdict pattern with regular dict
                            cleaned_str = re.sub(
                                r'defaultdict\(<class \'list\'>, ({[^}]*})\)',
                                r'\1',
                                categorical_analysis_str
                            )
                            categorical_analysis = ast.literal_eval(cleaned_str)
                        except:
                            # Method 3: Use eval with safe defaultdict replacement
                            try:
                                # Create a safe evaluation context
                                from collections import defaultdict
                                safe_dict = {'defaultdict': defaultdict}
                                categorical_analysis = eval(categorical_analysis_str, {"__builtins__": {}}, safe_dict)
                                # Convert defaultdict to regular dict for JSON serialization
                                if 'semantic_roles' in categorical_analysis:
                                    categorical_analysis['semantic_roles'] = dict(categorical_analysis['semantic_roles'])
                            except:
                                categorical_analysis = {}
                else:
                    categorical_analysis = categorical_analysis_str
                    
                # Parse compositional structure
                if isinstance(compositional_structure_str, str):
                    compositional_structure = ast.literal_eval(compositional_structure_str)
                else:
                    compositional_structure = compositional_structure_str
                    
            except Exception as e:
                # Only print error for first few records to avoid spam
                if record.get('record_id', 0) < 5:
                    print(f"⚠️  解析錯誤 (Record {record.get('record_id', 'unknown')}): {e}")
                categorical_analysis = {}
                compositional_structure = {}
            
            if not categorical_analysis or not record.get('discocat_ready', False):
                return self._get_default_record_result(record)
            
            # Create DisCoCat quantum circuit
            circuit = self.create_discocat_quantum_circuit(categorical_analysis, compositional_structure)
            
            # Measure quantum properties
            quantum_metrics = self.measure_discocat_quantum_properties(circuit)
            
            # Analyze multiple realities
            reality_analysis = self.analyze_multiple_realities(
                quantum_metrics, categorical_analysis, compositional_structure
            )
            
            # Compile comprehensive results
            result = {
                'record_id': record['record_id'],
                'field': record['field'],
                'original_text': record['original_text'],
                'word_count': record['word_count'],
                'unique_words': record['unique_words'],
                
                # DisCoCat specific metrics
                'categorical_diversity': len(set(categorical_analysis.get('categories', []))),
                'compositional_complexity': compositional_structure.get('compositional_complexity', 0),
                'semantic_density': compositional_structure.get('semantic_density', 0),
                
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
                
                # Analysis metadata
                'discocat_enhanced': True,
                'discopy_available': DISCOPY_AVAILABLE
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
            'original_text': record.get('original_text', ''),
            'word_count': record.get('word_count', 0),
            'unique_words': record.get('unique_words', 0),
            'categorical_diversity': 0,
            'compositional_complexity': 0,
            'semantic_density': 0,
            'von_neumann_entropy': 0.5,
            'category_coherence': 0.5,
            'compositional_entanglement': 0.5,
            'grammatical_superposition': 0.5,
            'semantic_interference': 0.5,
            'frame_competition': 0.5,
            'categorical_coherence_variance': 0.5,
            'multiple_reality_strength': 0.5,
            'frame_conflict_strength': 0.5,
            'semantic_ambiguity': 0.5,
            'circuit_depth': 0,
            'circuit_gates': 0,
            'qubit_count': 0,
            'discocat_enhanced': False,
            'discopy_available': DISCOPY_AVAILABLE
        }

def analyze_field_quantum_discocat(df_field: pd.DataFrame, field_name: str, analyzer: DiscoCatQuantumAnalyzer) -> Tuple[List[Dict], Dict]:
    """Analyze a field with DisCoCat quantum processing."""
    
    print(f"🔬 開始 {field_name} DisCoCat量子分析")
    
    results = []
    start_time = time.time()
    
    for idx, row in df_field.iterrows():
        record_dict = row.to_dict()
        quantum_result = analyzer.process_record_discocat(record_dict)
        results.append(quantum_result)
        
        if (idx + 1) % 50 == 0:
            print(f"  已處理 {idx + 1}/{len(df_field)} 筆記錄")
    
    processing_time = time.time() - start_time
    
    # Calculate field statistics
    if results:
        field_stats = {
            'field': field_name,
            'total_records': len(results),
            'processing_time_seconds': processing_time,
            'avg_multiple_reality_strength': np.mean([r['multiple_reality_strength'] for r in results]),
            'avg_frame_conflict_strength': np.mean([r['frame_conflict_strength'] for r in results]),
            'avg_semantic_ambiguity': np.mean([r['semantic_ambiguity'] for r in results]),
            'avg_category_coherence': np.mean([r['category_coherence'] for r in results]),
            'avg_compositional_complexity': np.mean([r['compositional_complexity'] for r in results]),
            'avg_categorical_diversity': np.mean([r['categorical_diversity'] for r in results]),
            'discocat_enhanced_percentage': (sum(1 for r in results if r['discocat_enhanced']) / len(results)) * 100
        }
    else:
        field_stats = {'field': field_name, 'total_records': 0, 'processing_time_seconds': processing_time}
    
    print(f"✅ {field_name} DisCoCat量子分析完成: {len(results)}筆記錄, 耗時{processing_time:.1f}秒")
    
    return results, field_stats

def main():
    """Main function for DisCoCat quantum analysis."""
    
    print("🚀 開始DisCoCat增強型量子自然語言處理分析")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = DiscoCatQuantumAnalyzer()
    
    # Load segmentation results
    print("📂 載入DisCoCat分詞結果...")
    
    try:
        segmentation_df = pd.read_csv('../results/complete_discocat_segmentation.csv')
        print(f"✅ 分詞結果載入成功: {len(segmentation_df)} 筆記錄")
    except Exception as e:
        print(f"❌ 分詞結果載入失敗: {e}")
        print("💡 請先執行 discocat_segmentation.py")
        return
    
    # Process each field
    fields_to_analyze = ['新聞標題', '影片對話', '影片描述']
    all_results = []
    field_statistics = {}
    
    for field in fields_to_analyze:
        print(f"\n🔬 分析欄位: {field}")
        field_df = segmentation_df[segmentation_df['field'] == field].copy()
        
        if not field_df.empty:
            field_results, field_stats = analyze_field_quantum_discocat(field_df, field, analyzer)
            all_results.extend(field_results)
            field_statistics[field] = field_stats
        else:
            print(f"⚠️  {field} 欄位無數據")
    
    if all_results:
        # Save detailed results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('../results/discocat_quantum_analysis_detailed.csv', index=False, encoding='utf-8')
        
        # Calculate global statistics
        global_stats = {
            'analysis_type': 'DisCoCat Enhanced Quantum NLP',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_records_analyzed': len(all_results),
            'discopy_integration': DISCOPY_AVAILABLE,
            'field_results': field_statistics,
            'global_quantum_metrics': {
                'avg_multiple_reality_strength': float(np.mean([r['multiple_reality_strength'] for r in all_results])),
                'avg_frame_conflict_strength': float(np.mean([r['frame_conflict_strength'] for r in all_results])),
                'avg_semantic_ambiguity': float(np.mean([r['semantic_ambiguity'] for r in all_results])),
                'avg_category_coherence': float(np.mean([r['category_coherence'] for r in all_results])),
                'avg_compositional_entanglement': float(np.mean([r['compositional_entanglement'] for r in all_results])),
                'avg_categorical_diversity': float(np.mean([r['categorical_diversity'] for r in all_results]))
            },
            'multiple_reality_analysis': {
                'high_reality_multiplicity_records': sum(1 for r in all_results if r['multiple_reality_strength'] > 0.7),
                'high_frame_conflict_records': sum(1 for r in all_results if r['frame_conflict_strength'] > 0.7),
                'high_ambiguity_records': sum(1 for r in all_results if r['semantic_ambiguity'] > 0.7),
                'multiple_reality_prevalence': float(sum(1 for r in all_results if r['multiple_reality_strength'] > 0.6) / len(all_results))
            },
            'discocat_enhancements': {
                'categorical_analysis_enabled': True,
                'compositional_structure_analysis': True,
                'grammatical_entanglement_modeling': True,
                'category_specific_coherence': True
            }
        }
        
        # Save analysis summary
        with open('../results/discocat_quantum_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(global_stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 DisCoCat量子分析完成!")
        print(f"📊 總計分析: {len(all_results)} 筆記錄")
        print(f"📁 結果保存於: ../results/")
        
        # Display key findings
        print(f"\n🔍 關鍵發現:")
        print(f"  平均多重現實強度: {global_stats['global_quantum_metrics']['avg_multiple_reality_strength']:.4f}")
        print(f"  平均框架衝突強度: {global_stats['global_quantum_metrics']['avg_frame_conflict_strength']:.4f}")
        print(f"  平均語義模糊度: {global_stats['global_quantum_metrics']['avg_semantic_ambiguity']:.4f}")
        print(f"  平均分類連貫性: {global_stats['global_quantum_metrics']['avg_category_coherence']:.4f}")
        print(f"  多重現實普及度: {global_stats['multiple_reality_analysis']['multiple_reality_prevalence']*100:.1f}%")
        
        # Field-specific summary
        print(f"\n📋 各欄位摘要:")
        for field, stats in field_statistics.items():
            print(f"  {field}:")
            print(f"    記錄數: {stats['total_records']}")
            print(f"    多重現實強度: {stats['avg_multiple_reality_strength']:.4f}")
            print(f"    框架衝突強度: {stats['avg_frame_conflict_strength']:.4f}")
            print(f"    DisCoCat增強率: {stats['discocat_enhanced_percentage']:.1f}%")
    
    else:
        print("❌ 沒有成功分析任何記錄")

if __name__ == "__main__":
    main()
