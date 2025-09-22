#!/usr/bin/env python3
"""
量子自然語言處理分析器
Quantum Natural Language Processing Analyzer

基於IBM Qiskit實現真實的量子自然語言處理，分析AI生成新聞中的「多重現實」現象。
Real quantum NLP implementation using IBM Qiskit to analyze "multiple realities" in AI-generated news.
"""

import pandas as pd
import numpy as np
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.quantum_info import entropy, Statevector, DensityMatrix
from qiskit.circuit.library import RYGate, CXGate, HGate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import Counter
import json
import time
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class QuantumNewsAnalyzer:
    """量子新聞分析器 - 實現QNLP分析AI生成新聞的多重現實現象"""
    
    def __init__(self, max_qubits: int = 8):
        """
        初始化量子分析器
        
        Args:
            max_qubits: 最大量子比特數
        """
        self.max_qubits = max_qubits
        self.backend = Aer.get_backend('statevector_simulator')
        self.qasm_backend = Aer.get_backend('qasm_simulator')
        
        # 量子分析結果存儲
        self.quantum_states = {}
        self.narrative_circuits = {}
        self.analysis_results = {}
        
        print(f"✅ 量子分析器初始化完成 (最大 {max_qubits} 量子比特)")
    
    def create_narrative_quantum_circuit(self, words: List[str], field_type: str) -> Tuple[QuantumCircuit, int]:
        """
        創建敘事量子電路 - 將文本編碼為量子態
        
        Args:
            words: 分詞結果
            field_type: 欄位類型 (新聞標題/影片對話/影片描述)
            
        Returns:
            Tuple[QuantumCircuit, int]: 量子電路和量子比特數
        """
        if not words:
            return None, 0
        
        # 限制量子比特數
        n_qubits = min(len(words), self.max_qubits)
        qc = QuantumCircuit(n_qubits, name=f"narrative_{field_type}")
        
        # 1. 初始化疊加態 - 模擬敘事的多重可能性
        for i in range(n_qubits):
            qc.h(i)
        
        # 2. 基於詞頻和語義權重的旋轉 - 編碼語義強度
        word_weights = self._calculate_semantic_weights(words[:n_qubits])
        
        for i, (word, weight) in enumerate(zip(words[:n_qubits], word_weights)):
            # 使用TF-IDF權重調制旋轉角度
            theta = weight * np.pi + np.pi/4
            phi = len(word) / 10 * np.pi  # 詞長影響相位
            
            qc.ry(theta, i)
            qc.rz(phi, i)
        
        # 3. 創建語義糾纏 - 模擬詞彙間的語義關聯
        self._create_semantic_entanglement(qc, n_qubits, words[:n_qubits])
        
        # 4. 根據欄位類型添加特定的量子操作
        self._add_field_specific_operations(qc, n_qubits, field_type)
        
        return qc, n_qubits
    
    def _calculate_semantic_weights(self, words: List[str]) -> List[float]:
        """計算語義權重"""
        # 使用詞頻和詞長計算權重
        word_freq = Counter(words)
        max_freq = max(word_freq.values()) if word_freq else 1
        
        weights = []
        for word in words:
            # 結合頻率和長度的權重
            freq_weight = word_freq[word] / max_freq
            length_weight = min(len(word) / 10, 1.0)
            semantic_weight = (freq_weight + length_weight) / 2
            weights.append(semantic_weight)
        
        return weights
    
    def _create_semantic_entanglement(self, qc: QuantumCircuit, n_qubits: int, words: List[str]):
        """創建語義糾纏 - 模擬詞彙間的語義關聯"""
        # 基於詞彙相似性創建糾纏
        for i in range(n_qubits - 1):
            for j in range(i + 1, n_qubits):
                # 計算詞彙相似性
                similarity = self._calculate_word_similarity(words[i], words[j])
                
                if similarity > 0.3:  # 相似性閾值
                    # 創建受控旋轉糾纏
                    angle = similarity * np.pi / 2
                    qc.cry(angle, i, j)
                    
                    # 添加相位糾纏
                    if similarity > 0.6:
                        qc.cz(i, j)
    
    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """計算詞彙相似性"""
        # 簡單的字符重疊相似性
        set1, set2 = set(word1), set(word2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0
        
        return intersection / union
    
    def _add_field_specific_operations(self, qc: QuantumCircuit, n_qubits: int, field_type: str):
        """根據欄位類型添加特定的量子操作"""
        if field_type == "新聞標題":
            # 新聞標題通常簡潔，添加更多的相位操作
            for i in range(n_qubits):
                qc.s(i)  # S門增加相位
                
        elif field_type == "影片對話":
            # 對話具有互動性，添加更多糾纏
            for i in range(0, n_qubits - 1, 2):
                if i + 1 < n_qubits:
                    qc.cx(i, i + 1)
                    
        elif field_type == "影片描述":
            # 描述性文本，添加T門增加複雜性
            for i in range(n_qubits):
                qc.t(i)
    
    def measure_quantum_narrative_properties(self, qc: QuantumCircuit, n_qubits: int, 
                                           words: List[str], field_type: str) -> Dict:
        """
        測量量子敘事特性 - 分析敘事的量子特徵
        
        Args:
            qc: 量子電路
            n_qubits: 量子比特數
            words: 詞彙列表
            field_type: 欄位類型
            
        Returns:
            Dict: 量子特性測量結果
        """
        if qc is None or n_qubits == 0:
            return self._empty_quantum_result()
        
        try:
            # 執行量子模擬
            job = execute(qc, self.backend)
            result = job.result()
            statevector = result.get_statevector()
            
            # 計算量子指標
            quantum_metrics = self._calculate_quantum_metrics(statevector, n_qubits)
            
            # 計算敘事特異性指標
            narrative_metrics = self._calculate_narrative_metrics(statevector, words, field_type)
            
            # 計算框架衝突指標
            framing_metrics = self._calculate_framing_conflict_metrics(statevector, words)
            
            # 綜合結果
            return {
                **quantum_metrics,
                **narrative_metrics,
                **framing_metrics,
                'circuit_depth': qc.depth(),
                'gate_count': len(qc.data),
                'field_type': field_type,
                'word_count': len(words),
                'qubit_count': n_qubits
            }
            
        except Exception as e:
            print(f"量子測量錯誤: {e}")
            return self._empty_quantum_result()
    
    def _calculate_quantum_metrics(self, statevector: Statevector, n_qubits: int) -> Dict:
        """計算基礎量子指標"""
        # von Neumann熵 - 敘事複雜度
        vn_entropy = entropy(statevector)
        
        # 量子連貫性 - 語義一致性
        amplitudes = np.abs(statevector.data)
        coherence = 1 - np.sum(amplitudes**4)  # 參與比
        
        # 量子糾纏度 - 語義關聯強度
        if n_qubits > 1:
            # 計算雙分割糾纏熵
            mid = n_qubits // 2
            entanglement = self._calculate_bipartite_entanglement(statevector, mid, n_qubits - mid)
        else:
            entanglement = 0
        
        # 量子干涉 - 敘事一致性
        phases = np.angle(statevector.data)
        phase_variance = np.var(phases)
        interference = 1 - (phase_variance / (np.pi**2)) if phase_variance > 0 else 1
        
        # 疊加強度 - 多重現實程度
        prob_dist = amplitudes**2
        superposition_strength = 1 - np.max(prob_dist)
        
        return {
            'von_neumann_entropy': float(vn_entropy),
            'quantum_coherence': float(coherence),
            'quantum_entanglement': float(entanglement),
            'quantum_interference': float(interference),
            'superposition_strength': float(superposition_strength)
        }
    
    def _calculate_bipartite_entanglement(self, statevector: Statevector, 
                                        subsystem_a_size: int, subsystem_b_size: int) -> float:
        """計算雙分割糾纏度"""
        try:
            # 創建密度矩陣
            rho = DensityMatrix(statevector)
            
            # 計算約化密度矩陣的熵
            subsystem_indices = list(range(subsystem_a_size))
            rho_a = rho.partial_trace(subsystem_indices)
            
            # 計算糾纏熵
            entanglement_entropy = entropy(rho_a)
            
            return float(entanglement_entropy)
            
        except:
            return 0.0
    
    def _calculate_narrative_metrics(self, statevector: Statevector, 
                                   words: List[str], field_type: str) -> Dict:
        """計算敘事特異性指標"""
        # 敘事分歧度 - 基於狀態分布的分散程度
        amplitudes = np.abs(statevector.data)
        prob_dist = amplitudes**2
        narrative_divergence = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
        
        # 語義密度 - 詞彙語義信息密度
        unique_words = len(set(words))
        total_words = len(words)
        semantic_density = unique_words / total_words if total_words > 0 else 0
        
        # 框架穩定性 - 基於相位一致性
        phases = np.angle(statevector.data)
        phase_coherence = 1 - np.std(phases) / np.pi
        
        # 敘事張力 - 基於幅度方差
        amplitude_tension = np.var(amplitudes)
        
        return {
            'narrative_divergence': float(narrative_divergence),
            'semantic_density': float(semantic_density),
            'frame_stability': float(phase_coherence),
            'narrative_tension': float(amplitude_tension)
        }
    
    def _calculate_framing_conflict_metrics(self, statevector: Statevector, words: List[str]) -> Dict:
        """計算框架衝突指標"""
        amplitudes = np.abs(statevector.data)
        
        # 框架競爭度 - 基於狀態競爭
        prob_dist = amplitudes**2
        max_prob = np.max(prob_dist)
        frame_competition = 1 - max_prob
        
        # 意義衝突度 - 基於幅度分布的不均勻性
        gini_coefficient = self._calculate_gini_coefficient(prob_dist)
        meaning_conflict = 1 - gini_coefficient
        
        # 語義模糊度 - 基於詞彙多樣性和量子不確定性
        word_diversity = len(set(words)) / len(words) if words else 0
        quantum_uncertainty = entropy(statevector)
        semantic_ambiguity = (word_diversity + quantum_uncertainty / np.log(len(amplitudes))) / 2
        
        return {
            'frame_competition': float(frame_competition),
            'meaning_conflict': float(meaning_conflict),
            'semantic_ambiguity': float(semantic_ambiguity)
        }
    
    def _calculate_gini_coefficient(self, prob_dist: np.ndarray) -> float:
        """計算基尼係數"""
        sorted_probs = np.sort(prob_dist)
        n = len(sorted_probs)
        
        if n == 0:
            return 0
        
        cumsum = np.cumsum(sorted_probs)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _empty_quantum_result(self) -> Dict:
        """返回空的量子結果"""
        return {
            'von_neumann_entropy': 0,
            'quantum_coherence': 0,
            'quantum_entanglement': 0,
            'quantum_interference': 0,
            'superposition_strength': 0,
            'narrative_divergence': 0,
            'semantic_density': 0,
            'frame_stability': 0,
            'narrative_tension': 0,
            'frame_competition': 0,
            'meaning_conflict': 0,
            'semantic_ambiguity': 0,
            'circuit_depth': 0,
            'gate_count': 0,
            'field_type': '',
            'word_count': 0,
            'qubit_count': 0
        }
    
    def analyze_news_dataset(self, segmentation_results: pd.DataFrame) -> Dict:
        """
        分析整個新聞數據集的量子特性
        
        Args:
            segmentation_results: 斷詞結果數據框
            
        Returns:
            Dict: 完整的量子分析結果
        """
        print("🔬 開始量子自然語言處理分析")
        print("=" * 60)
        
        start_time = time.time()
        
        # 按欄位分組分析
        field_results = {}
        all_quantum_results = []
        
        for field_type in segmentation_results['field'].unique():
            print(f"\n📊 分析 {field_type} 欄位的量子特性...")
            
            field_data = segmentation_results[segmentation_results['field'] == field_type]
            field_quantum_results = []
            
            for idx, row in field_data.iterrows():
                if pd.notna(row['words_list']) and row['words_list'].strip():
                    # 解析詞彙
                    words = [w.strip() for w in str(row['words_list']).split(',') if w.strip()]
                    
                    if words:
                        # 創建量子電路
                        qc, n_qubits = self.create_narrative_quantum_circuit(words, field_type)
                        
                        if qc is not None:
                            # 測量量子特性
                            quantum_props = self.measure_quantum_narrative_properties(
                                qc, n_qubits, words, field_type
                            )
                            
                            # 添加記錄信息
                            quantum_result = {
                                'record_id': row['record_id'],
                                'field': field_type,
                                'original_text': row['original_text'][:100] + '...' if len(row['original_text']) > 100 else row['original_text'],
                                **quantum_props
                            }
                            
                            field_quantum_results.append(quantum_result)
                            all_quantum_results.append(quantum_result)
                
                # 顯示進度
                if (len(field_quantum_results) + 1) % 50 == 0:
                    print(f"  處理進度: {len(field_quantum_results)}/{len(field_data)}")
            
            # 計算欄位統計
            if field_quantum_results:
                field_stats = self._calculate_field_quantum_statistics(field_quantum_results, field_type)
                field_results[field_type] = {
                    'statistics': field_stats,
                    'sample_count': len(field_quantum_results),
                    'individual_results': field_quantum_results
                }
                
                print(f"✅ {field_type} 量子分析完成: {len(field_quantum_results)} 筆記錄")
                self._print_field_quantum_summary(field_stats, field_type)
        
        # 計算跨欄位比較
        cross_field_analysis = self._calculate_cross_field_analysis(field_results)
        
        # 生成多重現實分析
        multiple_reality_analysis = self._analyze_multiple_realities(all_quantum_results)
        
        # 綜合結果
        complete_analysis = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'processing_time_seconds': time.time() - start_time,
            'total_records_analyzed': len(all_quantum_results),
            'field_results': field_results,
            'cross_field_analysis': cross_field_analysis,
            'multiple_reality_analysis': multiple_reality_analysis,
            'quantum_framework': 'IBM Qiskit',
            'max_qubits_used': self.max_qubits
        }
        
        print(f"\n🎉 量子自然語言處理分析完成!")
        print(f"⏱️  總處理時間: {time.time() - start_time:.1f} 秒")
        print(f"📊 分析記錄數: {len(all_quantum_results)}")
        
        return complete_analysis
    
    def _calculate_field_quantum_statistics(self, field_results: List[Dict], field_type: str) -> Dict:
        """計算欄位量子統計"""
        if not field_results:
            return {}
        
        # 提取各項指標
        metrics = [
            'von_neumann_entropy', 'quantum_coherence', 'quantum_entanglement',
            'quantum_interference', 'superposition_strength', 'narrative_divergence',
            'semantic_density', 'frame_stability', 'narrative_tension',
            'frame_competition', 'meaning_conflict', 'semantic_ambiguity'
        ]
        
        stats = {}
        for metric in metrics:
            values = [r[metric] for r in field_results if metric in r]
            if values:
                stats[f'avg_{metric}'] = np.mean(values)
                stats[f'std_{metric}'] = np.std(values)
                stats[f'min_{metric}'] = np.min(values)
                stats[f'max_{metric}'] = np.max(values)
        
        # 特殊統計
        stats['avg_circuit_depth'] = np.mean([r['circuit_depth'] for r in field_results])
        stats['avg_gate_count'] = np.mean([r['gate_count'] for r in field_results])
        stats['avg_qubit_count'] = np.mean([r['qubit_count'] for r in field_results])
        
        return stats
    
    def _print_field_quantum_summary(self, stats: Dict, field_type: str):
        """打印欄位量子摘要"""
        print(f"  📈 {field_type} 量子指標摘要:")
        print(f"    敘事複雜度 (von Neumann熵): {stats.get('avg_von_neumann_entropy', 0):.4f}")
        print(f"    語義一致性 (量子連貫性): {stats.get('avg_quantum_coherence', 0):.4f}")
        print(f"    多重現實程度 (疊加強度): {stats.get('avg_superposition_strength', 0):.4f}")
        print(f"    框架競爭度: {stats.get('avg_frame_competition', 0):.4f}")
        print(f"    語義模糊度: {stats.get('avg_semantic_ambiguity', 0):.4f}")
    
    def _calculate_cross_field_analysis(self, field_results: Dict) -> Dict:
        """計算跨欄位分析"""
        cross_analysis = {}
        
        fields = list(field_results.keys())
        
        # 欄位間比較
        for i, field1 in enumerate(fields):
            for j, field2 in enumerate(fields[i+1:], i+1):
                stats1 = field_results[field1]['statistics']
                stats2 = field_results[field2]['statistics']
                
                comparison_key = f"{field1}_vs_{field2}"
                cross_analysis[comparison_key] = {}
                
                # 比較主要指標
                key_metrics = [
                    'avg_von_neumann_entropy', 'avg_quantum_coherence', 
                    'avg_superposition_strength', 'avg_frame_competition'
                ]
                
                for metric in key_metrics:
                    if metric in stats1 and metric in stats2:
                        diff = stats2[metric] - stats1[metric]
                        cross_analysis[comparison_key][f'{metric}_difference'] = diff
        
        return cross_analysis
    
    def _analyze_multiple_realities(self, all_results: List[Dict]) -> Dict:
        """分析多重現實現象"""
        if not all_results:
            return {}
        
        # 高疊加強度記錄 - 多重現實現象明顯
        high_superposition = [r for r in all_results if r['superposition_strength'] > 0.5]
        
        # 高框架競爭記錄 - 框架衝突明顯
        high_frame_competition = [r for r in all_results if r['frame_competition'] > 0.6]
        
        # 高語義模糊記錄 - 意義不確定性高
        high_ambiguity = [r for r in all_results if r['semantic_ambiguity'] > 0.7]
        
        # 計算多重現實指標
        avg_superposition = np.mean([r['superposition_strength'] for r in all_results])
        avg_frame_competition = np.mean([r['frame_competition'] for r in all_results])
        avg_meaning_conflict = np.mean([r['meaning_conflict'] for r in all_results])
        
        return {
            'multiple_reality_prevalence': len(high_superposition) / len(all_results),
            'frame_conflict_prevalence': len(high_frame_competition) / len(all_results),
            'semantic_ambiguity_prevalence': len(high_ambiguity) / len(all_results),
            'avg_multiple_reality_strength': avg_superposition,
            'avg_frame_competition_strength': avg_frame_competition,
            'avg_meaning_conflict_strength': avg_meaning_conflict,
            'high_superposition_examples': len(high_superposition),
            'high_frame_competition_examples': len(high_frame_competition),
            'high_ambiguity_examples': len(high_ambiguity)
        }

def main():
    """主函數 - 執行量子新聞分析"""
    print("🚀 量子自然語言處理 - 多重現實分析")
    print("=" * 60)
    
    try:
        # 讀取斷詞結果
        print("📊 讀取斷詞分析結果...")
        segmentation_df = pd.read_csv('../results/complete_segmentation_results.csv')
        print(f"斷詞結果: {len(segmentation_df)} 筆記錄")
        
        # 初始化量子分析器
        analyzer = QuantumNewsAnalyzer(max_qubits=8)
        
        # 執行量子分析
        quantum_results = analyzer.analyze_news_dataset(segmentation_df)
        
        # 保存完整結果
        with open('../results/quantum_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(quantum_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存簡化的CSV結果
        all_individual_results = []
        for field_data in quantum_results['field_results'].values():
            all_individual_results.extend(field_data['individual_results'])
        
        if all_individual_results:
            results_df = pd.DataFrame(all_individual_results)
            results_df.to_csv('../results/quantum_analysis_detailed.csv', 
                            index=False, encoding='utf-8-sig')
        
        print(f"\n💾 量子分析結果已保存:")
        print(f"  完整結果: ../results/quantum_analysis_results.json")
        print(f"  詳細數據: ../results/quantum_analysis_detailed.csv")
        
        # 顯示關鍵發現
        print(f"\n🔍 關鍵發現:")
        mra = quantum_results['multiple_reality_analysis']
        print(f"  多重現實現象普及度: {mra['multiple_reality_prevalence']:.1%}")
        print(f"  框架衝突普及度: {mra['frame_conflict_prevalence']:.1%}")
        print(f"  語義模糊普及度: {mra['semantic_ambiguity_prevalence']:.1%}")
        
        print(f"\n✅ 量子自然語言處理分析完成！")
        
    except Exception as e:
        print(f"❌ 分析過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
