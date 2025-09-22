#!/usr/bin/env python3
"""
增強版量子自然語言處理分析器
Enhanced Quantum Natural Language Processing Analyzer
支持jieba和ChatGPT兩種斷詞方法的比較分析
"""

import pandas as pd
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import entropy, Statevector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedQNLPAnalyzer:
    """增強版QNLP分析器，支持多種斷詞方法"""
    
    def __init__(self):
        self.backend = Aer.get_backend('qasm_simulator')
        self.statevector_backend = Aer.get_backend('statevector_simulator')
        self.results = {}
        
    def create_quantum_circuit_from_segmentation(self, words: List[str], max_qubits: int = 6) -> Tuple[QuantumCircuit, int]:
        """根據斷詞結果創建量子電路"""
        if not words:
            return None, 0
        
        # 使用TF-IDF計算詞彙權重
        text = ' '.join(words)
        vectorizer = TfidfVectorizer(max_features=max_qubits)
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            weights = tfidf_matrix.toarray()[0]
        except:
            # 如果TF-IDF失敗，使用均勻權重
            n_qubits = min(len(words), max_qubits)
            weights = np.ones(n_qubits) / n_qubits
            feature_names = words[:n_qubits]
        
        n_qubits = len(weights)
        if n_qubits == 0:
            return None, 0
        
        # 創建量子電路
        qc = QuantumCircuit(n_qubits)
        
        # 初始化疊加態
        for i in range(n_qubits):
            qc.h(i)
        
        # 基於TF-IDF權重的旋轉
        for i, weight in enumerate(weights):
            if weight > 0:
                angle = weight * np.pi + np.pi/4
                qc.ry(angle, i)
        
        # 創建糾纏
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        
        # 添加相位門
        for i in range(n_qubits - 1):
            qc.cp(np.pi/4, i, (i + 1) % n_qubits)
        
        return qc, n_qubits
    
    def measure_quantum_properties(self, quantum_circuit: QuantumCircuit, n_qubits: int) -> Dict:
        """測量量子特性"""
        if quantum_circuit is None or n_qubits == 0:
            return {
                'entropy': 0,
                'coherence': 0,
                'interference': 0,
                'superposition_strength': 0
            }
        
        try:
            # 獲取狀態向量
            job = execute(quantum_circuit, self.statevector_backend)
            result = job.result()
            statevector = result.get_statevector()
            
            # 計算von Neumann熵（敘事複雜度）
            entropy_val = entropy(statevector)
            
            # 計算量子連貫性
            amplitudes = np.abs(statevector.data)
            coherence = 1 - np.sum(amplitudes**4)  # 參與比
            
            # 計算量子干涉
            phases = np.angle(statevector.data)
            phase_variance = np.var(phases)
            interference = 1 - (phase_variance / (np.pi**2)) if phase_variance > 0 else 1
            
            # 計算疊加強度
            prob_dist = amplitudes**2
            superposition_strength = 1 - np.max(prob_dist)  # 1 - 最大機率
            
            return {
                'entropy': float(entropy_val),
                'coherence': float(coherence),
                'interference': float(interference),
                'superposition_strength': float(superposition_strength)
            }
            
        except Exception as e:
            print(f"量子測量錯誤: {e}")
            return {
                'entropy': 0,
                'coherence': 0,
                'interference': 0,
                'superposition_strength': 0
            }
    
    def analyze_semantic_complexity(self, words: List[str]) -> float:
        """分析語義複雜度"""
        if not words:
            return 0
        
        # 基於詞彙多樣性和長度的複雜度
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words == 0:
            return 0
        
        # 詞彙豐富度
        richness = unique_words / total_words
        
        # 平均詞長
        avg_word_length = np.mean([len(word) for word in words])
        
        # 綜合複雜度
        complexity = (richness * 0.7 + min(avg_word_length / 5, 1) * 0.3)
        
        return float(complexity)
    
    def detect_narrative_superposition(self, segmentation_results: List[Dict]) -> float:
        """檢測敘事疊加現象"""
        if not segmentation_results:
            return 0
        
        # 收集所有詞彙
        all_words = []
        for result in segmentation_results:
            if 'words' in result and result['words']:
                all_words.extend(result['words'])
        
        if not all_words:
            return 0
        
        # 計算詞頻分布的熵
        word_counts = Counter(all_words)
        total_words = sum(word_counts.values())
        
        if total_words == 0:
            return 0
        
        # 計算機率分布的熵
        probs = [count / total_words for count in word_counts.values()]
        narrative_entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # 正規化到0-1範圍
        max_entropy = np.log2(len(word_counts))
        superposition = narrative_entropy / max_entropy if max_entropy > 0 else 0
        
        return float(superposition)
    
    def measure_semantic_entanglement(self, text1_words: List[str], text2_words: List[str]) -> float:
        """測量語義糾纏"""
        if not text1_words or not text2_words:
            return 0
        
        # 創建詞彙向量
        all_words = list(set(text1_words + text2_words))
        
        if len(all_words) < 2:
            return 0
        
        # 計算詞頻向量
        vec1 = [text1_words.count(word) for word in all_words]
        vec2 = [text2_words.count(word) for word in all_words]
        
        # 正規化
        vec1 = np.array(vec1) / (np.sum(vec1) + 1e-10)
        vec2 = np.array(vec2) / (np.sum(vec2) + 1e-10)
        
        # 計算餘弦相似度作為糾纏度量
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        
        # 轉換為糾纏強度（0-2範圍）
        entanglement = (1 + similarity) * 1.0
        
        return float(entanglement)
    
    def analyze_segmentation_method(self, segmentation_data: pd.DataFrame, method_name: str) -> Dict:
        """分析特定斷詞方法的結果"""
        print(f"\n🔬 分析 {method_name} 斷詞方法")
        print("=" * 40)
        
        results = {
            'method': method_name,
            'field_results': {},
            'overall_stats': {}
        }
        
        # 按欄位分析
        fields = segmentation_data['field'].unique()
        
        for field in fields:
            field_data = segmentation_data[segmentation_data['field'] == field]
            print(f"\n📊 分析 {field} 欄位 ({len(field_data)} 筆記錄)")
            
            field_results = {
                'quantum_properties': [],
                'semantic_complexity': [],
                'word_counts': [],
                'unique_word_counts': [],
                'text_lengths': []
            }
            
            # 逐筆分析
            for _, row in field_data.iterrows():
                if pd.isna(row['words_list']) or not row['words_list'].strip():
                    continue
                
                # 解析詞彙
                words = [w.strip() for w in str(row['words_list']).split(',') if w.strip()]
                
                if not words:
                    continue
                
                # 創建量子電路並測量
                qc, n_qubits = self.create_quantum_circuit_from_segmentation(words)
                quantum_props = self.measure_quantum_properties(qc, n_qubits)
                
                # 計算語義複雜度
                complexity = self.analyze_semantic_complexity(words)
                
                # 記錄結果
                field_results['quantum_properties'].append(quantum_props)
                field_results['semantic_complexity'].append(complexity)
                field_results['word_counts'].append(len(words))
                field_results['unique_word_counts'].append(len(set(words)))
                field_results['text_lengths'].append(len(' '.join(words)))
            
            # 計算欄位統計
            if field_results['quantum_properties']:
                field_stats = self._calculate_field_statistics(field_results)
                results['field_results'][field] = field_stats
                
                print(f"  量子連貫性: {field_stats['avg_coherence']:.4f} ± {field_stats['std_coherence']:.4f}")
                print(f"  量子干涉: {field_stats['avg_interference']:.4f} ± {field_stats['std_interference']:.4f}")
                print(f"  敘事複雜度: {field_stats['avg_entropy']:.4f} ± {field_stats['std_entropy']:.4f}")
                print(f"  語義複雜度: {field_stats['avg_semantic_complexity']:.4f}")
                print(f"  平均詞數: {field_stats['avg_word_count']:.1f}")
        
        # 計算整體統計
        results['overall_stats'] = self._calculate_overall_statistics(results['field_results'])
        
        return results
    
    def _calculate_field_statistics(self, field_results: Dict) -> Dict:
        """計算欄位統計"""
        quantum_props = field_results['quantum_properties']
        
        coherences = [qp['coherence'] for qp in quantum_props]
        interferences = [qp['interference'] for qp in quantum_props]
        entropies = [qp['entropy'] for qp in quantum_props]
        superpositions = [qp['superposition_strength'] for qp in quantum_props]
        
        return {
            'avg_coherence': np.mean(coherences),
            'std_coherence': np.std(coherences),
            'avg_interference': np.mean(interferences),
            'std_interference': np.std(interferences),
            'avg_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'avg_superposition': np.mean(superpositions),
            'std_superposition': np.std(superpositions),
            'avg_semantic_complexity': np.mean(field_results['semantic_complexity']),
            'avg_word_count': np.mean(field_results['word_counts']),
            'avg_unique_word_count': np.mean(field_results['unique_word_counts']),
            'avg_text_length': np.mean(field_results['text_lengths'])
        }
    
    def _calculate_overall_statistics(self, field_results: Dict) -> Dict:
        """計算整體統計"""
        if not field_results:
            return {}
        
        all_coherences = []
        all_interferences = []
        all_entropies = []
        all_superpositions = []
        
        for field_stats in field_results.values():
            all_coherences.append(field_stats['avg_coherence'])
            all_interferences.append(field_stats['avg_interference'])
            all_entropies.append(field_stats['avg_entropy'])
            all_superpositions.append(field_stats['avg_superposition'])
        
        return {
            'overall_avg_coherence': np.mean(all_coherences),
            'overall_avg_interference': np.mean(all_interferences),
            'overall_avg_entropy': np.mean(all_entropies),
            'overall_avg_superposition': np.mean(all_superpositions)
        }
    
    def compare_segmentation_methods(self, jieba_results: Dict, chatgpt_results: Dict) -> Dict:
        """比較兩種斷詞方法的QNLP結果"""
        print("\n🔍 比較jieba與ChatGPT的QNLP分析結果")
        print("=" * 50)
        
        comparison = {
            'method_comparison': {},
            'field_comparison': {},
            'insights': []
        }
        
        # 整體方法比較
        jieba_overall = jieba_results.get('overall_stats', {})
        chatgpt_overall = chatgpt_results.get('overall_stats', {})
        
        if jieba_overall and chatgpt_overall:
            comparison['method_comparison'] = {
                'coherence_diff': chatgpt_overall['overall_avg_coherence'] - jieba_overall['overall_avg_coherence'],
                'interference_diff': chatgpt_overall['overall_avg_interference'] - jieba_overall['overall_avg_interference'],
                'entropy_diff': chatgpt_overall['overall_avg_entropy'] - jieba_overall['overall_avg_entropy'],
                'superposition_diff': chatgpt_overall['overall_avg_superposition'] - jieba_overall['overall_avg_superposition']
            }
        
        # 欄位比較
        common_fields = set(jieba_results.get('field_results', {}).keys()) & set(chatgpt_results.get('field_results', {}).keys())
        
        for field in common_fields:
            jieba_field = jieba_results['field_results'][field]
            chatgpt_field = chatgpt_results['field_results'][field]
            
            comparison['field_comparison'][field] = {
                'coherence_diff': chatgpt_field['avg_coherence'] - jieba_field['avg_coherence'],
                'interference_diff': chatgpt_field['avg_interference'] - jieba_field['avg_interference'],
                'entropy_diff': chatgpt_field['avg_entropy'] - jieba_field['avg_entropy'],
                'word_count_diff': chatgpt_field['avg_word_count'] - jieba_field['avg_word_count'],
                'semantic_complexity_diff': chatgpt_field['avg_semantic_complexity'] - jieba_field['avg_semantic_complexity']
            }
        
        # 生成洞察
        comparison['insights'] = self._generate_comparison_insights(comparison)
        
        return comparison
    
    def _generate_comparison_insights(self, comparison: Dict) -> List[str]:
        """生成比較洞察"""
        insights = []
        
        method_comp = comparison.get('method_comparison', {})
        
        if method_comp:
            if method_comp['coherence_diff'] > 0.1:
                insights.append("ChatGPT斷詞產生更高的量子連貫性，表示語義一致性更強")
            elif method_comp['coherence_diff'] < -0.1:
                insights.append("jieba斷詞產生更高的量子連貫性，表示語義一致性更強")
            
            if method_comp['entropy_diff'] > 0.1:
                insights.append("ChatGPT斷詞顯示更高的敘事複雜度，可能反映更細緻的語義結構")
            elif method_comp['entropy_diff'] < -0.1:
                insights.append("jieba斷詞顯示更高的敘事複雜度")
            
            if method_comp['superposition_diff'] > 0.1:
                insights.append("ChatGPT斷詞展現更強的敘事疊加現象，支持多重現實理論")
            elif method_comp['superposition_diff'] < -0.1:
                insights.append("jieba斷詞展現更強的敘事疊加現象")
        
        return insights

def main():
    """主函數"""
    print("🔬 增強版量子自然語言處理分析")
    print("=" * 50)
    
    analyzer = EnhancedQNLPAnalyzer()
    
    try:
        # 讀取jieba結果
        print("📊 讀取jieba斷詞結果...")
        jieba_df = pd.read_csv('../jieba_segmentation_results.csv')
        print(f"jieba結果: {len(jieba_df)} 筆記錄")
        
        # 讀取ChatGPT結果
        chatgpt_files = [
            '../data/real_chatgpt_segmentation_complete.csv',
            '../real_chatgpt_segmentation_sample.csv'
        ]
        
        chatgpt_df = None
        for file_path in chatgpt_files:
            try:
                chatgpt_df = pd.read_csv(file_path)
                print(f"✅ 讀取ChatGPT結果: {file_path} ({len(chatgpt_df)} 筆記錄)")
                break
            except FileNotFoundError:
                continue
        
        if chatgpt_df is None:
            print("❌ 未找到ChatGPT斷詞結果，請先運行ChatGPT斷詞分析")
            return
        
        # 分析jieba結果
        jieba_analysis = analyzer.analyze_segmentation_method(jieba_df, "jieba")
        
        # 分析ChatGPT結果
        chatgpt_analysis = analyzer.analyze_segmentation_method(chatgpt_df, "ChatGPT")
        
        # 比較分析
        comparison = analyzer.compare_segmentation_methods(jieba_analysis, chatgpt_analysis)
        
        # 保存結果
        import json
        
        # 保存詳細分析結果
        analysis_results = {
            'jieba_analysis': jieba_analysis,
            'chatgpt_analysis': chatgpt_analysis,
            'comparison': comparison,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open('../results/qnlp_comparative_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 分析結果已保存: ../results/qnlp_comparative_analysis.json")
        
        # 顯示主要發現
        print(f"\n🔍 主要發現:")
        for insight in comparison['insights']:
            print(f"  • {insight}")
        
        # 顯示數值比較
        if comparison['method_comparison']:
            mc = comparison['method_comparison']
            print(f"\n📊 量子指標比較 (ChatGPT - jieba):")
            print(f"  量子連貫性差異: {mc['coherence_diff']:+.4f}")
            print(f"  量子干涉差異: {mc['interference_diff']:+.4f}")
            print(f"  敘事複雜度差異: {mc['entropy_diff']:+.4f}")
            print(f"  疊加強度差異: {mc['superposition_diff']:+.4f}")
        
        print(f"\n🎉 QNLP比較分析完成！")
        
    except Exception as e:
        print(f"❌ 分析過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
