#!/usr/bin/env python3
"""
Qiskit量子电路分析器 - 使用真实量子电路进行QNLP分析
基于DisCoCat理论的真实量子自然语言处理实现
"""

import pandas as pd
import numpy as np
from qiskit import QuantumCircuit, execute, Aer, ClassicalRegister
from qiskit.quantum_info import entropy, Statevector, DensityMatrix, partial_trace
from qiskit.circuit.library import RYGate, CXGate, HGate, RZGate
import json
import time
import os
from typing import Dict, List, Any, Tuple
import jieba
import jieba.posseg as pseg
import warnings
warnings.filterwarnings('ignore')

# 设置中文分词
jieba.set_dictionary('../data/dict.txt.big') if os.path.exists('../data/dict.txt.big') else None

class QiskitQuantumAnalyzer:
    """基于Qiskit量子电路的QNLP分析器"""
    
    def __init__(self):
        """初始化量子分析器"""
        print("🔧 初始化Qiskit量子分析器...")
        
        # 量子后端
        self.backend = Aer.get_backend('statevector_simulator')
        self.density_backend = Aer.get_backend('qasm_simulator')
        
        # 增强的类别映射与量子属性
        self.category_map = {
            'N': {'qubit': 0, 'angle': np.pi/8, 'weight': 1.0, 'phase': 0.0},      # 名词
            'V': {'qubit': 1, 'angle': np.pi/4, 'weight': 1.2, 'phase': np.pi/6}, # 动词
            'A': {'qubit': 2, 'angle': np.pi/6, 'weight': 0.8, 'phase': np.pi/4}, # 形容词
            'P': {'qubit': 3, 'angle': np.pi/3, 'weight': 0.9, 'phase': np.pi/3}, # 介词
            'D': {'qubit': 4, 'angle': np.pi/5, 'weight': 0.7, 'phase': np.pi/8}, # 副词
            'M': {'qubit': 5, 'angle': np.pi/7, 'weight': 0.6, 'phase': np.pi/5}, # 数词
            'Q': {'qubit': 6, 'angle': np.pi/9, 'weight': 0.5, 'phase': np.pi/7}, # 量词
            'R': {'qubit': 7, 'angle': np.pi/10, 'weight': 0.4, 'phase': np.pi/9} # 代词
        }
        
        # 情感词典
        self.emotion_lexicon = {
            'positive': ['成功', '获得', '优秀', '突破', '创新', '发展', '改善', '提升', '荣获', 
                        '卓越', '领先', '进步', '增长', '获奖', '肯定', '支持', '合作', '共赢',
                        '繁荣', '兴旺', '辉煌', '胜利', '喜悦', '满意', '赞扬', '表彰'],
            'negative': ['失败', '问题', '困难', '危机', '冲突', '争议', '批评', '质疑', '担忧',
                        '下降', '减少', '损失', '风险', '威胁', '挑战', '阻碍', '延迟', '取消',
                        '衰退', '恶化', '混乱', '灾难', '悲伤', '愤怒', '抗议', '谴责']
        }
        
        print("✅ Qiskit量子分析器初始化完成")

    def segment_and_pos_tag(self, text: str) -> Tuple[List[str], List[str]]:
        """分词和词性标注"""
        words = []
        pos_tags = []
        
        for word, flag in pseg.cut(text):
            if len(word.strip()) > 0:
                words.append(word)
                pos_tags.append(flag)
        
        return words, pos_tags

    def create_quantum_circuit(self, words: List[str], pos_tags: List[str], 
                             semantic_density: float = 0.0) -> QuantumCircuit:
        """创建量子电路基于语言分析"""
        
        if not words or not pos_tags:
            # 最小电路用于空输入
            circuit = QuantumCircuit(3)
            circuit.h(0)
            return circuit
        
        # 基于类别多样性确定量子比特数
        unique_categories = list(set(pos_tags))
        num_qubits = min(8, max(3, len(unique_categories) + 2))
        
        circuit = QuantumCircuit(num_qubits)
        
        # 1. 初始化：创建叠加态
        for i in range(num_qubits):
            circuit.h(i)
        
        # 2. 应用类别特定的旋转门
        from collections import Counter
        category_counts = Counter(pos_tags)
        
        for i, (cat, count) in enumerate(category_counts.items()):
            if cat in self.category_map and i < num_qubits - 1:
                cat_info = self.category_map[cat]
                # 基于频率和类型的旋转角度
                angle = cat_info['angle'] * (count / len(pos_tags)) * cat_info['weight']
                circuit.ry(angle, i)
                # 添加相位门
                circuit.rz(cat_info['phase'], i)
        
        # 3. 基于词汇关系创建纠缠
        word_freq = Counter(words)
        repeated_words = [word for word, count in word_freq.items() if count > 1]
        
        # 为重复词汇创建纠缠
        if len(repeated_words) > 0 and num_qubits > 2:
            for i in range(min(len(repeated_words), num_qubits - 1)):
                target = (i + 1) % num_qubits
                circuit.cx(i, target)
        
        # 4. 语义密度调制
        if semantic_density > 0 and num_qubits > 1:
            density_angle = semantic_density * np.pi / 4
            for i in range(num_qubits - 1):
                circuit.ry(density_angle, i)
        
        # 5. 情感极性纠缠
        positive_count = sum(1 for word in words if word in self.emotion_lexicon['positive'])
        negative_count = sum(1 for word in words if word in self.emotion_lexicon['negative'])
        
        if positive_count > 0 and negative_count > 0 and num_qubits > 2:
            # 情感冲突时创建特殊纠缠
            circuit.cx(0, num_qubits - 1)
            circuit.ry(np.pi * (positive_count - negative_count) / len(words), num_qubits - 1)
        
        return circuit

    def measure_quantum_properties(self, circuit: QuantumCircuit, 
                                 words: List[str], pos_tags: List[str]) -> Dict[str, float]:
        """测量量子电路的属性"""
        
        try:
            # 添加测量门到电路的副本
            measured_circuit = circuit.copy()
            measured_circuit.add_register(ClassicalRegister(circuit.num_qubits))
            measured_circuit.measure_all()
            
            # 执行量子电路获取状态向量
            job = execute(circuit, self.backend)
            result = job.result()
            statevector = result.get_statevector(circuit)
            
            # 确保状态向量是有效的
            if statevector is None or len(statevector.data) == 0:
                raise ValueError("Invalid statevector obtained")
            
            # 归一化状态向量
            statevector_data = np.array(statevector.data)
            norm = np.linalg.norm(statevector_data)
            if norm > 0:
                statevector_data = statevector_data / norm
            else:
                raise ValueError("Zero norm statevector")
            
            # 创建密度矩阵
            density_matrix = np.outer(statevector_data, np.conj(statevector_data))
            
            # 1. 冯纽曼熵 (量子信息熵)
            eigenvals = np.linalg.eigvals(density_matrix)
            eigenvals = eigenvals[eigenvals > 1e-12]  # 过滤小特征值
            if len(eigenvals) > 0:
                von_neumann_entropy = float(-np.sum(eigenvals * np.log2(eigenvals + 1e-12)))
            else:
                von_neumann_entropy = 0.0
            
            # 2. 量子纠缠度 (基于线性熵)
            num_qubits = circuit.num_qubits
            if num_qubits > 1:
                # 使用线性熵作为纠缠度量
                linear_entropy = 1.0 - np.trace(density_matrix @ density_matrix)
                entanglement_entropy = float(linear_entropy.real)
            else:
                entanglement_entropy = 0.0
            
            # 3. 量子叠加强度 (基于状态向量的幅度分布)
            amplitudes = np.abs(statevector_data)
            probabilities = amplitudes**2
            superposition_strength = float(4 * np.sum(probabilities * (1 - probabilities)))
            
            # 4. 量子相干性 (基于非对角元素)
            diagonal_elements = np.diag(density_matrix)
            off_diagonal = density_matrix - np.diag(diagonal_elements)
            coherence = float(np.sum(np.abs(off_diagonal)))
            
            # 5. 语义干涉强度 (基于相位信息)
            phases = np.angle(statevector_data)
            phase_variance = float(np.var(phases))
            semantic_interference = phase_variance / (2 * np.pi)
            
            # 6. 框架竞争强度 (基于概率分布的KL散度)
            probabilities_filtered = probabilities[probabilities > 1e-12]
            competition_entropy = float(min(1.0, von_neumann_entropy * 0.5))
            if len(probabilities_filtered) > 1:
                uniform_prob = 1.0 / len(probabilities_filtered)
                kl_divergence = np.sum(probabilities_filtered * np.log2((probabilities_filtered + 1e-12) / uniform_prob))
                max_kl = np.log2(len(probabilities_filtered))
                frame_competition_kl = float(1.0 - min(1.0, kl_divergence / max_kl))
            else:
                frame_competition_kl = 0.0
            
            # 7. 类别一致性 (基于词性标签分布)
            from collections import Counter
            pos_freq = Counter(pos_tags)
            pos_probs = np.array([count/len(pos_tags) for count in pos_freq.values()])
            if len(pos_probs) > 1:
                pos_entropy = -np.sum(pos_probs * np.log2(pos_probs + 1e-12))
                max_entropy = np.log2(len(pos_probs))
                category_coherence = float(1.0 - pos_entropy / max_entropy)
            else:
                category_coherence = 1.0
            
            # 8. 组合纠缠度 (词性多样性)
            pos_diversity = len(set(pos_tags))
            compositional_entanglement = float(pos_diversity / len(words))
            
            # 9. 类别一致性变异
            categorical_coherence_variance = float(np.var(pos_probs))
            
            return {
                'von_neumann_entropy': von_neumann_entropy,
                'quantum_entanglement': entanglement_entropy,
                'superposition_strength': superposition_strength,
                'quantum_coherence': coherence,
                'semantic_interference': semantic_interference,
                'frame_competition': competition_entropy,
                'frame_competition_kl': frame_competition_kl,
                'category_coherence': category_coherence,
                'compositional_entanglement': compositional_entanglement,
                'categorical_coherence_variance': categorical_coherence_variance
            }
            
        except Exception as e:
            print(f"❌ 量子电路执行错误: {e}")
            # 使用基于经典概率的回退计算
            return self._fallback_quantum_calculation(words, pos_tags)

    def _fallback_quantum_calculation(self, words: List[str], pos_tags: List[str]) -> Dict[str, float]:
        """回退到经典概率计算（当量子电路失败时）"""
        from collections import Counter
        
        # 基于词频的概率分布
        word_freq = Counter(words)
        total_words = sum(word_freq.values())
        probabilities = np.array([freq/total_words for freq in word_freq.values()])
        
        # 1. 经典熵（模拟冯纽曼熵）
        von_neumann_entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))
        
        # 2. 模拟量子纠缠（基于词汇重复）
        repeated_words = sum(1 for count in word_freq.values() if count > 1)
        quantum_entanglement = float(repeated_words / len(word_freq))
        
        # 3. 叠加强度（基于概率分布）
        superposition_strength = float(4 * np.sum(probabilities * (1 - probabilities)))
        
        # 4. 相干性（基于词汇多样性）
        unique_ratio = len(set(words)) / len(words)
        quantum_coherence = float(unique_ratio)
        
        # 5. 语义干涉（基于重复模式）
        repetition_variance = np.var(list(word_freq.values()))
        semantic_interference = float(repetition_variance / len(words))
        
        # 6. 框架竞争
        competition_entropy = float(min(1.0, von_neumann_entropy * 0.5))
        if len(probabilities) > 1:
            uniform_prob = 1.0 / len(probabilities)
            kl_divergence = np.sum(probabilities * np.log2((probabilities + 1e-12) / uniform_prob))
            max_kl = np.log2(len(probabilities))
            frame_competition_kl = float(1.0 - min(1.0, kl_divergence / max_kl))
        else:
            frame_competition_kl = 0.0
        
        # 7. 类别一致性
        pos_freq = Counter(pos_tags)
        pos_probs = np.array([count/len(pos_tags) for count in pos_freq.values()])
        if len(pos_probs) > 1:
            pos_entropy = -np.sum(pos_probs * np.log2(pos_probs + 1e-12))
            max_entropy = np.log2(len(pos_probs))
            category_coherence = float(1.0 - pos_entropy / max_entropy)
        else:
            category_coherence = 1.0
        
        # 8. 组合纠缠度
        pos_diversity = len(set(pos_tags))
        compositional_entanglement = float(pos_diversity / len(words))
        
        # 9. 类别一致性变异
        categorical_coherence_variance = float(np.var(pos_probs))
        
        return {
            'von_neumann_entropy': von_neumann_entropy,
            'quantum_entanglement': quantum_entanglement,
            'superposition_strength': superposition_strength,
            'quantum_coherence': quantum_coherence,
            'semantic_interference': semantic_interference,
            'frame_competition': competition_entropy,
            'frame_competition_kl': frame_competition_kl,
            'category_coherence': category_coherence,
            'compositional_entanglement': compositional_entanglement,
            'categorical_coherence_variance': categorical_coherence_variance
        }

    def analyze_multiple_realities(self, quantum_metrics: Dict, words: List[str]) -> Dict[str, float]:
        """分析多重现实现象"""
        
        # 计算语言复杂性因子
        word_count = len(words)
        unique_words = len(set(words))
        word_diversity = unique_words / max(word_count, 1)
        
        # 情感强度
        positive_count = sum(1 for word in words if word in self.emotion_lexicon['positive'])
        negative_count = sum(1 for word in words if word in self.emotion_lexicon['negative'])
        emotional_intensity = (positive_count + negative_count) / max(word_count, 1)
        
        # 多重现实强度 (基于量子叠加和纠缠)
        reality_strength = (
            quantum_metrics['superposition_strength'] * 0.30 +
            quantum_metrics['quantum_entanglement'] * 0.25 +
            quantum_metrics['semantic_interference'] * 0.20 +
            quantum_metrics['frame_competition'] * 0.15 +
            word_diversity * 0.10
        )
        
        # 框架冲突强度 (基于量子相干性和纠缠)
        conflict_strength = (
            quantum_metrics['compositional_entanglement'] * 0.35 +
            quantum_metrics['quantum_coherence'] * 0.25 +
            quantum_metrics['categorical_coherence_variance'] * 0.20 +
            emotional_intensity * 0.15 +
            (1.0 - quantum_metrics['category_coherence']) * 0.05
        )
        
        # 语义模糊度 (基于熵和干涉)
        ambiguity = (
            quantum_metrics['von_neumann_entropy'] * 0.40 +
            quantum_metrics['semantic_interference'] * 0.30 +
            (1.0 - quantum_metrics['category_coherence']) * 0.20 +
            word_diversity * 0.10
        )
        
        return {
            'multiple_reality_strength': float(reality_strength),
            'frame_conflict_strength': float(conflict_strength),
            'semantic_ambiguity': float(ambiguity)
        }

    def analyze_text_quantum(self, text: str, field_name: str = "text") -> Dict[str, Any]:
        """使用量子电路分析单个文本"""
        
        if not text or len(text.strip()) == 0:
            return None
        
        # 分词和词性标注
        words, pos_tags = self.segment_and_pos_tag(text)
        
        if len(words) == 0:
            return None
        
        # 计算语义密度
        semantic_density = len(set(words)) / len(words) * 10.0
        
        # 创建量子电路
        circuit = self.create_quantum_circuit(words, pos_tags, semantic_density)
        
        # 测量量子属性
        quantum_metrics = self.measure_quantum_properties(circuit, words, pos_tags)
        
        # 分析多重现实
        reality_metrics = self.analyze_multiple_realities(quantum_metrics, words)
        
        # 基本统计
        word_count = len(words)
        unique_words = len(set(words))
        categorical_diversity = len(set(pos_tags))
        compositional_complexity = categorical_diversity / max(word_count, 1) if word_count > 0 else 0
        
        return {
            'field': field_name,
            'original_text': text,
            'word_count': word_count,
            'unique_words': unique_words,
            'categorical_diversity': categorical_diversity,
            'compositional_complexity': float(compositional_complexity),
            'semantic_density': float(semantic_density),
            'quantum_circuit_qubits': circuit.num_qubits,
            **quantum_metrics,
            **reality_metrics,
            'analysis_version': 'qiskit_quantum_v1.0'
        }

    def process_ai_record(self, record: Dict) -> List[Dict]:
        """处理AI新闻记录"""
        results = []
        record_id = record.get('id', 0)
        
        # 分析三个字段
        fields = ['新聞標題', '影片對話', '影片描述']
        
        for field in fields:
            if field in record and record[field]:
                result = self.analyze_text_quantum(record[field], field)
                if result:
                    result['record_id'] = record_id
                    result['data_source'] = 'AI_Generated'
                    results.append(result)
        
        return results

    def process_journalist_record(self, record: Dict) -> List[Dict]:
        """处理记者新闻记录"""
        results = []
        record_id = record.get('id', 0)
        
        # 分析两个字段，但映射为统一的字段名
        field_mapping = {
            'title': '新聞標題',
            'content': '新聞內容'
        }
        
        for original_field, mapped_field in field_mapping.items():
            if original_field in record and record[original_field]:
                result = self.analyze_text_quantum(record[original_field], mapped_field)
                if result:
                    result['record_id'] = record_id
                    result['data_source'] = 'Journalist_Written'
                    results.append(result)
        
        return results

def main():
    """主函数：执行完整的Qiskit量子分析"""
    print("🚀 开始Qiskit量子电路分析...")
    
    # 初始化分析器
    analyzer = QiskitQuantumAnalyzer()
    
    # 1. 分析AI新闻数据
    print("\n📊 分析AI生成新闻...")
    ai_data_path = '../data/dataseet.xlsx'
    
    if os.path.exists(ai_data_path):
        ai_df = pd.read_excel(ai_data_path)
        print(f"✅ 加载AI数据: {len(ai_df)} 条记录")
        
        ai_results = []
        for idx, record in ai_df.iterrows():
            record_dict = record.to_dict()
            record_dict['id'] = idx
            results = analyzer.process_ai_record(record_dict)
            ai_results.extend(results)
        
        # 保存AI分析结果
        ai_results_df = pd.DataFrame(ai_results)
        ai_results_path = '../results/qiskit_ai_analysis_results.csv'
        ai_results_df.to_csv(ai_results_path, index=False, encoding='utf-8-sig')
        print(f"✅ AI分析结果已保存: {ai_results_path}")
        
        # 生成AI统计摘要
        ai_summary = {}
        for col in ['von_neumann_entropy', 'quantum_entanglement', 'superposition_strength', 
                   'quantum_coherence', 'semantic_interference', 'frame_competition',
                   'multiple_reality_strength', 'frame_conflict_strength', 'semantic_ambiguity']:
            ai_summary[col] = {
                'mean': float(ai_results_df[col].mean()),
                'std': float(ai_results_df[col].std()),
                'min': float(ai_results_df[col].min()),
                'max': float(ai_results_df[col].max()),
                'median': float(ai_results_df[col].median())
            }
        
        ai_summary_path = '../results/qiskit_ai_analysis_summary.json'
        with open(ai_summary_path, 'w', encoding='utf-8') as f:
            json.dump(ai_summary, f, ensure_ascii=False, indent=2)
        print(f"✅ AI统计摘要已保存: {ai_summary_path}")
    
    # 2. 分析记者新闻数据
    print("\n📊 分析记者撰写新闻...")
    journalist_data_path = '../data/cna.csv'
    
    if os.path.exists(journalist_data_path):
        journalist_df = pd.read_csv(journalist_data_path)
        print(f"✅ 加载记者数据: {len(journalist_df)} 条记录")
        
        journalist_results = []
        for idx, record in journalist_df.iterrows():
            record_dict = record.to_dict()
            record_dict['id'] = idx
            results = analyzer.process_journalist_record(record_dict)
            journalist_results.extend(results)
        
        # 保存记者分析结果
        journalist_results_df = pd.DataFrame(journalist_results)
        journalist_results_path = '../results/qiskit_journalist_analysis_results.csv'
        journalist_results_df.to_csv(journalist_results_path, index=False, encoding='utf-8-sig')
        print(f"✅ 记者分析结果已保存: {journalist_results_path}")
        
        # 生成记者统计摘要
        journalist_summary = {}
        for col in ['von_neumann_entropy', 'quantum_entanglement', 'superposition_strength', 
                   'quantum_coherence', 'semantic_interference', 'frame_competition',
                   'multiple_reality_strength', 'frame_conflict_strength', 'semantic_ambiguity']:
            journalist_summary[col] = {
                'mean': float(journalist_results_df[col].mean()),
                'std': float(journalist_results_df[col].std()),
                'min': float(journalist_results_df[col].min()),
                'max': float(journalist_results_df[col].max()),
                'median': float(journalist_results_df[col].median())
            }
        
        journalist_summary_path = '../results/qiskit_journalist_analysis_summary.json'
        with open(journalist_summary_path, 'w', encoding='utf-8') as f:
            json.dump(journalist_summary, f, ensure_ascii=False, indent=2)
        print(f"✅ 记者统计摘要已保存: {journalist_summary_path}")
    
    print("\n🎉 Qiskit量子电路分析完成！")

if __name__ == "__main__":
    main()
