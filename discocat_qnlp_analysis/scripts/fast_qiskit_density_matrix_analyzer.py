#!/usr/bin/env python3
"""
快速Qiskit量子分析器 - 使用密度矩陣計算熵版本
使用密度矩陣 (Density Matrix) 計算 von Neumann 熵
"""

import pandas as pd
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import json
import time
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path
import jieba
import jieba.posseg as pseg
import warnings
warnings.filterwarnings('ignore')

# 设置中文分词
script_dir = Path(__file__).parent
project_root = script_dir.parent
dict_path = project_root / 'data' / 'dict.txt.big'
jieba.set_dictionary(str(dict_path)) if dict_path.exists() else None

class FastQiskitDensityMatrixAnalyzer:
    """快速Qiskit量子分析器 - 使用密度矩陣版本"""
    
    def __init__(self):
        """初始化快速量子分析器"""
        print("🚀 初始化快速Qiskit量子分析器（密度矩陣版本）...")
        
        # 简化的类别映射
        self.category_map = {
            'N': 0.25,   # 名词
            'V': 0.5,    # 动词
            'A': 0.75,   # 形容词
            'P': 1.0,    # 介词
            'D': 0.3,    # 副词
            'M': 0.6,    # 数词
            'Q': 0.8,    # 量词
            'R': 0.4     # 代词
        }
        
        # 简化的情感词典
        self.positive_words = {'成功', '获得', '优秀', '突破', '创新', '发展', '改善', '提升', '荣获', '卓越', '领先', '进步'}
        self.negative_words = {'失败', '问题', '困难', '危机', '冲突', '争议', '批评', '质疑', '担忧', '下降', '减少', '损失'}
        
        print("✅ 快速Qiskit量子分析器初始化完成（使用密度矩陣計算熵）")

    def create_simple_quantum_circuit(self, words: List[str], pos_tags: List[str]) -> QuantumCircuit:
        """创建简化的量子电路"""
        
        # 限制量子比特数以提高速度
        num_qubits = min(4, max(2, len(set(pos_tags))))
        circuit = QuantumCircuit(num_qubits)
        
        # 1. 基础叠加
        for i in range(num_qubits):
            circuit.h(i)
        
        # 2. 基于词性的简单旋转
        pos_counts = {}
        for pos in pos_tags:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        for i, (pos, count) in enumerate(list(pos_counts.items())[:num_qubits]):
            if pos in self.category_map:
                angle = self.category_map[pos] * (count / len(pos_tags)) * np.pi / 4
                circuit.ry(angle, i)
        
        # 3. 简单纠缠（只在前两个量子比特间）
        if num_qubits > 1:
            circuit.cx(0, 1)
        
        return circuit

    def calculate_von_neumann_entropy_from_density_matrix(self, statevector_data: np.ndarray) -> float:
        """使用密度矩陣計算 von Neumann 熵"""
        try:
            # 歸一化狀態向量
            norm = np.linalg.norm(statevector_data)
            if norm > 0:
                statevector_data = statevector_data / norm
            else:
                return 0.0
            
            # 創建密度矩陣: ρ = |ψ⟩⟨ψ|
            density_matrix = np.outer(statevector_data, np.conj(statevector_data))
            
            # 計算密度矩陣的特徵值
            eigenvals = np.linalg.eigvals(density_matrix)
            eigenvals = eigenvals[eigenvals > 1e-12]  # 過濾小特徵值
            
            if len(eigenvals) > 0:
                # von Neumann 熵: S(ρ) = -Tr(ρ log₂ ρ) = -Σ λᵢ log₂ λᵢ
                von_neumann_entropy = float(-np.sum(eigenvals * np.log2(eigenvals + 1e-12)))
                return von_neumann_entropy
            else:
                return 0.0
        except Exception as e:
            print(f"⚠️  密度矩陣計算錯誤: {e}")
            return 0.0

    def fast_quantum_analysis(self, text: str, field_name: str = "text") -> Dict[str, Any]:
        """快速量子分析 - 使用密度矩陣計算熵"""
        
        if not text or len(text.strip()) == 0:
            return None
        
        # 分词和词性标注
        words = []
        pos_tags = []
        
        for word, flag in pseg.cut(text):
            if len(word.strip()) > 0:
                words.append(word)
                pos_tags.append(flag)
        
        if len(words) == 0:
            return None
        
        try:
            # 创建简化量子电路
            circuit = self.create_simple_quantum_circuit(words, pos_tags)
            
            # 执行量子电路 - 使用 Statevector 直接計算
            statevector = Statevector.from_instruction(circuit)
            
            # 獲取狀態向量數據
            statevector_data = np.array(statevector.data)
            
            # 快速量子指标计算
            amplitudes = np.abs(statevector_data)
            probabilities = amplitudes**2
            
            # 1. 量子熵 - 使用密度矩陣計算
            von_neumann_entropy = self.calculate_von_neumann_entropy_from_density_matrix(statevector_data)
            
            # 2. 叠加强度
            superposition_strength = float(4 * np.sum(probabilities * (1 - probabilities)))
            
            # 3. 量子相干性（基於密度矩陣）
            # 創建密度矩陣用於相干性計算
            norm = np.linalg.norm(statevector_data)
            if norm > 0:
                statevector_normalized = statevector_data / norm
                density_matrix = np.outer(statevector_normalized, np.conj(statevector_normalized))
                # 相干性：非對角元素的總和
                diagonal_elements = np.diag(density_matrix)
                off_diagonal = density_matrix - np.diag(diagonal_elements)
                quantum_coherence = float(np.sum(np.abs(off_diagonal)))
            else:
                quantum_coherence = float(1.0 - np.sum(probabilities**2))
            
            # 4. 语义干涉（基于词频方差）
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            semantic_interference = float(np.var(list(word_counts.values())) / len(words))
            
            # 5. 框架竞争
            if len(probabilities) > 1:
                uniform_prob = 1.0 / len(probabilities)
                kl_div = np.sum(probabilities * np.log2(probabilities / (uniform_prob + 1e-12)))
                max_kl = np.log2(len(probabilities))
                frame_competition = float(1.0 - (kl_div / max_kl)) if max_kl > 0 else 0.0
            else:
                frame_competition = 0.0
            
            # 6. 情感极性
            positive_count = sum(1 for word in words if word in self.positive_words)
            negative_count = sum(1 for word in words if word in self.negative_words)
            emotional_intensity = (positive_count + negative_count) / len(words)
            
            # 7. 多重现实强度
            reality_strength = (
                superposition_strength * 0.4 +
                semantic_interference * 0.3 +
                frame_competition * 0.2 +
                emotional_intensity * 0.1
            )
            
            # 基本统计
            word_count = len(words)
            unique_words = len(set(words))
            categorical_diversity = len(set(pos_tags))
            
            return {
                'field': field_name,
                'original_text': text[:100] + '...' if len(text) > 100 else text,  # 截断长文本
                'word_count': word_count,
                'unique_words': unique_words,
                'categorical_diversity': categorical_diversity,
                'quantum_circuit_qubits': circuit.num_qubits,
                'von_neumann_entropy': von_neumann_entropy,
                'superposition_strength': superposition_strength,
                'quantum_coherence': quantum_coherence,
                'semantic_interference': semantic_interference,
                'frame_competition': frame_competition,
                'emotional_intensity': float(emotional_intensity),
                'multiple_reality_strength': float(reality_strength),
                'analysis_version': 'fast_qiskit_density_matrix_v1.0'
            }
            
        except Exception as e:
            print(f"⚠️  量子电路失败，使用经典计算: {str(e)[:50]}...")
            # 快速回退到经典计算
            return self.classical_fallback(words, pos_tags, field_name, text)

    def classical_fallback(self, words: List[str], pos_tags: List[str], field_name: str, text: str) -> Dict[str, Any]:
        """经典计算回退"""
        
        # 词频分析
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        probabilities = np.array([count/len(words) for count in word_counts.values()])
        
        # 简化指标（經典計算仍使用概率熵）
        von_neumann_entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))
        superposition_strength = float(4 * np.sum(probabilities * (1 - probabilities)))
        quantum_coherence = float(len(set(words)) / len(words))
        semantic_interference = float(np.var(list(word_counts.values())) / len(words))
        
        # 框架竞争
        if len(probabilities) > 1:
            uniform_prob = 1.0 / len(probabilities)
            kl_div = np.sum(probabilities * np.log2(probabilities / (uniform_prob + 1e-12)))
            max_kl = np.log2(len(probabilities))
            frame_competition = float(1.0 - (kl_div / max_kl)) if max_kl > 0 else 0.0
        else:
            frame_competition = 0.0
        
        # 情感分析
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        emotional_intensity = (positive_count + negative_count) / len(words)
        
        reality_strength = (
            superposition_strength * 0.4 +
            semantic_interference * 0.3 +
            frame_competition * 0.2 +
            emotional_intensity * 0.1
        )
        
        return {
            'field': field_name,
            'original_text': text[:100] + '...' if len(text) > 100 else text,
            'word_count': len(words),
            'unique_words': len(set(words)),
            'categorical_diversity': len(set(pos_tags)),
            'quantum_circuit_qubits': 0,  # 标记为经典计算
            'von_neumann_entropy': von_neumann_entropy,
            'superposition_strength': superposition_strength,
            'quantum_coherence': quantum_coherence,
            'semantic_interference': semantic_interference,
            'frame_competition': frame_competition,
            'emotional_intensity': float(emotional_intensity),
            'multiple_reality_strength': float(reality_strength),
            'analysis_version': 'fast_classical_fallback_density_matrix_v1.0'
        }

    def process_record_batch(self, records: List[Dict], record_type: str) -> List[Dict]:
        """批处理记录"""
        results = []
        
        for record in records:
            record_id = record.get('id', 0)
            
            if record_type == 'ai':
                # AI记录的字段
                fields = ['新聞標題', '影片對話', '影片描述']
                for field in fields:
                    if field in record and record[field]:
                        result = self.fast_quantum_analysis(record[field], field)
                        if result:
                            result['record_id'] = record_id
                            result['data_source'] = 'AI_Generated'
                            results.append(result)
            
            elif record_type == 'journalist':
                # 记者记录的字段
                field_mapping = {'title': '新聞標題', 'content': '新聞內容'}
                for original_field, mapped_field in field_mapping.items():
                    if original_field in record and record[original_field]:
                        result = self.fast_quantum_analysis(record[original_field], mapped_field)
                        if result:
                            result['record_id'] = record_id
                            result['data_source'] = 'Journalist_Written'
                            results.append(result)
        
        return results

def main():
    """主函数：快速执行Qiskit量子分析（密度矩陣版本）"""
    print("🚀 开始快速Qiskit量子电路分析（使用密度矩陣計算熵）...")
    print("=" * 80)
    start_time = time.time()
    
    # 創建輸出目錄
    output_dir = project_root / '20251113_densityMatrix'
    output_dir.mkdir(exist_ok=True)
    results_dir = output_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # 初始化分析器
    analyzer = FastQiskitDensityMatrixAnalyzer()
    
    # 1. 分析AI新闻数据
    print("\n📊 分析AI生成新闻...")
    ai_data_path = project_root / 'data' / 'dataseet.xlsx'
    
    if ai_data_path.exists():
        ai_df = pd.read_excel(ai_data_path)
        print(f"✅ 加载AI数据: {len(ai_df)} 条记录")
        
        ai_records = []
        for idx, record in ai_df.iterrows():
            record_dict = record.to_dict()
            record_dict['id'] = idx
            ai_records.append(record_dict)
        
        # 批处理分析
        ai_results = analyzer.process_record_batch(ai_records, 'ai')
        
        # 保存结果
        if ai_results:
            ai_results_df = pd.DataFrame(ai_results)
            ai_results_path = results_dir / 'density_matrix_ai_analysis_results.csv'
            ai_results_df.to_csv(ai_results_path, index=False, encoding='utf-8-sig')
            print(f"✅ AI分析结果已保存: {ai_results_path}")
            
            # 生成统计摘要
            numeric_cols = ['von_neumann_entropy', 'superposition_strength', 'quantum_coherence', 
                           'semantic_interference', 'frame_competition', 'multiple_reality_strength']
            ai_summary = {}
            for col in numeric_cols:
                if col in ai_results_df.columns:
                    ai_summary[col] = {
                        'mean': float(ai_results_df[col].mean()),
                        'std': float(ai_results_df[col].std()),
                        'min': float(ai_results_df[col].min()),
                        'max': float(ai_results_df[col].max()),
                        'median': float(ai_results_df[col].median())
                    }
            
            ai_summary_path = results_dir / 'density_matrix_ai_analysis_summary.json'
            with open(ai_summary_path, 'w', encoding='utf-8') as f:
                json.dump(ai_summary, f, ensure_ascii=False, indent=2)
            print(f"✅ AI统计摘要已保存: {ai_summary_path}")
    else:
        print(f"❌ 找不到AI数据文件: {ai_data_path}")
        ai_results_df = None
    
    # 2. 分析记者新闻数据
    print("\n📊 分析记者撰写新闻...")
    journalist_data_path = project_root / 'data' / 'cna.csv'
    
    if journalist_data_path.exists():
        journalist_df = pd.read_csv(journalist_data_path)
        print(f"✅ 加载记者数据: {len(journalist_df)} 条记录")
        
        journalist_records = []
        for idx, record in journalist_df.iterrows():
            record_dict = record.to_dict()
            record_dict['id'] = idx
            journalist_records.append(record_dict)
        
        # 批处理分析
        journalist_results = analyzer.process_record_batch(journalist_records, 'journalist')
        
        # 保存结果
        if journalist_results:
            journalist_results_df = pd.DataFrame(journalist_results)
            journalist_results_path = results_dir / 'density_matrix_journalist_analysis_results.csv'
            journalist_results_df.to_csv(journalist_results_path, index=False, encoding='utf-8-sig')
            print(f"✅ 记者分析结果已保存: {journalist_results_path}")
            
            # 生成统计摘要
            numeric_cols = ['von_neumann_entropy', 'superposition_strength', 'quantum_coherence', 
                           'semantic_interference', 'frame_competition', 'multiple_reality_strength']
            journalist_summary = {}
            for col in numeric_cols:
                if col in journalist_results_df.columns:
                    journalist_summary[col] = {
                        'mean': float(journalist_results_df[col].mean()),
                        'std': float(journalist_results_df[col].std()),
                        'min': float(journalist_results_df[col].min()),
                        'max': float(journalist_results_df[col].max()),
                        'median': float(journalist_results_df[col].median())
                    }
            
            journalist_summary_path = results_dir / 'density_matrix_journalist_analysis_summary.json'
            with open(journalist_summary_path, 'w', encoding='utf-8') as f:
                json.dump(journalist_summary, f, ensure_ascii=False, indent=2)
            print(f"✅ 记者统计摘要已保存: {journalist_summary_path}")
    else:
        print(f"❌ 找不到记者数据文件: {journalist_data_path}")
        journalist_results_df = None
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 密度矩陣版本Qiskit量子电路分析完成！")
    print(f"⏱️  总耗时: {total_time:.2f} 秒")
    print(f"📊 分析模式: 使用密度矩陣計算 von Neumann 熵")
    print(f"🔬 使用技术: 密度矩陣 (ρ = |ψ⟩⟨ψ|) + 特徵值分解")
    if ai_results_df is not None and journalist_results_df is not None:
        print(f"📈 数据规模: AI新闻{len(ai_results_df)}条 + 记者新闻{len(journalist_results_df)}条")
    print(f"📁 結果保存目錄: {output_dir}")

if __name__ == "__main__":
    main()

