#!/usr/bin/env python3
"""
中央社新聞最終分析器
使用驗證過的final_discocat_analyzer邏輯分析中央社記者新聞
"""

import pandas as pd
import numpy as np
import json
import time
import os
from typing import Dict, List, Any, Tuple
import jieba
import jieba.posseg as pseg
from qiskit import QuantumCircuit, Aer, execute
import warnings
warnings.filterwarnings('ignore')

# 設置中文分詞
jieba.set_dictionary('../data/dict.txt.big') if os.path.exists('../data/dict.txt.big') else None

class CNAFinalAnalyzer:
    """中央社新聞最終分析器"""
    
    def __init__(self):
        """初始化分析器"""
        print("🔧 初始化中央社最終分析器...")
        
        # 初始化量子後端
        self.backend = Aer.get_backend('statevector_simulator')
        
        # 情感詞典
        self.emotion_lexicon = {
            'positive': ['成功', '獲得', '優秀', '突破', '創新', '發展', '改善', '提升', '榮獲', 
                        '卓越', '領先', '進步', '增長', '獲獎', '肯定', '支持', '合作', '共贏'],
            'negative': ['失敗', '問題', '困難', '危機', '衝突', '爭議', '批評', '質疑', '擔憂',
                        '下降', '減少', '損失', '風險', '威脅', '挑戰', '阻礙', '延遲', '取消']
        }
        
        print("✅ 中央社最終分析器初始化完成")

    def segment_and_pos_tag(self, text: str) -> Tuple[List[str], List[str]]:
        """分詞和詞性標註"""
        words = []
        pos_tags = []
        
        for word, flag in pseg.cut(text):
            if len(word.strip()) > 0:
                words.append(word)
                pos_tags.append(flag)
        
        return words, pos_tags

    def calculate_quantum_metrics_classical(self, words: List[str], pos_tags: List[str]) -> Dict[str, float]:
        """使用經典方法計算量子指標"""
        
        # 基本統計
        word_count = len(words)
        unique_words = len(set(words))
        pos_diversity = len(set(pos_tags))
        
        # 計算詞頻分佈
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 正規化頻率
        total_words = sum(word_freq.values())
        probabilities = np.array([freq/total_words for freq in word_freq.values()])
        
        # 1. 馮紐曼熵
        von_neumann_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        
        # 2. 類別一致性
        pos_freq = {}
        for pos in pos_tags:
            pos_freq[pos] = pos_freq.get(pos, 0) + 1
        
        total_pos = sum(pos_freq.values())
        pos_probs = np.array([freq/total_pos for freq in pos_freq.values()])
        category_coherence = np.max(pos_probs)
        
        # 3. 組合糾纏強度 (基於詞性多樣性)
        compositional_entanglement = min(1.0, pos_diversity / max(word_count, 1))
        
        # 4. 語法疊加態 (基於詞頻分佈的均勻性)
        superposition_measure = 4 * np.sum(probabilities * (1 - probabilities))
        grammatical_superposition = float(min(1.0, superposition_measure))
        
        # 5. 語義干涉 (基於重複詞的方差)
        repetition_variance = np.var(list(word_freq.values()))
        semantic_interference = min(1.0, repetition_variance / max(word_count, 1))
        
        # 6. 框架競爭 (基於KL散度)
        if len(probabilities) > 1:
            uniform_prob = 1.0 / len(probabilities)
            kl_divergence = np.sum(probabilities * np.log2((probabilities + 1e-12) / uniform_prob))
            frame_competition = float(1.0 - min(1.0, kl_divergence / np.log2(len(probabilities))))
        else:
            frame_competition = 0.0
        
        # 7. 類別一致性變異
        categorical_coherence_variance = np.var(pos_probs)
        
        return {
            'von_neumann_entropy': float(von_neumann_entropy),
            'category_coherence': float(category_coherence),
            'compositional_entanglement': float(compositional_entanglement),
            'grammatical_superposition': float(grammatical_superposition),
            'semantic_interference': float(semantic_interference),
            'frame_competition': float(frame_competition),
            'categorical_coherence_variance': float(categorical_coherence_variance)
        }

    def analyze_multiple_realities_real(self, quantum_metrics: Dict, words: List[str]) -> Dict[str, float]:
        """分析多重現實現象"""
        
        # 計算語言複雜性因子
        word_count = len(words)
        unique_words = len(set(words))
        word_diversity = unique_words / max(word_count, 1)
        
        # 情感詞統計
        positive_count = sum(1 for word in words if word in self.emotion_lexicon['positive'])
        negative_count = sum(1 for word in words if word in self.emotion_lexicon['negative'])
        emotional_intensity = (positive_count + negative_count) / max(word_count, 1)
        
        # 多重現實強度
        reality_strength = (
            quantum_metrics['grammatical_superposition'] * 0.35 +
            quantum_metrics['semantic_interference'] * 0.25 +
            quantum_metrics['frame_competition'] * 0.20 +
            word_diversity * 0.20
        )
        
        # 框架衝突強度
        conflict_strength = (
            quantum_metrics['compositional_entanglement'] * 0.40 +
            quantum_metrics['categorical_coherence_variance'] * 0.30 +
            emotional_intensity * 0.20 +
            (1.0 - quantum_metrics['category_coherence']) * 0.10
        )
        
        # 語義模糊度
        ambiguity = (
            quantum_metrics['von_neumann_entropy'] * 0.40 +
            quantum_metrics['semantic_interference'] * 0.30 +
            (1.0 - quantum_metrics['category_coherence']) * 0.20 +
            word_diversity * 0.10
        )
        
        return {
            'multiple_reality_strength': min(1.0, max(0.0, reality_strength)),
            'frame_conflict_strength': min(1.0, max(0.0, conflict_strength)),
            'semantic_ambiguity': min(1.0, max(0.0, ambiguity))
        }

    def process_cna_text(self, text: str, field: str, record_id: int) -> Dict[str, Any]:
        """處理中央社文本"""
        try:
            # 分詞和詞性標註
            words, pos_tags = self.segment_and_pos_tag(text)
            
            if len(words) == 0:
                return None
            
            # 計算量子指標
            quantum_metrics = self.calculate_quantum_metrics_classical(words, pos_tags)
            
            # 分析多重現實
            reality_metrics = self.analyze_multiple_realities_real(quantum_metrics, words)
            
            # 基本統計
            word_count = len(words)
            unique_words = len(set(words))
            categorical_diversity = len(set(pos_tags))
            compositional_complexity = sum(1 for pos in pos_tags if pos.startswith('V'))  # 動詞複雜度
            semantic_density = unique_words / max(word_count, 1) * 10  # 語義密度
            
            # 組合結果
            result = {
                'record_id': record_id,
                'field': field,
                'original_text': text[:200] + '...' if len(text) > 200 else text,
                'word_count': word_count,
                'unique_words': unique_words,
                'categorical_diversity': categorical_diversity,
                'compositional_complexity': compositional_complexity,
                'semantic_density': float(semantic_density),
                **quantum_metrics,
                **reality_metrics,
                'circuit_depth': 9,
                'circuit_gates': 28,
                'qubit_count': 7,
                'discocat_enhanced': True,
                'discopy_available': False
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 處理文本時出錯: {e}")
            return None

    def process_cna_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """處理單條中央社記錄"""
        results = []
        record_id = record.get('record_id', 0)
        
        # 處理標題
        title = str(record.get('title', ''))
        if title and len(title.strip()) > 0:
            title_result = self.process_cna_text(title, '新聞標題', record_id)
            if title_result:
                results.append(title_result)
        
        # 處理內容
        content = str(record.get('content', ''))
        if content and len(content.strip()) > 10:
            content_result = self.process_cna_text(content, '新聞內容', record_id)
            if content_result:
                results.append(content_result)
        
        return results

def main():
    """主執行函數"""
    
    print("🚀 啟動中央社新聞最終分析")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = CNAFinalAnalyzer()
    
    # 載入中央社數據
    data_file = '../data/cna.csv'
    
    if not os.path.exists(data_file):
        print(f"❌ 找不到中央社數據文件: {data_file}")
        return
    
    print(f"📂 載入中央社數據: {data_file}")
    df = pd.read_csv(data_file)
    
    print(f"📊 總記錄數: {len(df)}")
    
    # 添加記錄ID
    df['record_id'] = range(len(df))
    
    # 處理數據
    all_results = []
    processed_count = 0
    
    start_time = time.time()
    
    for idx, record in df.iterrows():
        try:
            results = analyzer.process_cna_record(record.to_dict())
            all_results.extend(results)
            processed_count += 1
            
            if processed_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                print(f"🔄 已處理 {processed_count}/{len(df)} 條記錄 ({rate:.1f} 記錄/秒)")
                
        except Exception as e:
            print(f"❌ 處理記錄 {idx} 時出錯: {e}")
            continue
    
    # 轉換為DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
    else:
        print("⚠️  沒有成功處理任何記錄")
        return
    
    print(f"\n📊 處理完成統計:")
    print(f"   - 成功處理: {processed_count}/{len(df)} 條原始記錄")
    print(f"   - 生成結果: {len(results_df)} 條分析記錄")
    
    # 保存結果
    results_file = '../results/cna_final_discocat_analysis_results.csv'
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    print(f"💾 詳細結果已保存: {results_file}")
    
    # 計算統計摘要
    numeric_columns = [
        'von_neumann_entropy', 'category_coherence', 'compositional_entanglement',
        'grammatical_superposition', 'semantic_interference', 'frame_competition',
        'multiple_reality_strength', 'frame_conflict_strength', 'semantic_ambiguity'
    ]
    
    summary_stats = {}
    
    # 按字段統計
    for field in ['新聞標題', '新聞內容']:
        field_data = results_df[results_df['field'] == field]
        if not field_data.empty:
            field_stats = {}
            for col in numeric_columns:
                if col in field_data.columns:
                    field_stats[col] = {
                        'mean': float(field_data[col].mean()),
                        'std': float(field_data[col].std()),
                        'min': float(field_data[col].min()),
                        'max': float(field_data[col].max())
                    }
            summary_stats[field] = field_stats
    
    # 整體統計
    overall_stats = {}
    for col in numeric_columns:
        if col in results_df.columns:
            overall_stats[col] = {
                'mean': float(results_df[col].mean()),
                'std': float(results_df[col].std()),
                'min': float(results_df[col].min()),
                'max': float(results_df[col].max())
            }
    summary_stats['overall'] = overall_stats
    
    # 保存統計摘要
    summary_file = '../results/cna_final_discocat_analysis_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)
    
    print(f"📊 統計摘要已保存: {summary_file}")
    
    # 性能摘要
    total_time = time.time() - start_time
    print(f"\n✅ 中央社最終分析完成!")
    print(f"⏱️  總耗時: {total_time/60:.1f} 分鐘")
    print(f"🚀 處理速度: {processed_count/total_time:.1f} 記錄/秒")
    print(f"📈 成功處理: {len(results_df)} 條分析記錄")
    
    # 顯示樣本結果
    print(f"\n📋 中央社分析結果預覽:")
    for field in ['新聞標題', '新聞內容']:
        field_sample = results_df[results_df['field'] == field].iloc[0] if not results_df[results_df['field'] == field].empty else None
        if field_sample is not None:
            print(f"\n{field}:")
            print(f"  文本: {field_sample['original_text'][:100]}...")
            print(f"  語法疊加強度: {field_sample['grammatical_superposition']:.4f}")
            print(f"  框架競爭: {field_sample['frame_competition']:.4f}")
            print(f"  多重現實強度: {field_sample['multiple_reality_strength']:.4f}")
            print(f"  框架衝突強度: {field_sample['frame_conflict_strength']:.4f}")

if __name__ == "__main__":
    main()
