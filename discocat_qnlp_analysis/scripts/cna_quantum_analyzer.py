#!/usr/bin/env python3
"""
中央社新聞量子框架分析器
專門用於分析台灣中央社記者撰寫的新聞內容
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
from qiskit.circuit.library import RealAmplitudes
import warnings
warnings.filterwarnings('ignore')

# 設置中文分詞
jieba.set_dictionary('../data/dict.txt.big') if os.path.exists('../data/dict.txt.big') else None

class CNAQuantumFrameAnalyzer:
    """中央社新聞量子框架分析器"""
    
    def __init__(self):
        """初始化分析器"""
        print("🔧 初始化中央社量子框架分析器...")
        
        # 初始化量子後端
        self.backend = Aer.get_backend('statevector_simulator')
        
        # 情感詞典
        self.emotion_lexicon = {
            'positive': ['成功', '獲得', '優秀', '突破', '創新', '發展', '改善', '提升', '榮獲', 
                        '卓越', '領先', '進步', '增長', '獲獎', '肯定', '支持', '合作', '共贏'],
            'negative': ['失敗', '問題', '困難', '危機', '衝突', '爭議', '批評', '質疑', '擔憂',
                        '下降', '減少', '損失', '風險', '威脅', '挑戰', '阻礙', '延遲', '取消']
        }
        
        # 改革框架詞典
        self.reform_lexicon = {
            'positive': ['改革', '創新', '變革', '轉型', '升級', '優化', '改進', '提升', '發展'],
            'reactive': ['應對', '回應', '處理', '解決', '因應', '調整', '修正', '補救', '改善'],
            'superficial': ['宣布', '聲明', '表示', '說明', '澄清', '回覆', '承諾', '保證', '強調']
        }
        
        # 語境修飾詞
        self.context_modifiers = {
            'intensifiers': ['非常', '極其', '十分', '相當', '特別', '尤其', '格外'],
            'diminishers': ['稍微', '略微', '有點', '些許', '輕微', '一定程度']
        }
        
        print("✅ 中央社量子框架分析器初始化完成")

    def extract_emotion_features(self, text: str) -> Tuple[float, float, int]:
        """提取情感特徵"""
        words = list(jieba.cut(text))
        word_count = len(words)
        
        positive_count = sum(1 for word in words if word in self.emotion_lexicon['positive'])
        negative_count = sum(1 for word in words if word in self.emotion_lexicon['negative'])
        
        # 正規化
        positive_intensity = positive_count / max(word_count, 1)
        negative_intensity = negative_count / max(word_count, 1)
        
        return positive_intensity, negative_intensity, word_count

    def analyze_syntactic_patterns(self, text: str, pos_tags: List[str]) -> Tuple[float, float, str]:
        """分析語法模式"""
        # 主動語態檢測
        active_indicators = ['VV', 'VC', 'VE']  # 動詞類別
        active_count = sum(1 for tag in pos_tags if tag in active_indicators)
        active_bonus = min(0.2, active_count / max(len(pos_tags), 1))
        
        # 未來時態檢測
        future_words = ['將', '會', '要', '即將', '預計', '計劃', '準備']
        future_count = sum(1 for word in jieba.cut(text) if word in future_words)
        future_bonus = min(0.1, future_count / max(len(pos_tags), 1))
        
        # 語境類型
        context_type = 'normal'
        if any(word in text for word in ['政府', '官方', '部門']):
            context_type = 'official'
        elif any(word in text for word in ['民眾', '公眾', '社會']):
            context_type = 'public'
            
        return active_bonus, future_bonus, context_type

    def construct_emotion_quantum_state(self, text: str) -> Tuple[np.ndarray, Dict]:
        """構建情感量子態"""
        # 詞性標註
        pos_tags = [pair.flag for pair in pseg.cut(text)]
        
        # 提取基礎特徵
        positive_intensity, negative_intensity, word_count = self.extract_emotion_features(text)
        active_bonus, future_bonus, context_type = self.analyze_syntactic_patterns(text, pos_tags)
        
        # 計算量子態振幅
        positive_base = min(1.0, positive_intensity)
        negative_base = min(1.0, negative_intensity)
        
        # 語法修正
        syntactic_modifier = 1.0 + active_bonus + future_bonus
        positive_amplitude = positive_base * syntactic_modifier
        negative_amplitude = negative_base
        
        # 中性成分
        neutral_amplitude = max(0.1, 1.0 - positive_intensity - negative_intensity)
        
        # 正規化
        raw_amplitudes = np.array([positive_amplitude, neutral_amplitude, negative_amplitude])
        norm = np.linalg.norm(raw_amplitudes)
        
        if norm > 0:
            emotion_state = raw_amplitudes / norm
        else:
            emotion_state = np.array([0.33, 0.34, 0.33])  # 均勻分布
        
        metadata = {
            'positive_intensity': positive_intensity,
            'negative_intensity': negative_intensity,
            'active_voice_bonus': active_bonus,
            'future_tense_bonus': future_bonus,
            'context_type': context_type
        }
        
        return emotion_state, metadata

    def construct_reform_quantum_state(self, text: str, context: str) -> Tuple[np.ndarray, Dict]:
        """構建改革量子態"""
        words = list(jieba.cut(text))
        
        # 計算各框架詞頻
        positive_count = sum(1 for word in words if word in self.reform_lexicon['positive'])
        reactive_count = sum(1 for word in words if word in self.reform_lexicon['reactive'])
        superficial_count = sum(1 for word in words if word in self.reform_lexicon['superficial'])
        
        total_reform_words = positive_count + reactive_count + superficial_count
        
        if total_reform_words > 0:
            # 基於詞頻分布
            positive_ratio = positive_count / total_reform_words
            reactive_ratio = reactive_count / total_reform_words
            superficial_ratio = superficial_count / total_reform_words
        else:
            # 默認均勻分布
            positive_ratio = reactive_ratio = superficial_ratio = 1/3
        
        # 語境調整
        context_modifier = 1.0
        if context == 'official':
            positive_ratio *= 1.2  # 官方語境增強積極改革
        elif context == 'public':
            reactive_ratio *= 1.1  # 民眾語境增強反應性
        
        # 構建量子態
        reform_amplitudes = np.array([positive_ratio, reactive_ratio, superficial_ratio])
        reform_amplitudes = reform_amplitudes / np.linalg.norm(reform_amplitudes)
        
        metadata = {
            'reform_word_count': total_reform_words,
            'positive_reform_ratio': positive_ratio,
            'reactive_reform_ratio': reactive_ratio,
            'superficial_reform_ratio': superficial_ratio
        }
        
        return reform_amplitudes, metadata

    def create_quantum_circuit_with_frames(self, emotion_state: np.ndarray, 
                                         reform_state: np.ndarray, 
                                         text_complexity: float) -> QuantumCircuit:
        """創建量子電路"""
        circuit = QuantumCircuit(6)
        
        # 使用旋轉門來設置量子態，而不是直接初始化
        # 情感框架 (qubits 0-2)
        theta_emotion = np.arccos(np.sqrt(emotion_state[0])) * 2
        phi_emotion = np.arccos(np.sqrt(emotion_state[1] / (emotion_state[1] + emotion_state[2] + 1e-10))) * 2
        
        circuit.ry(theta_emotion, 0)
        circuit.ry(phi_emotion, 1)
        
        # 改革框架 (qubits 3-5)
        theta_reform = np.arccos(np.sqrt(reform_state[0])) * 2
        phi_reform = np.arccos(np.sqrt(reform_state[1] / (reform_state[1] + reform_state[2] + 1e-10))) * 2
        
        circuit.ry(theta_reform, 3)
        circuit.ry(phi_reform, 4)
        
        # 根據文本複雜度添加糾纏
        if text_complexity > 0.5:
            circuit.cx(0, 3)  # 情感-改革糾纏
            circuit.cx(1, 4)
        
        return circuit

    def measure_quantum_frame_properties(self, circuit: QuantumCircuit, 
                                       emotion_state: np.ndarray, 
                                       reform_state: np.ndarray) -> Dict[str, float]:
        """測量量子框架屬性 - 簡化版本"""
        try:
            # 執行量子電路
            job = execute(circuit, self.backend, shots=1)
            result = job.result()
            statevector = result.get_statevector()
            
            # 計算機率分布
            probabilities = np.abs(statevector) ** 2
            valid_probs = probabilities[probabilities > 1e-10]
        except Exception as e:
            print(f"⚠️ 量子電路執行失敗，使用經典近似: {e}")
            # 使用經典近似
            combined_state = np.concatenate([emotion_state, reform_state])
            probabilities = combined_state ** 2
            valid_probs = probabilities[probabilities > 1e-10]
        
        metrics = {}
        
        # 1. 框架競爭強度 (冯纽曼熵 + KL 參考)
        emotion_entropy = -np.sum(emotion_state**2 * np.log2(emotion_state**2 + 1e-12))
        reform_entropy = -np.sum(reform_state**2 * np.log2(reform_state**2 + 1e-12))
        total_entropy = -np.sum(valid_probs * np.log2(valid_probs + 1e-12))
        metrics['frame_competition'] = float(min(1.0, total_entropy * 0.5))
        if len(valid_probs) > 1:
            uniform_prob = 1.0 / len(valid_probs)
            kl_divergence = np.sum(valid_probs * np.log2((valid_probs + 1e-12) / uniform_prob))
            max_kl = np.log2(len(valid_probs))
            metrics['frame_competition_kl'] = float(1.0 - min(1.0, kl_divergence / max_kl))
        else:
            metrics['frame_competition_kl'] = 0.0
        
        # 2. 框架糾纏強度
        metrics['frame_entanglement'] = float(max(0.0, total_entropy - emotion_entropy - reform_entropy))
        
        # 3. 馮紐曼熵
        metrics['von_neumann_entropy'] = float(total_entropy)
        
        # 4. 語義干涉
        phase_variance = np.var(np.angle(statevector[np.abs(statevector) > 1e-10]))
        metrics['semantic_interference'] = float(phase_variance / (np.pi**2))
        
        # 5. 框架強度
        metrics['emotion_frame_strength'] = float(np.max(emotion_state))
        metrics['reform_frame_strength'] = float(np.max(reform_state))
        
        return metrics

    def analyze_multiple_realities_with_frames(self, quantum_metrics: Dict, 
                                             emotion_metadata: Dict, 
                                             reform_metadata: Dict) -> Dict[str, float]:
        """基於量子框架分析多重現實"""
        
        # 框架多樣性
        frame_diversity = 0.0
        if emotion_metadata['positive_intensity'] > 0.1:
            frame_diversity += 0.3
        if emotion_metadata['negative_intensity'] > 0.1:
            frame_diversity += 0.3
        if reform_metadata['reform_word_count'] > 0:
            frame_diversity += 0.4
        
        # 多重現實強度
        reality_strength = (
            quantum_metrics['frame_competition'] * 0.4 +
            quantum_metrics['semantic_interference'] * 0.3 +
            frame_diversity * 0.3
        )
        
        # 框架衝突強度
        conflict_strength = (
            quantum_metrics['frame_entanglement'] * 0.5 +
            abs(emotion_metadata['positive_intensity'] - emotion_metadata['negative_intensity']) * 0.3 +
            quantum_metrics['frame_competition'] * 0.2
        )
        
        # 語義模糊度
        ambiguity = (
            quantum_metrics['von_neumann_entropy'] * 0.5 +
            quantum_metrics['semantic_interference'] * 0.3 +
            (1.0 - max(quantum_metrics['emotion_frame_strength'], quantum_metrics['reform_frame_strength'])) * 0.2
        )
        
        return {
            'multiple_reality_strength': min(1.0, max(0.0, reality_strength)),
            'frame_conflict_strength': min(1.0, max(0.0, conflict_strength)),
            'semantic_ambiguity': min(1.0, max(0.0, ambiguity))
        }

    def process_cna_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """處理單條中央社記錄"""
        try:
            # 提取文本
            title = str(record.get('title', ''))
            content = str(record.get('content', ''))
            
            # 分析標題
            title_emotion_state, title_emotion_meta = self.construct_emotion_quantum_state(title)
            title_reform_state, title_reform_meta = self.construct_reform_quantum_state(
                title, title_emotion_meta['context_type'])
            
            # 創建量子電路
            title_complexity = len(title) / 100.0  # 簡單的複雜度指標
            title_circuit = self.create_quantum_circuit_with_frames(
                title_emotion_state, title_reform_state, title_complexity)
            
            # 測量量子屬性
            title_quantum_metrics = self.measure_quantum_frame_properties(
                title_circuit, title_emotion_state, title_reform_state)
            
            # 分析多重現實
            title_reality_metrics = self.analyze_multiple_realities_with_frames(
                title_quantum_metrics, title_emotion_meta, title_reform_meta)
            
            # 分析內容 (如果有)
            if content and len(content) > 10:
                content_emotion_state, content_emotion_meta = self.construct_emotion_quantum_state(content)
                content_reform_state, content_reform_meta = self.construct_reform_quantum_state(
                    content, content_emotion_meta['context_type'])
                
                content_complexity = len(content) / 1000.0
                content_circuit = self.create_quantum_circuit_with_frames(
                    content_emotion_state, content_reform_state, content_complexity)
                
                content_quantum_metrics = self.measure_quantum_frame_properties(
                    content_circuit, content_emotion_state, content_reform_state)
                
                content_reality_metrics = self.analyze_multiple_realities_with_frames(
                    content_quantum_metrics, content_emotion_meta, content_reform_meta)
            else:
                # 使用標題數據
                content_emotion_state = title_emotion_state
                content_emotion_meta = title_emotion_meta
                content_reform_state = title_reform_state
                content_reform_meta = title_reform_meta
                content_quantum_metrics = title_quantum_metrics
                content_reality_metrics = title_reality_metrics
            
            # 組合結果
            results = []
            
            # 標題結果
            title_result = {
                'record_id': record.get('record_id', 0),
                'field': '新聞標題',
                'original_text': title,
                'word_count': len(list(jieba.cut(title))),
                'emotion_positive_amplitude': float(title_emotion_state[0]),
                'emotion_neutral_amplitude': float(title_emotion_state[1]),
                'emotion_negative_amplitude': float(title_emotion_state[2]),
                'reform_positive_amplitude': float(title_reform_state[0]),
                'reform_reactive_amplitude': float(title_reform_state[1]),
                'reform_superficial_amplitude': float(title_reform_state[2]),
                'positive_emotion_intensity': title_emotion_meta['positive_intensity'],
                'negative_emotion_intensity': title_emotion_meta['negative_intensity'],
                'active_voice_bonus': title_emotion_meta['active_voice_bonus'],
                'future_tense_bonus': title_emotion_meta['future_tense_bonus'],
                'context_type': title_emotion_meta['context_type'],
                'reform_word_count': title_reform_meta['reform_word_count'],
                **title_quantum_metrics,
                **title_reality_metrics,
                'circuit_depth': 2,
                'circuit_gates': 7,
                'qubit_count': 6,
                'quantum_frames_enabled': True,
                'analysis_version': 'cna_quantum_frames_v1.0'
            }
            results.append(title_result)
            
            # 內容結果
            content_result = {
                'record_id': record.get('record_id', 0),
                'field': '新聞內容',
                'original_text': content[:200] + '...' if len(content) > 200 else content,
                'word_count': len(list(jieba.cut(content))),
                'emotion_positive_amplitude': float(content_emotion_state[0]),
                'emotion_neutral_amplitude': float(content_emotion_state[1]),
                'emotion_negative_amplitude': float(content_emotion_state[2]),
                'reform_positive_amplitude': float(content_reform_state[0]),
                'reform_reactive_amplitude': float(content_reform_state[1]),
                'reform_superficial_amplitude': float(content_reform_state[2]),
                'positive_emotion_intensity': content_emotion_meta['positive_intensity'],
                'negative_emotion_intensity': content_emotion_meta['negative_intensity'],
                'active_voice_bonus': content_emotion_meta['active_voice_bonus'],
                'future_tense_bonus': content_emotion_meta['future_tense_bonus'],
                'context_type': content_emotion_meta['context_type'],
                'reform_word_count': content_reform_meta['reform_word_count'],
                **content_quantum_metrics,
                **content_reality_metrics,
                'circuit_depth': 2,
                'circuit_gates': 7,
                'qubit_count': 6,
                'quantum_frames_enabled': True,
                'analysis_version': 'cna_quantum_frames_v1.0'
            }
            results.append(content_result)
            
            return results
            
        except Exception as e:
            print(f"❌ 處理記錄時出錯: {e}")
            return []

def main():
    """主執行函數"""
    
    print("🚀 啟動中央社新聞量子框架分析")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = CNAQuantumFrameAnalyzer()
    
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
            
            if processed_count % 50 == 0:
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
        # 創建空的DataFrame
        results_df = pd.DataFrame()
        print("⚠️  沒有成功處理任何記錄")
        return
    
    print(f"\n📊 處理完成統計:")
    print(f"   - 成功處理: {processed_count}/{len(df)} 條原始記錄")
    print(f"   - 生成結果: {len(results_df)} 條分析記錄")
    
    # 保存結果
    results_file = '../results/cna_quantum_frame_analysis_results.csv'
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    print(f"💾 詳細結果已保存: {results_file}")
    
    # 計算統計摘要
    numeric_columns = [
        'emotion_positive_amplitude', 'emotion_neutral_amplitude', 'emotion_negative_amplitude',
        'reform_positive_amplitude', 'reform_reactive_amplitude', 'reform_superficial_amplitude',
        'positive_emotion_intensity', 'negative_emotion_intensity',
        'frame_competition', 'emotion_frame_strength', 'reform_frame_strength',
        'frame_entanglement', 'von_neumann_entropy', 'semantic_interference',
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
    summary_file = '../results/cna_quantum_frame_analysis_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)
    
    print(f"📊 統計摘要已保存: {summary_file}")
    
    # 性能摘要
    total_time = time.time() - start_time
    print(f"\n✅ 中央社量子框架分析完成!")
    print(f"⏱️  總耗時: {total_time/60:.1f} 分鐘")
    print(f"🚀 處理速度: {processed_count/total_time:.1f} 記錄/秒")
    print(f"📈 成功處理: {len(results_df)} 條分析記錄")
    
    # 顯示樣本結果
    print(f"\n📋 中央社量子框架分析結果預覽:")
    for field in ['新聞標題', '新聞內容']:
        field_sample = results_df[results_df['field'] == field].iloc[0] if not results_df[results_df['field'] == field].empty else None
        if field_sample is not None:
            print(f"\n{field}:")
            print(f"  文本: {field_sample['original_text'][:100]}...")
            print(f"  情感框架: +{field_sample['emotion_positive_amplitude']:.3f} ±{field_sample['emotion_neutral_amplitude']:.3f} -{field_sample['emotion_negative_amplitude']:.3f}")
            print(f"  改革框架: +{field_sample['reform_positive_amplitude']:.3f} ±{field_sample['reform_reactive_amplitude']:.3f} -{field_sample['reform_superficial_amplitude']:.3f}")
            print(f"  框架競爭: {field_sample['frame_competition']:.4f}")
            print(f"  框架糾纏: {field_sample['frame_entanglement']:.4f}")
            print(f"  多重現實強度: {field_sample['multiple_reality_strength']:.4f}")

if __name__ == "__main__":
    main()
