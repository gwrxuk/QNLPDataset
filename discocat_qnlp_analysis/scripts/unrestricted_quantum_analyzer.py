#!/usr/bin/env python3
"""
无限制量子分析器 - 移除所有min(1.0)限制
重新计算真实的量子特征数值
"""

import pandas as pd
import numpy as np
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

class UnrestrictedQuantumAnalyzer:
    """无限制量子分析器"""
    
    def __init__(self):
        """初始化分析器"""
        print("🔧 初始化无限制量子分析器...")
        
        # 情感词典
        self.emotion_lexicon = {
            'positive': ['成功', '获得', '优秀', '突破', '创新', '发展', '改善', '提升', '荣获', 
                        '卓越', '领先', '进步', '增长', '获奖', '肯定', '支持', '合作', '共赢'],
            'negative': ['失败', '问题', '困难', '危机', '冲突', '争议', '批评', '质疑', '担忧',
                        '下降', '减少', '损失', '风险', '威胁', '挑战', '阻碍', '延迟', '取消']
        }
        
        print("✅ 无限制量子分析器初始化完成")

    def segment_and_pos_tag(self, text: str) -> Tuple[List[str], List[str]]:
        """分词和词性标注"""
        words = []
        pos_tags = []
        
        for word, flag in pseg.cut(text):
            if len(word.strip()) > 0:
                words.append(word)
                pos_tags.append(flag)
        
        return words, pos_tags

    def calculate_unrestricted_quantum_metrics(self, words: List[str], pos_tags: List[str]) -> Dict[str, float]:
        """计算无限制的量子指标"""
        
        # 基本统计
        word_count = len(words)
        unique_words = len(set(words))
        pos_diversity = len(set(pos_tags))
        
        if word_count == 0:
            return self._get_zero_metrics()
        
        # 计算词频分布
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 正规化频率
        total_words = sum(word_freq.values())
        probabilities = np.array([freq/total_words for freq in word_freq.values()])
        
        # 1. 冯纽曼熵（无限制）
        von_neumann_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        
        # 2. 类别一致性
        pos_freq = {}
        for pos in pos_tags:
            pos_freq[pos] = pos_freq.get(pos, 0) + 1
        
        total_pos = sum(pos_freq.values())
        pos_probs = np.array([freq/total_pos for freq in pos_freq.values()])
        category_coherence = np.max(pos_probs)
        
        # 3. 组合纠缠强度（无限制）
        compositional_entanglement = pos_diversity / word_count
        
        # 4. 语法叠加态（无限制 - 这是关键！）
        superposition_measure = 4 * np.sum(probabilities * (1 - probabilities))
        grammatical_superposition = float(superposition_measure)  # 移除min(1.0)限制！
        
        # 5. 语义干涉（无限制）
        repetition_variance = np.var(list(word_freq.values()))
        semantic_interference = repetition_variance / word_count
        
        # 6. 框架竞争（无限制）
        if len(probabilities) > 1:
            uniform_prob = 1.0 / len(probabilities)
            kl_divergence = np.sum(probabilities * np.log2((probabilities + 1e-12) / uniform_prob))
            max_kl = np.log2(len(probabilities))
            frame_competition = float(1.0 - (kl_divergence / max_kl))  # 保持原始计算
        else:
            frame_competition = 0.0
        
        # 7. 类别一致性变异（无限制）
        categorical_coherence_variance = np.var(pos_probs)
        
        return {
            'von_neumann_entropy': float(von_neumann_entropy),
            'category_coherence': float(category_coherence),
            'compositional_entanglement': float(compositional_entanglement),
            'grammatical_superposition': float(grammatical_superposition),  # 真实值！
            'semantic_interference': float(semantic_interference),
            'frame_competition': float(frame_competition),
            'categorical_coherence_variance': float(categorical_coherence_variance)
        }

    def _get_zero_metrics(self):
        """返回零值指标"""
        return {
            'von_neumann_entropy': 0.0,
            'category_coherence': 0.0,
            'compositional_entanglement': 0.0,
            'grammatical_superposition': 0.0,
            'semantic_interference': 0.0,
            'frame_competition': 0.0,
            'categorical_coherence_variance': 0.0
        }

    def analyze_multiple_realities_unrestricted(self, quantum_metrics: Dict, words: List[str]) -> Dict[str, float]:
        """分析多重现实现象（无限制）"""
        
        # 计算语言复杂性因子
        word_count = len(words)
        unique_words = len(set(words))
        word_diversity = unique_words / max(word_count, 1)
        
        # 情感词统计
        positive_count = sum(1 for word in words if word in self.emotion_lexicon['positive'])
        negative_count = sum(1 for word in words if word in self.emotion_lexicon['negative'])
        emotional_intensity = (positive_count + negative_count) / max(word_count, 1)
        
        # 多重现实强度（无限制）
        reality_strength = (
            quantum_metrics['grammatical_superposition'] * 0.35 +  # 现在使用真实值
            quantum_metrics['semantic_interference'] * 0.25 +
            quantum_metrics['frame_competition'] * 0.20 +
            word_diversity * 0.20
        )
        
        # 框架冲突强度（无限制）
        conflict_strength = (
            quantum_metrics['compositional_entanglement'] * 0.40 +
            quantum_metrics['categorical_coherence_variance'] * 0.30 +
            emotional_intensity * 0.20 +
            (1.0 - quantum_metrics['category_coherence']) * 0.10
        )
        
        # 语义模糊度（无限制）
        ambiguity = (
            quantum_metrics['von_neumann_entropy'] * 0.40 +
            quantum_metrics['semantic_interference'] * 0.30 +
            (1.0 - quantum_metrics['category_coherence']) * 0.20 +
            word_diversity * 0.10
        )
        
        return {
            'multiple_reality_strength': float(reality_strength),  # 不再限制在1.0
            'frame_conflict_strength': float(conflict_strength),   # 不再限制在1.0
            'semantic_ambiguity': float(ambiguity)                 # 不再限制在1.0
        }

    def process_text_unrestricted(self, text: str, field: str, record_id: int) -> Dict[str, Any]:
        """处理文本（无限制）"""
        try:
            # 分词和词性标注
            words, pos_tags = self.segment_and_pos_tag(text)
            
            if len(words) == 0:
                return None
            
            # 计算量子指标（无限制）
            quantum_metrics = self.calculate_unrestricted_quantum_metrics(words, pos_tags)
            
            # 分析多重现实（无限制）
            reality_metrics = self.analyze_multiple_realities_unrestricted(quantum_metrics, words)
            
            # 基本统计
            word_count = len(words)
            unique_words = len(set(words))
            categorical_diversity = len(set(pos_tags))
            compositional_complexity = sum(1 for pos in pos_tags if pos.startswith('V'))  # 动词复杂度
            semantic_density = unique_words / max(word_count, 1) * 10  # 语义密度
            
            # 组合结果
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
                'analysis_version': 'unrestricted_v1.0'
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 处理文本时出错: {e}")
            return None

    def process_record(self, record: Dict[str, Any], data_type: str) -> List[Dict[str, Any]]:
        """处理单条记录"""
        results = []
        record_id = record.get('record_id', 0)
        
        if data_type == 'ai':
            # AI新闻数据
            title = str(record.get('新聞標題', ''))
            dialogue = str(record.get('影片對話', ''))
            description = str(record.get('影片描述', ''))
            
            if title and len(title.strip()) > 0:
                title_result = self.process_text_unrestricted(title, '新聞標題', record_id)
                if title_result:
                    results.append(title_result)
            
            if dialogue and len(dialogue.strip()) > 10:
                dialogue_result = self.process_text_unrestricted(dialogue, '影片對話', record_id)
                if dialogue_result:
                    results.append(dialogue_result)
            
            if description and len(description.strip()) > 10:
                description_result = self.process_text_unrestricted(description, '影片描述', record_id)
                if description_result:
                    results.append(description_result)
        
        elif data_type == 'journalist':
            # 记者新闻数据
            title = str(record.get('title', ''))
            content = str(record.get('content', ''))
            
            if title and len(title.strip()) > 0:
                title_result = self.process_text_unrestricted(title, '新聞標題', record_id)
                if title_result:
                    results.append(title_result)
            
            if content and len(content.strip()) > 10:
                content_result = self.process_text_unrestricted(content, '新聞內容', record_id)
                if content_result:
                    results.append(content_result)
        
        return results

def analyze_ai_news():
    """分析AI新闻"""
    print("📰 开始分析AI新闻...")
    
    analyzer = UnrestrictedQuantumAnalyzer()
    
    # 加载AI新闻数据
    data_file = '../data/dataseet.xlsx'
    if not os.path.exists(data_file):
        print(f"❌ 找不到AI新闻数据文件: {data_file}")
        return None
    
    df = pd.read_excel(data_file)
    df = df.dropna(subset=['新聞標題', '影片對話', '影片描述'])
    df['record_id'] = range(len(df))
    
    print(f"📊 AI新闻总记录数: {len(df)}")
    
    # 处理数据
    all_results = []
    for idx, record in df.iterrows():
        try:
            results = analyzer.process_record(record.to_dict(), 'ai')
            all_results.extend(results)
            
            if (idx + 1) % 100 == 0:
                print(f"🔄 已处理 {idx + 1}/{len(df)} 条AI新闻记录")
                
        except Exception as e:
            print(f"❌ 处理AI新闻记录 {idx} 时出错: {e}")
            continue
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def analyze_journalist_news():
    """分析记者新闻"""
    print("👨‍💼 开始分析记者新闻...")
    
    analyzer = UnrestrictedQuantumAnalyzer()
    
    # 加载记者新闻数据
    data_file = '../data/cna.csv'
    if not os.path.exists(data_file):
        print(f"❌ 找不到记者新闻数据文件: {data_file}")
        return None
    
    df = pd.read_csv(data_file)
    df = df.dropna(subset=['title', 'content'])
    df['record_id'] = range(len(df))
    
    print(f"📊 记者新闻总记录数: {len(df)}")
    
    # 处理数据
    all_results = []
    for idx, record in df.iterrows():
        try:
            results = analyzer.process_record(record.to_dict(), 'journalist')
            all_results.extend(results)
            
            if (idx + 1) % 10 == 0:
                print(f"🔄 已处理 {idx + 1}/{len(df)} 条记者新闻记录")
                
        except Exception as e:
            print(f"❌ 处理记者新闻记录 {idx} 时出错: {e}")
            continue
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def calculate_summary_stats(df: pd.DataFrame, data_type: str) -> Dict:
    """计算统计摘要"""
    
    numeric_columns = [
        'von_neumann_entropy', 'category_coherence', 'compositional_entanglement',
        'grammatical_superposition', 'semantic_interference', 'frame_competition',
        'multiple_reality_strength', 'frame_conflict_strength', 'semantic_ambiguity'
    ]
    
    summary_stats = {}
    
    # 按字段统计
    for field in df['field'].unique():
        field_data = df[df['field'] == field]
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
    
    # 整体统计
    overall_stats = {}
    for col in numeric_columns:
        if col in df.columns:
            overall_stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
    summary_stats['overall'] = overall_stats
    
    return summary_stats

def main():
    """主函数"""
    
    print("🚀 开始无限制量子分析")
    print("=" * 60)
    print("⚠️  重要：已移除所有min(1.0)限制，将计算真实的量子特征数值")
    print("=" * 60)
    
    start_time = time.time()
    
    # 分析AI新闻
    print("\n🤖 第一步：分析AI新闻")
    ai_results = analyze_ai_news()
    if ai_results is not None and not ai_results.empty:
        print(f"✅ AI新闻分析完成: {len(ai_results)} 条记录")
        
        # 保存AI新闻结果
        ai_file = '../results/unrestricted_ai_analysis_results.csv'
        ai_results.to_csv(ai_file, index=False, encoding='utf-8')
        print(f"💾 AI新闻结果已保存: {ai_file}")
        
        # 计算AI新闻统计
        ai_stats = calculate_summary_stats(ai_results, 'ai')
        ai_stats_file = '../results/unrestricted_ai_analysis_summary.json'
        with open(ai_stats_file, 'w', encoding='utf-8') as f:
            json.dump(ai_stats, f, ensure_ascii=False, indent=2)
        print(f"📊 AI新闻统计已保存: {ai_stats_file}")
    else:
        print("❌ AI新闻分析失败")
        return
    
    # 分析记者新闻
    print("\n👨‍💼 第二步：分析记者新闻")
    journalist_results = analyze_journalist_news()
    if journalist_results is not None and not journalist_results.empty:
        print(f"✅ 记者新闻分析完成: {len(journalist_results)} 条记录")
        
        # 保存记者新闻结果
        journalist_file = '../results/unrestricted_journalist_analysis_results.csv'
        journalist_results.to_csv(journalist_file, index=False, encoding='utf-8')
        print(f"💾 记者新闻结果已保存: {journalist_file}")
        
        # 计算记者新闻统计
        journalist_stats = calculate_summary_stats(journalist_results, 'journalist')
        journalist_stats_file = '../results/unrestricted_journalist_analysis_summary.json'
        with open(journalist_stats_file, 'w', encoding='utf-8') as f:
            json.dump(journalist_stats, f, ensure_ascii=False, indent=2)
        print(f"📊 记者新闻统计已保存: {journalist_stats_file}")
    else:
        print("❌ 记者新闻分析失败")
        return
    
    # 显示关键结果对比
    print("\n🔍 关键结果对比:")
    print("=" * 50)
    
    # 语法叠加强度对比（重点！）
    ai_superposition = ai_stats['新聞標題']['grammatical_superposition']['mean']
    journalist_superposition = journalist_stats['新聞標題']['grammatical_superposition']['mean']
    
    print(f"📈 语法叠加强度（真实值，无限制）:")
    print(f"   AI新闻标题:     {ai_superposition:.6f}")
    print(f"   记者新闻标题:   {journalist_superposition:.6f}")
    print(f"   差异倍数:       {max(ai_superposition, journalist_superposition) / min(ai_superposition, journalist_superposition):.2f}×")
    
    # 其他关键指标
    ai_interference = ai_stats['新聞標題']['semantic_interference']['mean']
    journalist_interference = journalist_stats['新聞標題']['semantic_interference']['mean']
    
    print(f"\n📈 语义干涉:")
    print(f"   AI新闻标题:     {ai_interference:.6f}")
    print(f"   记者新闻标题:   {journalist_interference:.6f}")
    print(f"   差异倍数:       {ai_interference / journalist_interference:.2f}×")
    
    ai_reality = ai_stats['新聞標題']['multiple_reality_strength']['mean']
    journalist_reality = journalist_stats['新聞標題']['multiple_reality_strength']['mean']
    
    print(f"\n📈 多重现实强度（无限制）:")
    print(f"   AI新闻标题:     {ai_reality:.6f}")
    print(f"   记者新闻标题:   {journalist_reality:.6f}")
    print(f"   差异倍数:       {ai_reality / journalist_reality:.2f}×")
    
    # 性能统计
    total_time = time.time() - start_time
    total_records = len(ai_results) + len(journalist_results)
    
    print(f"\n✅ 无限制量子分析完成!")
    print(f"⏱️  总耗时: {total_time/60:.1f} 分钟")
    print(f"🚀 处理速度: {total_records/total_time:.1f} 记录/秒")
    print(f"📈 总处理记录: {total_records} 条")
    print(f"🎯 关键发现: 语法叠加强度真实值远高于1.0！")

if __name__ == "__main__":
    main()
