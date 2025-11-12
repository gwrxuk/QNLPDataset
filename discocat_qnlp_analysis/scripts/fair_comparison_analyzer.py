#!/usr/bin/env python3
"""
公平对比分析器 - 确保AI新闻和记者新闻的字段对比公平性
AI数据: 新聞標題, 影片描述, 影片對話
CNA数据: title, content (分别对应标题和内容)
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

class FairComparisonAnalyzer:
    """公平对比分析器"""
    
    def __init__(self):
        """初始化分析器"""
        print("🔧 初始化公平对比分析器...")
        
        # 情感词典
        self.emotion_lexicon = {
            'positive': ['成功', '获得', '优秀', '突破', '创新', '发展', '改善', '提升', '荣获', 
                        '卓越', '领先', '进步', '增长', '获奖', '肯定', '支持', '合作', '共赢'],
            'negative': ['失败', '问题', '困难', '危机', '冲突', '争议', '批评', '质疑', '担忧',
                        '下降', '减少', '损失', '风险', '威胁', '挑战', '阻碍', '延迟', '取消']
        }
        
        print("✅ 公平对比分析器初始化完成")

    def segment_and_pos_tag(self, text: str) -> Tuple[List[str], List[str]]:
        """分词和词性标注"""
        words = []
        pos_tags = []
        
        for word, flag in pseg.cut(text):
            if len(word.strip()) > 0:
                words.append(word)
                pos_tags.append(flag)
        
        return words, pos_tags

    def calculate_quantum_metrics(self, words: List[str], pos_tags: List[str], use_restrictions: bool = True) -> Dict[str, float]:
        """计算量子指标（可选择是否使用限制）"""
        
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
        
        # 1. 冯纽曼熵
        von_neumann_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        
        # 2. 类别一致性
        pos_freq = {}
        for pos in pos_tags:
            pos_freq[pos] = pos_freq.get(pos, 0) + 1
        
        total_pos = sum(pos_freq.values())
        pos_probs = np.array([freq/total_pos for freq in pos_freq.values()])
        category_coherence = np.max(pos_probs)
        
        # 3. 组合纠缠强度
        compositional_entanglement = pos_diversity / word_count
        if use_restrictions:
            compositional_entanglement = min(1.0, compositional_entanglement)
        
        # 4. 语法叠加态 (关键差异！)
        superposition_measure = 4 * np.sum(probabilities * (1 - probabilities))
        if use_restrictions:
            grammatical_superposition = min(1.0, superposition_measure)  # 受限制版本
        else:
            grammatical_superposition = superposition_measure  # 无限制版本
        
        # 5. 语义干涉
        repetition_variance = np.var(list(word_freq.values()))
        semantic_interference = repetition_variance / word_count
        if use_restrictions:
            semantic_interference = min(1.0, semantic_interference)
        
        # 6. 框架竞争
        if len(probabilities) > 1:
            uniform_prob = 1.0 / len(probabilities)
            kl_divergence = np.sum(probabilities * np.log2((probabilities + 1e-12) / uniform_prob))
            max_kl = np.log2(len(probabilities))
            frame_competition = float(1.0 - (kl_divergence / max_kl))
        else:
            frame_competition = 0.0
        
        # 7. 类别一致性变异
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

    def analyze_multiple_realities(self, quantum_metrics: Dict, words: List[str], use_restrictions: bool = True) -> Dict[str, float]:
        """分析多重现实现象"""
        
        # 计算语言复杂性因子
        word_count = len(words)
        unique_words = len(set(words))
        word_diversity = unique_words / max(word_count, 1)
        
        # 情感词统计
        positive_count = sum(1 for word in words if word in self.emotion_lexicon['positive'])
        negative_count = sum(1 for word in words if word in self.emotion_lexicon['negative'])
        emotional_intensity = (positive_count + negative_count) / max(word_count, 1)
        
        # 多重现实强度
        reality_strength = (
            quantum_metrics['grammatical_superposition'] * 0.35 +
            quantum_metrics['semantic_interference'] * 0.25 +
            quantum_metrics['frame_competition'] * 0.20 +
            word_diversity * 0.20
        )
        
        # 框架冲突强度
        conflict_strength = (
            quantum_metrics['compositional_entanglement'] * 0.40 +
            quantum_metrics['categorical_coherence_variance'] * 0.30 +
            emotional_intensity * 0.20 +
            (1.0 - quantum_metrics['category_coherence']) * 0.10
        )
        
        # 语义模糊度
        ambiguity = (
            quantum_metrics['von_neumann_entropy'] * 0.40 +
            quantum_metrics['semantic_interference'] * 0.30 +
            (1.0 - quantum_metrics['category_coherence']) * 0.20 +
            word_diversity * 0.10
        )
        
        # 是否应用限制
        if use_restrictions:
            return {
                'multiple_reality_strength': min(1.0, max(0.0, reality_strength)),
                'frame_conflict_strength': min(1.0, max(0.0, conflict_strength)),
                'semantic_ambiguity': min(1.0, max(0.0, ambiguity))
            }
        else:
            return {
                'multiple_reality_strength': float(reality_strength),
                'frame_conflict_strength': float(conflict_strength),
                'semantic_ambiguity': float(ambiguity)
            }

    def process_text(self, text: str, field: str, record_id: int, use_restrictions: bool = True) -> Dict[str, Any]:
        """处理文本"""
        try:
            # 分词和词性标注
            words, pos_tags = self.segment_and_pos_tag(text)
            
            if len(words) == 0:
                return None
            
            # 计算量子指标
            quantum_metrics = self.calculate_quantum_metrics(words, pos_tags, use_restrictions)
            
            # 分析多重现实
            reality_metrics = self.analyze_multiple_realities(quantum_metrics, words, use_restrictions)
            
            # 基本统计
            word_count = len(words)
            unique_words = len(set(words))
            categorical_diversity = len(set(pos_tags))
            compositional_complexity = sum(1 for pos in pos_tags if pos.startswith('V'))
            semantic_density = unique_words / max(word_count, 1) * 10
            
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
                'analysis_version': f'fair_comparison_{"restricted" if use_restrictions else "unrestricted"}_v1.0'
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 处理文本时出错: {e}")
            return None

def analyze_ai_news(use_restrictions: bool = True):
    """分析AI新闻"""
    version_name = "受限制" if use_restrictions else "无限制"
    print(f"📰 开始分析AI新闻（{version_name}版本）...")
    
    analyzer = FairComparisonAnalyzer()
    
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
            record_dict = record.to_dict()
            record_id = record_dict.get('record_id', 0)
            
            # 处理三个字段
            title = str(record_dict.get('新聞標題', ''))
            dialogue = str(record_dict.get('影片對話', ''))
            description = str(record_dict.get('影片描述', ''))
            
            if title and len(title.strip()) > 0:
                title_result = analyzer.process_text(title, '新聞標題', record_id, use_restrictions)
                if title_result:
                    all_results.append(title_result)
            
            if dialogue and len(dialogue.strip()) > 10:
                dialogue_result = analyzer.process_text(dialogue, '影片對話', record_id, use_restrictions)
                if dialogue_result:
                    all_results.append(dialogue_result)
            
            if description and len(description.strip()) > 10:
                description_result = analyzer.process_text(description, '影片描述', record_id, use_restrictions)
                if description_result:
                    all_results.append(description_result)
            
            if (idx + 1) % 50 == 0:
                print(f"🔄 已处理 {idx + 1}/{len(df)} 条AI新闻记录")
                
        except Exception as e:
            print(f"❌ 处理AI新闻记录 {idx} 时出错: {e}")
            continue
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def analyze_journalist_news(use_restrictions: bool = True):
    """分析记者新闻"""
    version_name = "受限制" if use_restrictions else "无限制"
    print(f"👨‍💼 开始分析记者新闻（{version_name}版本）...")
    
    analyzer = FairComparisonAnalyzer()
    
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
            record_dict = record.to_dict()
            record_id = record_dict.get('record_id', 0)
            
            # 处理两个字段
            title = str(record_dict.get('title', ''))
            content = str(record_dict.get('content', ''))
            
            if title and len(title.strip()) > 0:
                title_result = analyzer.process_text(title, '新聞標題', record_id, use_restrictions)
                if title_result:
                    all_results.append(title_result)
            
            if content and len(content.strip()) > 10:
                content_result = analyzer.process_text(content, '新聞內容', record_id, use_restrictions)
                if content_result:
                    all_results.append(content_result)
            
            if (idx + 1) % 5 == 0:
                print(f"🔄 已处理 {idx + 1}/{len(df)} 条记者新闻记录")
                
        except Exception as e:
            print(f"❌ 处理记者新闻记录 {idx} 时出错: {e}")
            continue
    
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def calculate_summary_stats(df: pd.DataFrame, data_type: str, version: str) -> Dict:
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
    
    print("🚀 开始公平对比分析")
    print("=" * 80)
    print("📋 分析对比:")
    print("   AI数据: 新聞標題, 影片對話, 影片描述")
    print("   CNA数据: title (新聞標題), content (新聞內容)")
    print("   版本: 受限制版本 + 无限制版本")
    print("=" * 80)
    
    start_time = time.time()
    
    # 分析所有版本
    versions = [
        (True, "restricted", "受限制"),
        (False, "unrestricted", "无限制")
    ]
    
    all_results = {}
    
    for use_restrictions, version_key, version_name in versions:
        print(f"\n🔄 开始{version_name}版本分析...")
        
        # 分析AI新闻
        ai_results = analyze_ai_news(use_restrictions)
        if ai_results is not None and not ai_results.empty:
            print(f"✅ AI新闻{version_name}分析完成: {len(ai_results)} 条记录")
            
            # 保存AI新闻结果
            ai_file = f'../results/fair_comparison_ai_{version_key}_results.csv'
            ai_results.to_csv(ai_file, index=False, encoding='utf-8')
            
            # 计算AI新闻统计
            ai_stats = calculate_summary_stats(ai_results, 'ai', version_key)
            ai_stats_file = f'../results/fair_comparison_ai_{version_key}_summary.json'
            with open(ai_stats_file, 'w', encoding='utf-8') as f:
                json.dump(ai_stats, f, ensure_ascii=False, indent=2)
            
            all_results[f'ai_{version_key}'] = {
                'data': ai_results,
                'stats': ai_stats,
                'file': ai_file,
                'stats_file': ai_stats_file
            }
        
        # 分析记者新闻
        journalist_results = analyze_journalist_news(use_restrictions)
        if journalist_results is not None and not journalist_results.empty:
            print(f"✅ 记者新闻{version_name}分析完成: {len(journalist_results)} 条记录")
            
            # 保存记者新闻结果
            journalist_file = f'../results/fair_comparison_journalist_{version_key}_results.csv'
            journalist_results.to_csv(journalist_file, index=False, encoding='utf-8')
            
            # 计算记者新闻统计
            journalist_stats = calculate_summary_stats(journalist_results, 'journalist', version_key)
            journalist_stats_file = f'../results/fair_comparison_journalist_{version_key}_summary.json'
            with open(journalist_stats_file, 'w', encoding='utf-8') as f:
                json.dump(journalist_stats, f, ensure_ascii=False, indent=2)
            
            all_results[f'journalist_{version_key}'] = {
                'data': journalist_results,
                'stats': journalist_stats,
                'file': journalist_file,
                'stats_file': journalist_stats_file
            }
    
    # 显示关键结果对比
    print("\n🔍 关键结果对比预览:")
    print("=" * 80)
    
    if 'ai_restricted' in all_results and 'journalist_restricted' in all_results:
        ai_restricted_title = all_results['ai_restricted']['stats']['新聞標題']['grammatical_superposition']['mean']
        journalist_restricted_title = all_results['journalist_restricted']['stats']['新聞標題']['grammatical_superposition']['mean']
        
        print(f"📈 受限制版本 - 语法叠加强度（新聞標題）:")
        print(f"   AI新闻:     {ai_restricted_title:.6f}")
        print(f"   记者新闻:   {journalist_restricted_title:.6f}")
        print(f"   差异倍数:   {ai_restricted_title/journalist_restricted_title:.2f}×")
    
    if 'ai_unrestricted' in all_results and 'journalist_unrestricted' in all_results:
        ai_unrestricted_title = all_results['ai_unrestricted']['stats']['新聞標題']['grammatical_superposition']['mean']
        journalist_unrestricted_title = all_results['journalist_unrestricted']['stats']['新聞標題']['grammatical_superposition']['mean']
        
        print(f"\n📈 无限制版本 - 语法叠加强度（新聞標題）:")
        print(f"   AI新闻:     {ai_unrestricted_title:.6f}")
        print(f"   记者新闻:   {journalist_unrestricted_title:.6f}")
        print(f"   差异倍数:   {ai_unrestricted_title/journalist_unrestricted_title:.2f}×")
    
    # 保存综合结果摘要
    summary = {
        'analysis_info': {
            'ai_fields': ['新聞標題', '影片對話', '影片描述'],
            'journalist_fields': ['新聞標題', '新聞內容'],
            'versions': ['restricted', 'unrestricted'],
            'analysis_date': '2024-09-26'
        },
        'file_mapping': {key: val['file'] for key, val in all_results.items()},
        'stats_mapping': {key: val['stats_file'] for key, val in all_results.items()}
    }
    
    summary_file = '../results/fair_comparison_analysis_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 性能统计
    total_time = time.time() - start_time
    total_records = sum(len(result['data']) for result in all_results.values())
    
    print(f"\n✅ 公平对比分析完成!")
    print(f"⏱️  总耗时: {total_time/60:.1f} 分钟")
    print(f"🚀 处理速度: {total_records/total_time:.1f} 记录/秒")
    print(f"📈 总处理记录: {total_records} 条")
    print(f"📄 综合摘要: {summary_file}")

if __name__ == "__main__":
    main()
