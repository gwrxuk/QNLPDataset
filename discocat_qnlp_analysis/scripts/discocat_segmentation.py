#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DisCoCat Enhanced Chinese Text Segmentation and Grammatical Analysis
====================================================================

This script implements Chinese text segmentation with DisCoCat (Distributional 
Compositional Categorical) preprocessing for quantum natural language processing.

DisCoCat extends traditional word segmentation by:
1. Identifying grammatical categories and syntactic relationships
2. Creating categorical semantic representations
3. Preparing text for monoidal category analysis
4. Supporting compositional semantic modeling

Author: QNLP Research Team
Date: 2025-09-20
"""

import pandas as pd
import jieba
import jieba.posseg as pseg  # Part-of-speech tagging
import re
import json
import os
import time
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any
import numpy as np

# DisCoCat and categorical semantics imports
try:
    from discopy import Word, Ty
    from discopy.grammar.pregroup import Diagram
    DISCOPY_AVAILABLE = True
except ImportError:
    DISCOPY_AVAILABLE = False
    print("⚠️  DisCoPy not available - using simplified categorical analysis")

class DiscoCatSegmenter:
    """
    DisCoCat-enhanced Chinese text segmentation with categorical semantics.
    
    This class combines traditional jieba segmentation with categorical grammar
    analysis suitable for quantum natural language processing.
    """
    
    def __init__(self):
        """Initialize the DisCoCat segmenter with Chinese grammatical categories."""
        
        # Chinese grammatical category mappings for DisCoCat
        self.pos_to_category = {
            # Nouns and noun phrases
            'n': 'N',      # 名词
            'nr': 'N',     # 人名
            'ns': 'N',     # 地名
            'nt': 'N',     # 机构名
            'nz': 'N',     # 其他专名
            
            # Verbs and verb phrases  
            'v': 'V',      # 动词
            'vd': 'V',     # 副动词
            'vn': 'V',     # 名动词
            'a': 'A',      # 形容词
            
            # Function words
            'r': 'R',      # 代词
            'p': 'P',      # 介词
            'c': 'C',      # 连词
            'u': 'U',      # 助词
            'e': 'E',      # 叹词
            
            # Modifiers
            'd': 'D',      # 副词
            'b': 'B',      # 区别词
            'f': 'F',      # 方位词
            
            # Others
            'm': 'M',      # 数词
            'q': 'Q',      # 量词
            'x': 'X',      # 非语素字
            'w': 'W',      # 标点符号
        }
        
        # DisCoCat type assignments for Chinese grammar
        self.category_types = {
            'N': Ty('n') if DISCOPY_AVAILABLE else 'n',           # Noun
            'V': Ty('n').r @ Ty('s') @ Ty('n').l if DISCOPY_AVAILABLE else 'v',  # Transitive verb
            'A': Ty('n') @ Ty('n').l if DISCOPY_AVAILABLE else 'a',   # Adjective
            'D': Ty('s') @ Ty('s').l if DISCOPY_AVAILABLE else 'd',   # Adverb
            'P': Ty('n').r @ Ty('n') @ Ty('n').l if DISCOPY_AVAILABLE else 'p',  # Preposition
            'R': Ty('n') if DISCOPY_AVAILABLE else 'r',           # Pronoun
            'C': Ty('s').r @ Ty('s') @ Ty('s').l if DISCOPY_AVAILABLE else 'c',  # Conjunction
        }
        
        print("🔧 DisCoCat分詞器初始化完成")
        if DISCOPY_AVAILABLE:
            print("✅ DisCoPy可用 - 完整categorical語義分析")
        else:
            print("⚠️  DisCoPy不可用 - 使用簡化categorical分析")

    def clean_text(self, text: str) -> str:
        """Clean and normalize Chinese text for processing."""
        if pd.isna(text) or text == "":
            return ""
        
        text = str(text).strip()
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Chinese punctuation
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9\s，。！？；：""''（）【】《》、]', '', text)
        
        return text

    def segment_with_pos(self, text: str) -> List[Tuple[str, str]]:
        """Segment text and return words with part-of-speech tags."""
        clean_text = self.clean_text(text)
        if not clean_text:
            return []
        
        # Use jieba's part-of-speech segmentation
        words_with_pos = list(pseg.cut(clean_text))
        
        return [(word, flag) for word, flag in words_with_pos if word.strip()]

    def create_categorical_representation(self, words_with_pos: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Create DisCoCat categorical representation of the segmented text."""
        
        categorical_analysis = {
            'words': [],
            'categories': [],
            'types': [],
            'compositional_structure': [],
            'semantic_roles': defaultdict(list)
        }
        
        for word, pos in words_with_pos:
            # Map POS tag to grammatical category
            category = self.pos_to_category.get(pos, 'X')  # Default to X for unknown
            
            # Get DisCoCat type
            discocat_type = self.category_types.get(category, 'x')
            
            word_analysis = {
                'word': word,
                'pos': pos,
                'category': category,
                'type': str(discocat_type),
                'length': len(word)
            }
            
            categorical_analysis['words'].append(word)
            categorical_analysis['categories'].append(category)
            categorical_analysis['types'].append(str(discocat_type))
            categorical_analysis['compositional_structure'].append(word_analysis)
            
            # Group by semantic roles
            categorical_analysis['semantic_roles'][category].append(word)
        
        return categorical_analysis

    def analyze_compositional_structure(self, categorical_rep: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the compositional structure for quantum processing."""
        
        structure_analysis = {
            'noun_phrases': [],
            'verb_phrases': [],
            'prepositional_phrases': [],
            'compositional_complexity': 0,
            'category_transitions': [],
            'semantic_density': 0
        }
        
        categories = categorical_rep['categories']
        words = categorical_rep['words']
        
        # Identify phrases and compositional patterns
        i = 0
        while i < len(categories):
            category = categories[i]
            word = words[i]
            
            if category == 'N':
                # Look for noun phrases (N + N, A + N, etc.)
                np_words = [word]
                j = i + 1
                while j < len(categories) and categories[j] in ['N', 'A', 'B']:
                    np_words.append(words[j])
                    j += 1
                
                if len(np_words) > 1:
                    structure_analysis['noun_phrases'].append(' '.join(np_words))
                i = j
            
            elif category == 'V':
                # Look for verb phrases
                vp_words = [word]
                j = i + 1
                while j < len(categories) and categories[j] in ['D', 'A']:
                    vp_words.append(words[j])
                    j += 1
                
                if len(vp_words) > 1:
                    structure_analysis['verb_phrases'].append(' '.join(vp_words))
                i = j
            
            elif category == 'P':
                # Look for prepositional phrases
                pp_words = [word]
                j = i + 1
                while j < len(categories) and categories[j] in ['N', 'R']:
                    pp_words.append(words[j])
                    j += 1
                
                if len(pp_words) > 1:
                    structure_analysis['prepositional_phrases'].append(' '.join(pp_words))
                i = j
            
            else:
                i += 1
        
        # Calculate compositional complexity
        unique_categories = len(set(categories))
        category_transitions = sum(1 for i in range(len(categories)-1) 
                                 if categories[i] != categories[i+1])
        
        structure_analysis['compositional_complexity'] = unique_categories * (category_transitions + 1)
        structure_analysis['category_transitions'] = category_transitions
        structure_analysis['semantic_density'] = len(words) / max(unique_categories, 1)
        
        return structure_analysis

    def process_text_discocat(self, text: str, field: str, record_id: int) -> Dict[str, Any]:
        """Process a single text with full DisCoCat analysis."""
        
        # Basic segmentation with POS
        words_with_pos = self.segment_with_pos(text)
        
        if not words_with_pos:
            return {
                'record_id': record_id,
                'field': field,
                'original_text': text,
                'cleaned_text': '',
                'words_list': [],
                'pos_list': [],
                'word_count': 0,
                'unique_words': 0,
                'categorical_analysis': {},
                'compositional_structure': {},
                'discocat_ready': False
            }
        
        # Create categorical representation
        categorical_rep = self.create_categorical_representation(words_with_pos)
        
        # Analyze compositional structure
        compositional_analysis = self.analyze_compositional_structure(categorical_rep)
        
        # Extract basic statistics
        words = [word for word, pos in words_with_pos]
        pos_tags = [pos for word, pos in words_with_pos]
        
        result = {
            'record_id': record_id,
            'field': field,
            'original_text': text,
            'cleaned_text': self.clean_text(text),
            'words_list': words,
            'pos_list': pos_tags,
            'word_count': len(words),
            'unique_words': len(set(words)),
            'categorical_analysis': categorical_rep,
            'compositional_structure': compositional_analysis,
            'discocat_ready': True
        }
        
        return result

def analyze_field_discocat(df_field: pd.DataFrame, field_name: str, segmenter: DiscoCatSegmenter) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Analyze a specific field with DisCoCat enhancement."""
    
    print(f"🔍 開始DisCoCat分析: {field_name}")
    
    results = []
    vocabulary_stats = defaultdict(int)
    category_stats = defaultdict(int)
    compositional_stats = {
        'total_noun_phrases': 0,
        'total_verb_phrases': 0,
        'total_prep_phrases': 0,
        'avg_compositional_complexity': 0,
        'avg_semantic_density': 0
    }
    
    start_time = time.time()
    
    for idx, row in df_field.iterrows():
        text = row['text']
        record_id = row['record_id']
        
        # Process with DisCoCat
        analysis = segmenter.process_text_discocat(text, field_name, record_id)
        results.append(analysis)
        
        # Update vocabulary statistics
        for word in analysis['words_list']:
            vocabulary_stats[word] += 1
        
        # Update categorical statistics
        if analysis['discocat_ready']:
            for category in analysis['categorical_analysis']['categories']:
                category_stats[category] += 1
            
            # Update compositional statistics
            comp_struct = analysis['compositional_structure']
            compositional_stats['total_noun_phrases'] += len(comp_struct['noun_phrases'])
            compositional_stats['total_verb_phrases'] += len(comp_struct['verb_phrases'])
            compositional_stats['total_prep_phrases'] += len(comp_struct['prepositional_phrases'])
            compositional_stats['avg_compositional_complexity'] += comp_struct['compositional_complexity']
            compositional_stats['avg_semantic_density'] += comp_struct['semantic_density']
        
        if (idx + 1) % 50 == 0:
            print(f"  已處理 {idx + 1}/{len(df_field)} 筆記錄")
    
    # Calculate averages
    total_records = len(results)
    if total_records > 0:
        compositional_stats['avg_compositional_complexity'] /= total_records
        compositional_stats['avg_semantic_density'] /= total_records
    
    processing_time = time.time() - start_time
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    summary_stats = {
        'field': field_name,
        'total_records': total_records,
        'total_words': sum(len(r['words_list']) for r in results),
        'vocabulary_size': len(vocabulary_stats),
        'avg_words_per_record': np.mean([len(r['words_list']) for r in results]) if results else 0,
        'most_common_words': dict(Counter(vocabulary_stats).most_common(20)),
        'category_distribution': dict(category_stats),
        'compositional_statistics': compositional_stats,
        'processing_time_seconds': processing_time
    }
    
    print(f"✅ {field_name} DisCoCat分析完成: {total_records}筆記錄, 耗時{processing_time:.1f}秒")
    
    return results_df, vocabulary_stats, summary_stats

def main():
    """Main function for DisCoCat-enhanced segmentation analysis."""
    
    print("🚀 開始DisCoCat增強型中文分詞分析")
    print("=" * 60)
    
    # Initialize DisCoCat segmenter
    segmenter = DiscoCatSegmenter()
    
    # Load dataset
    print("📂 載入數據集...")
    try:
        df = pd.read_excel('../data/dataseet.xlsx')
        print(f"✅ 數據集載入成功: {len(df)} 筆原始記錄")
    except Exception as e:
        print(f"❌ 數據集載入失敗: {e}")
        return
    
    # Prepare data for analysis
    fields_to_analyze = ['新聞標題', '影片對話', '影片描述']
    
    # Create ID column and melt the dataframe to long format
    df['id'] = range(len(df))
    df_melted = df.melt(id_vars=['id'], 
                        value_vars=fields_to_analyze, 
                        var_name='field', 
                        value_name='text')
    
    # Filter out empty texts
    df_filtered = df_melted.dropna(subset=['text'])
    df_filtered = df_filtered[df_filtered['text'].str.strip() != '']
    df_filtered['record_id'] = range(len(df_filtered))
    
    print(f"📊 準備分析 {len(df_filtered)} 筆有效記錄")
    
    # Analyze each field
    all_results = []
    field_summaries = {}
    
    for field in fields_to_analyze:
        print(f"\n🔬 分析欄位: {field}")
        field_df = df_filtered[df_filtered['field'] == field].copy()
        
        if not field_df.empty:
            field_results, vocab_stats, summary = analyze_field_discocat(field_df, field, segmenter)
            all_results.append(field_results)
            field_summaries[field] = summary
            
            # Save field-specific results
            field_results.to_csv(f'../results/{field}_discocat_segmentation.csv', index=False, encoding='utf-8')
            print(f"💾 {field} 結果已保存")
        else:
            print(f"⚠️  {field} 欄位無有效記錄")
    
    # Combine all results
    if all_results:
        complete_results = pd.concat(all_results, ignore_index=True)
        complete_results.to_csv('../results/complete_discocat_segmentation.csv', index=False, encoding='utf-8')
        
        # Global statistics
        global_stats = {
            'analysis_type': 'DisCoCat Enhanced Segmentation',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_records_analyzed': len(complete_results),
            'fields_analyzed': fields_to_analyze,
            'field_summaries': field_summaries,
            'global_statistics': {
                'total_word_tokens': sum(s['total_words'] for s in field_summaries.values()),
                'vocabulary_size': len(set().union(*[s['most_common_words'].keys() for s in field_summaries.values()])),
                'avg_words_per_record': np.mean([s['avg_words_per_record'] for s in field_summaries.values()]),
                'total_compositional_complexity': sum(s['compositional_statistics']['avg_compositional_complexity'] for s in field_summaries.values()),
                'discocat_ready_percentage': (len(complete_results[complete_results['discocat_ready'] == True]) / len(complete_results)) * 100
            },
            'discocat_features': {
                'categorical_analysis_enabled': True,
                'compositional_structure_analysis': True,
                'semantic_role_identification': True,
                'discopy_integration': DISCOPY_AVAILABLE
            }
        }
        
        # Save global analysis summary
        with open('../results/discocat_segmentation_summary.json', 'w', encoding='utf-8') as f:
            json.dump(global_stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 DisCoCat分詞分析完成!")
        print(f"📊 總計分析: {len(complete_results)} 筆記錄")
        print(f"📁 結果保存於: ../results/")
        print(f"🔧 DisCoCat特性: {'完整' if DISCOPY_AVAILABLE else '簡化'}模式")
        
        # Display summary
        print(f"\n📈 分析摘要:")
        for field, stats in field_summaries.items():
            comp_stats = stats['compositional_statistics']
            print(f"  {field}:")
            print(f"    記錄數: {stats['total_records']}")
            print(f"    詞彙量: {stats['vocabulary_size']}")
            print(f"    平均複合複雜度: {comp_stats['avg_compositional_complexity']:.2f}")
            print(f"    平均語義密度: {comp_stats['avg_semantic_density']:.2f}")
    
    else:
        print("❌ 沒有成功分析任何欄位")

if __name__ == "__main__":
    main()
