#!/usr/bin/env python3
"""
Jieba中文斷詞分析腳本
Chinese Word Segmentation Analysis Script using Jieba
"""

import pandas as pd
import jieba
import jieba.posseg as pseg
import numpy as np
from collections import Counter
import re

def clean_text(text):
    """清理文本，移除特殊字符和多餘空格"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # 移除英文字母、數字、標點符號，保留中文字符
    text = re.sub(r'[^\u4e00-\u9fff\s]', '', text)
    # 移除多餘空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def segment_text_detailed(text):
    """使用jieba進行詳細斷詞分析"""
    if not text:
        return {
            'words': [],
            'pos_tags': [],
            'word_count': 0,
            'unique_words': 0,
            'segmented_text': ''
        }
    
    # 使用精確模式斷詞
    words = list(jieba.cut(text, cut_all=False))
    
    # 詞性標註
    pos_tagged = list(pseg.cut(text))
    pos_tags = [f"{word}/{flag}" for word, flag in pos_tagged]
    
    # 過濾空詞和單字符標點
    words_filtered = [w for w in words if len(w.strip()) > 0]
    
    return {
        'words': words_filtered,
        'pos_tags': pos_tags,
        'word_count': len(words_filtered),
        'unique_words': len(set(words_filtered)),
        'segmented_text': ' / '.join(words_filtered)
    }

def analyze_field_segmentation(df, field_name):
    """分析單一欄位的斷詞結果"""
    print(f"\n正在分析 {field_name} 欄位...")
    
    results = []
    
    for idx, text in enumerate(df[field_name]):
        if pd.isna(text):
            results.append({
                'record_id': idx,
                'field': field_name,
                'original_text': '',
                'cleaned_text': '',
                'segmented_text': '',
                'pos_tags': '',
                'word_count': 0,
                'unique_word_count': 0,
                'words_list': ''
            })
            continue
        
        # 清理文本
        cleaned = clean_text(text)
        
        # 斷詞分析
        seg_result = segment_text_detailed(cleaned)
        
        results.append({
            'record_id': idx,
            'field': field_name,
            'original_text': str(text)[:200] + ('...' if len(str(text)) > 200 else ''),
            'cleaned_text': cleaned,
            'segmented_text': seg_result['segmented_text'],
            'pos_tags': ' | '.join(seg_result['pos_tags']),
            'word_count': seg_result['word_count'],
            'unique_word_count': seg_result['unique_words'],
            'words_list': ', '.join(seg_result['words'])
        })
        
        if (idx + 1) % 50 == 0:
            print(f"已處理 {idx + 1} 筆記錄...")
    
    return results

def generate_vocabulary_analysis(all_results):
    """生成詞彙統計分析"""
    all_words = []
    field_word_counts = {}
    
    for result in all_results:
        if result['words_list']:
            words = [w.strip() for w in result['words_list'].split(',') if w.strip()]
            all_words.extend(words)
            
            field = result['field']
            if field not in field_word_counts:
                field_word_counts[field] = []
            field_word_counts[field].extend(words)
    
    # 整體詞頻統計
    word_freq = Counter(all_words)
    
    # 各欄位詞頻統計
    field_freq = {}
    for field, words in field_word_counts.items():
        field_freq[field] = Counter(words)
    
    return word_freq, field_freq

def save_vocabulary_stats(word_freq, field_freq):
    """保存詞彙統計結果"""
    
    # 整體高頻詞
    top_words = word_freq.most_common(100)
    vocab_df = pd.DataFrame(top_words, columns=['詞彙', '頻次'])
    vocab_df['相對頻率'] = vocab_df['頻次'] / vocab_df['頻次'].sum()
    vocab_df.to_csv('jieba_vocabulary_stats.csv', index=False, encoding='utf-8-sig')
    
    # 各欄位高頻詞
    field_vocab_data = []
    for field, freq_counter in field_freq.items():
        top_field_words = freq_counter.most_common(50)
        for word, count in top_field_words:
            field_vocab_data.append({
                '欄位': field,
                '詞彙': word,
                '頻次': count,
                '欄位內排名': len([w for w, c in top_field_words if c > count]) + 1
            })
    
    field_vocab_df = pd.DataFrame(field_vocab_data)
    field_vocab_df.to_csv('jieba_field_vocabulary.csv', index=False, encoding='utf-8-sig')
    
    return vocab_df, field_vocab_df

def main():
    """主要執行函數"""
    print("開始jieba中文斷詞分析...")
    
    try:
        # 讀取數據
        print("讀取數據集...")
        df = pd.read_excel('dataseet.xlsx')
        print(f"數據集形狀: {df.shape}")
        
        # 目標欄位
        target_fields = ['新聞標題', '影片對話', '影片描述']
        
        # 檢查欄位存在性
        available_fields = []
        for field in target_fields:
            if field in df.columns:
                available_fields.append(field)
                print(f"✓ 找到欄位: {field}")
            else:
                print(f"✗ 未找到欄位: {field}")
        
        if not available_fields:
            print("錯誤: 未找到任何目標欄位")
            return
        
        # 執行斷詞分析
        all_results = []
        
        for field in available_fields:
            field_results = analyze_field_segmentation(df, field)
            all_results.extend(field_results)
        
        # 保存詳細斷詞結果
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('jieba_segmentation_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n詳細斷詞結果已保存至: jieba_segmentation_results.csv")
        
        # 生成詞彙統計
        print("生成詞彙統計分析...")
        word_freq, field_freq = generate_vocabulary_analysis(all_results)
        vocab_df, field_vocab_df = save_vocabulary_stats(word_freq, field_freq)
        
        # 生成摘要統計
        summary_stats = []
        for field in available_fields:
            field_data = [r for r in all_results if r['field'] == field]
            field_words = []
            for r in field_data:
                if r['words_list']:
                    field_words.extend([w.strip() for w in r['words_list'].split(',') if w.strip()])
            
            summary_stats.append({
                '欄位': field,
                '記錄數': len(field_data),
                '非空記錄數': len([r for r in field_data if r['cleaned_text']]),
                '總詞數': len(field_words),
                '唯一詞數': len(set(field_words)),
                '平均每筆記錄詞數': np.mean([r['word_count'] for r in field_data if r['word_count'] > 0]),
                '最高頻詞': field_freq[field].most_common(1)[0][0] if field_words else '',
                '最高頻詞次數': field_freq[field].most_common(1)[0][1] if field_words else 0
            })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv('jieba_summary_stats.csv', index=False, encoding='utf-8-sig')
        
        # 輸出結果摘要
        print("\n=== 斷詞分析結果摘要 ===")
        print(f"總共處理記錄數: {len(all_results)}")
        print(f"總詞彙數: {len(word_freq)}")
        print(f"生成檔案:")
        print("  - jieba_segmentation_results.csv (詳細斷詞結果)")
        print("  - jieba_vocabulary_stats.csv (整體詞彙統計)")
        print("  - jieba_field_vocabulary.csv (各欄位詞彙統計)")
        print("  - jieba_summary_stats.csv (摘要統計)")
        
        print("\n=== 各欄位統計 ===")
        for _, row in summary_df.iterrows():
            print(f"{row['欄位']}:")
            print(f"  記錄數: {row['記錄數']}")
            print(f"  總詞數: {row['總詞數']}")
            print(f"  唯一詞數: {row['唯一詞數']}")
            print(f"  平均詞數: {row['平均每筆記錄詞數']:.1f}")
            print(f"  最高頻詞: {row['最高頻詞']} ({row['最高頻詞次數']}次)")
        
        print("\n=== 整體高頻詞 TOP 10 ===")
        for i, (word, count) in enumerate(word_freq.most_common(10), 1):
            print(f"{i:2d}. {word} ({count}次)")
        
        print("\n斷詞分析完成!")
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
