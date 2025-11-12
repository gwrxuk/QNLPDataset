#!/usr/bin/env python3
"""
CNA資料集字數長度分析器
分析cna.csv中標題和內容的平均字數長度
"""

import pandas as pd
import numpy as np
import jieba
import statistics

def analyze_word_length():
    """分析CNA資料集的字數長度"""
    print("📊 分析CNA資料集字數長度...")
    
    # 載入CNA資料
    cna_df = pd.read_csv('/Users/junghualiu/case/a2a/qnlp/discocat_qnlp_analysis/data/cna.csv')
    print(f"✅ 載入CNA資料: {len(cna_df)} 條記錄")
    
    # 分析標題字數
    title_lengths = []
    content_lengths = []
    
    print("\n📝 分析標題字數...")
    for idx, row in cna_df.iterrows():
        title = str(row['title']) if pd.notna(row['title']) else ""
        content = str(row['content']) if pd.notna(row['content']) else ""
        
        # 計算字數（中文字符數）
        title_char_count = len(title.strip())
        content_char_count = len(content.strip())
        
        # 使用jieba分詞計算詞數
        title_words = list(jieba.cut(title.strip()))
        content_words = list(jieba.cut(content.strip()))
        
        # 過濾空白詞
        title_words = [w for w in title_words if w.strip()]
        content_words = [w for w in content_words if w.strip()]
        
        title_word_count = len(title_words)
        content_word_count = len(content_words)
        
        title_lengths.append({
            'char_count': title_char_count,
            'word_count': title_word_count,
            'text': title[:50] + '...' if len(title) > 50 else title
        })
        
        content_lengths.append({
            'char_count': content_char_count,
            'word_count': content_word_count,
            'text': content[:100] + '...' if len(content) > 100 else content
        })
    
    # 計算統計數據
    title_char_counts = [item['char_count'] for item in title_lengths]
    title_word_counts = [item['word_count'] for item in title_lengths]
    content_char_counts = [item['char_count'] for item in content_lengths]
    content_word_counts = [item['word_count'] for item in content_lengths]
    
    # 標題統計
    title_stats = {
        'char_count': {
            'mean': np.mean(title_char_counts),
            'median': np.median(title_char_counts),
            'std': np.std(title_char_counts),
            'min': np.min(title_char_counts),
            'max': np.max(title_char_counts)
        },
        'word_count': {
            'mean': np.mean(title_word_counts),
            'median': np.median(title_word_counts),
            'std': np.std(title_word_counts),
            'min': np.min(title_word_counts),
            'max': np.max(title_word_counts)
        }
    }
    
    # 內容統計
    content_stats = {
        'char_count': {
            'mean': np.mean(content_char_counts),
            'median': np.median(content_char_counts),
            'std': np.std(content_char_counts),
            'min': np.min(content_char_counts),
            'max': np.max(content_char_counts)
        },
        'word_count': {
            'mean': np.mean(content_word_counts),
            'median': np.median(content_word_counts),
            'std': np.std(content_word_counts),
            'min': np.min(content_word_counts),
            'max': np.max(content_word_counts)
        }
    }
    
    # 打印結果
    print("\n" + "="*60)
    print("📊 CNA資料集字數長度分析結果")
    print("="*60)
    
    print(f"\n📰 **標題 (title) 統計** ({len(cna_df)} 條記錄)")
    print("-" * 40)
    print(f"字符數統計:")
    print(f"  平均值: {title_stats['char_count']['mean']:.2f} 字符")
    print(f"  中位數: {title_stats['char_count']['median']:.2f} 字符")
    print(f"  標準差: {title_stats['char_count']['std']:.2f}")
    print(f"  最小值: {title_stats['char_count']['min']} 字符")
    print(f"  最大值: {title_stats['char_count']['max']} 字符")
    
    print(f"\n詞數統計 (jieba分詞):")
    print(f"  平均值: {title_stats['word_count']['mean']:.2f} 詞")
    print(f"  中位數: {title_stats['word_count']['median']:.2f} 詞")
    print(f"  標準差: {title_stats['word_count']['std']:.2f}")
    print(f"  最小值: {title_stats['word_count']['min']} 詞")
    print(f"  最大值: {title_stats['word_count']['max']} 詞")
    
    print(f"\n📄 **內容 (content) 統計** ({len(cna_df)} 條記錄)")
    print("-" * 40)
    print(f"字符數統計:")
    print(f"  平均值: {content_stats['char_count']['mean']:.2f} 字符")
    print(f"  中位數: {content_stats['char_count']['median']:.2f} 字符")
    print(f"  標準差: {content_stats['char_count']['std']:.2f}")
    print(f"  最小值: {content_stats['char_count']['min']} 字符")
    print(f"  最大值: {content_stats['char_count']['max']} 字符")
    
    print(f"\n詞數統計 (jieba分詞):")
    print(f"  平均值: {content_stats['word_count']['mean']:.2f} 詞")
    print(f"  中位數: {content_stats['word_count']['median']:.2f} 詞")
    print(f"  標準差: {content_stats['word_count']['std']:.2f}")
    print(f"  最小值: {content_stats['word_count']['min']} 詞")
    print(f"  最大值: {content_stats['word_count']['max']} 詞")
    
    # 顯示一些示例
    print(f"\n📝 **標題示例** (前5條)")
    print("-" * 40)
    for i in range(min(5, len(title_lengths))):
        item = title_lengths[i]
        print(f"{i+1}. {item['text']}")
        print(f"   字符數: {item['char_count']}, 詞數: {item['word_count']}")
    
    print(f"\n📄 **內容示例** (前3條)")
    print("-" * 40)
    for i in range(min(3, len(content_lengths))):
        item = content_lengths[i]
        print(f"{i+1}. {item['text']}")
        print(f"   字符數: {item['char_count']}, 詞數: {item['word_count']}")
    
    # 比較分析
    print(f"\n🔍 **比較分析**")
    print("-" * 40)
    char_ratio = content_stats['char_count']['mean'] / title_stats['char_count']['mean']
    word_ratio = content_stats['word_count']['mean'] / title_stats['word_count']['mean']
    
    print(f"內容與標題的字符數比例: {char_ratio:.2f}:1")
    print(f"內容與標題的詞數比例: {word_ratio:.2f}:1")
    print(f"標題平均每詞字符數: {title_stats['char_count']['mean']/title_stats['word_count']['mean']:.2f}")
    print(f"內容平均每詞字符數: {content_stats['char_count']['mean']/content_stats['word_count']['mean']:.2f}")
    
    # 保存詳細結果
    results = {
        'dataset_info': {
            'total_records': len(cna_df),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'title_statistics': title_stats,
        'content_statistics': content_stats,
        'comparison': {
            'content_to_title_char_ratio': char_ratio,
            'content_to_title_word_ratio': word_ratio,
            'title_avg_chars_per_word': title_stats['char_count']['mean']/title_stats['word_count']['mean'],
            'content_avg_chars_per_word': content_stats['char_count']['mean']/content_stats['word_count']['mean']
        }
    }
    
    # 保存到JSON文件
    import json
    
    # 轉換numpy類型為Python原生類型
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results_converted = convert_numpy_types(results)
    
    with open('../results/cna_word_length_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 詳細結果已保存到: ../results/cna_word_length_analysis.json")
    
    return results

def main():
    """主函數"""
    print("🚀 開始CNA資料集字數長度分析...")
    
    try:
        results = analyze_word_length()
        print("\n🎉 分析完成！")
        
        # 簡要摘要
        print(f"\n📋 **快速摘要**")
        print(f"標題平均字符數: {results['title_statistics']['char_count']['mean']:.1f}")
        print(f"標題平均詞數: {results['title_statistics']['word_count']['mean']:.1f}")
        print(f"內容平均字符數: {results['content_statistics']['char_count']['mean']:.1f}")
        print(f"內容平均詞數: {results['content_statistics']['word_count']['mean']:.1f}")
        
    except Exception as e:
        print(f"❌ 分析失敗: {str(e)}")

if __name__ == "__main__":
    main()
