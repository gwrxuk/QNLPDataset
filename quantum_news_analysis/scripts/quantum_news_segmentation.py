#!/usr/bin/env python3
"""
量子新聞分析 - 中文斷詞模組
Quantum News Analysis - Chinese Word Segmentation Module

本模組使用jieba進行真實的中文斷詞處理，為後續的量子自然語言處理分析做準備。
This module uses jieba for real Chinese word segmentation, preparing data for quantum NLP analysis.
"""

import pandas as pd
import numpy as np
import jieba
import jieba.posseg as pseg
import re
from collections import Counter
from typing import List, Dict, Tuple
import json
import time

class QuantumNewsSegmenter:
    """量子新聞斷詞器 - 專門處理新聞文本的中文斷詞"""
    
    def __init__(self):
        """初始化斷詞器"""
        self.setup_jieba()
        self.processed_count = 0
        self.total_words = 0
        self.vocabulary = set()
        
    def setup_jieba(self):
        """設置jieba斷詞器的新聞領域詞典"""
        # 添加新聞常用詞彙
        news_words = [
            '人工智慧', 'AI', '機器學習', '深度學習', '大數據',
            '區塊鏈', '虛擬實境', '擴增實境', '物聯網', '5G',
            '新聞標題', '影片對話', '影片描述', '媒體報導',
            '議題設定', '框架建構', '敘事分析', '語意框架'
        ]
        
        for word in news_words:
            jieba.add_word(word)
        
        print("✅ jieba新聞詞典已載入")
    
    def clean_text(self, text: str) -> str:
        """清理文本，保留中文、數字和重要標點"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # 保留中文、數字、重要標點符號
        text = re.sub(r'[^\u4e00-\u9fff0-9！？。，、；：「」『』（）\[\]《》\-]', ' ', text)
        
        # 移除多餘空白
        text = re.sub(r'\s+', '', text)
        
        return text
    
    def segment_text(self, text: str) -> Dict:
        """
        對單一文本進行斷詞分析
        
        Args:
            text: 待分析的文本
            
        Returns:
            Dict: 包含斷詞結果的字典
        """
        if not text or not text.strip():
            return self._empty_result()
        
        # 清理文本
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return self._empty_result()
        
        # 使用jieba進行斷詞和詞性標註
        words = []
        pos_tags = []
        
        for word, flag in pseg.cut(cleaned_text):
            if len(word.strip()) > 0:
                words.append(word)
                pos_tags.append(f"{word}/{flag}")
        
        # 計算統計信息
        word_count = len(words)
        unique_words = len(set(words))
        word_freq = Counter(words)
        
        # 更新全局統計
        self.total_words += word_count
        self.vocabulary.update(words)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'words': words,
            'segmented_text': ' / '.join(words),
            'pos_tags': ' | '.join(pos_tags),
            'word_count': word_count,
            'unique_word_count': unique_words,
            'words_list': ', '.join(words),
            'word_frequencies': dict(word_freq),
            'lexical_diversity': unique_words / word_count if word_count > 0 else 0
        }
    
    def _empty_result(self) -> Dict:
        """返回空結果"""
        return {
            'original_text': '',
            'cleaned_text': '',
            'words': [],
            'segmented_text': '',
            'pos_tags': '',
            'word_count': 0,
            'unique_word_count': 0,
            'words_list': '',
            'word_frequencies': {},
            'lexical_diversity': 0
        }
    
    def analyze_field(self, df: pd.DataFrame, field_name: str) -> List[Dict]:
        """
        分析特定欄位的所有文本
        
        Args:
            df: 數據框
            field_name: 欄位名稱
            
        Returns:
            List[Dict]: 分析結果列表
        """
        print(f"\n📊 分析 {field_name} 欄位")
        print("=" * 50)
        
        if field_name not in df.columns:
            print(f"❌ 欄位 {field_name} 不存在")
            return []
        
        field_data = df[field_name].dropna()
        print(f"📝 有效記錄數: {len(field_data)}")
        
        results = []
        
        for idx, (record_idx, text) in enumerate(field_data.items()):
            if idx % 50 == 0:
                print(f"處理進度: {idx + 1}/{len(field_data)}")
            
            # 進行斷詞分析
            seg_result = self.segment_text(str(text))
            
            # 添加記錄信息
            result = {
                'record_id': record_idx,
                'field': field_name,
                **seg_result
            }
            
            results.append(result)
            self.processed_count += 1
        
        print(f"✅ {field_name} 分析完成: {len(results)} 筆記錄")
        return results
    
    def generate_field_statistics(self, results: List[Dict], field_name: str) -> Dict:
        """生成欄位統計信息"""
        if not results:
            return {}
        
        word_counts = [r['word_count'] for r in results]
        unique_word_counts = [r['unique_word_count'] for r in results]
        lexical_diversities = [r['lexical_diversity'] for r in results]
        
        # 收集所有詞彙
        all_words = []
        for result in results:
            all_words.extend(result['words'])
        
        word_freq = Counter(all_words)
        
        stats = {
            'field_name': field_name,
            'total_records': len(results),
            'avg_word_count': np.mean(word_counts),
            'std_word_count': np.std(word_counts),
            'min_word_count': np.min(word_counts),
            'max_word_count': np.max(word_counts),
            'avg_unique_words': np.mean(unique_word_counts),
            'avg_lexical_diversity': np.mean(lexical_diversities),
            'total_vocabulary_size': len(set(all_words)),
            'total_word_tokens': len(all_words),
            'top_10_words': word_freq.most_common(10)
        }
        
        return stats
    
    def get_global_statistics(self) -> Dict:
        """獲取全局統計信息"""
        return {
            'total_processed_records': self.processed_count,
            'total_word_tokens': self.total_words,
            'vocabulary_size': len(self.vocabulary),
            'avg_words_per_record': self.total_words / self.processed_count if self.processed_count > 0 else 0
        }

def main():
    """主函數 - 執行完整的新聞文本斷詞分析"""
    print("🚀 量子新聞分析 - 中文斷詞處理")
    print("=" * 60)
    
    start_time = time.time()
    
    # 初始化斷詞器
    segmenter = QuantumNewsSegmenter()
    
    try:
        # 讀取數據
        print("📊 讀取新聞數據集...")
        df = pd.read_excel('../data/dataseet.xlsx')
        print(f"數據集形狀: {df.shape}")
        print(f"可用欄位: {list(df.columns)}")
        
        # 目標欄位
        target_fields = ['新聞標題', '影片對話', '影片描述']
        available_fields = [field for field in target_fields if field in df.columns]
        
        if not available_fields:
            print("❌ 未找到目標欄位")
            return
        
        print(f"✅ 將分析欄位: {available_fields}")
        
        # 分析各欄位
        all_results = []
        field_statistics = {}
        
        for field in available_fields:
            field_results = segmenter.analyze_field(df, field)
            all_results.extend(field_results)
            
            # 生成欄位統計
            field_stats = segmenter.generate_field_statistics(field_results, field)
            field_statistics[field] = field_stats
            
            # 保存欄位結果
            if field_results:
                field_df = pd.DataFrame(field_results)
                filename = f"../results/{field}_segmentation_results.csv"
                field_df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"💾 {field} 結果已保存: {filename}")
        
        # 保存完整結果
        if all_results:
            # 主要結果
            results_df = pd.DataFrame(all_results)
            results_df.to_csv('../results/complete_segmentation_results.csv', 
                            index=False, encoding='utf-8-sig')
            print(f"\n💾 完整斷詞結果已保存: ../results/complete_segmentation_results.csv")
            
            # 統計摘要
            global_stats = segmenter.get_global_statistics()
            
            analysis_summary = {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'processing_time_seconds': time.time() - start_time,
                'global_statistics': global_stats,
                'field_statistics': field_statistics,
                'fields_analyzed': available_fields,
                'total_records_processed': len(all_results)
            }
            
            with open('../results/segmentation_analysis_summary.json', 'w', encoding='utf-8') as f:
                json.dump(analysis_summary, f, ensure_ascii=False, indent=2, default=str)
            
            # 顯示摘要
            print(f"\n📈 分析摘要:")
            print(f"  處理記錄數: {global_stats['total_processed_records']}")
            print(f"  總詞彙tokens: {global_stats['total_word_tokens']:,}")
            print(f"  詞彙表大小: {global_stats['vocabulary_size']:,}")
            print(f"  平均詞數/記錄: {global_stats['avg_words_per_record']:.1f}")
            print(f"  處理時間: {time.time() - start_time:.1f} 秒")
            
            # 各欄位統計
            for field, stats in field_statistics.items():
                print(f"\n  {field} 統計:")
                print(f"    記錄數: {stats['total_records']}")
                print(f"    平均詞數: {stats['avg_word_count']:.1f} ± {stats['std_word_count']:.1f}")
                print(f"    詞彙多樣性: {stats['avg_lexical_diversity']:.3f}")
                print(f"    高頻詞彙: {[word for word, count in stats['top_10_words'][:5]]}")
            
            print(f"\n✅ 中文斷詞分析完成！")
            print(f"📁 結果保存在: ../results/")
            
        else:
            print("❌ 沒有成功處理任何記錄")
    
    except Exception as e:
        print(f"❌ 處理過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
