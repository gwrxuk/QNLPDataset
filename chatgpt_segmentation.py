#!/usr/bin/env python3
"""
使用ChatGPT進行中文斷詞分析腳本
Chinese Word Segmentation Analysis Script using ChatGPT API
"""

import pandas as pd
import numpy as np
import openai
import time
import json
import re
from collections import Counter
import os
from typing import List, Dict, Any

# 設定OpenAI API
# 請在環境變數中設定您的API密鑰: export OPENAI_API_KEY="your-api-key"
openai.api_key = os.getenv('OPENAI_API_KEY')

class ChatGPTSegmenter:
    """使用ChatGPT進行中文斷詞的類別"""
    
    def __init__(self):
        self.model = "gpt-3.5-turbo"  # 或使用 "gpt-4" 獲得更好效果
        self.request_count = 0
        self.max_requests_per_minute = 20  # API限制
        
    def create_segmentation_prompt(self, text: str) -> str:
        """創建斷詞提示詞"""
        prompt = f"""請對以下中文文本進行精確的斷詞分析。要求：

1. 將文本分割成有意義的詞彙單位
2. 使用空格分隔每個詞
3. 保持原文的語義完整性
4. 對於專有名詞、人名、地名要保持完整
5. 只返回斷詞結果，不要其他說明

文本：{text}

斷詞結果："""
        return prompt
    
    def create_analysis_prompt(self, text: str) -> str:
        """創建詞性分析提示詞"""
        prompt = f"""請對以下中文文本進行斷詞並標註詞性。要求：

1. 進行精確斷詞
2. 為每個詞標註詞性（名詞n、動詞v、形容詞a、副詞ad、介詞p、連詞c、助詞u、數詞m、量詞q等）
3. 格式：詞/詞性
4. 用空格分隔每個詞性標註對
5. 只返回結果，不要其他說明

文本：{text}

結果："""
        return prompt
    
    def segment_with_chatgpt(self, text: str, include_pos: bool = True) -> Dict[str, Any]:
        """使用ChatGPT進行斷詞"""
        if not text or not text.strip():
            return {
                'segmented_text': '',
                'pos_tags': '',
                'words': [],
                'word_count': 0,
                'unique_words': 0,
                'error': None
            }
        
        try:
            # 控制請求頻率
            if self.request_count >= self.max_requests_per_minute:
                print("達到API請求限制，等待60秒...")
                time.sleep(60)
                self.request_count = 0
            
            # 清理文本
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return {
                    'segmented_text': '',
                    'pos_tags': '',
                    'words': [],
                    'word_count': 0,
                    'unique_words': 0,
                    'error': None
                }
            
            if include_pos:
                prompt = self.create_analysis_prompt(cleaned_text)
            else:
                prompt = self.create_segmentation_prompt(cleaned_text)
            
            # 調用ChatGPT API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1,  # 低溫度確保一致性
                top_p=0.9
            )
            
            self.request_count += 1
            result_text = response.choices[0].message.content.strip()
            
            # 解析結果
            if include_pos:
                return self.parse_pos_result(result_text)
            else:
                return self.parse_segmentation_result(result_text)
                
        except Exception as e:
            print(f"ChatGPT API錯誤: {e}")
            return {
                'segmented_text': '',
                'pos_tags': '',
                'words': [],
                'word_count': 0,
                'unique_words': 0,
                'error': str(e)
            }
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # 保留中文字符、標點符號和空格
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\s]', '', text)
        # 移除多餘空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def parse_pos_result(self, result_text: str) -> Dict[str, Any]:
        """解析詞性標註結果"""
        words = []
        pos_tags = []
        
        # 分割並解析每個詞性標註對
        items = result_text.split()
        for item in items:
            if '/' in item:
                parts = item.split('/')
                if len(parts) >= 2:
                    word = parts[0].strip()
                    pos = parts[1].strip()
                    if word:
                        words.append(word)
                        pos_tags.append(f"{word}/{pos}")
        
        segmented_text = ' / '.join(words)
        pos_text = ' | '.join(pos_tags)
        
        return {
            'segmented_text': segmented_text,
            'pos_tags': pos_text,
            'words': words,
            'word_count': len(words),
            'unique_words': len(set(words)),
            'error': None
        }
    
    def parse_segmentation_result(self, result_text: str) -> Dict[str, Any]:
        """解析純斷詞結果"""
        words = [w.strip() for w in result_text.split() if w.strip()]
        
        segmented_text = ' / '.join(words)
        
        return {
            'segmented_text': segmented_text,
            'pos_tags': '',
            'words': words,
            'word_count': len(words),
            'unique_words': len(set(words)),
            'error': None
        }

def analyze_field_with_chatgpt(df: pd.DataFrame, field_name: str, segmenter: ChatGPTSegmenter) -> List[Dict]:
    """使用ChatGPT分析單一欄位"""
    print(f"\n正在使用ChatGPT分析 {field_name} 欄位...")
    
    results = []
    total_records = len(df[field_name])
    
    for idx, text in enumerate(df[field_name]):
        print(f"處理進度: {idx + 1}/{total_records} ({((idx + 1)/total_records*100):.1f}%)", end='\r')
        
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
                'words_list': '',
                'api_error': None
            })
            continue
        
        # 使用ChatGPT進行斷詞
        seg_result = segmenter.segment_with_chatgpt(str(text), include_pos=True)
        
        results.append({
            'record_id': idx,
            'field': field_name,
            'original_text': str(text)[:200] + ('...' if len(str(text)) > 200 else ''),
            'cleaned_text': segmenter.clean_text(str(text)),
            'segmented_text': seg_result['segmented_text'],
            'pos_tags': seg_result['pos_tags'],
            'word_count': seg_result['word_count'],
            'unique_word_count': seg_result['unique_words'],
            'words_list': ', '.join(seg_result['words']),
            'api_error': seg_result['error']
        })
        
        # 每10個請求暫停一下
        if (idx + 1) % 10 == 0:
            time.sleep(2)
    
    print(f"\n{field_name} 欄位分析完成!")
    return results

def generate_chatgpt_vocabulary_analysis(all_results: List[Dict]) -> tuple:
    """生成ChatGPT斷詞的詞彙統計"""
    all_words = []
    field_word_counts = {}
    
    for result in all_results:
        if result['words_list'] and not result['api_error']:
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

def save_chatgpt_vocabulary_stats(word_freq: Counter, field_freq: Dict) -> tuple:
    """保存ChatGPT斷詞的詞彙統計"""
    
    # 整體高頻詞
    top_words = word_freq.most_common(100)
    vocab_df = pd.DataFrame(top_words, columns=['詞彙', '頻次'])
    vocab_df['相對頻率'] = vocab_df['頻次'] / vocab_df['頻次'].sum()
    vocab_df.to_csv('chatgpt_vocabulary_stats.csv', index=False, encoding='utf-8-sig')
    
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
    field_vocab_df.to_csv('chatgpt_field_vocabulary.csv', index=False, encoding='utf-8-sig')
    
    return vocab_df, field_vocab_df

def main():
    """主要執行函數"""
    print("開始使用ChatGPT進行中文斷詞分析...")
    
    # 檢查API密鑰
    if not openai.api_key:
        print("錯誤: 請設定OPENAI_API_KEY環境變數")
        print("設定方法: export OPENAI_API_KEY='your-api-key'")
        return
    
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
        
        # 初始化ChatGPT斷詞器
        segmenter = ChatGPTSegmenter()
        
        # 執行斷詞分析
        all_results = []
        
        # 為了節省API調用，可以選擇只分析部分數據
        sample_size = input(f"是否要分析全部數據？輸入樣本數量(1-{len(df)})或按Enter分析全部: ")
        if sample_size.strip():
            try:
                sample_size = int(sample_size)
                df = df.head(sample_size)
                print(f"將分析前{sample_size}筆記錄")
            except ValueError:
                print("無效輸入，將分析全部數據")
        
        for field in available_fields:
            field_results = analyze_field_with_chatgpt(df, field, segmenter)
            all_results.extend(field_results)
        
        # 保存詳細斷詞結果
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('chatgpt_segmentation_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n詳細ChatGPT斷詞結果已保存至: chatgpt_segmentation_results.csv")
        
        # 生成詞彙統計
        print("生成詞彙統計分析...")
        word_freq, field_freq = generate_chatgpt_vocabulary_analysis(all_results)
        
        if word_freq:
            vocab_df, field_vocab_df = save_chatgpt_vocabulary_stats(word_freq, field_freq)
            
            # 生成摘要統計
            summary_stats = []
            for field in available_fields:
                field_data = [r for r in all_results if r['field'] == field and not r['api_error']]
                field_words = []
                for r in field_data:
                    if r['words_list']:
                        field_words.extend([w.strip() for w in r['words_list'].split(',') if w.strip()])
                
                error_count = len([r for r in all_results if r['field'] == field and r['api_error']])
                
                summary_stats.append({
                    '欄位': field,
                    '記錄數': len([r for r in all_results if r['field'] == field]),
                    '成功處理數': len(field_data),
                    'API錯誤數': error_count,
                    '總詞數': len(field_words),
                    '唯一詞數': len(set(field_words)),
                    '平均每筆記錄詞數': np.mean([r['word_count'] for r in field_data if r['word_count'] > 0]) if field_data else 0,
                    '最高頻詞': field_freq[field].most_common(1)[0][0] if field_words else '',
                    '最高頻詞次數': field_freq[field].most_common(1)[0][1] if field_words else 0
                })
            
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_csv('chatgpt_summary_stats.csv', index=False, encoding='utf-8-sig')
            
            # 輸出結果摘要
            print("\n=== ChatGPT斷詞分析結果摘要 ===")
            print(f"總共處理記錄數: {len(all_results)}")
            print(f"總詞彙數: {len(word_freq)}")
            print(f"生成檔案:")
            print("  - chatgpt_segmentation_results.csv (詳細斷詞結果)")
            print("  - chatgpt_vocabulary_stats.csv (整體詞彙統計)")
            print("  - chatgpt_field_vocabulary.csv (各欄位詞彙統計)")
            print("  - chatgpt_summary_stats.csv (摘要統計)")
            
            print("\n=== 各欄位統計 ===")
            for _, row in summary_df.iterrows():
                print(f"{row['欄位']}:")
                print(f"  處理成功: {row['成功處理數']}/{row['記錄數']}")
                print(f"  總詞數: {row['總詞數']}")
                print(f"  唯一詞數: {row['唯一詞數']}")
                print(f"  平均詞數: {row['平均每筆記錄詞數']:.1f}")
                print(f"  最高頻詞: {row['最高頻詞']} ({row['最高頻詞次數']}次)")
            
            if word_freq:
                print("\n=== ChatGPT整體高頻詞 TOP 10 ===")
                for i, (word, count) in enumerate(word_freq.most_common(10), 1):
                    print(f"{i:2d}. {word} ({count}次)")
        
        print(f"\nChatGPT斷詞分析完成! API調用次數: {segmenter.request_count}")
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
