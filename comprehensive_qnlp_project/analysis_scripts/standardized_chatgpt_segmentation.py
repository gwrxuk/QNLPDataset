#!/usr/bin/env python3
"""
標準化ChatGPT中文斷詞分析 - 與jieba格式完全一致
Standardized ChatGPT Chinese Word Segmentation - Consistent with jieba format
"""

from openai import OpenAI
import pandas as pd
import numpy as np
import os
import time
import re
from typing import List, Dict, Optional
from collections import Counter

class StandardizedChatGPTSegmenter:
    """標準化ChatGPT斷詞器 - 與jieba格式一致"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None
        self.request_count = 0
        self.total_tokens = 0
        self.error_count = 0
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def clean_text(self, text: str) -> str:
        """清理文本 - 與jieba保持一致"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        # 移除英文部分但保留標點
        text = re.sub(r'[A-Za-z][A-Za-z\s]*[A-Za-z]', '', text)
        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 移除特殊字符但保留中文標點
        text = re.sub(r'["""''""''「」『』]', '', text)
        
        return text
    
    def create_segmentation_prompt(self, text: str) -> str:
        """創建標準化斷詞提示詞"""
        return f"""請對以下中文文本進行精確斷詞並標註詞性。

要求：
1. 只處理中文部分，忽略英文
2. 精確斷詞，保持語義完整性
3. 專有名詞保持完整
4. 詞性標註使用標準標記：n(名詞) v(動詞) a(形容詞) ad(副詞) p(介詞) c(連詞) u(助詞) m(數詞) r(代詞) w(標點) ns(地名) nr(人名) nt(機構名)
5. 格式：詞/詞性
6. 用空格分隔
7. 只返回斷詞結果，不要解釋

文本：{text}

斷詞結果："""
    
    def segment_text(self, text: str, retry_count: int = 2) -> Dict:
        """使用ChatGPT進行標準化文本斷詞"""
        
        if not self.client:
            return self._error_result('API客戶端未初始化')
        
        if not text or not text.strip():
            return self._empty_result()
        
        # 清理文本
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return self._empty_result()
        
        for attempt in range(retry_count + 1):
            try:
                prompt = self.create_segmentation_prompt(cleaned_text)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是專業的中文自然語言處理專家，擅長精確的中文斷詞和詞性標註。請嚴格按照要求格式輸出。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=500,
                    temperature=0.1,
                    top_p=0.9
                )
                
                self.request_count += 1
                self.total_tokens += response.usage.total_tokens
                
                result_text = response.choices[0].message.content.strip()
                return self._parse_segmentation_result(result_text, cleaned_text, response.usage.total_tokens)
                
            except Exception as e:
                if attempt < retry_count:
                    print(f"⚠️  API調用失敗，重試中... (第{attempt + 1}次)")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    self.error_count += 1
                    return self._error_result(f"API調用失敗: {str(e)}")
    
    def _parse_segmentation_result(self, result_text: str, cleaned_text: str, tokens_used: int) -> Dict:
        """解析斷詞結果 - 標準化格式"""
        words = []
        pos_tags = []
        
        # 分割並解析每個詞性標註對
        items = result_text.split()
        for item in items:
            if '/' in item:
                parts = item.split('/', 1)
                if len(parts) == 2:
                    word = parts[0].strip()
                    pos = parts[1].strip()
                    if word and word not in ['', ' ']:
                        words.append(word)
                        pos_tags.append(f"{word}/{pos}")
        
        # 創建標準化格式
        segmented_text = ' / '.join(words)
        pos_tags_str = ' | '.join(pos_tags)
        words_list = ', '.join(words)
        
        return {
            'error': None,
            'cleaned_text': cleaned_text,
            'segmented_text': segmented_text,
            'pos_tags': pos_tags_str,
            'word_count': len(words),
            'unique_word_count': len(set(words)),
            'words_list': words_list,
            'tokens_used': tokens_used,
            'raw_response': result_text
        }
    
    def _empty_result(self) -> Dict:
        """空結果 - 標準化格式"""
        return {
            'error': None,
            'cleaned_text': '',
            'segmented_text': '',
            'pos_tags': '',
            'word_count': 0,
            'unique_word_count': 0,
            'words_list': '',
            'tokens_used': 0,
            'raw_response': ''
        }
    
    def _error_result(self, error_msg: str) -> Dict:
        """錯誤結果 - 標準化格式"""
        return {
            'error': error_msg,
            'cleaned_text': '',
            'segmented_text': '',
            'pos_tags': '',
            'word_count': 0,
            'unique_word_count': 0,
            'words_list': '',
            'tokens_used': 0,
            'raw_response': ''
        }

def analyze_field_with_standardized_chatgpt(df: pd.DataFrame, field_name: str, 
                                           segmenter: StandardizedChatGPTSegmenter,
                                           max_records: Optional[int] = None) -> List[Dict]:
    """使用標準化ChatGPT分析欄位 - 與jieba格式一致"""
    
    print(f"\n🤖 使用標準化ChatGPT分析 {field_name} 欄位")
    print("=" * 50)
    
    if not segmenter.client:
        print("❌ ChatGPT API未設定")
        return []
    
    # 限制處理數量
    field_data = df[field_name].dropna()
    if max_records:
        field_data = field_data.head(max_records)
        print(f"📊 處理前 {max_records} 筆記錄")
    else:
        print(f"📊 處理全部 {len(field_data)} 筆記錄")
    
    results = []
    
    for idx, (original_idx, text) in enumerate(field_data.items()):
        print(f"處理進度: {idx + 1}/{len(field_data)} - {str(text)[:30]}...")
        
        seg_result = segmenter.segment_text(str(text))
        
        # 創建與jieba完全一致的格式
        results.append({
            'record_id': original_idx,
            'field': field_name,
            'original_text': str(text),
            'cleaned_text': seg_result['cleaned_text'],
            'segmented_text': seg_result['segmented_text'],
            'pos_tags': seg_result['pos_tags'],
            'word_count': seg_result['word_count'],
            'unique_word_count': seg_result['unique_word_count'],
            'words_list': seg_result['words_list'],
            'tokens_used': seg_result['tokens_used'],
            'api_error': seg_result['error'],
            'raw_response': seg_result['raw_response']
        })
        
        # 避免API限制
        time.sleep(1.2)
    
    # 顯示統計
    successful = len([r for r in results if not r['api_error']])
    total_tokens = sum(r['tokens_used'] for r in results if r['tokens_used'] > 0)
    
    print(f"\n📈 {field_name} 處理完成:")
    print(f"  成功處理: {successful}/{len(results)}")
    print(f"  總tokens: {total_tokens}")
    print(f"  平均詞數: {sum(r['word_count'] for r in results if r['word_count'] > 0) / max(successful, 1):.1f}")
    
    return results

def main():
    """主函數"""
    print("🚀 標準化ChatGPT中文斷詞分析")
    print("=" * 50)
    
    # 檢查API密鑰
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ 請設定OPENAI_API_KEY環境變數")
        return
    
    print(f"✅ API密鑰已設定: {api_key[:10]}...")
    
    try:
        # 讀取數據
        print("\n📊 讀取數據集...")
        df = pd.read_excel('../datasets/dataseet.xlsx')
        print(f"數據集形狀: {df.shape}")
        
        # 目標欄位
        target_fields = ['新聞標題', '影片對話', '影片描述']
        available_fields = [field for field in target_fields if field in df.columns]
        
        if not available_fields:
            print("❌ 未找到目標欄位")
            return
        
        print(f"✅ 找到欄位: {', '.join(available_fields)}")
        
        # 詢問處理範圍
        max_records_input = input(f"每個欄位處理多少筆記錄？(Enter=全部, 數字=限制筆數): ").strip()
        
        max_records = None
        if max_records_input.isdigit():
            max_records = int(max_records_input)
            print(f"📝 將處理每個欄位的前 {max_records} 筆記錄")
        else:
            print("📝 將處理全部記錄")
        
        # 初始化標準化ChatGPT斷詞器
        segmenter = StandardizedChatGPTSegmenter(api_key=api_key)
        
        # 處理各欄位
        all_results = []
        
        for field in available_fields:
            field_results = analyze_field_with_standardized_chatgpt(
                df, field, segmenter, max_records
            )
            all_results.extend(field_results)
            
            # 保存中間結果
            if field_results:
                field_df = pd.DataFrame(field_results)
                filename = f"../segmentation_results/chatgpt_standardized_{field}_segmentation.csv"
                field_df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"💾 {field} 結果已保存: {filename}")
        
        # 保存完整結果 - 與jieba格式完全一致
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # 確保欄位順序與jieba一致
            column_order = [
                'record_id', 'field', 'original_text', 'cleaned_text', 
                'segmented_text', 'pos_tags', 'word_count', 
                'unique_word_count', 'words_list'
            ]
            
            # 只保留標準欄位
            standard_df = results_df[column_order].copy()
            standard_df.to_csv('../segmentation_results/chatgpt_standardized_segmentation_results.csv', 
                              index=False, encoding='utf-8-sig')
            
            print(f"\n💾 標準化結果已保存: ../segmentation_results/chatgpt_standardized_segmentation_results.csv")
            
            # 生成統計摘要
            successful_results = [r for r in all_results if not r['api_error']]
            
            print(f"\n🎉 標準化處理完成摘要:")
            print(f"  總記錄數: {len(all_results)}")
            print(f"  成功處理: {len(successful_results)}")
            print(f"  平均詞數: {np.mean([r['word_count'] for r in successful_results]) if successful_results else 0:.1f}")
            print(f"  格式: 與jieba完全一致")
            
        else:
            print("❌ 沒有成功處理任何記錄")
    
    except Exception as e:
        print(f"❌ 處理過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
