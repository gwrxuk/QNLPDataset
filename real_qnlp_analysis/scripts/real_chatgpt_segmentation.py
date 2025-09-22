#!/usr/bin/env python3
"""
真實ChatGPT中文斷詞完整分析
Real ChatGPT Chinese Word Segmentation Complete Analysis
"""

from openai import OpenAI
import pandas as pd
import numpy as np
import os
import time
import json
from typing import List, Dict, Optional
from collections import Counter
import re

class RealChatGPTSegmenter:
    """真實ChatGPT斷詞器"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None
        self.request_count = 0
        self.total_tokens = 0
        self.error_count = 0
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def create_segmentation_prompt(self, text: str) -> str:
        """創建專業斷詞提示詞"""
        return f"""請對以下中文文本進行專業斷詞並標註詞性。

要求：
1. 精確斷詞，保持語義完整性和自然性
2. 專有名詞、人名、地名、機構名保持完整
3. 數字、英文單詞單獨處理
4. 詞性標註使用標準標記：
   - n(名詞) v(動詞) a(形容詞) ad(副詞) p(介詞)
   - c(連詞) u(助詞) m(數詞) r(代詞) w(標點)
   - ns(地名) nr(人名) nt(機構名) nz(其他專名)
5. 格式：詞/詞性
6. 用空格分隔每個詞性標註對
7. 只返回斷詞結果，不要任何解釋

文本：{text}

斷詞結果："""
    
    def segment_text(self, text: str, retry_count: int = 2) -> Dict:
        """使用ChatGPT進行文本斷詞"""
        
        if not self.client:
            return self._error_result('API客戶端未初始化，請檢查API密鑰')
        
        if not text or not text.strip():
            return self._empty_result()
        
        # 清理文本
        cleaned_text = self._clean_text(text)
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
                            "content": "你是專業的中文自然語言處理專家，擅長精確的中文斷詞和詞性標註。你的分析準確、一致，遵循學術標準。"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=800,
                    temperature=0.1,  # 低溫度確保一致性
                    top_p=0.9
                )
                
                self.request_count += 1
                self.total_tokens += response.usage.total_tokens
                
                result_text = response.choices[0].message.content.strip()
                return self._parse_segmentation_result(result_text, response.usage.total_tokens)
                
            except Exception as e:
                if attempt < retry_count:
                    print(f"⚠️  API調用失敗，重試中... (第{attempt + 1}次)")
                    time.sleep(2 ** attempt)  # 指數退避
                    continue
                else:
                    self.error_count += 1
                    return self._error_result(f"API調用失敗: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        # 移除過多的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _parse_segmentation_result(self, result_text: str, tokens_used: int) -> Dict:
        """解析斷詞結果"""
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
        
        return {
            'error': None,
            'words': words,
            'segmented_text': ' / '.join(words),
            'pos_tags': ' | '.join(pos_tags),
            'word_count': len(words),
            'unique_words': len(set(words)),
            'tokens_used': tokens_used,
            'raw_response': result_text
        }
    
    def _empty_result(self) -> Dict:
        """空結果"""
        return {
            'error': None,
            'words': [],
            'segmented_text': '',
            'pos_tags': '',
            'word_count': 0,
            'unique_words': 0,
            'tokens_used': 0,
            'raw_response': ''
        }
    
    def _error_result(self, error_msg: str) -> Dict:
        """錯誤結果"""
        return {
            'error': error_msg,
            'words': [],
            'segmented_text': '',
            'pos_tags': '',
            'word_count': 0,
            'unique_words': 0,
            'tokens_used': 0,
            'raw_response': ''
        }
    
    def batch_segment(self, texts: List[str], delay: float = 1.0, 
                     progress_callback=None) -> List[Dict]:
        """批量斷詞"""
        results = []
        
        for i, text in enumerate(texts):
            if progress_callback:
                progress_callback(i + 1, len(texts), text)
            
            result = self.segment_text(text)
            results.append(result)
            
            # API限制控制
            if delay > 0 and i < len(texts) - 1:
                time.sleep(delay)
        
        return results
    
    def get_stats(self) -> Dict:
        """獲取統計信息"""
        return {
            'total_requests': self.request_count,
            'total_tokens': self.total_tokens,
            'error_count': self.error_count,
            'success_rate': (self.request_count - self.error_count) / max(self.request_count, 1),
            'avg_tokens_per_request': self.total_tokens / max(self.request_count, 1)
        }

def analyze_field_with_real_chatgpt(df: pd.DataFrame, field_name: str, 
                                   segmenter: RealChatGPTSegmenter,
                                   max_records: Optional[int] = None) -> List[Dict]:
    """使用真實ChatGPT分析欄位"""
    
    print(f"\n🤖 使用真實ChatGPT分析 {field_name} 欄位")
    print("=" * 50)
    
    if not segmenter.client:
        print("❌ ChatGPT API未設定，請檢查OPENAI_API_KEY環境變數")
        return []
    
    # 限制處理數量（如果指定）
    field_data = df[field_name].dropna()
    if max_records:
        field_data = field_data.head(max_records)
        print(f"📊 處理前 {max_records} 筆記錄")
    else:
        print(f"📊 處理全部 {len(field_data)} 筆記錄")
    
    results = []
    
    def progress_callback(current, total, text):
        progress = (current / total) * 100
        print(f"進度: {current}/{total} ({progress:.1f}%) - {text[:30]}...")
    
    # 批量處理
    segmentation_results = segmenter.batch_segment(
        field_data.tolist(), 
        delay=1.2,  # 避免API限制
        progress_callback=progress_callback
    )
    
    # 組織結果
    for idx, (original_idx, text) in enumerate(field_data.items()):
        seg_result = segmentation_results[idx]
        
        results.append({
            'record_id': original_idx,
            'field': field_name,
            'original_text': str(text),
            'segmented_text': seg_result['segmented_text'],
            'pos_tags': seg_result['pos_tags'],
            'word_count': seg_result['word_count'],
            'unique_word_count': seg_result['unique_words'],
            'words_list': ', '.join(seg_result['words']),
            'tokens_used': seg_result['tokens_used'],
            'api_error': seg_result['error'],
            'raw_response': seg_result['raw_response']
        })
    
    # 顯示統計
    stats = segmenter.get_stats()
    successful = len([r for r in results if not r['api_error']])
    
    print(f"\n📈 {field_name} 處理完成:")
    print(f"  成功處理: {successful}/{len(results)}")
    print(f"  總tokens: {stats['total_tokens']}")
    print(f"  平均tokens/筆: {stats['avg_tokens_per_request']:.1f}")
    print(f"  成功率: {stats['success_rate']:.1%}")
    
    return results

def main():
    """主函數"""
    print("🚀 真實ChatGPT中文斷詞完整分析")
    print("=" * 50)
    
    # 檢查API密鑰
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ 請設定OPENAI_API_KEY環境變數")
        print("設定方法: export OPENAI_API_KEY='your-api-key'")
        return
    
    print(f"✅ API密鑰已設定: {api_key[:10]}...")
    
    try:
        # 讀取數據
        print("\n📊 讀取數據集...")
        df = pd.read_excel('../dataseet.xlsx')
        print(f"數據集形狀: {df.shape}")
        
        # 目標欄位
        target_fields = ['新聞標題', '影片對話', '影片描述']
        available_fields = [field for field in target_fields if field in df.columns]
        
        if not available_fields:
            print("❌ 未找到目標欄位")
            return
        
        print(f"✅ 找到欄位: {', '.join(available_fields)}")
        
        # 詢問處理範圍
        print(f"\n⚙️  處理設定:")
        max_records_input = input(f"每個欄位處理多少筆記錄？(Enter=全部, 數字=限制筆數): ").strip()
        
        max_records = None
        if max_records_input.isdigit():
            max_records = int(max_records_input)
            print(f"📝 將處理每個欄位的前 {max_records} 筆記錄")
        else:
            print("📝 將處理全部記錄")
        
        # 初始化ChatGPT斷詞器
        segmenter = RealChatGPTSegmenter(api_key=api_key)
        
        # 處理各欄位
        all_results = []
        
        for field in available_fields:
            field_results = analyze_field_with_real_chatgpt(
                df, field, segmenter, max_records
            )
            all_results.extend(field_results)
            
            # 保存中間結果
            if field_results:
                field_df = pd.DataFrame(field_results)
                filename = f"../data/chatgpt_{field}_segmentation.csv"
                field_df.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"💾 {field} 結果已保存: {filename}")
        
        # 保存完整結果
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv('../data/real_chatgpt_segmentation_complete.csv', 
                            index=False, encoding='utf-8-sig')
            print(f"\n💾 完整結果已保存: ../data/real_chatgpt_segmentation_complete.csv")
            
            # 生成統計摘要
            final_stats = segmenter.get_stats()
            successful_results = [r for r in all_results if not r['api_error']]
            
            summary = {
                'total_records': len(all_results),
                'successful_records': len(successful_results),
                'total_tokens': final_stats['total_tokens'],
                'total_cost_estimate': (final_stats['total_tokens'] / 1000) * 0.002,  # GPT-3.5-turbo價格
                'avg_words_per_record': np.mean([r['word_count'] for r in successful_results]) if successful_results else 0,
                'fields_processed': available_fields,
                'processing_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 保存統計摘要
            with open('../data/chatgpt_processing_summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"\n🎉 處理完成摘要:")
            print(f"  總記錄數: {summary['total_records']}")
            print(f"  成功處理: {summary['successful_records']}")
            print(f"  總tokens: {summary['total_tokens']}")
            print(f"  估計成本: ${summary['total_cost_estimate']:.4f}")
            print(f"  平均詞數: {summary['avg_words_per_record']:.1f}")
            print(f"  處理時間: {summary['processing_time']}")
            
        else:
            print("❌ 沒有成功處理任何記錄")
    
    except Exception as e:
        print(f"❌ 處理過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
