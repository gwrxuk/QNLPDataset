#!/usr/bin/env python3
"""
現代版ChatGPT中文斷詞 (使用最新OpenAI API)
Modern ChatGPT Chinese Word Segmentation (Latest OpenAI API)
"""

from openai import OpenAI
import os
from typing import List, Dict, Optional
import time

class ChatGPTSegmenter:
    """使用最新OpenAI API的ChatGPT斷詞器"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        初始化ChatGPT斷詞器
        
        Args:
            api_key: OpenAI API密鑰，如果為None則從環境變數讀取
            model: 使用的模型，推薦 "gpt-3.5-turbo" 或 "gpt-4"
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        
        self.request_count = 0
        
    def create_segmentation_prompt(self, text: str, include_pos: bool = True) -> str:
        """創建斷詞提示詞"""
        
        if include_pos:
            return f"""請對以下中文文本進行專業的斷詞分析並標註詞性。

要求：
1. 精確斷詞，保持語義完整性
2. 專有名詞、人名、地名保持完整
3. 為每個詞標註詞性：
   - n: 名詞, v: 動詞, a: 形容詞, ad: 副詞
   - p: 介詞, c: 連詞, u: 助詞, m: 數詞
   - r: 代詞, ns: 地名, nr: 人名
4. 格式：詞/詞性
5. 用空格分隔每個詞性標註對
6. 只返回斷詞結果，不要其他解釋

文本：{text}

斷詞結果："""
        else:
            return f"""請對以下中文文本進行專業斷詞。

要求：
1. 精確分割成有意義的詞彙單位
2. 保持語義完整性和自然性
3. 專有名詞要保持完整
4. 用空格分隔每個詞
5. 只返回斷詞結果，不要其他解釋

文本：{text}

斷詞結果："""
    
    def segment_text(self, text: str, include_pos: bool = True, 
                    temperature: float = 0.1, max_tokens: int = 1000) -> Dict:
        """
        使用ChatGPT進行文本斷詞
        
        Args:
            text: 要斷詞的文本
            include_pos: 是否包含詞性標註
            temperature: 控制隨機性 (0.0-1.0)
            max_tokens: 最大token數
            
        Returns:
            包含斷詞結果的字典
        """
        
        if not self.client:
            return {
                'error': 'API密鑰未設定，請設定OPENAI_API_KEY環境變數',
                'words': [],
                'segmented_text': '',
                'pos_tags': '',
                'word_count': 0,
                'unique_words': 0
            }
        
        if not text or not text.strip():
            return {
                'error': None,
                'words': [],
                'segmented_text': '',
                'pos_tags': '',
                'word_count': 0,
                'unique_words': 0
            }
        
        try:
            # 創建提示詞
            prompt = self.create_segmentation_prompt(text.strip(), include_pos)
            
            # 呼叫ChatGPT API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一個專業的中文自然語言處理專家，擅長精確的中文斷詞和詞性標註。你的回答簡潔準確，只提供要求的結果。"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            self.request_count += 1
            
            # 解析結果
            result_text = response.choices[0].message.content.strip()
            
            if include_pos:
                return self._parse_pos_result(result_text)
            else:
                return self._parse_simple_result(result_text)
                
        except Exception as e:
            return {
                'error': f"API調用錯誤: {str(e)}",
                'words': [],
                'segmented_text': '',
                'pos_tags': '',
                'word_count': 0,
                'unique_words': 0
            }
    
    def _parse_pos_result(self, result_text: str) -> Dict:
        """解析帶詞性標註的結果"""
        words = []
        pos_tags = []
        
        # 分割並解析每個詞性標註對
        items = result_text.split()
        for item in items:
            if '/' in item:
                parts = item.split('/', 1)  # 只分割第一個斜線
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
            'unique_words': len(set(words))
        }
    
    def _parse_simple_result(self, result_text: str) -> Dict:
        """解析簡單斷詞結果"""
        words = [w.strip() for w in result_text.split() if w.strip()]
        
        return {
            'error': None,
            'words': words,
            'segmented_text': ' / '.join(words),
            'pos_tags': '',
            'word_count': len(words),
            'unique_words': len(set(words))
        }
    
    def batch_segment(self, texts: List[str], include_pos: bool = True, 
                     delay: float = 1.0) -> List[Dict]:
        """
        批量斷詞
        
        Args:
            texts: 文本列表
            include_pos: 是否包含詞性標註
            delay: 請求間隔時間(秒)
            
        Returns:
            斷詞結果列表
        """
        results = []
        
        for i, text in enumerate(texts):
            print(f"處理進度: {i+1}/{len(texts)} - {text[:30]}...")
            
            result = self.segment_text(text, include_pos)
            results.append(result)
            
            # 避免API限制
            if delay > 0 and i < len(texts) - 1:
                time.sleep(delay)
        
        return results

def demo_usage():
    """演示使用方法"""
    
    print("🤖 現代版ChatGPT中文斷詞演示")
    print("=" * 50)
    
    # 初始化斷詞器
    segmenter = ChatGPTSegmenter(model="gpt-3.5-turbo")
    
    # 測試文本
    test_texts = [
        "日本能源轉向增核能降火力台灣應借鏡",
        "人工智慧技術快速發展改變社會生活",
        "台灣政府推動綠能政策發展再生能源",
        "ChatGPT等大型語言模型革命性突破",
        "量子計算與傳統計算的根本差異"
    ]
    
    print(f"\n📊 API狀態: {'✅ 已設定' if segmenter.client else '❌ 未設定API密鑰'}")
    
    if not segmenter.client:
        print("\n⚠️  請設定OPENAI_API_KEY環境變數:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 測試 {i}: {text}")
        
        # 進行斷詞
        result = segmenter.segment_text(text, include_pos=True)
        
        if result['error']:
            print(f"❌ 錯誤: {result['error']}")
        else:
            print(f"✅ 斷詞: {result['segmented_text']}")
            print(f"📊 詞性: {result['pos_tags'][:100]}...")  # 限制顯示長度
            print(f"📈 統計: {result['word_count']}詞, {result['unique_words']}唯一詞")
    
    print(f"\n📊 總API調用次數: {segmenter.request_count}")

def interactive_segmentation():
    """互動式斷詞"""
    
    segmenter = ChatGPTSegmenter()
    
    if not segmenter.client:
        print("❌ 請先設定OPENAI_API_KEY環境變數")
        return
    
    print("🎯 互動式ChatGPT中文斷詞")
    print("輸入 'quit' 或 'exit' 退出")
    print("=" * 40)
    
    while True:
        text = input("\n請輸入要斷詞的中文文本: ").strip()
        
        if text.lower() in ['quit', 'exit', '退出']:
            break
        
        if not text:
            print("⚠️  請輸入有效文本")
            continue
        
        print("🔄 處理中...")
        result = segmenter.segment_text(text, include_pos=True)
        
        if result['error']:
            print(f"❌ 錯誤: {result['error']}")
        else:
            print(f"\n✅ 斷詞結果: {result['segmented_text']}")
            print(f"📊 詞性標註: {result['pos_tags']}")
            print(f"📈 詞數統計: {result['word_count']} 詞，{result['unique_words']} 唯一詞")
    
    print(f"\n👋 再見！總共處理了 {segmenter.request_count} 次請求")

if __name__ == "__main__":
    import sys
    
    print("🚀 ChatGPT中文斷詞工具 (現代版)")
    print("選擇功能:")
    print("1. 演示功能")
    print("2. 互動式斷詞")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("請選擇 (1-2): ").strip()
    
    if choice == "1":
        demo_usage()
    elif choice == "2":
        interactive_segmentation()
    else:
        print("❌ 無效選項，執行演示功能")
        demo_usage()
