#!/usr/bin/env python3
"""
簡單的ChatGPT中文斷詞示例
Simple ChatGPT Chinese Word Segmentation Example
"""

import openai
import os
from typing import List, Dict

# 設定API密鑰
openai.api_key = os.getenv('OPENAI_API_KEY')

def call_chatgpt_for_segmentation(text: str, include_pos: bool = True) -> Dict:
    """
    呼叫ChatGPT進行中文斷詞
    
    Args:
        text: 要斷詞的中文文本
        include_pos: 是否包含詞性標註
    
    Returns:
        包含斷詞結果的字典
    """
    
    # 構建提示詞
    if include_pos:
        prompt = f"""請對以下中文文本進行斷詞並標註詞性。

要求：
1. 精確斷詞，保持語義完整
2. 為每個詞標註詞性（名詞n、動詞v、形容詞a、副詞ad、介詞p、連詞c、助詞u、數詞m、代詞r等）
3. 格式：詞/詞性
4. 用空格分隔每個詞性標註對
5. 只返回斷詞結果，不要其他說明

文本：{text}

斷詞結果："""
    else:
        prompt = f"""請對以下中文文本進行精確斷詞。

要求：
1. 將文本分割成有意義的詞彙單位
2. 用空格分隔每個詞
3. 保持語義完整性
4. 專有名詞要保持完整
5. 只返回斷詞結果，不要其他說明

文本：{text}

斷詞結果："""
    
    try:
        # 呼叫ChatGPT API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 或使用 "gpt-4" 獲得更好效果
            messages=[
                {
                    "role": "system", 
                    "content": "你是一個專業的中文自然語言處理專家，擅長中文斷詞和詞性標註。"
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=1000,
            temperature=0.1,  # 低溫度確保一致性
            top_p=0.9
        )
        
        # 解析結果
        result_text = response.choices[0].message.content.strip()
        
        if include_pos:
            return parse_pos_result(result_text)
        else:
            return parse_simple_result(result_text)
            
    except Exception as e:
        return {
            'error': str(e),
            'words': [],
            'segmented_text': '',
            'pos_tags': ''
        }

def parse_pos_result(result_text: str) -> Dict:
    """解析帶詞性的斷詞結果"""
    words = []
    pos_tags = []
    
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
    
    return {
        'words': words,
        'segmented_text': ' / '.join(words),
        'pos_tags': ' | '.join(pos_tags),
        'word_count': len(words),
        'unique_words': len(set(words)),
        'error': None
    }

def parse_simple_result(result_text: str) -> Dict:
    """解析簡單斷詞結果"""
    words = [w.strip() for w in result_text.split() if w.strip()]
    
    return {
        'words': words,
        'segmented_text': ' / '.join(words),
        'pos_tags': '',
        'word_count': len(words),
        'unique_words': len(set(words)),
        'error': None
    }

def demo_chatgpt_segmentation():
    """演示ChatGPT斷詞功能"""
    
    # 檢查API密鑰
    if not openai.api_key:
        print("❌ 請設定OPENAI_API_KEY環境變數")
        print("設定方法: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # 測試文本
    test_texts = [
        "日本能源轉向增核能降火力台灣應借鏡",
        "麥當勞性侵案後改革董事長發聲承諾改善",
        "Google示警AI詐騙電話橫行Gmail用戶警惕",
        "台灣政府推動綠能政策發展再生能源產業",
        "人工智慧技術快速發展改變社會生活型態"
    ]
    
    print("🤖 ChatGPT中文斷詞演示")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 測試文本 {i}: {text}")
        
        # 呼叫ChatGPT進行斷詞
        result = call_chatgpt_for_segmentation(text, include_pos=True)
        
        if result['error']:
            print(f"❌ 錯誤: {result['error']}")
        else:
            print(f"✅ 斷詞結果: {result['segmented_text']}")
            print(f"📊 詞性標註: {result['pos_tags']}")
            print(f"📈 詞數統計: {result['word_count']} 詞，{result['unique_words']} 唯一詞")

def batch_segmentation_example():
    """批量斷詞示例"""
    
    if not openai.api_key:
        print("❌ 請設定OPENAI_API_KEY環境變數")
        return
    
    # 從檔案讀取文本進行批量處理
    try:
        import pandas as pd
        
        # 讀取數據
        df = pd.read_excel('dataseet.xlsx')
        
        print("📊 批量ChatGPT斷詞處理")
        print("=" * 40)
        
        # 處理前5筆新聞標題作為示例
        results = []
        
        for i in range(min(5, len(df))):
            title = str(df.iloc[i]['新聞標題'])
            print(f"\n處理第 {i+1} 筆: {title[:50]}...")
            
            result = call_chatgpt_for_segmentation(title, include_pos=True)
            
            if not result['error']:
                results.append({
                    '原文': title,
                    '斷詞結果': result['segmented_text'],
                    '詞性標註': result['pos_tags'],
                    '詞數': result['word_count']
                })
                print(f"✅ 完成: {result['word_count']} 個詞")
            else:
                print(f"❌ 失敗: {result['error']}")
        
        # 保存結果
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv('chatgpt_batch_segmentation_example.csv', 
                            index=False, encoding='utf-8-sig')
            print(f"\n💾 結果已保存至: chatgpt_batch_segmentation_example.csv")
            
            # 顯示結果摘要
            print(f"\n📈 處理摘要:")
            print(f"  成功處理: {len(results)} 筆")
            print(f"  平均詞數: {sum(r['詞數'] for r in results) / len(results):.1f}")
        
    except Exception as e:
        print(f"❌ 批量處理錯誤: {e}")

if __name__ == "__main__":
    print("🚀 ChatGPT中文斷詞工具")
    print("請選擇功能:")
    print("1. 演示斷詞功能")
    print("2. 批量處理示例")
    print("3. 自定義文本斷詞")
    
    choice = input("\n請輸入選項 (1-3): ").strip()
    
    if choice == "1":
        demo_chatgpt_segmentation()
    elif choice == "2":
        batch_segmentation_example()
    elif choice == "3":
        custom_text = input("請輸入要斷詞的中文文本: ").strip()
        if custom_text:
            if not openai.api_key:
                print("❌ 請設定OPENAI_API_KEY環境變數")
            else:
                result = call_chatgpt_for_segmentation(custom_text, include_pos=True)
                if result['error']:
                    print(f"❌ 錯誤: {result['error']}")
                else:
                    print(f"\n✅ 斷詞結果: {result['segmented_text']}")
                    print(f"📊 詞性標註: {result['pos_tags']}")
                    print(f"📈 詞數統計: {result['word_count']} 詞")
        else:
            print("❌ 請輸入有效的文本")
    else:
        print("❌ 無效選項")
