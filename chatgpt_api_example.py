#!/usr/bin/env python3
"""
ChatGPT API調用斷詞的具體示例
Specific Example of ChatGPT API Call for Word Segmentation
"""

from openai import OpenAI
import os
import json

def show_api_call_structure():
    """展示ChatGPT API調用的具體結構"""
    
    print("🔧 ChatGPT API調用結構示例")
    print("=" * 50)
    
    # 1. 設定API客戶端
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ 請設定環境變數: export OPENAI_API_KEY='your-key'")
        return
    
    client = OpenAI(api_key=api_key)
    
    # 2. 準備要斷詞的文本
    text = "人工智慧技術快速發展改變社會生活型態"
    
    # 3. 構建提示詞
    prompt = f"""請對以下中文文本進行斷詞並標註詞性。

要求：
1. 精確斷詞，保持語義完整
2. 詞性標註：n(名詞) v(動詞) a(形容詞) ad(副詞) p(介詞) c(連詞) u(助詞) m(數詞) r(代詞)
3. 格式：詞/詞性
4. 用空格分隔
5. 只返回結果，不要解釋

文本：{text}

結果："""
    
    print(f"📝 輸入文本: {text}")
    print(f"📋 提示詞預覽:\n{prompt[:100]}...")
    
    try:
        # 4. 進行API調用
        print("\n🔄 正在調用ChatGPT API...")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "你是專業的中文NLP專家，擅長精確斷詞和詞性標註。"
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
        
        # 5. 解析回應
        result = response.choices[0].message.content.strip()
        
        print("\n✅ API調用成功!")
        print(f"📤 ChatGPT回應: {result}")
        
        # 6. 解析斷詞結果
        words = []
        pos_tags = []
        
        items = result.split()
        for item in items:
            if '/' in item:
                word, pos = item.split('/', 1)
                words.append(word.strip())
                pos_tags.append(f"{word.strip()}/{pos.strip()}")
        
        print(f"\n📊 解析結果:")
        print(f"  斷詞: {' / '.join(words)}")
        print(f"  詞性: {' | '.join(pos_tags)}")
        print(f"  詞數: {len(words)}")
        
        # 7. 顯示API使用信息
        print(f"\n💰 API使用信息:")
        print(f"  模型: {response.model}")
        print(f"  輸入tokens: {response.usage.prompt_tokens}")
        print(f"  輸出tokens: {response.usage.completion_tokens}")
        print(f"  總tokens: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"❌ API調用失敗: {e}")

def show_batch_processing_example():
    """展示批量處理的API調用示例"""
    
    print("\n🔄 批量處理示例")
    print("=" * 30)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ 需要API密鑰")
        return
    
    client = OpenAI(api_key=api_key)
    
    # 測試文本列表
    texts = [
        "台灣政府推動綠能政策",
        "ChatGPT改變人工智慧發展",
        "量子計算突破傳統限制"
    ]
    
    results = []
    
    for i, text in enumerate(texts, 1):
        print(f"\n處理第 {i} 個文本: {text}")
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是中文斷詞專家。"},
                    {"role": "user", "content": f"請斷詞：{text}\n格式：詞/詞性 詞/詞性"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            results.append({
                'original': text,
                'segmented': result,
                'tokens_used': response.usage.total_tokens
            })
            
            print(f"✅ 結果: {result}")
            print(f"💰 使用tokens: {response.usage.total_tokens}")
            
        except Exception as e:
            print(f"❌ 處理失敗: {e}")
    
    # 總結
    if results:
        total_tokens = sum(r['tokens_used'] for r in results)
        print(f"\n📊 批量處理總結:")
        print(f"  處理文本數: {len(results)}")
        print(f"  總tokens使用: {total_tokens}")
        print(f"  平均每個文本: {total_tokens/len(results):.1f} tokens")

def show_cost_estimation():
    """展示成本估算"""
    
    print("\n💰 ChatGPT斷詞成本估算")
    print("=" * 30)
    
    # GPT-3.5-turbo 價格 (2024年參考價格)
    price_per_1k_input = 0.0015  # USD
    price_per_1k_output = 0.002  # USD
    
    print(f"📋 GPT-3.5-turbo 價格:")
    print(f"  輸入: ${price_per_1k_input}/1K tokens")
    print(f"  輸出: ${price_per_1k_output}/1K tokens")
    
    # 估算不同文本量的成本
    scenarios = [
        {"name": "短文本 (10-20字)", "input_tokens": 50, "output_tokens": 30},
        {"name": "中文本 (50-100字)", "input_tokens": 120, "output_tokens": 80},
        {"name": "長文本 (200-300字)", "input_tokens": 300, "output_tokens": 200},
    ]
    
    print(f"\n📊 不同文本長度的成本估算:")
    
    for scenario in scenarios:
        input_cost = (scenario["input_tokens"] / 1000) * price_per_1k_input
        output_cost = (scenario["output_tokens"] / 1000) * price_per_1k_output
        total_cost = input_cost + output_cost
        
        print(f"\n  {scenario['name']}:")
        print(f"    輸入tokens: {scenario['input_tokens']}")
        print(f"    輸出tokens: {scenario['output_tokens']}")
        print(f"    單次成本: ${total_cost:.6f}")
        print(f"    1000次成本: ${total_cost * 1000:.3f}")

def main():
    """主函數"""
    
    print("🎯 ChatGPT API斷詞調用示例")
    print("請選擇功能:")
    print("1. 基本API調用結構")
    print("2. 批量處理示例")  
    print("3. 成本估算")
    print("4. 全部執行")
    
    choice = input("\n請選擇 (1-4): ").strip()
    
    if choice == "1":
        show_api_call_structure()
    elif choice == "2":
        show_batch_processing_example()
    elif choice == "3":
        show_cost_estimation()
    elif choice == "4":
        show_api_call_structure()
        show_batch_processing_example()
        show_cost_estimation()
    else:
        print("❌ 無效選項")

if __name__ == "__main__":
    main()
