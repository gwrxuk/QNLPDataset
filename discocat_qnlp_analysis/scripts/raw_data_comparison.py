#!/usr/bin/env python3
"""
AI vs 記者新聞原始數值比較分析
"""

import pandas as pd
import json
import numpy as np

def load_and_compare_raw_data():
    """載入並比較原始數據"""
    
    print("📊 載入原始數據進行比較分析...")
    
    # 載入數據
    with open('../results/final_discocat_analysis_summary.json', 'r', encoding='utf-8') as f:
        ai_data = json.load(f)
    
    with open('../results/cna_final_discocat_analysis_summary.json', 'r', encoding='utf-8') as f:
        journalist_data = json.load(f)
    
    # 關鍵量子指標
    metrics = [
        'grammatical_superposition',
        'frame_competition', 
        'multiple_reality_strength',
        'frame_conflict_strength',
        'semantic_interference',
        'von_neumann_entropy',
        'category_coherence',
        'compositional_entanglement'
    ]
    
    metric_names_chinese = [
        '語法疊加強度',
        '框架競爭強度',
        '多重現實強度', 
        '框架衝突強度',
        '語義干涉',
        '馮紐曼熵',
        '類別一致性',
        '組合糾纏強度'
    ]
    
    # 創建詳細比較表
    comparison_data = []
    
    for field_pair in [('新聞標題', '新聞標題'), ('影片對話', '新聞內容')]:
        ai_field, journalist_field = field_pair
        
        print(f"\n📋 {ai_field} vs {journalist_field} 比較:")
        print("=" * 80)
        
        for i, metric in enumerate(metrics):
            ai_val = ai_data[ai_field][metric]['mean']
            ai_std = ai_data[ai_field][metric]['std']
            journalist_val = journalist_data[journalist_field][metric]['mean']
            journalist_std = journalist_data[journalist_field][metric]['std']
            
            # 計算差異倍數
            if journalist_val != 0:
                ratio = ai_val / journalist_val
            else:
                ratio = float('inf') if ai_val > 0 else 0
            
            comparison_data.append({
                '文本類型': f'{ai_field} vs {journalist_field}',
                '量子指標': metric_names_chinese[i],
                'AI新聞均值': f'{ai_val:.6f}',
                'AI新聞標準差': f'{ai_std:.6f}',
                '記者新聞均值': f'{journalist_val:.6f}',
                '記者新聞標準差': f'{journalist_std:.6f}',
                '差異倍數(AI/記者)': f'{ratio:.2f}' if ratio != float('inf') else '∞',
                '絕對差異': f'{abs(ai_val - journalist_val):.6f}'
            })
            
            print(f"{metric_names_chinese[i]:>8}: AI={ai_val:>10.6f} | 記者={journalist_val:>10.6f} | 倍數={ratio:>8.2f}")
    
    # 保存詳細比較表
    df = pd.DataFrame(comparison_data)
    output_file = '../results/raw_data_detailed_comparison.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n💾 詳細原始數據比較表已保存: {output_file}")
    
    return df

def analyze_significant_differences():
    """分析顯著差異"""
    
    print("\n🔍 顯著差異分析:")
    print("=" * 50)
    
    # 載入數據
    with open('../results/final_discocat_analysis_summary.json', 'r', encoding='utf-8') as f:
        ai_data = json.load(f)
    
    with open('../results/cna_final_discocat_analysis_summary.json', 'r', encoding='utf-8') as f:
        journalist_data = json.load(f)
    
    # 分析標題數據的關鍵差異
    title_differences = []
    
    # 1. 語義干涉 - 最大差異
    ai_interference = ai_data['新聞標題']['semantic_interference']['mean']
    journalist_interference = journalist_data['新聞標題']['semantic_interference']['mean']
    interference_ratio = ai_interference / journalist_interference if journalist_interference > 0 else float('inf')
    
    title_differences.append({
        '指標': '語義干涉',
        'AI新聞': ai_interference,
        '記者新聞': journalist_interference,
        '差異倍數': interference_ratio,
        '解釋': 'AI新聞語義相互作用更複雜'
    })
    
    # 2. 馮紐曼熵 - 資訊密度
    ai_entropy = ai_data['新聞標題']['von_neumann_entropy']['mean']
    journalist_entropy = journalist_data['新聞標題']['von_neumann_entropy']['mean']
    entropy_ratio = journalist_entropy / ai_entropy
    
    title_differences.append({
        '指標': '馮紐曼熵',
        'AI新聞': ai_entropy,
        '記者新聞': journalist_entropy,
        '差異倍數': entropy_ratio,
        '解釋': '記者新聞資訊密度更高'
    })
    
    # 3. 組合糾纏強度
    ai_entanglement = ai_data['新聞標題']['compositional_entanglement']['mean']
    journalist_entanglement = journalist_data['新聞標題']['compositional_entanglement']['mean']
    entanglement_ratio = journalist_entanglement / ai_entanglement
    
    title_differences.append({
        '指標': '組合糾纏強度',
        'AI新聞': ai_entanglement,
        '記者新聞': journalist_entanglement,
        '差異倍數': entanglement_ratio,
        '解釋': '記者新聞語法成分關聯更強'
    })
    
    # 4. 類別一致性
    ai_coherence = ai_data['新聞標題']['category_coherence']['mean']
    journalist_coherence = journalist_data['新聞標題']['category_coherence']['mean']
    coherence_ratio = journalist_coherence / ai_coherence
    
    title_differences.append({
        '指標': '類別一致性',
        'AI新聞': ai_coherence,
        '記者新聞': journalist_coherence,
        '差異倍數': coherence_ratio,
        '解釋': '記者新聞詞性使用更一致'
    })
    
    # 顯示分析結果
    for diff in title_differences:
        print(f"\n📈 {diff['指標']}:")
        print(f"   AI新聞:     {diff['AI新聞']:.6f}")
        print(f"   記者新聞:   {diff['記者新聞']:.6f}")
        print(f"   差異倍數:   {diff['差異倍數']:.2f}×")
        print(f"   解釋:       {diff['解釋']}")
    
    return title_differences

def create_ranking_analysis():
    """創建排名分析"""
    
    print("\n🏆 量子特徵強度排名分析:")
    print("=" * 50)
    
    # 載入數據
    with open('../results/final_discocat_analysis_summary.json', 'r', encoding='utf-8') as f:
        ai_data = json.load(f)
    
    with open('../results/cna_final_discocat_analysis_summary.json', 'r', encoding='utf-8') as f:
        journalist_data = json.load(f)
    
    # 標題數據排名
    metrics_for_ranking = [
        ('語法疊加強度', 'grammatical_superposition'),
        ('框架競爭強度', 'frame_competition'),
        ('多重現實強度', 'multiple_reality_strength'),
        ('框架衝突強度', 'frame_conflict_strength'),
        ('語義干涉', 'semantic_interference'),
        ('馮紐曼熵', 'von_neumann_entropy'),
        ('類別一致性', 'category_coherence'),
        ('組合糾纏強度', 'compositional_entanglement')
    ]
    
    ai_ranking = []
    journalist_ranking = []
    
    for name, metric in metrics_for_ranking:
        ai_val = ai_data['新聞標題'][metric]['mean']
        journalist_val = journalist_data['新聞標題'][metric]['mean']
        
        ai_ranking.append((name, ai_val))
        journalist_ranking.append((name, journalist_val))
    
    # 按數值排序
    ai_ranking.sort(key=lambda x: x[1], reverse=True)
    journalist_ranking.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🤖 AI新聞量子特徵強度排名:")
    for i, (name, value) in enumerate(ai_ranking, 1):
        print(f"   {i:2d}. {name:<12}: {value:.6f}")
    
    print("\n👨‍💼 記者新聞量子特徵強度排名:")
    for i, (name, value) in enumerate(journalist_ranking, 1):
        print(f"   {i:2d}. {name:<12}: {value:.6f}")

def main():
    """主函數"""
    
    print("🚀 開始原始數值比較分析")
    print("=" * 60)
    
    # 詳細數據比較
    comparison_df = load_and_compare_raw_data()
    
    # 顯著差異分析
    significant_diffs = analyze_significant_differences()
    
    # 排名分析
    create_ranking_analysis()
    
    print(f"\n✅ 原始數值比較分析完成!")
    print(f"📄 詳細比較表: ../results/raw_data_detailed_comparison.csv")
    print(f"🔍 關鍵發現:")
    print(f"   • 語義干涉: AI新聞是記者新聞的 {significant_diffs[0]['差異倍數']:.0f} 倍")
    print(f"   • 馮紐曼熵: 記者新聞是AI新聞的 {significant_diffs[1]['差異倍數']:.2f} 倍")
    print(f"   • 組合糾纏: 記者新聞是AI新聞的 {significant_diffs[2]['差異倍數']:.2f} 倍")
    print(f"   • 類別一致性: 記者新聞是AI新聞的 {significant_diffs[3]['差異倍數']:.2f} 倍")

if __name__ == "__main__":
    main()
