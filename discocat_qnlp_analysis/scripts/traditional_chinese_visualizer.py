#!/usr/bin/env python3
"""
繁體中文可視化分析器
重新生成所有圖表，確保使用繁體中文顯示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import platform

def setup_traditional_chinese_font():
    """設置繁體中文字體以確保正確顯示"""
    print("🔧 設置繁體中文字體...")
    
    # 檢測操作系統並設置相應的繁體中文字體
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        fonts = ['Arial Unicode MS', 'PingFang TC', 'Heiti TC', 'STSong', 'LiHei Pro']
        print("🍎 檢測到macOS系統")
    elif system == 'Windows':
        fonts = ['Microsoft JhengHei', 'PMingLiU', 'MingLiU', 'DFKai-SB']
        print("🪟 檢測到Windows系統")
    else:  # Linux
        fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK TC', 'AR PL UMing TW']
        print("🐧 檢測到Linux系統")
    
    # 設置matplotlib參數
    plt.rcParams['axes.unicode_minus'] = False
    
    # 嘗試設置字體
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            print(f"✅ 設置字體: {font}")
            break
        except:
            continue
    
    # 驗證繁體中文字體設置
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 測試繁體中文字符
    fig, ax = plt.subplots(figsize=(10, 6))
    test_text = "測試繁體中文字符顯示：量子自然語言處理分析\n馮紐曼熵、語義干涉、框架競爭強度"
    ax.text(0.5, 0.5, test_text, ha='center', va='center', fontsize=16)
    ax.set_title('繁體中文字體測試圖表', fontsize=18, fontweight='bold')
    ax.axis('off')
    
    # 保存測試圖
    test_path = '../20250927-image/traditional_chinese_font_test.png'
    plt.savefig(test_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 繁體中文字體測試圖已保存: {test_path}")

def load_full_dataset_results():
    """載入完整資料集的分析結果"""
    print("📂 載入完整資料集分析結果...")
    
    # 載入統計摘要
    ai_summary_path = '../results/full_qiskit_ai_analysis_summary.json'
    journalist_summary_path = '../results/full_qiskit_journalist_analysis_summary.json'
    field_level_path = '../results/full_field_level_quantum_analysis.json'
    
    with open(ai_summary_path, 'r', encoding='utf-8') as f:
        ai_summary = json.load(f)
    
    with open(journalist_summary_path, 'r', encoding='utf-8') as f:
        journalist_summary = json.load(f)
        
    with open(field_level_path, 'r', encoding='utf-8') as f:
        field_level_data = json.load(f)
    
    print("✅ 資料載入完成")
    return ai_summary, journalist_summary, field_level_data

def create_comprehensive_comparison_tc(ai_summary, journalist_summary, field_level_data):
    """創建綜合對比圖表（繁體中文）"""
    print("📊 生成綜合對比圖表（繁體中文）...")
    
    # 量子指標（繁體中文）
    metrics = ['von_neumann_entropy', 'superposition_strength', 'quantum_coherence', 
               'semantic_interference', 'frame_competition', 'multiple_reality_strength']
    
    metric_names = {
        'von_neumann_entropy': '馮紐曼熵',
        'superposition_strength': '量子疊加強度',
        'quantum_coherence': '量子相干性',
        'semantic_interference': '語義干涉',
        'frame_competition': '框架競爭',
        'multiple_reality_strength': '多重現實強度'
    }
    
    # 創建2x3的子圖布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('完整資料集量子特徵對比分析\n(基於934個文本片段的Qiskit量子電路分析)', 
                 fontsize=20, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # 資料準備
        ai_mean = ai_summary[metric]['mean']
        ai_std = ai_summary[metric]['std']
        journalist_mean = journalist_summary[metric]['mean']
        journalist_std = journalist_summary[metric]['std']
        
        # 柱狀圖
        categories = ['AI生成新聞\n(298條記錄)', '記者撰寫新聞\n(20條記錄)']
        means = [ai_mean, journalist_mean]
        stds = [ai_std, journalist_std]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(categories, means, yerr=stds, capsize=8, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加數值標籤
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_title(f'{metric_names[metric]}', fontsize=14, fontweight='bold')
        ax.set_ylabel('數值', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 設置Y軸範圍
        max_val = max(means) + max(stds)
        min_val = min(means) - max(stds)
        margin = (max_val - min_val) * 0.15
        ax.set_ylim(max(0, min_val - margin), max_val + margin)
    
    plt.tight_layout()
    plt.savefig('../20250927-image/comprehensive_quantum_comparison_tc.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 綜合對比圖表（繁體中文）已保存")

def create_field_level_heatmap_tc(field_level_data):
    """創建欄位級別熱力圖（繁體中文）"""
    print("🔥 生成欄位級別熱力圖（繁體中文）...")
    
    # 準備資料
    metrics = ['von_neumann_entropy', 'superposition_strength', 'quantum_coherence', 
               'semantic_interference', 'multiple_reality_strength']
    
    # AI資料
    ai_fields = ['新聞標題', '影片對話', '影片描述']
    ai_data = []
    ai_labels = []
    
    for field in ai_fields:
        row = []
        for metric in metrics:
            mean_val = field_level_data['AI_Generated'][field][metric]['mean']
            row.append(mean_val)
        ai_data.append(row)
        ai_labels.append(f'AI-{field}')
    
    # 記者資料
    journalist_fields = ['新聞標題', '新聞內容']
    journalist_data = []
    journalist_labels = []
    
    for field in journalist_fields:
        row = []
        for metric in metrics:
            mean_val = field_level_data['Journalist_Written'][field][metric]['mean']
            row.append(mean_val)
        journalist_data.append(row)
        journalist_labels.append(f'記者-{field}')
    
    # 合併資料
    all_data = np.array(ai_data + journalist_data)
    all_labels = ai_labels + journalist_labels
    
    # 創建熱力圖
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metric_names = ['馮紐曼熵', '量子疊加強度', '量子相干性', '語義干涉', '多重現實強度']
    
    # 使用seaborn創建熱力圖
    sns.heatmap(all_data, 
                xticklabels=metric_names,
                yticklabels=all_labels,
                annot=True, 
                fmt='.4f',
                cmap='RdYlBu_r',
                center=None,
                ax=ax,
                cbar_kws={'label': '量子特徵值'})
    
    ax.set_title('欄位級別量子特徵熱力圖\n(完整資料集分析)', fontsize=16, fontweight='bold')
    ax.set_xlabel('量子指標', fontsize=12)
    ax.set_ylabel('資料來源與欄位', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../20250927-image/field_level_heatmap_tc.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 欄位級別熱力圖（繁體中文）已保存")

def create_radar_chart_tc(field_level_data):
    """創建雷達圖對比（繁體中文）"""
    print("🎯 生成雷達圖對比（繁體中文）...")
    
    metrics = ['von_neumann_entropy', 'superposition_strength', 'quantum_coherence', 
               'semantic_interference', 'multiple_reality_strength']
    metric_names = ['馮紐曼熵', '量子疊加強度', '量子相干性', '語義干涉', '多重現實強度']
    
    # 資料準備 - 選擇代表性欄位進行對比
    ai_dialogue = []  # AI影片對話
    journalist_content = []  # 記者新聞內容
    
    for metric in metrics:
        ai_val = field_level_data['AI_Generated']['影片對話'][metric]['mean']
        journalist_val = field_level_data['Journalist_Written']['新聞內容'][metric]['mean']
        
        # 歸一化到0-1範圍用於雷達圖顯示
        max_val = max(ai_val, journalist_val)
        if max_val > 0:
            ai_dialogue.append(ai_val / max_val)
            journalist_content.append(journalist_val / max_val)
        else:
            ai_dialogue.append(0)
            journalist_content.append(0)
    
    # 創建雷達圖
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 閉合雷達圖
    
    ai_dialogue += ai_dialogue[:1]
    journalist_content += journalist_content[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 繪製雷達圖
    ax.plot(angles, ai_dialogue, 'o-', linewidth=3, label='AI影片對話 (298條)', color='#FF6B6B')
    ax.fill(angles, ai_dialogue, alpha=0.25, color='#FF6B6B')
    
    ax.plot(angles, journalist_content, 'o-', linewidth=3, label='記者新聞內容 (20條)', color='#4ECDC4')
    ax.fill(angles, journalist_content, alpha=0.25, color='#4ECDC4')
    
    # 設置標籤
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=12)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    plt.title('量子特徵雷達圖對比\n(長文本欄位代表性分析)', fontsize=16, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.savefig('../20250927-image/quantum_radar_comparison_tc.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 雷達圖對比（繁體中文）已保存")

def create_distribution_analysis_tc(field_level_data):
    """創建分佈分析圖（繁體中文）"""
    print("📈 生成分佈分析圖（繁體中文）...")
    
    # 選擇關鍵指標進行分佈分析
    key_metrics = ['von_neumann_entropy', 'semantic_interference', 'multiple_reality_strength']
    metric_names = ['馮紐曼熵', '語義干涉', '多重現實強度']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('關鍵量子指標的欄位分佈分析\n(完整資料集)', fontsize=16, fontweight='bold')
    
    for i, (metric, name) in enumerate(zip(key_metrics, metric_names)):
        ax = axes[i]
        
        # 收集所有欄位的資料
        field_names = []
        values = []
        colors = []
        
        # AI資料
        for field in ['新聞標題', '影片對話', '影片描述']:
            mean_val = field_level_data['AI_Generated'][field][metric]['mean']
            std_val = field_level_data['AI_Generated'][field][metric]['std']
            count = field_level_data['AI_Generated'][field][metric]['count']
            
            field_names.append(f'AI-{field}\n({count}條)')
            values.append(mean_val)
            colors.append('#FF6B6B')
        
        # 記者資料
        for field in ['新聞標題', '新聞內容']:
            mean_val = field_level_data['Journalist_Written'][field][metric]['mean']
            std_val = field_level_data['Journalist_Written'][field][metric]['std']
            count = field_level_data['Journalist_Written'][field][metric]['count']
            
            field_names.append(f'記者-{field}\n({count}條)')
            values.append(mean_val)
            colors.append('#4ECDC4')
        
        # 創建柱狀圖
        bars = ax.bar(range(len(field_names)), values, color=colors, alpha=0.8, edgecolor='black')
        
        # 添加數值標籤
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_ylabel('數值', fontsize=12)
        ax.set_xticks(range(len(field_names)))
        ax.set_xticklabels(field_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../20250927-image/distribution_analysis_tc.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 分佈分析圖（繁體中文）已保存")

def create_summary_statistics_table_tc():
    """創建統計摘要表格（繁體中文）"""
    print("📋 生成統計摘要表格（繁體中文）...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # 表格資料（繁體中文）
    table_data = [
        ['資料來源', '欄位', '記錄數', '馮紐曼熵', '量子疊加強度', '量子相干性', '語義干涉', '多重現實強度'],
        ['AI生成', '新聞標題', '298', '3.9967±0.0579', '3.7492±0.0145', '0.9373±0.0036', '0.0014±0.0026', '1.7001±0.0059'],
        ['AI生成', '影片對話', '298', '4.0000±0.0000', '3.7500±0.0000', '0.9375±0.0000', '0.0178±0.0042', '1.7054±0.0013'],
        ['AI生成', '影片描述', '298', '4.0000±0.0000', '3.7500±0.0000', '0.9375±0.0000', '0.0111±0.0039', '1.7033±0.0012'],
        ['記者撰寫', '新聞標題', '20', '3.8500±0.3663', '3.7125±0.0916', '0.9281±0.0229', '0.0008±0.0022', '1.6856±0.0369'],
        ['記者撰寫', '新聞內容', '20', '4.0000±0.0000', '3.7500±0.0000', '0.9375±0.0000', '0.0177±0.0060', '1.7054±0.0018']
    ]
    
    # 創建表格
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                     loc='center', cellLoc='center')
    
    # 設置表格樣式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)
    
    # 設置標題行樣式
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 設置資料行樣式
    for i in range(1, len(table_data)):
        color = '#FFE5E5' if 'AI生成' in table_data[i][0] else '#E5F9F6'
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)
    
    plt.title('完整資料集量子特徵統計摘要表\n(平均值±標準差)', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('../20250927-image/statistics_summary_table_tc.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 統計摘要表格（繁體中文）已保存")

def create_qubit_distribution_chart_tc():
    """創建量子位元分佈圖表（繁體中文）"""
    print("📊 生成量子位元分佈圖表（繁體中文）...")
    
    # 載入量子位元分佈資料
    with open('../20250927-image/qubit_distribution_data.json', 'r', encoding='utf-8') as f:
        qubit_data = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('量子位元分佈統計\n(基於934個文本片段)', fontsize=18, fontweight='bold')
    
    # AI生成新聞分佈
    ai_dist = qubit_data['ai_stats']['qubit_distribution']
    ai_qubits = list(ai_dist.keys())
    ai_counts = list(ai_dist.values())
    ai_total = sum(ai_counts)
    ai_percentages = [count/ai_total*100 for count in ai_counts]
    
    colors1 = ['#FF9999', '#FF6B6B', '#FF4444']
    bars1 = ax1.bar([f'{q}個量子位元' for q in ai_qubits], ai_counts, 
                   color=colors1[:len(ai_qubits)], alpha=0.8, edgecolor='black')
    
    # 添加百分比標籤
    for bar, count, pct in zip(bars1, ai_counts, ai_percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{count}條\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('AI生成新聞 (894條記錄)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('記錄數量', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 記者撰寫新聞分佈
    journalist_dist = qubit_data['journalist_stats']['qubit_distribution']
    journalist_qubits = list(journalist_dist.keys())
    journalist_counts = list(journalist_dist.values())
    journalist_total = sum(journalist_counts)
    journalist_percentages = [count/journalist_total*100 for count in journalist_counts]
    
    colors2 = ['#99E5E5', '#4ECDC4', '#44C4C4']
    bars2 = ax2.bar([f'{q}個量子位元' for q in journalist_qubits], journalist_counts, 
                   color=colors2[:len(journalist_qubits)], alpha=0.8, edgecolor='black')
    
    # 添加百分比標籤
    for bar, count, pct in zip(bars2, journalist_counts, journalist_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}條\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('記者撰寫新聞 (40條記錄)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('記錄數量', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../20250927-image/qubit_distribution_chart_tc.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 量子位元分佈圖表（繁體中文）已保存")

def main():
    """主函數"""
    print("🚀 開始完整資料集可視化分析（繁體中文）...")
    print(f"📊 分析規模: 934個文本片段")
    
    # 確保輸出目錄存在
    Path('../20250927-image').mkdir(exist_ok=True)
    
    # 設置繁體中文字體
    setup_traditional_chinese_font()
    
    # 載入資料
    ai_summary, journalist_summary, field_level_data = load_full_dataset_results()
    
    # 生成各種可視化圖表（繁體中文版本）
    create_comprehensive_comparison_tc(ai_summary, journalist_summary, field_level_data)
    create_field_level_heatmap_tc(field_level_data)
    create_radar_chart_tc(field_level_data)
    create_distribution_analysis_tc(field_level_data)
    create_summary_statistics_table_tc()
    create_qubit_distribution_chart_tc()
    
    print("\n🎉 完整資料集可視化分析（繁體中文）完成！")
    print("📂 所有圖表已保存到: ../20250927-image/")
    print("📊 生成的繁體中文圖表:")
    print("   1. traditional_chinese_font_test.png - 繁體中文字體測試")
    print("   2. comprehensive_quantum_comparison_tc.png - 綜合量子特徵對比")
    print("   3. field_level_heatmap_tc.png - 欄位級別熱力圖")
    print("   4. quantum_radar_comparison_tc.png - 量子特徵雷達圖")
    print("   5. distribution_analysis_tc.png - 分佈分析圖")
    print("   6. statistics_summary_table_tc.png - 統計摘要表格")
    print("   7. qubit_distribution_chart_tc.png - 量子位元分佈圖表")

if __name__ == "__main__":
    main()
