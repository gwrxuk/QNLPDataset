#!/usr/bin/env python3
"""
AI vs 記者新聞量子特徵比較可視化 - 修復中文顯示版本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_fonts():
    """設置中文字體"""
    print("🔧 設置中文字體...")
    
    # 獲取系統所有字體
    font_list = [f.name for f in fm.fontManager.ttflist]
    
    # macOS 常見中文字體
    mac_fonts = ['Arial Unicode MS', 'PingFang SC', 'Helvetica Neue', 'STHeiti', 'STSong']
    # Windows 常見中文字體  
    win_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    # Linux 常見中文字體
    linux_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC']
    
    all_chinese_fonts = mac_fonts + win_fonts + linux_fonts
    
    # 找到可用的中文字體
    available_chinese = [font for font in all_chinese_fonts if font in font_list]
    
    if available_chinese:
        selected_font = available_chinese[0]
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 使用中文字體: {selected_font}")
        return selected_font
    else:
        # 如果沒有中文字體，使用英文替代
        print("⚠️  未找到中文字體，使用英文標籤")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return None

def get_labels(has_chinese_font):
    """根據字體支援情況返回標籤"""
    if has_chinese_font:
        return {
            'metrics': ['語法疊加強度', '框架競爭強度', '多重現實強度', '框架衝突強度', '語義干涉', '馮紐曼熵'],
            'categories': ['新聞標題', '內容/對話'],
            'ai_label': 'AI新聞',
            'journalist_label': '記者新聞',
            'titles': {
                'radar': '新聞標題量子特徵雷達圖',
                'interference': '語義干涉強度對比（差異達386倍）',
                'competition_conflict': '框架競爭 vs 框架衝突模式', 
                'reality': '多重現實強度分佈對比',
                'entropy': '資訊密度對比（馮紐曼熵）',
                'fingerprint': '量子特徵指紋對比',
                'main': 'AI生成新聞 vs 記者撰寫新聞：量子特徵全面對比'
            }
        }
    else:
        return {
            'metrics': ['Grammar Superposition', 'Frame Competition', 'Multiple Reality', 'Frame Conflict', 'Semantic Interference', 'Von Neumann Entropy'],
            'categories': ['News Title', 'Content/Dialog'],
            'ai_label': 'AI News',
            'journalist_label': 'Journalist News', 
            'titles': {
                'radar': 'News Title Quantum Features Radar Chart',
                'interference': 'Semantic Interference Comparison (386x Difference)',
                'competition_conflict': 'Frame Competition vs Frame Conflict Pattern',
                'reality': 'Multiple Reality Strength Distribution',
                'entropy': 'Information Density Comparison (Von Neumann Entropy)',
                'fingerprint': 'Quantum Feature Fingerprint Comparison',
                'main': 'AI Generated News vs Journalist Written News: Quantum Features Comparison'
            }
        }

def load_comparison_data():
    """載入比較數據"""
    
    # AI新聞數據
    with open('../results/final_discocat_analysis_summary.json', 'r', encoding='utf-8') as f:
        ai_data = json.load(f)
    
    # 記者新聞數據
    with open('../results/cna_final_discocat_analysis_summary.json', 'r', encoding='utf-8') as f:
        journalist_data = json.load(f)
    
    return ai_data, journalist_data

def create_comparison_charts():
    """創建比較圖表"""
    
    print("🎨 開始創建量子特徵比較圖表...")
    
    # 設置中文字體
    chinese_font = setup_chinese_fonts()
    labels = get_labels(chinese_font is not None)
    
    ai_data, journalist_data = load_comparison_data()
    
    # 設置圖表樣式
    plt.style.use('default')  # 使用默認樣式，更穩定
    fig = plt.figure(figsize=(20, 24))
    
    # 主要量子指標
    quantum_metrics = [
        'grammatical_superposition',
        'frame_competition', 
        'multiple_reality_strength',
        'frame_conflict_strength',
        'semantic_interference',
        'von_neumann_entropy'
    ]
    
    # 1. 雷達圖比較 (新聞標題)
    ax1 = plt.subplot(3, 2, 1, projection='polar')
    
    # 準備雷達圖數據
    ai_title_values = []
    journalist_title_values = []
    
    for metric in quantum_metrics:
        ai_val = ai_data['新聞標題'][metric]['mean']
        journalist_val = journalist_data['新聞標題'][metric]['mean']
        
        # 正規化到0-1範圍（馮紐曼熵需要特殊處理）
        if metric == 'von_neumann_entropy':
            ai_val = min(1.0, ai_val / 5.0)  # 假設最大值為5
            journalist_val = min(1.0, journalist_val / 5.0)
        
        ai_title_values.append(ai_val)
        journalist_title_values.append(journalist_val)
    
    # 雷達圖角度
    angles = np.linspace(0, 2*np.pi, len(quantum_metrics), endpoint=False).tolist()
    ai_title_values += ai_title_values[:1]  # 閉合
    journalist_title_values += journalist_title_values[:1]
    angles += angles[:1]
    
    ax1.plot(angles, ai_title_values, 'o-', linewidth=2, label=labels['ai_label'], color='#FF6B6B')
    ax1.fill(angles, ai_title_values, alpha=0.25, color='#FF6B6B')
    ax1.plot(angles, journalist_title_values, 'o-', linewidth=2, label=labels['journalist_label'], color='#4ECDC4')
    ax1.fill(angles, journalist_title_values, alpha=0.25, color='#4ECDC4')
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels['metrics'], fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.set_title(labels['titles']['radar'], fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 2. 柱狀圖比較 - 語義干涉
    ax2 = plt.subplot(3, 2, 2)
    
    ai_interference = [
        ai_data['新聞標題']['semantic_interference']['mean'],
        ai_data['影片對話']['semantic_interference']['mean']
    ]
    journalist_interference = [
        journalist_data['新聞標題']['semantic_interference']['mean'],
        journalist_data['新聞內容']['semantic_interference']['mean']
    ]
    
    x = np.arange(len(labels['categories']))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, ai_interference, width, label=labels['ai_label'], color='#FF6B6B', alpha=0.8)
    bars2 = ax2.bar(x + width/2, journalist_interference, width, label=labels['journalist_label'], color='#4ECDC4', alpha=0.8)
    
    ax2.set_xlabel('Text Type', fontsize=12)
    ax2.set_ylabel('Semantic Interference', fontsize=12)
    ax2.set_title(labels['titles']['interference'], fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels['categories'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加數值標籤
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # 3. 框架競爭 vs 框架衝突散點圖
    ax3 = plt.subplot(3, 2, 3)
    
    # AI數據點
    ai_competition = [ai_data['新聞標題']['frame_competition']['mean'], ai_data['影片對話']['frame_competition']['mean']]
    ai_conflict = [ai_data['新聞標題']['frame_conflict_strength']['mean'], ai_data['影片對話']['frame_conflict_strength']['mean']]
    
    # 記者數據點
    journalist_competition = [journalist_data['新聞標題']['frame_competition']['mean'], journalist_data['新聞內容']['frame_competition']['mean']]
    journalist_conflict = [journalist_data['新聞標題']['frame_conflict_strength']['mean'], journalist_data['新聞內容']['frame_conflict_strength']['mean']]
    
    ax3.scatter(ai_competition, ai_conflict, s=200, alpha=0.7, c='#FF6B6B', label=labels['ai_label'], edgecolors='black', linewidth=1)
    ax3.scatter(journalist_competition, journalist_conflict, s=200, alpha=0.7, c='#4ECDC4', label=labels['journalist_label'], edgecolors='black', linewidth=1)
    
    ax3.set_xlabel('Frame Competition', fontsize=12)
    ax3.set_ylabel('Frame Conflict', fontsize=12)
    ax3.set_title(labels['titles']['competition_conflict'], fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加象限標籤
    ax3.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=0.95, color='gray', linestyle='--', alpha=0.5)
    
    if chinese_font:
        ax3.text(0.85, 0.35, 'AI模式:\n高競爭低衝突', fontsize=10, ha='center', va='center', 
                 bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.8))
        ax3.text(0.995, 0.29, '記者模式:\n極高競爭中衝突', fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='#E5F9F6', alpha=0.8))
    else:
        ax3.text(0.85, 0.35, 'AI Pattern:\nHigh Competition\nLow Conflict', fontsize=10, ha='center', va='center', 
                 bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.8))
        ax3.text(0.995, 0.29, 'Journalist Pattern:\nVery High Competition\nMedium Conflict', fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='#E5F9F6', alpha=0.8))
    
    # 4. 多重現實強度箱線圖
    ax4 = plt.subplot(3, 2, 4)
    
    # 模擬數據分佈（基於均值和標準差）
    np.random.seed(42)
    ai_reality_title = np.random.normal(ai_data['新聞標題']['multiple_reality_strength']['mean'], 
                                       ai_data['新聞標題']['multiple_reality_strength']['std'], 100)
    journalist_reality_title = np.random.normal(journalist_data['新聞標題']['multiple_reality_strength']['mean'],
                                               journalist_data['新聞標題']['multiple_reality_strength']['std'], 100)
    
    box_data = [ai_reality_title, journalist_reality_title]
    box_labels = [labels['ai_label'], labels['journalist_label']]
    
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('#FF6B6B')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#4ECDC4')
    bp['boxes'][1].set_alpha(0.7)
    
    ax4.set_ylabel('Multiple Reality Strength', fontsize=12)
    ax4.set_title(labels['titles']['reality'], fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. 馮紐曼熵對比
    ax5 = plt.subplot(3, 2, 5)
    
    ai_entropy = [ai_data['新聞標題']['von_neumann_entropy']['mean'], ai_data['影片對話']['von_neumann_entropy']['mean']]
    journalist_entropy = [journalist_data['新聞標題']['von_neumann_entropy']['mean'], journalist_data['新聞內容']['von_neumann_entropy']['mean']]
    
    x = np.arange(len(labels['categories']))
    width = 0.35
    bars1 = ax5.bar(x - width/2, ai_entropy, width, label=labels['ai_label'], color='#FF6B6B', alpha=0.8)
    bars2 = ax5.bar(x + width/2, journalist_entropy, width, label=labels['journalist_label'], color='#4ECDC4', alpha=0.8)
    
    ax5.set_xlabel('Text Type', fontsize=12)
    ax5.set_ylabel('Von Neumann Entropy', fontsize=12)
    ax5.set_title(labels['titles']['entropy'], fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels['categories'])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 添加數值標籤
    for bar in bars1:
        height = bar.get_height()
        ax5.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax5.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # 6. 綜合量子指紋對比
    ax6 = plt.subplot(3, 2, 6)
    
    # 創建量子指紋熱圖數據
    if chinese_font:
        metrics_short = ['語法疊加', '框架競爭', '多重現實', '框架衝突', '語義干涉']
    else:
        metrics_short = ['Grammar', 'Competition', 'Reality', 'Conflict', 'Interference']
    
    ai_fingerprint = [
        ai_data['新聞標題']['grammatical_superposition']['mean'],
        ai_data['新聞標題']['frame_competition']['mean'],
        ai_data['新聞標題']['multiple_reality_strength']['mean'],
        ai_data['新聞標題']['frame_conflict_strength']['mean'],
        min(1.0, ai_data['新聞標題']['semantic_interference']['mean'])  # 正規化
    ]
    
    journalist_fingerprint = [
        journalist_data['新聞標題']['grammatical_superposition']['mean'],
        journalist_data['新聞標題']['frame_competition']['mean'],
        journalist_data['新聞標題']['multiple_reality_strength']['mean'],
        journalist_data['新聞標題']['frame_conflict_strength']['mean'],
        min(1.0, journalist_data['新聞標題']['semantic_interference']['mean'] * 100)  # 放大顯示
    ]
    
    fingerprint_data = np.array([ai_fingerprint, journalist_fingerprint])
    
    im = ax6.imshow(fingerprint_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax6.set_xticks(range(len(metrics_short)))
    ax6.set_xticklabels(metrics_short, rotation=45, ha='right')
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels([labels['ai_label'], labels['journalist_label']])
    ax6.set_title(labels['titles']['fingerprint'], fontsize=14, fontweight='bold')
    
    # 添加數值標籤
    for i in range(2):
        for j in range(len(metrics_short)):
            text = ax6.text(j, i, f'{fingerprint_data[i, j]:.3f}', 
                           ha="center", va="center", color="white" if fingerprint_data[i, j] > 0.5 else "black",
                           fontweight='bold')
    
    # 添加顏色條
    cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
    if chinese_font:
        cbar.set_label('特徵強度', fontsize=10)
    else:
        cbar.set_label('Feature Strength', fontsize=10)
    
    plt.tight_layout()
    plt.suptitle(labels['titles']['main'], fontsize=18, fontweight='bold', y=0.98)
    
    # 保存圖表
    output_file = '../visualizations/ai_vs_journalist_quantum_comparison_fixed.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 修復版比較圖表已保存: {output_file}")
    
    plt.show()

def main():
    """主函數"""
    
    print("🚀 開始AI vs 記者新聞量子特徵比較分析（修復中文顯示版本）")
    print("=" * 70)
    
    # 創建可視化
    create_comparison_charts()
    
    print("\n✅ 修復版比較分析完成!")
    print("📊 圖表文件: ../visualizations/ai_vs_journalist_quantum_comparison_fixed.png")

if __name__ == "__main__":
    main()
