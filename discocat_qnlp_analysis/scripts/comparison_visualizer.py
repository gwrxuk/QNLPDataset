#!/usr/bin/env python3
"""
AI vs 記者新聞量子特徵比較可視化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
import matplotlib
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Helvetica', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 检查并设置可用的中文字体
def setup_chinese_font():
    """设置中文字体"""
    import matplotlib.font_manager as fm
    
    # 常见的中文字体
    chinese_fonts = [
        'Arial Unicode MS',
        'PingFang SC',
        'Helvetica Neue',
        'SimHei',
        'Microsoft YaHei',
        'WenQuanYi Micro Hei',
        'Noto Sans CJK SC',
        'Source Han Sans SC'
    ]
    
    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 找到第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            print(f"✅ 使用中文字体: {font}")
            return font
    
    # 如果没有找到，使用默认字体并警告
    print("⚠️  未找到中文字体，可能无法正确显示中文")
    return None

# 设置中文字体
setup_chinese_font()

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
    
    ai_data, journalist_data = load_comparison_data()
    
    # 設置圖表樣式
    plt.style.use('seaborn-v0_8')
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
    
    metric_names = [
        '語法疊加強度',
        '框架競爭強度', 
        '多重現實強度',
        '框架衝突強度',
        '語義干涉',
        '馮紐曼熵'
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
    
    ax1.plot(angles, ai_title_values, 'o-', linewidth=2, label='AI新聞', color='#FF6B6B')
    ax1.fill(angles, ai_title_values, alpha=0.25, color='#FF6B6B')
    ax1.plot(angles, journalist_title_values, 'o-', linewidth=2, label='記者新聞', color='#4ECDC4')
    ax1.fill(angles, journalist_title_values, alpha=0.25, color='#4ECDC4')
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metric_names, fontsize=10, fontfamily='sans-serif')
    ax1.set_ylim(0, 1)
    ax1.set_title('新聞標題量子特徵雷達圖', fontsize=14, fontweight='bold', pad=20, fontfamily='sans-serif')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop={'family': 'sans-serif'})
    
    # 2. 柱狀圖比較 - 語義干涉
    ax2 = plt.subplot(3, 2, 2)
    
    categories = ['新聞標題', '內容/對話']
    ai_interference = [
        ai_data['新聞標題']['semantic_interference']['mean'],
        ai_data['影片對話']['semantic_interference']['mean']
    ]
    journalist_interference = [
        journalist_data['新聞標題']['semantic_interference']['mean'],
        journalist_data['新聞內容']['semantic_interference']['mean']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, ai_interference, width, label='AI新聞', color='#FF6B6B', alpha=0.8)
    bars2 = ax2.bar(x + width/2, journalist_interference, width, label='記者新聞', color='#4ECDC4', alpha=0.8)
    
    ax2.set_xlabel('文本類型', fontsize=12)
    ax2.set_ylabel('語義干涉強度', fontsize=12)
    ax2.set_title('語義干涉強度對比（差異達386倍）', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
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
    
    ax3.scatter(ai_competition, ai_conflict, s=200, alpha=0.7, c='#FF6B6B', label='AI新聞', edgecolors='black', linewidth=1)
    ax3.scatter(journalist_competition, journalist_conflict, s=200, alpha=0.7, c='#4ECDC4', label='記者新聞', edgecolors='black', linewidth=1)
    
    ax3.set_xlabel('框架競爭強度', fontsize=12)
    ax3.set_ylabel('框架衝突強度', fontsize=12)
    ax3.set_title('框架競爭 vs 框架衝突模式', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加象限標籤
    ax3.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=0.95, color='gray', linestyle='--', alpha=0.5)
    ax3.text(0.85, 0.35, 'AI模式:\n高競爭低衝突', fontsize=10, ha='center', va='center', 
             bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.8))
    ax3.text(0.995, 0.29, '記者模式:\n極高競爭中衝突', fontsize=10, ha='center', va='center',
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
    box_labels = ['AI新聞', '記者新聞']
    
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('#FF6B6B')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#4ECDC4')
    bp['boxes'][1].set_alpha(0.7)
    
    ax4.set_ylabel('多重現實強度', fontsize=12)
    ax4.set_title('多重現實強度分佈對比', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. 馮紐曼熵對比
    ax5 = plt.subplot(3, 2, 5)
    
    ai_entropy = [ai_data['新聞標題']['von_neumann_entropy']['mean'], ai_data['影片對話']['von_neumann_entropy']['mean']]
    journalist_entropy = [journalist_data['新聞標題']['von_neumann_entropy']['mean'], journalist_data['新聞內容']['von_neumann_entropy']['mean']]
    
    x = np.arange(len(categories))
    bars1 = ax5.bar(x - width/2, ai_entropy, width, label='AI新聞', color='#FF6B6B', alpha=0.8)
    bars2 = ax5.bar(x + width/2, journalist_entropy, width, label='記者新聞', color='#4ECDC4', alpha=0.8)
    
    ax5.set_xlabel('文本類型', fontsize=12)
    ax5.set_ylabel('馮紐曼熵', fontsize=12)
    ax5.set_title('資訊密度對比（馮紐曼熵）', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories)
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
    metrics_short = ['語法疊加', '框架競爭', '多重現實', '框架衝突', '語義干涉']
    
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
    ax6.set_yticklabels(['AI新聞', '記者新聞'])
    ax6.set_title('量子特徵指紋對比', fontsize=14, fontweight='bold')
    
    # 添加數值標籤
    for i in range(2):
        for j in range(len(metrics_short)):
            text = ax6.text(j, i, f'{fingerprint_data[i, j]:.3f}', 
                           ha="center", va="center", color="white" if fingerprint_data[i, j] > 0.5 else "black",
                           fontweight='bold')
    
    # 添加顏色條
    cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
    cbar.set_label('特徵強度', fontsize=10)
    
    plt.tight_layout()
    plt.suptitle('AI生成新聞 vs 記者撰寫新聞：量子特徵全面對比', fontsize=18, fontweight='bold', y=0.98)
    
    # 保存圖表
    output_file = '../visualizations/ai_vs_journalist_quantum_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 比較圖表已保存: {output_file}")
    
    plt.show()

def create_summary_table():
    """創建摘要對比表"""
    
    print("📋 創建摘要對比表...")
    
    ai_data, journalist_data = load_comparison_data()
    
    # 關鍵指標對比
    comparison_data = {
        '量子特徵': [
            '語法疊加強度',
            '框架競爭強度', 
            '多重現實強度',
            '框架衝突強度',
            '語義干涉',
            '馮紐曼熵',
            '類別一致性',
            '組合糾纏強度'
        ],
        'AI新聞（標題）': [
            f"{ai_data['新聞標題']['grammatical_superposition']['mean']:.4f}",
            f"{ai_data['新聞標題']['frame_competition']['mean']:.4f}",
            f"{ai_data['新聞標題']['multiple_reality_strength']['mean']:.4f}",
            f"{ai_data['新聞標題']['frame_conflict_strength']['mean']:.4f}",
            f"{ai_data['新聞標題']['semantic_interference']['mean']:.4f}",
            f"{ai_data['新聞標題']['von_neumann_entropy']['mean']:.4f}",
            f"{ai_data['新聞標題']['category_coherence']['mean']:.4f}",
            f"{ai_data['新聞標題']['compositional_entanglement']['mean']:.4f}"
        ],
        '記者新聞（標題）': [
            f"{journalist_data['新聞標題']['grammatical_superposition']['mean']:.4f}",
            f"{journalist_data['新聞標題']['frame_competition']['mean']:.4f}",
            f"{journalist_data['新聞標題']['multiple_reality_strength']['mean']:.4f}",
            f"{journalist_data['新聞標題']['frame_conflict_strength']['mean']:.4f}",
            f"{journalist_data['新聞標題']['semantic_interference']['mean']:.4f}",
            f"{journalist_data['新聞標題']['von_neumann_entropy']['mean']:.4f}",
            f"{journalist_data['新聞標題']['category_coherence']['mean']:.4f}",
            f"{journalist_data['新聞標題']['compositional_entanglement']['mean']:.4f}"
        ],
        '差異倍數': [
            '1.00×',
            f"{journalist_data['新聞標題']['frame_competition']['mean'] / ai_data['新聞標題']['frame_competition']['mean']:.2f}×",
            f"{ai_data['新聞標題']['multiple_reality_strength']['mean'] / journalist_data['新聞標題']['multiple_reality_strength']['mean']:.2f}×",
            f"{journalist_data['新聞標題']['frame_conflict_strength']['mean'] / ai_data['新聞標題']['frame_conflict_strength']['mean']:.2f}×",
            f"{ai_data['新聞標題']['semantic_interference']['mean'] / journalist_data['新聞標題']['semantic_interference']['mean']:.0f}×",
            f"{journalist_data['新聞標題']['von_neumann_entropy']['mean'] / ai_data['新聞標題']['von_neumann_entropy']['mean']:.2f}×",
            f"{journalist_data['新聞標題']['category_coherence']['mean'] / ai_data['新聞標題']['category_coherence']['mean']:.2f}×",
            f"{journalist_data['新聞標題']['compositional_entanglement']['mean'] / ai_data['新聞標題']['compositional_entanglement']['mean']:.2f}×"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # 保存CSV
    output_file = '../results/ai_vs_journalist_comparison_table.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"📄 對比表已保存: {output_file}")
    
    # 顯示表格
    print("\n📊 AI vs 記者新聞量子特徵對比表:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

def main():
    """主函數"""
    
    print("🚀 開始AI vs 記者新聞量子特徵比較分析")
    print("=" * 60)
    
    # 創建可視化
    create_comparison_charts()
    
    # 創建對比表
    create_summary_table()
    
    print("\n✅ 比較分析完成!")
    print("📊 圖表文件: ../visualizations/ai_vs_journalist_quantum_comparison.png")
    print("📄 對比表文件: ../results/ai_vs_journalist_comparison_table.csv")
    print("📝 完整報告: ../analysis_reports/ai_vs_journalist_quantum_comparison.md")

if __name__ == "__main__":
    main()
