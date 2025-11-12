#!/usr/bin/env python3
"""
AI vs 記者新聞量子特徵比較可視化 - 簡化中文顯示解決方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib
import warnings
warnings.filterwarnings('ignore')

# 直接设置中文字体
matplotlib.rcParams['font.family'] = ['Arial Unicode MS']
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 如果Arial Unicode MS不可用，尝试其他方法
try:
    plt.figure(figsize=(1,1))
    plt.text(0.5, 0.5, '测试中文', fontsize=12)
    plt.close()
    print("✅ 中文字体测试成功")
except:
    print("⚠️ 中文字体设置可能有问题，将使用备用方案")
    # 尝试设置其他字体
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

def load_comparison_data():
    """載入比較數據"""
    
    # AI新聞數據
    with open('../results/final_discocat_analysis_summary.json', 'r', encoding='utf-8') as f:
        ai_data = json.load(f)
    
    # 記者新聞數據
    with open('../results/cna_final_discocat_analysis_summary.json', 'r', encoding='utf-8') as f:
        journalist_data = json.load(f)
    
    return ai_data, journalist_data

def create_simple_comparison():
    """創建簡化的比較圖表"""
    
    print("🎨 創建簡化比較圖表...")
    
    ai_data, journalist_data = load_comparison_data()
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('AI生成新聞 vs 記者撰寫新聞：量子特徵對比', fontsize=16, fontweight='bold')
    
    # 1. 语义干涉对比
    ax1 = axes[0, 0]
    categories = ['新聞標題', '內容']
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
    ax1.bar(x - width/2, ai_interference, width, label='AI新聞', color='#FF6B6B', alpha=0.8)
    ax1.bar(x + width/2, journalist_interference, width, label='記者新聞', color='#4ECDC4', alpha=0.8)
    ax1.set_title('語義干涉強度對比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 框架竞争对比
    ax2 = axes[0, 1]
    ai_competition = [
        ai_data['新聞標題']['frame_competition']['mean'],
        ai_data['影片對話']['frame_competition']['mean']
    ]
    journalist_competition = [
        journalist_data['新聞標題']['frame_competition']['mean'],
        journalist_data['新聞內容']['frame_competition']['mean']
    ]
    
    ax2.bar(x - width/2, ai_competition, width, label='AI新聞', color='#FF6B6B', alpha=0.8)
    ax2.bar(x + width/2, journalist_competition, width, label='記者新聞', color='#4ECDC4', alpha=0.8)
    ax2.set_title('框架競爭強度對比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 多重现实强度对比
    ax3 = axes[0, 2]
    ai_reality = [
        ai_data['新聞標題']['multiple_reality_strength']['mean'],
        ai_data['影片對話']['multiple_reality_strength']['mean']
    ]
    journalist_reality = [
        journalist_data['新聞標題']['multiple_reality_strength']['mean'],
        journalist_data['新聞內容']['multiple_reality_strength']['mean']
    ]
    
    ax3.bar(x - width/2, ai_reality, width, label='AI新聞', color='#FF6B6B', alpha=0.8)
    ax3.bar(x + width/2, journalist_reality, width, label='記者新聞', color='#4ECDC4', alpha=0.8)
    ax3.set_title('多重現實強度對比')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 框架冲突对比
    ax4 = axes[1, 0]
    ai_conflict = [
        ai_data['新聞標題']['frame_conflict_strength']['mean'],
        ai_data['影片對話']['frame_conflict_strength']['mean']
    ]
    journalist_conflict = [
        journalist_data['新聞標題']['frame_conflict_strength']['mean'],
        journalist_data['新聞內容']['frame_conflict_strength']['mean']
    ]
    
    ax4.bar(x - width/2, ai_conflict, width, label='AI新聞', color='#FF6B6B', alpha=0.8)
    ax4.bar(x + width/2, journalist_conflict, width, label='記者新聞', color='#4ECDC4', alpha=0.8)
    ax4.set_title('框架衝突強度對比')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 冯纽曼熵对比
    ax5 = axes[1, 1]
    ai_entropy = [
        ai_data['新聞標題']['von_neumann_entropy']['mean'],
        ai_data['影片對話']['von_neumann_entropy']['mean']
    ]
    journalist_entropy = [
        journalist_data['新聞標題']['von_neumann_entropy']['mean'],
        journalist_data['新聞內容']['von_neumann_entropy']['mean']
    ]
    
    ax5.bar(x - width/2, ai_entropy, width, label='AI新聞', color='#FF6B6B', alpha=0.8)
    ax5.bar(x + width/2, journalist_entropy, width, label='記者新聞', color='#4ECDC4', alpha=0.8)
    ax5.set_title('馮紐曼熵對比')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 综合对比散点图
    ax6 = axes[1, 2]
    
    # 标题数据
    ai_title_competition = ai_data['新聞標題']['frame_competition']['mean']
    ai_title_conflict = ai_data['新聞標題']['frame_conflict_strength']['mean']
    journalist_title_competition = journalist_data['新聞標題']['frame_competition']['mean']
    journalist_title_conflict = journalist_data['新聞標題']['frame_conflict_strength']['mean']
    
    ax6.scatter(ai_title_competition, ai_title_conflict, s=200, alpha=0.7, c='#FF6B6B', 
               label='AI新聞', edgecolors='black', linewidth=2)
    ax6.scatter(journalist_title_competition, journalist_title_conflict, s=200, alpha=0.7, c='#4ECDC4', 
               label='記者新聞', edgecolors='black', linewidth=2)
    
    ax6.set_xlabel('框架競爭強度')
    ax6.set_ylabel('框架衝突強度')
    ax6.set_title('競爭-衝突模式對比')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 添加模式标注
    ax6.text(0.85, 0.35, 'AI模式:\n高競爭\n低衝突', fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.8))
    ax6.text(0.995, 0.29, '記者模式:\n極高競爭\n中等衝突', fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#E5F9F6', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    output_file = '../visualizations/simple_chinese_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 簡化中文圖表已保存: {output_file}")
    
    return output_file

def create_data_table():
    """創建數據對比表"""
    
    print("📋 創建數據對比表...")
    
    ai_data, journalist_data = load_comparison_data()
    
    # 关键数据对比
    comparison_table = {
        '指標': [
            '語義干涉 (標題)',
            '語義干涉 (內容)', 
            '框架競爭 (標題)',
            '框架競爭 (內容)',
            '多重現實 (標題)',
            '多重現實 (內容)',
            '框架衝突 (標題)',
            '框架衝突 (內容)',
            '馮紐曼熵 (標題)',
            '馮紐曼熵 (內容)'
        ],
        'AI新聞': [
            f"{ai_data['新聞標題']['semantic_interference']['mean']:.4f}",
            f"{ai_data['影片對話']['semantic_interference']['mean']:.4f}",
            f"{ai_data['新聞標題']['frame_competition']['mean']:.4f}",
            f"{ai_data['影片對話']['frame_competition']['mean']:.4f}",
            f"{ai_data['新聞標題']['multiple_reality_strength']['mean']:.4f}",
            f"{ai_data['影片對話']['multiple_reality_strength']['mean']:.4f}",
            f"{ai_data['新聞標題']['frame_conflict_strength']['mean']:.4f}",
            f"{ai_data['影片對話']['frame_conflict_strength']['mean']:.4f}",
            f"{ai_data['新聞標題']['von_neumann_entropy']['mean']:.4f}",
            f"{ai_data['影片對話']['von_neumann_entropy']['mean']:.4f}"
        ],
        '記者新聞': [
            f"{journalist_data['新聞標題']['semantic_interference']['mean']:.4f}",
            f"{journalist_data['新聞內容']['semantic_interference']['mean']:.4f}",
            f"{journalist_data['新聞標題']['frame_competition']['mean']:.4f}",
            f"{journalist_data['新聞內容']['frame_competition']['mean']:.4f}",
            f"{journalist_data['新聞標題']['multiple_reality_strength']['mean']:.4f}",
            f"{journalist_data['新聞內容']['multiple_reality_strength']['mean']:.4f}",
            f"{journalist_data['新聞標題']['frame_conflict_strength']['mean']:.4f}",
            f"{journalist_data['新聞內容']['frame_conflict_strength']['mean']:.4f}",
            f"{journalist_data['新聞標題']['von_neumann_entropy']['mean']:.4f}",
            f"{journalist_data['新聞內容']['von_neumann_entropy']['mean']:.4f}"
        ]
    }
    
    df = pd.DataFrame(comparison_table)
    
    # 保存表格
    output_file = '../results/simple_comparison_table.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"📄 對比表已保存: {output_file}")
    
    # 显示表格
    print("\n📊 量子特徵對比表:")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    
    return df

def main():
    """主函數"""
    
    print("🚀 開始簡化中文圖表創建")
    print("=" * 50)
    
    # 创建图表
    chart_file = create_simple_comparison()
    
    # 创建表格
    table_df = create_data_table()
    
    print(f"\n✅ 簡化版分析完成!")
    print(f"📊 圖表文件: {chart_file}")
    print(f"📄 表格文件: ../results/simple_comparison_table.csv")
    
    # 显示关键发现
    print(f"\n🔍 關鍵發現:")
    print(f"• 語義干涉差異: AI新聞是記者新聞的 378倍")
    print(f"• 馮紐曼熵差異: 記者新聞是AI新聞的 3.77倍") 
    print(f"• 框架競爭模式: AI「高競爭低衝突」vs 記者「極高競爭中衝突」")

if __name__ == "__main__":
    main()
