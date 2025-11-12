#!/usr/bin/env python3
"""
最终无限制版本可视化器 - 只展示无限制量子分析结果
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_font():
    """设置中文字体支持"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.weight'] = 'normal'
    print("✅ 配置中文字体支持")

def load_unrestricted_data():
    """加载无限制版本数据"""
    files = {
        'ai': '../results/fair_comparison_ai_unrestricted_summary.json',
        'journalist': '../results/fair_comparison_journalist_unrestricted_summary.json'
    }
    
    data = {}
    for key, file_path in files.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data[key] = json.load(f)
            print(f"✅ 加载数据: {key}")
        except Exception as e:
            print(f"❌ 加载失败 {key}: {e}")
    
    return data

def create_main_comparison_chart(data):
    """创建主要对比图表"""
    
    # 语法叠加强度对比
    ai_superposition = data['ai']['新聞標題']['grammatical_superposition']['mean']
    journalist_superposition = data['journalist']['新聞標題']['grammatical_superposition']['mean']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 语法叠加强度对比
    categories = ['AI新闻', '记者新闻']
    values = [ai_superposition, journalist_superposition]
    colors = ['#ff6b6b', '#4ecdc4']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.8, width=0.6)
    ax1.set_title('语法叠加强度对比 - 新聞標題字段', fontsize=14, pad=20)
    ax1.set_ylabel('语法叠加强度', fontsize=12)
    ax1.set_ylim(0, 4.0)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签和理论最大值线
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.6f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加理论最大值线
    ax1.axhline(y=4.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(0.5, 4.1, '理论最大值 (4.0)', ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 添加差异说明
    diff_percent = ((ai_superposition - journalist_superposition) / journalist_superposition) * 100
    ax1.text(0.5, 3.5, f'AI比记者高 {diff_percent:.2f}%', 
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 2. 各字段语法叠加强度对比
    ai_fields = ['新聞標題', '影片對話', '影片描述']
    ai_values = [data['ai'][field]['grammatical_superposition']['mean'] for field in ai_fields]
    
    journalist_fields = ['新聞標題', '新聞內容']
    journalist_values = [data['journalist'][field]['grammatical_superposition']['mean'] for field in journalist_fields]
    
    # 绘制AI新闻
    x_ai = np.arange(len(ai_fields))
    bars_ai = ax2.bar(x_ai - 0.2, ai_values, width=0.35, label='AI新闻', 
                     color='#ff6b6b', alpha=0.8)
    
    # 绘制记者新闻
    x_journalist = np.arange(len(journalist_fields))
    bars_journalist = ax2.bar(x_journalist + 0.2, journalist_values, width=0.35, 
                             label='记者新闻', color='#4ecdc4', alpha=0.8)
    
    ax2.set_title('各字段语法叠加强度对比', fontsize=14, pad=20)
    ax2.set_ylabel('语法叠加强度', fontsize=12)
    ax2.set_ylim(0, 4.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 设置x轴标签
    all_fields = ['新聞標題', '影片對話/新聞內容', '影片描述']
    ax2.set_xticks(range(len(all_fields)))
    ax2.set_xticklabels(all_fields, rotation=0)
    
    # 添加数值标签
    for bar, val in zip(bars_ai, ai_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, val in zip(bars_journalist, journalist_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('AI生成新闻 vs 记者撰写新闻：量子自然语言处理分析', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('../visualizations/final_unrestricted_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ 生成主要对比图")

def create_comprehensive_metrics_table(data):
    """创建综合指标对比表"""
    
    # 准备数据
    metrics_data = []
    metrics_labels = [
        ('语法叠加强度', 'grammatical_superposition'),
        ('多重现实强度', 'multiple_reality_strength'), 
        ('语义模糊度', 'semantic_ambiguity'),
        ('框架竞争强度', 'frame_competition'),
        ('语义干涉', 'semantic_interference'),
        ('冯纽曼熵', 'von_neumann_entropy'),
        ('类别一致性', 'category_coherence'),
        ('组合纠缠强度', 'compositional_entanglement'),
        ('框架冲突强度', 'frame_conflict_strength')
    ]
    
    for label, metric in metrics_labels:
        ai_val = data['ai']['新聞標題'][metric]['mean']
        journalist_val = data['journalist']['新聞標題'][metric]['mean']
        diff = ai_val / journalist_val if journalist_val > 0 else 0
        metrics_data.append([label, ai_val, journalist_val, diff])
    
    # 创建表格样式的图
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格数据
    table_data = []
    table_data.append(['量子指标', 'AI新闻', '记者新闻', '差异倍数', '优势方'])
    for row in metrics_data:
        advantage = 'AI' if row[3] > 1 else '记者' if row[3] < 1 else '相等'
        table_data.append([row[0], f'{row[1]:.6f}', f'{row[2]:.6f}', f'{row[3]:.3f}×', advantage])
    
    # 创建表格
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                    cellLoc='center', loc='center', 
                    colWidths=[0.25, 0.2, 0.2, 0.15, 0.2])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行样式
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
            
            # 突出显示差异倍数列
            if j == 3:  # 差异倍数列
                diff_val = float(table_data[i][j].replace('×', ''))
                if diff_val > 1.1:
                    table[(i, j)].set_facecolor('#ffeb3b')  # 黄色突出显示
                elif diff_val < 0.9:
                    table[(i, j)].set_facecolor('#e3f2fd')  # 浅蓝色
    
    plt.title('AI新闻 vs 记者新闻：量子指标全面对比 (新聞標題字段)', 
             fontsize=16, pad=30, fontweight='bold')
    plt.savefig('../visualizations/final_unrestricted_metrics_table.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ 生成综合指标对比表")

def create_radar_chart(data):
    """创建雷达图对比"""
    
    # 选择关键指标
    key_metrics = [
        'grammatical_superposition', 'multiple_reality_strength', 'semantic_ambiguity',
        'frame_competition', 'semantic_interference', 'von_neumann_entropy'
    ]
    
    key_metrics_cn = [
        '语法叠加强度', '多重现实强度', '语义模糊度',
        '框架竞争强度', '语义干涉', '冯纽曼熵'
    ]
    
    # 获取数据
    ai_values = [data['ai']['新聞標題'][metric]['mean'] for metric in key_metrics]
    journalist_values = [data['journalist']['新聞標題'][metric]['mean'] for metric in key_metrics]
    
    # 为了显示效果，对数据进行归一化（相对于各自的最大值）
    max_values = [max(ai_val, j_val) for ai_val, j_val in zip(ai_values, journalist_values)]
    ai_normalized = [ai_val/max_val for ai_val, max_val in zip(ai_values, max_values)]
    journalist_normalized = [j_val/max_val for j_val, max_val in zip(journalist_values, max_values)]
    
    # 设置角度
    angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
    ai_normalized += ai_normalized[:1]  # 闭合
    journalist_normalized += journalist_normalized[:1]
    angles += angles[:1]
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 绘制AI新闻
    ax.plot(angles, ai_normalized, 'o-', linewidth=3, label='AI新闻', color='#ff6b6b', alpha=0.8)
    ax.fill(angles, ai_normalized, alpha=0.25, color='#ff6b6b')
    
    # 绘制记者新闻
    ax.plot(angles, journalist_normalized, 'o-', linewidth=3, label='记者新闻', color='#4ecdc4', alpha=0.8)
    ax.fill(angles, journalist_normalized, alpha=0.25, color='#4ecdc4')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(key_metrics_cn, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
    ax.grid(True)
    
    # 添加图例和标题
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
    plt.title('量子特征雷达图对比 (新聞標題字段)\n相对于各指标最大值归一化', 
             fontsize=14, pad=30, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../visualizations/final_unrestricted_radar.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ 生成雷达图对比")

def create_summary_statistics(data):
    """创建统计摘要"""
    
    print("\n" + "="*80)
    print("📊 AI新闻 vs 记者新闻：量子特征统计摘要")
    print("="*80)
    
    # 新聞標題字段对比
    print("\n📈 新聞標題字段核心指标对比:")
    print("-"*60)
    print(f"{'指标':<20} {'AI新闻':<15} {'记者新闻':<15} {'差异':<10}")
    print("-"*60)
    
    key_metrics = [
        ('语法叠加强度', 'grammatical_superposition'),
        ('多重现实强度', 'multiple_reality_strength'),
        ('语义模糊度', 'semantic_ambiguity'),
        ('框架竞争强度', 'frame_competition'),
        ('冯纽曼熵', 'von_neumann_entropy')
    ]
    
    for cn_name, metric in key_metrics:
        ai_val = data['ai']['新聞標題'][metric]['mean']
        j_val = data['journalist']['新聞標題'][metric]['mean']
        diff = ((ai_val - j_val) / j_val) * 100 if j_val != 0 else 0
        
        print(f"{cn_name:<20} {ai_val:<15.6f} {j_val:<15.6f} {diff:>+7.2f}%")
    
    # 关键发现
    ai_sup = data['ai']['新聞標題']['grammatical_superposition']['mean']
    j_sup = data['journalist']['新聞標題']['grammatical_superposition']['mean']
    
    print(f"\n🔍 关键发现:")
    print(f"• 语法叠加强度: AI新闻达到理论最大值的 {(ai_sup/4.0)*100:.1f}%")
    print(f"• 语法叠加强度: 记者新闻达到理论最大值的 {(j_sup/4.0)*100:.1f}%")
    print(f"• AI新闻在语法叠加强度上比记者新闻高 {((ai_sup-j_sup)/j_sup)*100:.2f}%")
    
    # 各字段对比
    print(f"\n📋 各字段语法叠加强度:")
    print(f"AI新闻:")
    for field in ['新聞標題', '影片對話', '影片描述']:
        val = data['ai'][field]['grammatical_superposition']['mean']
        print(f"  • {field}: {val:.6f} ({(val/4.0)*100:.1f}%)")
    
    print(f"记者新闻:")
    for field in ['新聞標題', '新聞內容']:
        val = data['journalist'][field]['grammatical_superposition']['mean']
        print(f"  • {field}: {val:.6f} ({(val/4.0)*100:.1f}%)")

def main():
    """主函数"""
    
    print("🎨 开始生成最终无限制版本可视化...")
    
    # 设置中文字体
    setup_chinese_font()
    
    # 创建可视化目录
    Path('../visualizations').mkdir(exist_ok=True)
    
    # 加载数据
    data = load_unrestricted_data()
    
    if not data:
        print("❌ 无法加载数据，退出程序")
        return
    
    print("\n开始生成各种图表...")
    
    # 生成各种可视化
    create_main_comparison_chart(data)
    create_comprehensive_metrics_table(data)
    create_radar_chart(data)
    create_summary_statistics(data)
    
    print(f"\n✅ 最终无限制版本可视化已生成完成!")
    print(f"📁 保存位置: ../visualizations/")
    print(f"   • final_unrestricted_comparison.png - 主要对比图")
    print(f"   • final_unrestricted_metrics_table.png - 综合指标对比表")
    print(f"   • final_unrestricted_radar.png - 雷达图对比")

if __name__ == "__main__":
    main()
