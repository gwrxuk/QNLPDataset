#!/usr/bin/env python3
"""
修复版中文可视化器 - 专门解决中文字符显示问题
"""

import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_font():
    """强制设置中文字体支持"""
    
    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    
    # 对于macOS系统，尝试使用系统字体
    try:
        import platform
        if platform.system() == 'Darwin':  # macOS
            # 使用macOS系统中文字体
            plt.rcParams['font.family'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
            print("✅ 配置macOS中文字体")
        else:
            # 其他系统
            plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            print("✅ 配置通用中文字体")
    except:
        plt.rcParams['font.family'] = ['DejaVu Sans']
        print("⚠️ 使用默认字体")
    
    # 额外的字体设置
    plt.rcParams['font.weight'] = 'normal'

def load_summary_data():
    """加载汇总数据"""
    files = {
        'ai_restricted': '../results/fair_comparison_ai_restricted_summary.json',
        'ai_unrestricted': '../results/fair_comparison_ai_unrestricted_summary.json',
        'journalist_restricted': '../results/fair_comparison_journalist_restricted_summary.json',
        'journalist_unrestricted': '../results/fair_comparison_journalist_unrestricted_summary.json'
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

def create_simple_comparison_chart(data):
    """创建简单的对比图表"""
    
    # 准备数据
    ai_unrestricted_title = data['ai_unrestricted']['新聞標題']['grammatical_superposition']['mean']
    journalist_unrestricted_title = data['journalist_unrestricted']['新聞標題']['grammatical_superposition']['mean']
    
    ai_restricted_title = data['ai_restricted']['新聞標題']['grammatical_superposition']['mean']
    journalist_restricted_title = data['journalist_restricted']['新聞標題']['grammatical_superposition']['mean']
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 受限制版本对比
    categories1 = ['AI新闻', '记者新闻']
    values1 = [ai_restricted_title, journalist_restricted_title]
    colors1 = ['#ff6b6b', '#4ecdc4']
    
    bars1 = ax1.bar(categories1, values1, color=colors1, alpha=0.8, width=0.6)
    ax1.set_title('受限制版本 - 语法叠加强度对比', fontsize=14, pad=20)
    ax1.set_ylabel('语法叠加强度', fontsize=12)
    ax1.set_ylim(0, 1.2)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars1, values1):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.6f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 无限制版本对比
    categories2 = ['AI新闻', '记者新闻']
    values2 = [ai_unrestricted_title, journalist_unrestricted_title]
    colors2 = ['#ff6b6b', '#4ecdc4']
    
    bars2 = ax2.bar(categories2, values2, color=colors2, alpha=0.8, width=0.6)
    ax2.set_title('无限制版本 - 语法叠加强度对比', fontsize=14, pad=20)
    ax2.set_ylabel('语法叠加强度', fontsize=12)
    ax2.set_ylim(0, 4.0)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars2, values2):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.6f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加差异说明
    diff_percent = ((ai_unrestricted_title - journalist_unrestricted_title) / journalist_unrestricted_title) * 100
    ax2.text(0.5, 3.5, f'AI比记者高 {diff_percent:.2f}%', 
            ha='center', va='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.suptitle('AI生成新闻 vs 记者撰写新闻：语法叠加强度对比', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('../visualizations/fixed_chinese_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ 生成语法叠加强度对比图")

def create_metrics_table_chart(data):
    """创建指标对比表格图"""
    
    # 准备数据
    metrics_data = []
    metrics_labels = [
        ('语法叠加强度', 'grammatical_superposition'),
        ('多重现实强度', 'multiple_reality_strength'), 
        ('语义模糊度', 'semantic_ambiguity'),
        ('框架竞争强度', 'frame_competition'),
        ('语义干涉', 'semantic_interference'),
        ('冯纽曼熵', 'von_neumann_entropy')
    ]
    
    for label, metric in metrics_labels:
        ai_u = data['ai_unrestricted']['新聞標題'][metric]['mean']
        j_u = data['journalist_unrestricted']['新聞標題'][metric]['mean']
        diff = ai_u / j_u if j_u > 0 else 0
        metrics_data.append([label, ai_u, j_u, diff])
    
    # 创建表格样式的图
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格数据
    table_data = []
    table_data.append(['量子指标', 'AI新闻', '记者新闻', '差异倍数'])
    for row in metrics_data:
        table_data.append([row[0], f'{row[1]:.6f}', f'{row[2]:.6f}', f'{row[3]:.3f}×'])
    
    # 创建表格
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                    cellLoc='center', loc='center', 
                    colWidths=[0.3, 0.25, 0.25, 0.2])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
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
    
    plt.title('AI新闻 vs 记者新闻：量子指标详细对比 (无限制版本)', 
             fontsize=16, pad=20, fontweight='bold')
    plt.savefig('../visualizations/fixed_chinese_metrics_table.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ 生成量子指标对比表")

def create_field_comparison_chart(data):
    """创建字段对比图"""
    
    # AI新闻各字段数据
    ai_fields = ['新聞標題', '影片對話', '影片描述']
    ai_values = [data['ai_unrestricted'][field]['grammatical_superposition']['mean'] for field in ai_fields]
    
    # 记者新闻各字段数据  
    journalist_fields = ['新聞標題', '新聞內容']
    journalist_values = [data['journalist_unrestricted'][field]['grammatical_superposition']['mean'] for field in journalist_fields]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # AI新闻字段对比
    bars1 = ax1.bar(range(len(ai_fields)), ai_values, color=['#ff6b6b', '#ff9f43', '#feca57'], alpha=0.8)
    ax1.set_title('AI新闻各字段 - 语法叠加强度', fontsize=14, pad=20)
    ax1.set_ylabel('语法叠加强度', fontsize=12)
    ax1.set_xticks(range(len(ai_fields)))
    ax1.set_xticklabels(ai_fields, fontsize=11)
    ax1.set_ylim(0, 4.0)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars1, ai_values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 记者新闻字段对比
    bars2 = ax2.bar(range(len(journalist_fields)), journalist_values, color=['#4ecdc4', '#10ac84'], alpha=0.8)
    ax2.set_title('记者新闻各字段 - 语法叠加强度', fontsize=14, pad=20)
    ax2.set_ylabel('语法叠加强度', fontsize=12)
    ax2.set_xticks(range(len(journalist_fields)))
    ax2.set_xticklabels(journalist_fields, fontsize=11)
    ax2.set_ylim(0, 4.0)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars2, journalist_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('各字段语法叠加强度对比 (无限制版本)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('../visualizations/fixed_chinese_field_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ 生成字段对比图")

def create_difference_analysis_chart(data):
    """创建差异分析图"""
    
    # 准备指标数据
    metrics = [
        ('语法叠加强度', 'grammatical_superposition'),
        ('多重现实强度', 'multiple_reality_strength'),
        ('语义模糊度', 'semantic_ambiguity'),
        ('框架竞争强度', 'frame_competition'),
        ('语义干涉', 'semantic_interference')
    ]
    
    # 计算差异倍数
    metric_names = []
    differences = []
    
    for name, metric in metrics:
        ai_val = data['ai_unrestricted']['新聞標題'][metric]['mean']
        journalist_val = data['journalist_unrestricted']['新聞標題'][metric]['mean']
        diff = ai_val / journalist_val if journalist_val > 0 else 0
        metric_names.append(name)
        differences.append(diff)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建水平条形图
    colors = ['#ff6b6b' if d > 1 else '#4ecdc4' for d in differences]
    bars = ax.barh(range(len(metric_names)), differences, color=colors, alpha=0.8)
    
    # 设置标签和标题
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names, fontsize=12)
    ax.set_xlabel('差异倍数 (AI/记者)', fontsize=12)
    ax.set_title('AI新闻 vs 记者新闻：量子指标差异倍数分析', fontsize=14, pad=20, fontweight='bold')
    
    # 添加基准线
    ax.axvline(x=1, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(1.02, len(metric_names)-0.5, '基准线\n(相等)', ha='left', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 添加数值标签
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{diff:.3f}×', ha='left', va='center', fontsize=11, fontweight='bold')
    
    # 设置x轴范围
    ax.set_xlim(0, max(differences) * 1.15)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('../visualizations/fixed_chinese_difference_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ 生成差异分析图")

def main():
    """主函数"""
    
    print("🎨 开始生成修复版中文可视化...")
    
    # 设置中文字体
    setup_chinese_font()
    
    # 创建可视化目录
    Path('../visualizations').mkdir(exist_ok=True)
    
    # 加载数据
    data = load_summary_data()
    
    if not data:
        print("❌ 无法加载数据，退出程序")
        return
    
    print("\n开始生成各种图表...")
    
    # 生成各种可视化
    create_simple_comparison_chart(data)
    create_metrics_table_chart(data)
    create_field_comparison_chart(data)
    create_difference_analysis_chart(data)
    
    print(f"\n✅ 所有修复版可视化已生成完成!")
    print(f"📁 保存位置: ../visualizations/")
    print(f"   • fixed_chinese_comparison.png - 语法叠加强度对比")
    print(f"   • fixed_chinese_metrics_table.png - 量子指标对比表")
    print(f"   • fixed_chinese_field_comparison.png - 字段对比图")
    print(f"   • fixed_chinese_difference_analysis.png - 差异分析图")
    
    # 输出关键数据摘要
    print(f"\n📊 关键发现摘要:")
    ai_sup = data['ai_unrestricted']['新聞標題']['grammatical_superposition']['mean']
    j_sup = data['journalist_unrestricted']['新聞標題']['grammatical_superposition']['mean']
    diff_percent = ((ai_sup - j_sup) / j_sup) * 100
    
    print(f"• 语法叠加强度: AI新闻 {ai_sup:.6f} vs 记者新闻 {j_sup:.6f}")
    print(f"• AI比记者高 {diff_percent:.2f}%")
    print(f"• AI达到理论最大值的 {(ai_sup/4.0)*100:.1f}%")
    print(f"• 记者达到理论最大值的 {(j_sup/4.0)*100:.1f}%")

if __name__ == "__main__":
    main()
