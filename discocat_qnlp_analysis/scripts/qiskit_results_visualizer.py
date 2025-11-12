#!/usr/bin/env python3
"""
Qiskit量子分析结果可视化
生成基于真实量子电路分析的对比图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def setup_chinese_font():
    """设置中文字体"""
    import platform
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
    elif system == 'Windows':
        fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi']
    else:  # Linux
        fonts = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    
    for font in fonts:
        plt.rcParams['font.sans-serif'].insert(0, font)

def load_analysis_data():
    """加载Qiskit分析数据"""
    
    # 加载统计摘要
    ai_summary_path = '../results/fast_qiskit_ai_analysis_summary.json'
    journalist_summary_path = '../results/fast_qiskit_journalist_analysis_summary.json'
    
    with open(ai_summary_path, 'r', encoding='utf-8') as f:
        ai_summary = json.load(f)
    
    with open(journalist_summary_path, 'r', encoding='utf-8') as f:
        journalist_summary = json.load(f)
    
    return ai_summary, journalist_summary

def create_comparison_charts(ai_summary, journalist_summary):
    """创建对比图表"""
    
    setup_chinese_font()
    
    # 提取指标数据
    metrics = ['von_neumann_entropy', 'superposition_strength', 'quantum_coherence', 
               'semantic_interference', 'frame_competition', 'multiple_reality_strength']
    
    metric_names = {
        'von_neumann_entropy': '冯纽曼熵',
        'superposition_strength': '量子叠加强度',
        'quantum_coherence': '量子相干性',
        'semantic_interference': '语义干涉',
        'frame_competition': '框架竞争',
        'multiple_reality_strength': '多重现实强度'
    }
    
    # 创建综合对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('基于Qiskit量子电路的QNLP分析结果对比', fontsize=20, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # 数据准备
        ai_mean = ai_summary[metric]['mean']
        ai_std = ai_summary[metric]['std']
        journalist_mean = journalist_summary[metric]['mean']
        journalist_std = journalist_summary[metric]['std']
        
        # 柱状图
        categories = ['AI生成新闻', '记者撰写新闻']
        means = [ai_mean, journalist_mean]
        stds = [ai_std, journalist_std]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(categories, means, yerr=stds, capsize=5, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{metric_names[metric]}', fontsize=14, fontweight='bold')
        ax.set_ylabel('数值', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 设置Y轴范围
        max_val = max(means) + max(stds)
        min_val = min(means) - max(stds)
        margin = (max_val - min_val) * 0.1
        ax.set_ylim(min_val - margin, max_val + margin)
    
    plt.tight_layout()
    plt.savefig('../visualizations/qiskit_quantum_analysis_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 创建雷达图
    create_radar_chart(ai_summary, journalist_summary, metric_names)
    
    # 创建详细对比表
    create_comparison_table(ai_summary, journalist_summary, metric_names)

def create_radar_chart(ai_summary, journalist_summary, metric_names):
    """创建雷达图"""
    
    metrics = list(metric_names.keys())
    
    # 归一化数据用于雷达图显示
    ai_values = []
    journalist_values = []
    
    for metric in metrics:
        ai_val = ai_summary[metric]['mean']
        journalist_val = journalist_summary[metric]['mean']
        
        # 获取两者的最大值用于归一化
        max_val = max(ai_val, journalist_val)
        if max_val > 0:
            ai_values.append(ai_val / max_val)
            journalist_values.append(journalist_val / max_val)
        else:
            ai_values.append(0)
            journalist_values.append(0)
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    ai_values += ai_values[:1]
    journalist_values += journalist_values[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 绘制雷达图
    ax.plot(angles, ai_values, 'o-', linewidth=2, label='AI生成新闻', color='#FF6B6B')
    ax.fill(angles, ai_values, alpha=0.25, color='#FF6B6B')
    
    ax.plot(angles, journalist_values, 'o-', linewidth=2, label='记者撰写新闻', color='#4ECDC4')
    ax.fill(angles, journalist_values, alpha=0.25, color='#4ECDC4')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric_names[metric] for metric in metrics], fontsize=12)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    plt.title('量子特征雷达图对比\n(基于Qiskit量子电路)', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig('../visualizations/qiskit_quantum_radar_chart.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_comparison_table(ai_summary, journalist_summary, metric_names):
    """创建详细对比表"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    metrics = list(metric_names.keys())
    table_data = []
    
    for metric in metrics:
        ai_data = ai_summary[metric]
        journalist_data = journalist_summary[metric]
        
        # 计算差异
        mean_diff = ai_data['mean'] - journalist_data['mean']
        diff_pct = (mean_diff / journalist_data['mean']) * 100 if journalist_data['mean'] != 0 else 0
        
        row = [
            metric_names[metric],
            f"{ai_data['mean']:.4f}",
            f"{ai_data['std']:.4f}",
            f"{journalist_data['mean']:.4f}",
            f"{journalist_data['std']:.4f}",
            f"{mean_diff:+.4f}",
            f"{diff_pct:+.2f}%"
        ]
        table_data.append(row)
    
    # 创建表格
    columns = ['量子指标', 'AI均值', 'AI标准差', '记者均值', '记者标准差', '差异', '差异百分比']
    
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # 设置表格样式
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 为数据行设置交替颜色
    for i in range(1, len(table_data) + 1):
        color = '#F0F0F0' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Qiskit量子电路分析详细对比表', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('../visualizations/qiskit_quantum_comparison_table.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """主函数"""
    print("🎨 开始生成Qiskit量子分析结果可视化...")
    
    # 确保输出目录存在
    Path('../visualizations').mkdir(exist_ok=True)
    
    # 加载数据
    ai_summary, journalist_summary = load_analysis_data()
    
    # 生成图表
    create_comparison_charts(ai_summary, journalist_summary)
    
    print("✅ 可视化图表已生成:")
    print("   📊 ../visualizations/qiskit_quantum_analysis_comparison.png")
    print("   🎯 ../visualizations/qiskit_quantum_radar_chart.png") 
    print("   📋 ../visualizations/qiskit_quantum_comparison_table.png")
    
    print("\n🎉 Qiskit量子分析可视化完成！")

if __name__ == "__main__":
    main()
