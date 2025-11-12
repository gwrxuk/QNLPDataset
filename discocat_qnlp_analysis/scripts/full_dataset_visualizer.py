#!/usr/bin/env python3
"""
完整数据集可视化分析器
生成基于934个文本片段的量子特征可视化图表
测试中文字符显示效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import platform

def setup_chinese_font():
    """设置中文字体以确保正确显示"""
    print("🔧 设置中文字体...")
    
    # 检测操作系统并设置相应的中文字体
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STSong']
        print("🍎 检测到macOS系统")
    elif system == 'Windows':
        fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun']
        print("🪟 检测到Windows系统")
    else:  # Linux
        fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
        print("🐧 检测到Linux系统")
    
    # 设置matplotlib参数
    plt.rcParams['axes.unicode_minus'] = False
    
    # 尝试设置字体
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            print(f"✅ 设置字体: {font}")
            break
        except:
            continue
    
    # 验证中文字体设置
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 测试中文字符
    fig, ax = plt.subplots(figsize=(8, 6))
    test_text = "测试中文字符显示：量子自然语言处理分析"
    ax.text(0.5, 0.5, test_text, ha='center', va='center', fontsize=16)
    ax.set_title('中文字体测试图表', fontsize=18, fontweight='bold')
    ax.axis('off')
    
    # 保存测试图
    test_path = '../20250927-image/chinese_font_test.png'
    plt.savefig(test_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 中文字体测试图已保存: {test_path}")

def load_full_dataset_results():
    """加载完整数据集的分析结果"""
    print("📂 加载完整数据集分析结果...")
    
    # 加载统计摘要
    ai_summary_path = '../results/full_qiskit_ai_analysis_summary.json'
    journalist_summary_path = '../results/full_qiskit_journalist_analysis_summary.json'
    field_level_path = '../results/full_field_level_quantum_analysis.json'
    
    with open(ai_summary_path, 'r', encoding='utf-8') as f:
        ai_summary = json.load(f)
    
    with open(journalist_summary_path, 'r', encoding='utf-8') as f:
        journalist_summary = json.load(f)
        
    with open(field_level_path, 'r', encoding='utf-8') as f:
        field_level_data = json.load(f)
    
    print("✅ 数据加载完成")
    return ai_summary, journalist_summary, field_level_data

def create_comprehensive_comparison(ai_summary, journalist_summary, field_level_data):
    """创建综合对比图表"""
    print("📊 生成综合对比图表...")
    
    # 量子指标
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
    
    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('完整数据集量子特征对比分析\n(基于934个文本片段的Qiskit量子电路分析)', 
                 fontsize=20, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # 数据准备
        ai_mean = ai_summary[metric]['mean']
        ai_std = ai_summary[metric]['std']
        journalist_mean = journalist_summary[metric]['mean']
        journalist_std = journalist_summary[metric]['std']
        
        # 柱状图
        categories = ['AI生成新闻\n(298条记录)', '记者撰写新闻\n(20条记录)']
        means = [ai_mean, journalist_mean]
        stds = [ai_std, journalist_std]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(categories, means, yerr=stds, capsize=8, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_title(f'{metric_names[metric]}', fontsize=14, fontweight='bold')
        ax.set_ylabel('数值', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 设置Y轴范围
        max_val = max(means) + max(stds)
        min_val = min(means) - max(stds)
        margin = (max_val - min_val) * 0.15
        ax.set_ylim(max(0, min_val - margin), max_val + margin)
    
    plt.tight_layout()
    plt.savefig('../20250927-image/comprehensive_quantum_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 综合对比图表已保存")

def create_field_level_heatmap(field_level_data):
    """创建字段级别热力图"""
    print("🔥 生成字段级别热力图...")
    
    # 准备数据
    metrics = ['von_neumann_entropy', 'superposition_strength', 'quantum_coherence', 
               'semantic_interference', 'multiple_reality_strength']
    
    # AI数据
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
    
    # 记者数据
    journalist_fields = ['新聞標題', '新聞內容']
    journalist_data = []
    journalist_labels = []
    
    for field in journalist_fields:
        row = []
        for metric in metrics:
            mean_val = field_level_data['Journalist_Written'][field][metric]['mean']
            row.append(mean_val)
        journalist_data.append(row)
        journalist_labels.append(f'记者-{field}')
    
    # 合并数据
    all_data = np.array(ai_data + journalist_data)
    all_labels = ai_labels + journalist_labels
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metric_names = ['冯纽曼熵', '量子叠加强度', '量子相干性', '语义干涉', '多重现实强度']
    
    # 使用seaborn创建热力图
    sns.heatmap(all_data, 
                xticklabels=metric_names,
                yticklabels=all_labels,
                annot=True, 
                fmt='.4f',
                cmap='RdYlBu_r',
                center=None,
                ax=ax,
                cbar_kws={'label': '量子特征值'})
    
    ax.set_title('字段级别量子特征热力图\n(完整数据集分析)', fontsize=16, fontweight='bold')
    ax.set_xlabel('量子指标', fontsize=12)
    ax.set_ylabel('数据源与字段', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../20250927-image/field_level_heatmap.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 字段级别热力图已保存")

def create_radar_chart(field_level_data):
    """创建雷达图对比"""
    print("🎯 生成雷达图对比...")
    
    metrics = ['von_neumann_entropy', 'superposition_strength', 'quantum_coherence', 
               'semantic_interference', 'multiple_reality_strength']
    metric_names = ['冯纽曼熵', '量子叠加强度', '量子相干性', '语义干涉', '多重现实强度']
    
    # 数据准备 - 选择代表性字段进行对比
    ai_dialogue = []  # AI影片對話
    journalist_content = []  # 记者新聞內容
    
    for metric in metrics:
        ai_val = field_level_data['AI_Generated']['影片對話'][metric]['mean']
        journalist_val = field_level_data['Journalist_Written']['新聞內容'][metric]['mean']
        
        # 归一化到0-1范围用于雷达图显示
        max_val = max(ai_val, journalist_val)
        if max_val > 0:
            ai_dialogue.append(ai_val / max_val)
            journalist_content.append(journalist_val / max_val)
        else:
            ai_dialogue.append(0)
            journalist_content.append(0)
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    ai_dialogue += ai_dialogue[:1]
    journalist_content += journalist_content[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 绘制雷达图
    ax.plot(angles, ai_dialogue, 'o-', linewidth=3, label='AI影片對話 (298条)', color='#FF6B6B')
    ax.fill(angles, ai_dialogue, alpha=0.25, color='#FF6B6B')
    
    ax.plot(angles, journalist_content, 'o-', linewidth=3, label='记者新聞內容 (20条)', color='#4ECDC4')
    ax.fill(angles, journalist_content, alpha=0.25, color='#4ECDC4')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=12)
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    plt.title('量子特征雷达图对比\n(长文本字段代表性分析)', fontsize=16, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.savefig('../20250927-image/quantum_radar_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 雷达图对比已保存")

def create_distribution_analysis(field_level_data):
    """创建分布分析图"""
    print("📈 生成分布分析图...")
    
    # 选择关键指标进行分布分析
    key_metrics = ['von_neumann_entropy', 'semantic_interference', 'multiple_reality_strength']
    metric_names = ['冯纽曼熵', '语义干涉', '多重现实强度']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('关键量子指标的字段分布分析\n(完整数据集)', fontsize=16, fontweight='bold')
    
    for i, (metric, name) in enumerate(zip(key_metrics, metric_names)):
        ax = axes[i]
        
        # 收集所有字段的数据
        field_names = []
        values = []
        colors = []
        
        # AI数据
        for field in ['新聞標題', '影片對話', '影片描述']:
            mean_val = field_level_data['AI_Generated'][field][metric]['mean']
            std_val = field_level_data['AI_Generated'][field][metric]['std']
            count = field_level_data['AI_Generated'][field][metric]['count']
            
            field_names.append(f'AI-{field}\n({count}条)')
            values.append(mean_val)
            colors.append('#FF6B6B')
        
        # 记者数据
        for field in ['新聞標題', '新聞內容']:
            mean_val = field_level_data['Journalist_Written'][field][metric]['mean']
            std_val = field_level_data['Journalist_Written'][field][metric]['std']
            count = field_level_data['Journalist_Written'][field][metric]['count']
            
            field_names.append(f'记者-{field}\n({count}条)')
            values.append(mean_val)
            colors.append('#4ECDC4')
        
        # 创建柱状图
        bars = ax.bar(range(len(field_names)), values, color=colors, alpha=0.8, edgecolor='black')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_ylabel('数值', fontsize=12)
        ax.set_xticks(range(len(field_names)))
        ax.set_xticklabels(field_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../20250927-image/distribution_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 分布分析图已保存")

def create_summary_statistics_table():
    """创建统计摘要表格"""
    print("📋 生成统计摘要表格...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # 表格数据
    table_data = [
        ['数据源', '字段', '记录数', '冯纽曼熵', '量子叠加强度', '量子相干性', '语义干涉', '多重现实强度'],
        ['AI生成', '新聞標題', '298', '3.9967±0.0579', '3.7492±0.0145', '0.9373±0.0036', '0.0014±0.0026', '1.7001±0.0059'],
        ['AI生成', '影片對話', '298', '4.0000±0.0000', '3.7500±0.0000', '0.9375±0.0000', '0.0178±0.0042', '1.7054±0.0013'],
        ['AI生成', '影片描述', '298', '4.0000±0.0000', '3.7500±0.0000', '0.9375±0.0000', '0.0111±0.0039', '1.7033±0.0012'],
        ['记者撰写', '新聞標題', '20', '3.8500±0.3663', '3.7125±0.0916', '0.9281±0.0229', '0.0008±0.0022', '1.6856±0.0369'],
        ['记者撰写', '新聞內容', '20', '4.0000±0.0000', '3.7500±0.0000', '0.9375±0.0000', '0.0177±0.0060', '1.7054±0.0018']
    ]
    
    # 创建表格
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                     loc='center', cellLoc='center')
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)
    
    # 设置标题行样式
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行样式
    for i in range(1, len(table_data)):
        color = '#FFE5E5' if 'AI生成' in table_data[i][0] else '#E5F9F6'
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)
    
    plt.title('完整数据集量子特征统计摘要表\n(均值±标准差)', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('../20250927-image/statistics_summary_table.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ 统计摘要表格已保存")

def main():
    """主函数"""
    print("🚀 开始完整数据集可视化分析...")
    print(f"📊 分析规模: 934个文本片段")
    
    # 确保输出目录存在
    Path('../20250927-image').mkdir(exist_ok=True)
    
    # 设置中文字体
    setup_chinese_font()
    
    # 加载数据
    ai_summary, journalist_summary, field_level_data = load_full_dataset_results()
    
    # 生成各种可视化图表
    create_comprehensive_comparison(ai_summary, journalist_summary, field_level_data)
    create_field_level_heatmap(field_level_data)
    create_radar_chart(field_level_data)
    create_distribution_analysis(field_level_data)
    create_summary_statistics_table()
    
    print("\n🎉 完整数据集可视化分析完成！")
    print("📂 所有图表已保存到: ../20250927-image/")
    print("📊 生成的图表:")
    print("   1. chinese_font_test.png - 中文字体测试")
    print("   2. comprehensive_quantum_comparison.png - 综合量子特征对比")
    print("   3. field_level_heatmap.png - 字段级别热力图")
    print("   4. quantum_radar_comparison.png - 量子特征雷达图")
    print("   5. distribution_analysis.png - 分布分析图")
    print("   6. statistics_summary_table.png - 统计摘要表格")

if __name__ == "__main__":
    main()
