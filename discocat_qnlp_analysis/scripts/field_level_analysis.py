#!/usr/bin/env python3
"""
字段级别量子特征分析
按照具体字段（新聞標題、影片對話、影片描述、新聞內容）分别统计量子特征
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_field_level_statistics():
    """分析字段级别的量子特征统计"""
    
    # 加载完整数据集的分析结果
    ai_data = pd.read_csv('../results/full_qiskit_ai_analysis_results.csv')
    journalist_data = pd.read_csv('../results/full_qiskit_journalist_analysis_results.csv')
    
    # 量子指标
    quantum_metrics = [
        'von_neumann_entropy', 'superposition_strength', 'quantum_coherence',
        'semantic_interference', 'frame_competition', 'multiple_reality_strength'
    ]
    
    results = {}
    
    # 分析AI数据
    print("📊 分析AI生成新闻的字段级别量子特征...")
    ai_fields = ['新聞標題', '影片對話', '影片描述']
    
    results['AI_Generated'] = {}
    for field in ai_fields:
        field_data = ai_data[ai_data['field'] == field]
        if len(field_data) > 0:
            field_stats = {}
            for metric in quantum_metrics:
                if metric in field_data.columns:
                    field_stats[metric] = {
                        'mean': float(field_data[metric].mean()),
                        'std': float(field_data[metric].std()),
                        'min': float(field_data[metric].min()),
                        'max': float(field_data[metric].max()),
                        'median': float(field_data[metric].median()),
                        'count': int(len(field_data))
                    }
            results['AI_Generated'][field] = field_stats
            print(f"  ✅ {field}: {len(field_data)} 条记录")
    
    # 分析记者数据
    print("\n📊 分析记者撰写新闻的字段级别量子特征...")
    journalist_fields = ['新聞標題', '新聞內容']
    
    results['Journalist_Written'] = {}
    for field in journalist_fields:
        field_data = journalist_data[journalist_data['field'] == field]
        if len(field_data) > 0:
            field_stats = {}
            for metric in quantum_metrics:
                if metric in field_data.columns:
                    field_stats[metric] = {
                        'mean': float(field_data[metric].mean()),
                        'std': float(field_data[metric].std()),
                        'min': float(field_data[metric].min()),
                        'max': float(field_data[metric].max()),
                        'median': float(field_data[metric].median()),
                        'count': int(len(field_data))
                    }
            results['Journalist_Written'][field] = field_stats
            print(f"  ✅ {field}: {len(field_data)} 条记录")
    
    # 保存结果
    output_path = '../results/full_field_level_quantum_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 完整字段级别分析结果已保存: {output_path}")
    return results

def create_field_comparison_table(results):
    """创建字段对比表格"""
    
    print("\n📋 生成字段级别对比表格...")
    
    # 创建Markdown表格
    markdown_content = []
    
    # AI数据表格
    markdown_content.append("## AI生成新闻的字段级别量子特征\n")
    
    # 表头
    ai_fields = list(results['AI_Generated'].keys())
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
    
    for metric in metrics:
        markdown_content.append(f"### {metric_names[metric]} ({metric})\n")
        markdown_content.append("| 字段 | 均值 | 标准差 | 最小值 | 最大值 | 中位数 | 记录数 |")
        markdown_content.append("|------|------|--------|--------|--------|--------|--------|")
        
        for field in ai_fields:
            if field in results['AI_Generated'] and metric in results['AI_Generated'][field]:
                stats = results['AI_Generated'][field][metric]
                markdown_content.append(
                    f"| **{field}** | {stats['mean']:.4f} | {stats['std']:.4f} | "
                    f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['median']:.4f} | {stats['count']} |"
                )
        markdown_content.append("")
    
    # 记者数据表格
    markdown_content.append("## 记者撰写新闻的字段级别量子特征\n")
    
    journalist_fields = list(results['Journalist_Written'].keys())
    
    for metric in metrics:
        markdown_content.append(f"### {metric_names[metric]} ({metric})\n")
        markdown_content.append("| 字段 | 均值 | 标准差 | 最小值 | 最大值 | 中位数 | 记录数 |")
        markdown_content.append("|------|------|--------|--------|--------|--------|--------|")
        
        for field in journalist_fields:
            if field in results['Journalist_Written'] and metric in results['Journalist_Written'][field]:
                stats = results['Journalist_Written'][field][metric]
                markdown_content.append(
                    f"| **{field}** | {stats['mean']:.4f} | {stats['std']:.4f} | "
                    f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['median']:.4f} | {stats['count']} |"
                )
        markdown_content.append("")
    
    # 字段间对比
    markdown_content.append("## 字段间量子特征对比\n")
    
    # 创建综合对比表
    markdown_content.append("### 综合对比表\n")
    markdown_content.append("| 数据源 | 字段 | 冯纽曼熵 | 量子叠加强度 | 量子相干性 | 语义干涉 | 框架竞争 | 多重现实强度 |")
    markdown_content.append("|--------|------|----------|-------------|------------|----------|----------|--------------|")
    
    # AI数据行
    for field in ai_fields:
        if field in results['AI_Generated']:
            row = f"| **AI生成** | {field} |"
            for metric in metrics:
                if metric in results['AI_Generated'][field]:
                    mean_val = results['AI_Generated'][field][metric]['mean']
                    row += f" {mean_val:.4f} |"
                else:
                    row += " N/A |"
            markdown_content.append(row)
    
    # 记者数据行
    for field in journalist_fields:
        if field in results['Journalist_Written']:
            row = f"| **记者撰写** | {field} |"
            for metric in metrics:
                if metric in results['Journalist_Written'][field]:
                    mean_val = results['Journalist_Written'][field][metric]['mean']
                    row += f" {mean_val:.4f} |"
                else:
                    row += " N/A |"
            markdown_content.append(row)
    
    # 保存Markdown文件
    output_path = '../analysis_reports/full_field_level_quantum_comparison.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_content))
    
    print(f"✅ 完整字段级别对比表格已保存: {output_path}")

def main():
    """主函数"""
    print("🚀 开始完整数据集字段级别量子特征分析...")
    
    # 确保输出目录存在
    Path('../results').mkdir(exist_ok=True)
    Path('../analysis_reports').mkdir(exist_ok=True)
    
    # 分析字段级别统计
    results = analyze_field_level_statistics()
    
    # 创建对比表格
    create_field_comparison_table(results)
    
    print("\n🎉 完整数据集字段级别量子特征分析完成！")
    print("📊 分析规模:")
    print("   - AI新闻: 298条记录 × 3个字段 = 894个文本片段")
    print("   - 记者新闻: 20条记录 × 2个字段 = 40个文本片段")
    print("   - 总计: 934个文本片段的量子特征分析")

if __name__ == "__main__":
    main()
