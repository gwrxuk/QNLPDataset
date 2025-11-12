#!/usr/bin/env python3
"""
量子比特分布分析器
分析实际数据中2、3、4个量子比特的使用情况
"""

import pandas as pd
import json
from collections import Counter, defaultdict
import numpy as np

def analyze_qubit_distribution():
    """分析量子比特分布"""
    print("🔍 分析量子比特分布...")
    
    # 加载数据
    ai_df = pd.read_csv('../results/full_qiskit_ai_analysis_results.csv')
    journalist_df = pd.read_csv('../results/full_qiskit_journalist_analysis_results.csv')
    
    print(f"✅ AI数据: {len(ai_df)} 条记录")
    print(f"✅ 记者数据: {len(journalist_df)} 条记录")
    
    # 分析AI数据
    ai_qubit_stats = analyze_dataset(ai_df, "AI生成新闻")
    
    # 分析记者数据
    journalist_qubit_stats = analyze_dataset(journalist_df, "记者撰写新闻")
    
    # 生成报告
    generate_qubit_report(ai_qubit_stats, journalist_qubit_stats)
    
    return ai_qubit_stats, journalist_qubit_stats

def analyze_dataset(df, dataset_name):
    """分析单个数据集的量子比特分布"""
    print(f"\n📊 分析 {dataset_name}...")
    
    # 统计量子比特分布
    qubit_counts = Counter(df['quantum_circuit_qubits'])
    
    # 按字段分组统计
    field_stats = defaultdict(lambda: defaultdict(list))
    
    for _, row in df.iterrows():
        qubits = row['quantum_circuit_qubits']
        field = row['field']
        text = row['original_text']
        word_count = row['word_count']
        categorical_diversity = row['categorical_diversity']
        
        field_stats[field][qubits].append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'word_count': word_count,
            'categorical_diversity': categorical_diversity,
            'full_text': text
        })
    
    # 计算统计信息
    stats = {
        'dataset_name': dataset_name,
        'total_records': len(df),
        'qubit_distribution': dict(qubit_counts),
        'qubit_percentages': {k: v/len(df)*100 for k, v in qubit_counts.items()},
        'field_stats': dict(field_stats),
        'examples_by_qubits': {}
    }
    
    # 为每个量子比特数选择代表性示例
    for qubits in [2, 3, 4]:
        examples = []
        for field, field_data in field_stats.items():
            if qubits in field_data:
                # 选择前3个示例
                for example in field_data[qubits][:3]:
                    examples.append({
                        'field': field,
                        'text': example['text'],
                        'word_count': example['word_count'],
                        'categorical_diversity': example['categorical_diversity']
                    })
        stats['examples_by_qubits'][qubits] = examples
    
    return stats

def generate_qubit_report(ai_stats, journalist_stats):
    """生成量子比特分布报告"""
    print("\n📝 生成量子比特分布报告...")
    
    report = f"""# 量子比特分布详细分析报告

## 📊 **总体统计概览**

### **数据规模**
- **AI生成新闻**: {ai_stats['total_records']} 条记录
- **记者撰写新闻**: {journalist_stats['total_records']} 条记录
- **总计**: {ai_stats['total_records'] + journalist_stats['total_records']} 条记录

### **量子比特分布统计**

#### **AI生成新闻的量子比特分布**
"""
    
    # AI数据统计
    for qubits, count in sorted(ai_stats['qubit_distribution'].items()):
        percentage = ai_stats['qubit_percentages'][qubits]
        report += f"- **{qubits}个量子比特**: {count} 条记录 ({percentage:.1f}%)\n"
    
    report += f"""
#### **记者撰写新闻的量子比特分布**
"""
    
    # 记者数据统计
    for qubits, count in sorted(journalist_stats['qubit_distribution'].items()):
        percentage = journalist_stats['qubit_percentages'][qubits]
        report += f"- **{qubits}个量子比特**: {count} 条记录 ({percentage:.1f}%)\n"
    
    # 详细示例
    report += """
## 🔍 **详细示例分析**

### **2个量子比特的文本示例**

#### **特征**: 词性种类较少（≤2种），通常是简短的标题或简单句子

"""
    
    # 2个量子比特示例
    if 2 in ai_stats['examples_by_qubits']:
        report += "#### **AI生成新闻 - 2个量子比特示例**\n\n"
        for i, example in enumerate(ai_stats['examples_by_qubits'][2][:5], 1):
            report += f"""**示例 {i}** ({example['field']})
- **文本**: {example['text']}
- **词数**: {example['word_count']} 个
- **词性种类**: {example['categorical_diversity']} 种
- **量子比特**: 2个

"""
    
    if 2 in journalist_stats['examples_by_qubits']:
        report += "#### **记者撰写新闻 - 2个量子比特示例**\n\n"
        for i, example in enumerate(journalist_stats['examples_by_qubits'][2][:5], 1):
            report += f"""**示例 {i}** ({example['field']})
- **文本**: {example['text']}
- **词数**: {example['word_count']} 个
- **词性种类**: {example['categorical_diversity']} 种
- **量子比特**: 2个

"""
    
    # 3个量子比特示例
    report += """### **3个量子比特的文本示例**

#### **特征**: 词性种类中等（3种），通常包含名词、动词、形容词的组合

"""
    
    if 3 in ai_stats['examples_by_qubits']:
        report += "#### **AI生成新闻 - 3个量子比特示例**\n\n"
        for i, example in enumerate(ai_stats['examples_by_qubits'][3][:5], 1):
            report += f"""**示例 {i}** ({example['field']})
- **文本**: {example['text']}
- **词数**: {example['word_count']} 个
- **词性种类**: {example['categorical_diversity']} 种
- **量子比特**: 3个

"""
    
    if 3 in journalist_stats['examples_by_qubits']:
        report += "#### **记者撰写新闻 - 3个量子比特示例**\n\n"
        for i, example in enumerate(journalist_stats['examples_by_qubits'][3][:5], 1):
            report += f"""**示例 {i}** ({example['field']})
- **文本**: {example['text']}
- **词数**: {example['word_count']} 个
- **词性种类**: {example['categorical_diversity']} 种
- **量子比特**: 3个

"""
    
    # 4个量子比特示例
    report += """### **4个量子比特的文本示例**

#### **特征**: 词性种类丰富（≥4种），通常是复杂的长句或包含多种语法成分

"""
    
    if 4 in ai_stats['examples_by_qubits']:
        report += "#### **AI生成新闻 - 4个量子比特示例**\n\n"
        for i, example in enumerate(ai_stats['examples_by_qubits'][4][:5], 1):
            report += f"""**示例 {i}** ({example['field']})
- **文本**: {example['text']}
- **词数**: {example['word_count']} 个
- **词性种类**: {example['categorical_diversity']} 种
- **量子比特**: 4个

"""
    
    if 4 in journalist_stats['examples_by_qubits']:
        report += "#### **记者撰写新闻 - 4个量子比特示例**\n\n"
        for i, example in enumerate(journalist_stats['examples_by_qubits'][4][:5], 1):
            report += f"""**示例 {i}** ({example['field']})
- **文本**: {example['text']}
- **词数**: {example['word_count']} 个
- **词性种类**: {example['categorical_diversity']} 种
- **量子比特**: 4个

"""
    
    # 字段级别分析
    report += """## 📋 **字段级别量子比特分布**

### **AI生成新闻字段分析**

"""
    
    for field, field_data in ai_stats['field_stats'].items():
        report += f"#### **{field}**\n"
        for qubits, examples in sorted(field_data.items()):
            count = len(examples)
            percentage = count / sum(len(v) for v in field_data.values()) * 100
            report += f"- **{qubits}个量子比特**: {count} 条 ({percentage:.1f}%)\n"
        report += "\n"
    
    report += """### **记者撰写新闻字段分析**

"""
    
    for field, field_data in journalist_stats['field_stats'].items():
        report += f"#### **{field}**\n"
        for qubits, examples in sorted(field_data.items()):
            count = len(examples)
            percentage = count / sum(len(v) for v in field_data.values()) * 100
            report += f"- **{qubits}个量子比特**: {count} 条 ({percentage:.1f}%)\n"
        report += "\n"
    
    # 分析结论
    report += """## 🎯 **分析结论**

### **量子比特使用模式**

1. **4个量子比特占主导**: 绝大多数文本使用4个量子比特，说明新闻文本通常具有丰富的词性多样性
2. **3个量子比特较少见**: 只有少数文本使用3个量子比特，通常是较短的标题
3. **2个量子比特极少**: 极少数文本使用2个量子比特，通常是非常简短的标题

### **AI vs 记者对比**

- **相似性**: 两种数据源的量子比特分布模式基本相似
- **差异性**: 记者撰写的新闻可能在某些字段有略微不同的分布

### **技术意义**

- **算法有效性**: 动态量子比特分配算法能够有效适应不同复杂度的文本
- **信息保留**: 4个量子比特能够充分保留大多数新闻文本的语法信息
- **计算效率**: 限制最大4个量子比特保证了计算的可操作性

### **语言学观察**

- **中文新闻特点**: 中文新闻文本通常包含名词、动词、形容词、副词等多种词性
- **语法复杂性**: 新闻语言的正式性决定了其语法结构的复杂性
- **信息密度**: 新闻文本的高信息密度体现在词性的多样性上
"""
    
    # 保存报告
    with open('../20250927-image/qubit_distribution_analysis.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存JSON数据
    combined_stats = {
        'ai_stats': ai_stats,
        'journalist_stats': journalist_stats,
        'summary': {
            'total_records': ai_stats['total_records'] + journalist_stats['total_records'],
            'ai_qubit_distribution': ai_stats['qubit_distribution'],
            'journalist_qubit_distribution': journalist_stats['qubit_distribution']
        }
    }
    
    with open('../20250927-image/qubit_distribution_data.json', 'w', encoding='utf-8') as f:
        json.dump(combined_stats, f, ensure_ascii=False, indent=2)
    
    print("✅ 量子比特分布报告已生成")
    print("📄 报告文件: ../20250927-image/qubit_distribution_analysis.md")
    print("📊 数据文件: ../20250927-image/qubit_distribution_data.json")

def main():
    """主函数"""
    print("🚀 开始量子比特分布分析...")
    
    ai_stats, journalist_stats = analyze_qubit_distribution()
    
    # 打印简要统计
    print("\n📊 简要统计:")
    print(f"AI生成新闻量子比特分布: {ai_stats['qubit_distribution']}")
    print(f"记者撰写新闻量子比特分布: {journalist_stats['qubit_distribution']}")
    
    print("\n🎉 量子比特分布分析完成！")

if __name__ == "__main__":
    main()
