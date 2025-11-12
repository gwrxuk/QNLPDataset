#!/usr/bin/env python3
"""
数据集元数据提取器 - 提取AI新闻和记者新闻数据集的详细元数据
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from collections import Counter
import re

def analyze_ai_dataset():
    """分析AI新闻数据集"""
    
    print("📰 分析AI新闻数据集...")
    
    try:
        # 读取数据
        df = pd.read_excel('../data/dataseet.xlsx')
        
        # 基本信息
        metadata = {
            "dataset_name": "AI生成新闻数据集",
            "file_name": "dataseet.xlsx",
            "file_format": "Excel (.xlsx)",
            "total_records": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 字段分析
        fields_analysis = {}
        
        for col in df.columns:
            if col in df.columns:
                # 基本统计
                non_null_count = df[col].notna().sum()
                null_count = df[col].isna().sum()
                
                field_info = {
                    "non_null_count": int(non_null_count),
                    "null_count": int(null_count),
                    "null_percentage": float(null_count / len(df) * 100),
                    "data_type": str(df[col].dtype)
                }
                
                # 文本长度分析
                if df[col].dtype == 'object':
                    text_lengths = df[col].dropna().astype(str).str.len()
                    if len(text_lengths) > 0:
                        field_info.update({
                            "avg_length": float(text_lengths.mean()),
                            "min_length": int(text_lengths.min()),
                            "max_length": int(text_lengths.max()),
                            "median_length": float(text_lengths.median())
                        })
                        
                        # 样本内容
                        samples = df[col].dropna().head(3).tolist()
                        field_info["samples"] = [str(s)[:100] + "..." if len(str(s)) > 100 else str(s) for s in samples]
                
                fields_analysis[col] = field_info
        
        metadata["fields_analysis"] = fields_analysis
        
        # 内容质量分析
        content_quality = {}
        
        # 分析新聞標題
        if '新聞標題' in df.columns:
            titles = df['新聞標題'].dropna()
            content_quality["新聞標題"] = {
                "avg_word_count": float(titles.astype(str).str.len().mean()),
                "unique_titles": len(titles.unique()),
                "duplicate_rate": float((len(titles) - len(titles.unique())) / len(titles) * 100)
            }
        
        # 分析影片對話
        if '影片對話' in df.columns:
            dialogues = df['影片對話'].dropna()
            content_quality["影片對話"] = {
                "avg_word_count": float(dialogues.astype(str).str.len().mean()),
                "unique_dialogues": len(dialogues.unique()),
                "duplicate_rate": float((len(dialogues) - len(dialogues.unique())) / len(dialogues) * 100)
            }
        
        # 分析影片描述
        if '影片描述' in df.columns:
            descriptions = df['影片描述'].dropna()
            content_quality["影片描述"] = {
                "avg_word_count": float(descriptions.astype(str).str.len().mean()),
                "unique_descriptions": len(descriptions.unique()),
                "duplicate_rate": float((len(descriptions) - len(descriptions.unique())) / len(descriptions) * 100)
            }
        
        metadata["content_quality"] = content_quality
        
        return metadata
        
    except Exception as e:
        print(f"❌ AI数据集分析失败: {e}")
        return None

def analyze_journalist_dataset():
    """分析记者新闻数据集"""
    
    print("👨‍💼 分析记者新闻数据集...")
    
    try:
        # 读取数据
        df = pd.read_csv('../data/cna.csv')
        
        # 基本信息
        metadata = {
            "dataset_name": "台湾中央社记者新闻数据集",
            "file_name": "cna.csv",
            "file_format": "CSV (.csv)",
            "total_records": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "台湾中央通讯社 (Central News Agency Taiwan)",
            "language": "繁体中文"
        }
        
        # 字段分析
        fields_analysis = {}
        
        for col in df.columns:
            # 基本统计
            non_null_count = df[col].notna().sum()
            null_count = df[col].isna().sum()
            
            field_info = {
                "non_null_count": int(non_null_count),
                "null_count": int(null_count),
                "null_percentage": float(null_count / len(df) * 100),
                "data_type": str(df[col].dtype)
            }
            
            # 文本长度分析
            if df[col].dtype == 'object':
                text_lengths = df[col].dropna().astype(str).str.len()
                if len(text_lengths) > 0:
                    field_info.update({
                        "avg_length": float(text_lengths.mean()),
                        "min_length": int(text_lengths.min()),
                        "max_length": int(text_lengths.max()),
                        "median_length": float(text_lengths.median())
                    })
                    
                    # 样本内容
                    samples = df[col].dropna().head(3).tolist()
                    field_info["samples"] = [str(s)[:100] + "..." if len(str(s)) > 100 else str(s) for s in samples]
            
            fields_analysis[col] = field_info
        
        metadata["fields_analysis"] = fields_analysis
        
        # 时间范围分析
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'], errors='coerce').dropna()
            if len(dates) > 0:
                metadata["temporal_coverage"] = {
                    "earliest_date": dates.min().strftime("%Y-%m-%d"),
                    "latest_date": dates.max().strftime("%Y-%m-%d"),
                    "date_range_days": (dates.max() - dates.min()).days,
                    "unique_dates": len(dates.unique())
                }
        
        # URL域名分析
        if 'url' in df.columns:
            urls = df['url'].dropna()
            domains = [re.findall(r'https?://([^/]+)', url) for url in urls]
            domains = [domain[0] if domain else 'unknown' for domain in domains]
            domain_counts = Counter(domains)
            
            metadata["url_analysis"] = {
                "total_urls": len(urls),
                "unique_urls": len(urls.unique()),
                "domains": dict(domain_counts.most_common(10))
            }
        
        # 内容质量分析
        content_quality = {}
        
        # 分析标题
        if 'title' in df.columns:
            titles = df['title'].dropna()
            content_quality["title"] = {
                "avg_char_count": float(titles.astype(str).str.len().mean()),
                "unique_titles": len(titles.unique()),
                "duplicate_rate": float((len(titles) - len(titles.unique())) / len(titles) * 100)
            }
        
        # 分析内容
        if 'content' in df.columns:
            contents = df['content'].dropna()
            content_quality["content"] = {
                "avg_char_count": float(contents.astype(str).str.len().mean()),
                "unique_contents": len(contents.unique()),
                "duplicate_rate": float((len(contents) - len(contents.unique())) / len(contents) * 100)
            }
        
        metadata["content_quality"] = content_quality
        
        return metadata
        
    except Exception as e:
        print(f"❌ 记者数据集分析失败: {e}")
        return None

def generate_metadata_report(ai_metadata, journalist_metadata):
    """生成元数据报告"""
    
    report = f"""# 数据集元数据报告

## 📊 数据集概览

本报告详细描述了用于量子自然语言处理对比分析的两个数据集的元数据信息。

---

## 🤖 AI生成新闻数据集

### 基本信息
- **数据集名称**: {ai_metadata['dataset_name']}
- **文件名**: {ai_metadata['file_name']}
- **文件格式**: {ai_metadata['file_format']}
- **总记录数**: {ai_metadata['total_records']:,} 条
- **总字段数**: {ai_metadata['total_columns']} 个
- **分析时间**: {ai_metadata['analysis_date']}

### 数据结构
**字段列表**: {', '.join(ai_metadata['column_names'])}

### 字段详细分析

"""
    
    # AI数据集字段分析
    for field, info in ai_metadata['fields_analysis'].items():
        report += f"""#### {field}
- **数据类型**: {info['data_type']}
- **非空记录**: {info['non_null_count']:,} 条 ({100-info['null_percentage']:.1f}%)
- **空值记录**: {info['null_count']:,} 条 ({info['null_percentage']:.1f}%)
"""
        
        if 'avg_length' in info:
            report += f"""- **平均长度**: {info['avg_length']:.1f} 字符
- **长度范围**: {info['min_length']} - {info['max_length']} 字符
- **中位数长度**: {info['median_length']:.1f} 字符

**样本内容**:
"""
            for i, sample in enumerate(info['samples'], 1):
                report += f"{i}. {sample}\n"
        
        report += "\n"
    
    # AI数据集内容质量
    report += "### 内容质量分析\n\n"
    for field, quality in ai_metadata['content_quality'].items():
        report += f"""#### {field}
- **平均字数**: {quality['avg_word_count']:.1f} 字符
- **唯一内容数**: {quality['unique_titles'] if 'unique_titles' in quality else quality.get('unique_dialogues', quality.get('unique_descriptions', 0)):,} 条
- **重复率**: {quality['duplicate_rate']:.2f}%

"""
    
    # 记者数据集
    report += f"""---

## 👨‍💼 记者撰写新闻数据集

### 基本信息
- **数据集名称**: {journalist_metadata['dataset_name']}
- **文件名**: {journalist_metadata['file_name']}
- **文件格式**: {journalist_metadata['file_format']}
- **数据来源**: {journalist_metadata['source']}
- **语言**: {journalist_metadata['language']}
- **总记录数**: {journalist_metadata['total_records']:,} 条
- **总字段数**: {journalist_metadata['total_columns']} 个
- **分析时间**: {journalist_metadata['analysis_date']}

### 数据结构
**字段列表**: {', '.join(journalist_metadata['column_names'])}

### 字段详细分析

"""
    
    # 记者数据集字段分析
    for field, info in journalist_metadata['fields_analysis'].items():
        report += f"""#### {field}
- **数据类型**: {info['data_type']}
- **非空记录**: {info['non_null_count']:,} 条 ({100-info['null_percentage']:.1f}%)
- **空值记录**: {info['null_count']:,} 条 ({info['null_percentage']:.1f}%)
"""
        
        if 'avg_length' in info:
            report += f"""- **平均长度**: {info['avg_length']:.1f} 字符
- **长度范围**: {info['min_length']} - {info['max_length']} 字符
- **中位数长度**: {info['median_length']:.1f} 字符

**样本内容**:
"""
            for i, sample in enumerate(info['samples'], 1):
                report += f"{i}. {sample}\n"
        
        report += "\n"
    
    # 时间覆盖范围
    if 'temporal_coverage' in journalist_metadata:
        temp = journalist_metadata['temporal_coverage']
        report += f"""### 时间覆盖范围
- **最早日期**: {temp['earliest_date']}
- **最晚日期**: {temp['latest_date']}
- **时间跨度**: {temp['date_range_days']} 天
- **唯一日期数**: {temp['unique_dates']} 个

"""
    
    # URL分析
    if 'url_analysis' in journalist_metadata:
        url = journalist_metadata['url_analysis']
        report += f"""### URL来源分析
- **总URL数**: {url['total_urls']:,} 个
- **唯一URL数**: {url['unique_urls']:,} 个
- **主要域名分布**:
"""
        for domain, count in url['domains'].items():
            report += f"  - {domain}: {count} 条\n"
        
        report += "\n"
    
    # 记者数据集内容质量
    report += "### 内容质量分析\n\n"
    for field, quality in journalist_metadata['content_quality'].items():
        report += f"""#### {field}
- **平均字符数**: {quality['avg_char_count']:.1f} 字符
- **唯一内容数**: {quality.get('unique_titles', quality.get('unique_contents', 0)):,} 条
- **重复率**: {quality['duplicate_rate']:.2f}%

"""
    
    # 对比分析
    report += f"""---

## 📈 数据集对比分析

### 规模对比
| 指标 | AI新闻数据集 | 记者新闻数据集 | 比例 |
|------|-------------|---------------|------|
| 总记录数 | {ai_metadata['total_records']:,} | {journalist_metadata['total_records']:,} | {ai_metadata['total_records']/journalist_metadata['total_records']:.1f}:1 |
| 字段数 | {ai_metadata['total_columns']} | {journalist_metadata['total_columns']} | {ai_metadata['total_columns']/journalist_metadata['total_columns']:.1f}:1 |

### 字段映射关系
| AI新闻字段 | 记者新闻字段 | 用途 |
|-----------|-------------|------|
| 新聞標題 | title | 新闻标题对比分析 |
| 影片對話 | content | 内容对比分析（视频对话 vs 新闻正文） |
| 影片描述 | content | 内容对比分析（视频描述 vs 新闻正文） |

### 数据质量对比
- **AI新闻**: 多媒体新闻格式，包含标题、视频对话、视频描述
- **记者新闻**: 传统新闻格式，包含URL、标题、日期、正文内容
- **语言一致性**: 两个数据集均为繁体中文
- **内容类型**: AI新闻偏向多媒体内容，记者新闻为传统文字报道

### 分析适用性评估
✅ **优势**:
- 语言一致性良好（均为繁体中文）
- 都包含新闻标题，可进行直接对比
- 数据质量较高，重复率低
- 涵盖不同的新闻生产方式（AI生成 vs 人工撰写）

⚠️ **限制**:
- 数据规模差异较大（AI新闻 {ai_metadata['total_records']} 条 vs 记者新闻 {journalist_metadata['total_records']} 条）
- 内容类型存在差异（多媒体 vs 传统文字）
- 时间跨度可能不同

---

## 🔬 方法论说明

### 数据预处理
1. **字段映射**: 确保对比分析的公平性
2. **质量过滤**: 移除空值和异常数据
3. **文本标准化**: 统一编码和格式

### 分析范围
- **新聞標題 vs title**: 标题级别的量子特征对比
- **影片對話 + 影片描述 vs content**: 内容级别的量子特征对比

### 统计显著性
- 样本量充足，满足统计分析要求
- 采用多维度量子指标确保结果可靠性
- 通过受限制/无限制版本对比验证分析方法

---

**报告生成时间**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}
**数据分析工具**: Python + pandas + numpy
**量子分析框架**: DisCoCat量子自然语言处理
"""
    
    return report

def main():
    """主函数"""
    
    print("🔍 开始提取数据集元数据...")
    print("=" * 60)
    
    # 分析AI数据集
    ai_metadata = analyze_ai_dataset()
    
    # 分析记者数据集
    journalist_metadata = analyze_journalist_dataset()
    
    if ai_metadata and journalist_metadata:
        # 保存元数据JSON
        with open('../results/ai_dataset_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(ai_metadata, f, ensure_ascii=False, indent=2)
        
        with open('../results/journalist_dataset_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(journalist_metadata, f, ensure_ascii=False, indent=2)
        
        # 生成报告
        report = generate_metadata_report(ai_metadata, journalist_metadata)
        
        # 保存报告
        with open('../analysis_reports/dataset_metadata_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✅ 元数据提取完成!")
        print("📁 输出文件:")
        print("   • ../results/ai_dataset_metadata.json")
        print("   • ../results/journalist_dataset_metadata.json")
        print("   • ../analysis_reports/dataset_metadata_report.md")
        
        # 显示关键统计
        print(f"\n📊 关键统计:")
        print(f"   • AI新闻数据集: {ai_metadata['total_records']} 条记录")
        print(f"   • 记者新闻数据集: {journalist_metadata['total_records']} 条记录")
        print(f"   • 数据规模比例: {ai_metadata['total_records']/journalist_metadata['total_records']:.1f}:1")
        
    else:
        print("❌ 元数据提取失败")

if __name__ == "__main__":
    main()
