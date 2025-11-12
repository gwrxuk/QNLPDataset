#!/usr/bin/env python3
"""
公平对比可视化器 - 生成AI vs 记者新闻的详细对比图表
"""

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
def setup_chinese_font():
    """设置中文字体"""
    
    # 尝试系统中文字体
    chinese_fonts = [
        'Arial Unicode MS',
        'STHeiti',
        'SimHei', 
        'Microsoft YaHei',
        'PingFang SC',
        'Hiragino Sans GB',
        'Source Han Sans CN',
        'Noto Sans CJK SC'
    ]
    
    font_found = False
    
    # 首先尝试系统字体
    for font_name in chinese_fonts:
        try:
            # 测试字体是否可用
            test_font = fm.FontProperties(family=font_name)
            if test_font.get_name() in [f.name for f in fm.fontManager.ttflist]:
                plt.rcParams['font.family'] = [font_name]
                print(f"✅ 使用中文字体: {font_name}")
                font_found = True
                break
        except Exception as e:
            continue
    
    # 如果系统字体不可用，尝试文件路径
    if not font_found:
        font_paths = [
            '/System/Library/Fonts/Arial Unicode MS.ttf',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
            '/Library/Fonts/Arial Unicode MS.ttf'
        ]
        
        for font_path in font_paths:
            try:
                if Path(font_path).exists():
                    font_prop = fm.FontProperties(fname=font_path)
                    plt.rcParams['font.family'] = [font_prop.get_name()]
                    print(f"✅ 使用字体文件: {font_path}")
                    font_found = True
                    break
            except Exception as e:
                continue
    
    # 最后的备选方案
    if not font_found:
        print("⚠️ 未找到合适的中文字体，使用默认设置")
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    
    # 设置其他字体参数
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.weight'] = 'normal'
    
    # 测试中文显示
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试中文', ha='center', va='center')
        plt.close(fig)
        print("✅ 中文字体测试通过")
    except Exception as e:
        print(f"⚠️ 中文字体测试失败: {e}")
        # 强制设置为支持中文的字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

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

def create_grammatical_superposition_comparison(data):
    """创建语法叠加强度对比图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('语法叠加强度对比分析 - 受限制 vs 无限制版本', fontsize=16, fontweight='bold', 
                fontproperties='Arial Unicode MS')
    
    # 1. 受限制版本 - 新聞標題对比
    ai_restricted_title = data['ai_restricted']['新聞標題']['grammatical_superposition']['mean']
    journalist_restricted_title = data['journalist_restricted']['新聞標題']['grammatical_superposition']['mean']
    
    ax1.bar(['AI新闻', '记者新闻'], [ai_restricted_title, journalist_restricted_title], 
           color=['#ff6b6b', '#4ecdc4'], alpha=0.8)
    ax1.set_title('受限制版本 - 新聞標題', fontweight='bold')
    ax1.set_ylabel('语法叠加强度')
    ax1.set_ylim(0, 1.2)
    for i, v in enumerate([ai_restricted_title, journalist_restricted_title]):
        ax1.text(i, v + 0.02, f'{v:.6f}', ha='center', fontweight='bold')
    
    # 2. 无限制版本 - 新聞標題对比
    ai_unrestricted_title = data['ai_unrestricted']['新聞標題']['grammatical_superposition']['mean']
    journalist_unrestricted_title = data['journalist_unrestricted']['新聞標題']['grammatical_superposition']['mean']
    
    ax2.bar(['AI新闻', '记者新闻'], [ai_unrestricted_title, journalist_unrestricted_title], 
           color=['#ff6b6b', '#4ecdc4'], alpha=0.8)
    ax2.set_title('无限制版本 - 新聞標題', fontweight='bold')
    ax2.set_ylabel('语法叠加强度')
    ax2.set_ylim(0, 4.0)
    for i, v in enumerate([ai_unrestricted_title, journalist_unrestricted_title]):
        ax2.text(i, v + 0.05, f'{v:.6f}', ha='center', fontweight='bold')
    
    # 3. AI新闻各字段对比 (无限制版本)
    ai_fields = ['新聞標題', '影片對話', '影片描述']
    ai_values = [data['ai_unrestricted'][field]['grammatical_superposition']['mean'] for field in ai_fields]
    
    ax3.bar(ai_fields, ai_values, color='#ff6b6b', alpha=0.8)
    ax3.set_title('AI新闻各字段 - 语法叠加强度 (无限制)', fontweight='bold')
    ax3.set_ylabel('语法叠加强度')
    ax3.set_ylim(0, 4.0)
    ax3.tick_params(axis='x', rotation=45)
    for i, v in enumerate(ai_values):
        ax3.text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 4. 记者新闻各字段对比 (无限制版本)
    journalist_fields = ['新聞標題', '新聞內容']
    journalist_values = [data['journalist_unrestricted'][field]['grammatical_superposition']['mean'] for field in journalist_fields]
    
    ax4.bar(journalist_fields, journalist_values, color='#4ecdc4', alpha=0.8)
    ax4.set_title('记者新闻各字段 - 语法叠加强度 (无限制)', fontweight='bold')
    ax4.set_ylabel('语法叠加强度')
    ax4.set_ylim(0, 4.0)
    ax4.tick_params(axis='x', rotation=45)
    for i, v in enumerate(journalist_values):
        ax4.text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../visualizations/fair_comparison_grammatical_superposition.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_metrics_comparison(data):
    """创建综合指标对比热力图"""
    
    # 准备数据
    metrics = [
        'von_neumann_entropy', 'category_coherence', 'compositional_entanglement',
        'grammatical_superposition', 'semantic_interference', 'frame_competition',
        'multiple_reality_strength', 'frame_conflict_strength', 'semantic_ambiguity'
    ]
    
    metric_names_cn = [
        '冯纽曼熵', '类别一致性', '组合纠缠强度',
        '语法叠加强度', '语义干涉', '框架竞争强度',
        '多重现实强度', '框架冲突强度', '语义模糊度'
    ]
    
    # 创建对比矩阵 (新聞標題字段)
    comparison_data = []
    
    for metric in metrics:
        ai_restricted = data['ai_restricted']['新聞標題'][metric]['mean']
        ai_unrestricted = data['ai_unrestricted']['新聞標題'][metric]['mean']
        journalist_restricted = data['journalist_restricted']['新聞標題'][metric]['mean']
        journalist_unrestricted = data['journalist_unrestricted']['新聞標題'][metric]['mean']
        
        comparison_data.append([ai_restricted, ai_unrestricted, journalist_restricted, journalist_unrestricted])
    
    comparison_df = pd.DataFrame(comparison_data, 
                               index=metric_names_cn,
                               columns=['AI受限制', 'AI无限制', '记者受限制', '记者无限制'])
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(comparison_df, annot=True, fmt='.4f', cmap='RdYlBu_r', 
               cbar_kws={'label': '数值大小'}, ax=ax)
    ax.set_title('量子指标综合对比 - 新聞標題字段', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('数据类型和版本', fontweight='bold')
    ax.set_ylabel('量子指标', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../visualizations/fair_comparison_comprehensive_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_field_wise_radar_chart(data):
    """创建按字段的雷达图对比"""
    
    # 选择关键指标
    key_metrics = [
        'grammatical_superposition', 'multiple_reality_strength', 'semantic_ambiguity',
        'frame_competition', 'semantic_interference', 'von_neumann_entropy'
    ]
    
    key_metrics_cn = [
        '语法叠加强度', '多重现实强度', '语义模糊度',
        '框架竞争强度', '语义干涉', '冯纽曼熵'
    ]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    fig.suptitle('量子特征雷达图对比 (无限制版本)', fontsize=16, fontweight='bold')
    
    # 1. AI新闻 - 新聞標題
    ai_title_values = [data['ai_unrestricted']['新聞標題'][metric]['mean'] for metric in key_metrics]
    ai_title_values_norm = [v/max(ai_title_values) for v in ai_title_values]  # 归一化到[0,1]
    
    angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
    ai_title_values_norm += ai_title_values_norm[:1]  # 闭合
    angles += angles[:1]
    
    ax1.plot(angles, ai_title_values_norm, 'o-', linewidth=2, color='#ff6b6b', alpha=0.8)
    ax1.fill(angles, ai_title_values_norm, alpha=0.25, color='#ff6b6b')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(key_metrics_cn, fontsize=10)
    ax1.set_title('AI新闻 - 新聞標題', fontweight='bold', pad=20)
    ax1.set_ylim(0, 1)
    
    # 2. 记者新闻 - 新聞標題
    journalist_title_values = [data['journalist_unrestricted']['新聞標題'][metric]['mean'] for metric in key_metrics]
    journalist_title_values_norm = [v/max(journalist_title_values) for v in journalist_title_values]
    journalist_title_values_norm += journalist_title_values_norm[:1]
    
    ax2.plot(angles, journalist_title_values_norm, 'o-', linewidth=2, color='#4ecdc4', alpha=0.8)
    ax2.fill(angles, journalist_title_values_norm, alpha=0.25, color='#4ecdc4')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(key_metrics_cn, fontsize=10)
    ax2.set_title('记者新闻 - 新聞標題', fontweight='bold', pad=20)
    ax2.set_ylim(0, 1)
    
    # 3. AI新闻 - 影片對話
    ai_dialogue_values = [data['ai_unrestricted']['影片對話'][metric]['mean'] for metric in key_metrics]
    ai_dialogue_values_norm = [v/max(ai_dialogue_values) for v in ai_dialogue_values]
    ai_dialogue_values_norm += ai_dialogue_values_norm[:1]
    
    ax3.plot(angles, ai_dialogue_values_norm, 'o-', linewidth=2, color='#ff9f43', alpha=0.8)
    ax3.fill(angles, ai_dialogue_values_norm, alpha=0.25, color='#ff9f43')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(key_metrics_cn, fontsize=10)
    ax3.set_title('AI新闻 - 影片對話', fontweight='bold', pad=20)
    ax3.set_ylim(0, 1)
    
    # 4. 记者新闻 - 新聞內容
    journalist_content_values = [data['journalist_unrestricted']['新聞內容'][metric]['mean'] for metric in key_metrics]
    journalist_content_values_norm = [v/max(journalist_content_values) for v in journalist_content_values]
    journalist_content_values_norm += journalist_content_values_norm[:1]
    
    ax4.plot(angles, journalist_content_values_norm, 'o-', linewidth=2, color='#10ac84', alpha=0.8)
    ax4.fill(angles, journalist_content_values_norm, alpha=0.25, color='#10ac84')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(key_metrics_cn, fontsize=10)
    ax4.set_title('记者新闻 - 新聞內容', fontweight='bold', pad=20)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('../visualizations/fair_comparison_radar_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_difference_analysis(data):
    """创建差异分析图"""
    
    # 计算差异倍数
    metrics = [
        'von_neumann_entropy', 'category_coherence', 'compositional_entanglement',
        'grammatical_superposition', 'semantic_interference', 'frame_competition',
        'multiple_reality_strength', 'frame_conflict_strength', 'semantic_ambiguity'
    ]
    
    metric_names_cn = [
        '冯纽曼熵', '类别一致性', '组合纠缠强度',
        '语法叠加强度', '语义干涉', '框架竞争强度',
        '多重现实强度', '框架冲突强度', '语义模糊度'
    ]
    
    # 计算新聞標題字段的差异倍数
    differences_restricted = []
    differences_unrestricted = []
    
    for metric in metrics:
        # 受限制版本差异
        ai_restricted = data['ai_restricted']['新聞標題'][metric]['mean']
        journalist_restricted = data['journalist_restricted']['新聞標題'][metric]['mean']
        diff_restricted = ai_restricted / max(journalist_restricted, 1e-6)
        differences_restricted.append(diff_restricted)
        
        # 无限制版本差异
        ai_unrestricted = data['ai_unrestricted']['新聞標題'][metric]['mean']
        journalist_unrestricted = data['journalist_unrestricted']['新聞標題'][metric]['mean']
        diff_unrestricted = ai_unrestricted / max(journalist_unrestricted, 1e-6)
        differences_unrestricted.append(diff_unrestricted)
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('AI新闻 vs 记者新闻差异倍数分析 (新聞標題字段)', fontsize=16, fontweight='bold')
    
    x = np.arange(len(metric_names_cn))
    width = 0.35
    
    # 受限制版本差异
    bars1 = ax1.bar(x, differences_restricted, width, label='受限制版本', color='#ff6b6b', alpha=0.8)
    ax1.set_title('受限制版本差异倍数', fontweight='bold')
    ax1.set_ylabel('差异倍数 (AI/记者)')
    ax1.set_xlabel('量子指标')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names_cn, rotation=45, ha='right')
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, diff in zip(bars1, differences_restricted):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{diff:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 无限制版本差异
    bars2 = ax2.bar(x, differences_unrestricted, width, label='无限制版本', color='#4ecdc4', alpha=0.8)
    ax2.set_title('无限制版本差异倍数', fontweight='bold')
    ax2.set_ylabel('差异倍数 (AI/记者)')
    ax2.set_xlabel('量子指标')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names_cn, rotation=45, ha='right')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, diff in zip(bars2, differences_unrestricted):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{diff:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../visualizations/fair_comparison_difference_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(data):
    """创建汇总对比表"""
    
    print("\n" + "="*100)
    print("🎯 公平对比分析 - 详细数值对比表")
    print("="*100)
    
    # 新聞標題字段对比
    print("\n📊 新聞標題字段对比:")
    print("-"*80)
    print(f"{'指标':<20} {'AI受限制':<12} {'AI无限制':<12} {'记者受限制':<12} {'记者无限制':<12} {'差异倍数':<10}")
    print("-"*80)
    
    metrics = [
        ('语法叠加强度', 'grammatical_superposition'),
        ('多重现实强度', 'multiple_reality_strength'),
        ('语义模糊度', 'semantic_ambiguity'),
        ('框架竞争强度', 'frame_competition'),
        ('语义干涉', 'semantic_interference'),
        ('冯纽曼熵', 'von_neumann_entropy')
    ]
    
    for cn_name, metric in metrics:
        ai_r = data['ai_restricted']['新聞標題'][metric]['mean']
        ai_u = data['ai_unrestricted']['新聞標題'][metric]['mean']
        j_r = data['journalist_restricted']['新聞標題'][metric]['mean']
        j_u = data['journalist_unrestricted']['新聞標題'][metric]['mean']
        diff = ai_u / j_u
        
        print(f"{cn_name:<20} {ai_r:<12.6f} {ai_u:<12.6f} {j_r:<12.6f} {j_u:<12.6f} {diff:<10.3f}×")
    
    print("\n📈 关键发现:")
    print("-"*50)
    
    # 语法叠加强度分析
    ai_sup_u = data['ai_unrestricted']['新聞標題']['grammatical_superposition']['mean']
    j_sup_u = data['journalist_unrestricted']['新聞標題']['grammatical_superposition']['mean']
    sup_diff = ((ai_sup_u - j_sup_u) / j_sup_u) * 100
    
    print(f"• 语法叠加强度: AI比记者高 {sup_diff:.2f}%")
    print(f"• AI达到理论最大值的 {(ai_sup_u/4.0)*100:.1f}%")
    print(f"• 记者达到理论最大值的 {(j_sup_u/4.0)*100:.1f}%")
    
    # 多重现实强度分析
    ai_mr_u = data['ai_unrestricted']['新聞標題']['multiple_reality_strength']['mean']
    j_mr_u = data['journalist_unrestricted']['新聞標題']['multiple_reality_strength']['mean']
    mr_diff = ((ai_mr_u - j_mr_u) / j_mr_u) * 100
    
    print(f"• 多重现实强度: AI比记者高 {mr_diff:.2f}%")

def main():
    """主函数"""
    
    print("🎨 开始生成公平对比可视化...")
    
    # 设置中文字体
    setup_chinese_font()
    
    # 创建可视化目录
    Path('../visualizations').mkdir(exist_ok=True)
    
    # 加载数据
    data = load_summary_data()
    
    if not data:
        print("❌ 无法加载数据，退出程序")
        return
    
    # 生成各种可视化
    print("\n📊 生成语法叠加强度对比图...")
    create_grammatical_superposition_comparison(data)
    
    print("\n🔥 生成综合指标热力图...")
    create_comprehensive_metrics_comparison(data)
    
    print("\n📡 生成雷达图对比...")
    create_field_wise_radar_chart(data)
    
    print("\n📈 生成差异分析图...")
    create_difference_analysis(data)
    
    print("\n📋 生成汇总对比表...")
    create_summary_table(data)
    
    print(f"\n✅ 所有可视化已生成完成!")
    print(f"📁 保存位置: ../visualizations/")
    print(f"   • fair_comparison_grammatical_superposition.png")
    print(f"   • fair_comparison_comprehensive_heatmap.png") 
    print(f"   • fair_comparison_radar_charts.png")
    print(f"   • fair_comparison_difference_analysis.png")

if __name__ == "__main__":
    main()
