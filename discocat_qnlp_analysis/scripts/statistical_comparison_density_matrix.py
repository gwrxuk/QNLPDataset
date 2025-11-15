#!/usr/bin/env python3
"""
統計比較分析器 - 密度矩陣版本結果
計算 Cohen's d 和統計顯著性
"""

import pandas as pd
import numpy as np
import json
import scipy.stats as stats
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 導入統計比較分析器類
import sys
sys.path.append(str(Path(__file__).parent))
from statistical_comparison_analyzer import StatisticalComparisonAnalyzer

def main():
    """主函數 - 分析密度矩陣版本結果"""
    print("🚀 開始統計比較分析（密度矩陣版本）...")
    print("=" * 80)
    
    # 初始化分析器
    analyzer = StatisticalComparisonAnalyzer()
    
    # 讀取數據
    print("📊 讀取數據...")
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / '20251113_densityMatrix'
    
    ai_data_path = output_dir / 'results' / 'density_matrix_ai_analysis_results.csv'
    journalist_data_path = output_dir / 'results' / 'density_matrix_journalist_analysis_results.csv'
    
    if not ai_data_path.exists():
        print(f"❌ 找不到 AI 數據文件: {ai_data_path}")
        return
    
    if not journalist_data_path.exists():
        print(f"❌ 找不到記者數據文件: {journalist_data_path}")
        return
    
    ai_data = pd.read_csv(ai_data_path)
    journalist_data = pd.read_csv(journalist_data_path)
    
    print(f"✅ AI 數據: {len(ai_data)} 條記錄")
    print(f"✅ 記者數據: {len(journalist_data)} 條記錄")
    
    # 字段映射：將記者的「新聞內容」對應到多個 AI 字段進行比較
    # 映射 1: 新聞內容 -> 影片描述
    # 映射 2: 新聞內容 -> 影片對話 (需要創建額外的映射)
    field_mapping = {
        '新聞內容': '影片描述'  # 記者字段 -> AI 字段（主要映射）
    }
    
    # 創建記者數據的副本並應用字段映射
    journalist_data_mapped = journalist_data.copy()
    journalist_data_mapped['field'] = journalist_data_mapped['field'].replace(field_mapping)
    
    # 為「影片對話 vs 新聞內容」創建額外的映射數據
    # 創建第二個副本，將「新聞內容」映射到「影片對話」
    journalist_data_mapped_dialogue = journalist_data.copy()
    journalist_data_mapped_dialogue['field'] = journalist_data_mapped_dialogue['field'].replace({
        '新聞內容': '影片對話'  # 新聞內容 -> 影片對話
    })
    
    print(f"\n📋 字段映射:")
    for old_field, new_field in field_mapping.items():
        print(f"  {old_field} → {new_field}")
    
    print(f"\n📊 映射後的記者數據字段（映射1: 新聞內容→影片描述）: {sorted(journalist_data_mapped['field'].unique())}")
    print(f"📊 映射後的記者數據字段（映射2: 新聞內容→影片對話）: {sorted(journalist_data_mapped_dialogue['field'].unique())}")
    print(f"📊 AI 數據字段: {sorted(ai_data['field'].unique())}")
    
    # 整體比較（使用映射後的數據）
    print("\n📈 執行整體比較...")
    overall_results = analyzer.compare_groups(ai_data, journalist_data_mapped)
    
    # 按字段比較（使用映射後的數據）
    print("📈 執行按字段比較...")
    field_results = {}
    
    # 映射1: 新聞內容 -> 影片描述
    for field in ai_data['field'].unique():
        if field in journalist_data_mapped['field'].values:
            print(f"  - 比較字段: {field} (對應記者: {[k for k, v in field_mapping.items() if v == field] if field in field_mapping.values() else '新聞標題'})")
            field_results[field] = analyzer.compare_groups(ai_data, journalist_data_mapped, field=field)
    
    # 映射2: 新聞內容 -> 影片對話（特殊處理）
    dialogue_field = '影片對話'
    if dialogue_field in ai_data['field'].values and '新聞內容' in journalist_data['field'].values:
        print(f"  - 比較字段: {dialogue_field} (對應記者: 新聞內容)")
        field_results[f'{dialogue_field}_vs_新聞內容'] = analyzer.compare_groups(
            ai_data, journalist_data_mapped_dialogue, field=dialogue_field
        )
    
    # 生成報告
    print("\n📄 生成報告...")
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # 整體報告
    overall_report_path = reports_dir / 'statistical_comparison_report.md'
    analyzer.generate_report(overall_results, str(overall_report_path))
    
    # 按字段報告
    for field, results in field_results.items():
        # 處理特殊命名的字段（如 影片對話_vs_新聞內容）
        report_field_name = field.replace('_vs_', '_vs_').replace(' ', '_')
        field_report_path = reports_dir / f'statistical_comparison_report_{report_field_name}.md'
        analyzer.generate_report(results, str(field_report_path))
    
    # 保存 JSON 結果
    json_output = {
        'overall': overall_results,
        'by_field': field_results,
        'field_mapping': {
            '新聞內容→影片描述': '主要映射',
            '新聞內容→影片對話': '額外映射（AI 影片對話 vs 記者新聞內容）'
        },
        'analysis_method': 'density_matrix',
        'description': '使用密度矩陣 (ρ = |ψ⟩⟨ψ|) 計算 von Neumann 熵'
    }
    
    json_output_path = reports_dir / 'statistical_comparison_results.json'
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ JSON 結果已保存: {json_output_path}")
    
    # 顯示關鍵結果
    print("\n🔍 關鍵結果:")
    print("=" * 80)
    for metric, result in overall_results.items():
        print(f"\n{result['metric_name']}:")
        print(f"  Cohen's d = {result['cohens_d']:.4f} ({result['effect_size_interpretation']}效應)")
        print(f"  t 檢驗: p = {result['t_test']['p_value']:.4e}, 顯著 = {result['t_test']['significant']}")
        print(f"  變異性比率 = {result['variability_ratio']:.4f}")
    
    print("\n✅ 統計比較分析完成!")
    print(f"📄 報告已保存: {overall_report_path}")

if __name__ == "__main__":
    main()

