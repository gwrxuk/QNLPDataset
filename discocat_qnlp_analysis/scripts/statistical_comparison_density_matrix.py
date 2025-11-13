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
    
    # 整體比較
    print("\n📈 執行整體比較...")
    overall_results = analyzer.compare_groups(ai_data, journalist_data)
    
    # 按字段比較
    print("📈 執行按字段比較...")
    field_results = {}
    for field in ai_data['field'].unique():
        if field in journalist_data['field'].values:
            print(f"  - 比較字段: {field}")
            field_results[field] = analyzer.compare_groups(ai_data, journalist_data, field=field)
    
    # 生成報告
    print("\n📄 生成報告...")
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # 整體報告
    overall_report_path = reports_dir / 'statistical_comparison_report.md'
    analyzer.generate_report(overall_results, str(overall_report_path))
    
    # 按字段報告
    for field, results in field_results.items():
        field_report_path = reports_dir / f'statistical_comparison_report_{field}.md'
        analyzer.generate_report(results, str(field_report_path))
    
    # 保存 JSON 結果
    json_output = {
        'overall': overall_results,
        'by_field': field_results,
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

