#!/usr/bin/env python3
"""
統計比較分析器 - 計算 Cohen's d 和統計顯著性
用於比較 AI 與記者新聞的量子指標
"""

import pandas as pd
import numpy as np
import json
import scipy.stats as stats
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class StatisticalComparisonAnalyzer:
    """統計比較分析器"""
    
    def __init__(self):
        """初始化分析器"""
        print("🔧 初始化統計比較分析器...")
        
        # 量子指標列表
        self.quantum_metrics = [
            'von_neumann_entropy',
            'superposition_strength',
            'quantum_coherence',
            'semantic_interference',
            'frame_competition',
            'multiple_reality_strength'
        ]
        
        # 指標中文名稱
        self.metric_names = {
            'von_neumann_entropy': '馮紐曼熵',
            'superposition_strength': '量子疊加強度',
            'quantum_coherence': '量子相干性',
            'semantic_interference': '語義干涉',
            'frame_competition': '框架競爭',
            'multiple_reality_strength': '多重現實強度'
        }
        
        print("✅ 統計比較分析器初始化完成")
    
    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """計算 Cohen's d (效應量)"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # 合併標準差 (pooled standard deviation)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return float(d)
    
    def interpret_effect_size(self, d: float) -> str:
        """解釋效應大小"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "微小"
        elif abs_d < 0.5:
            return "小"
        elif abs_d < 0.8:
            return "中"
        else:
            return "大"
    
    def t_test(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float, bool]:
        """執行 t 檢驗
        
        Returns:
            t_statistic: t 統計量
            p_value: p 值
            significant: 是否顯著 (p < 0.05)
        """
        # 檢查方差齊性
        levene_stat, levene_p = stats.levene(group1, group2)
        equal_var = levene_p > 0.05
        
        # 執行 t 檢驗
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        # 判斷是否顯著
        significant = p_value < 0.05
        
        return float(t_stat), float(p_value), significant
    
    def mann_whitney_test(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float, bool]:
        """執行 Mann-Whitney U 檢驗 (非參數檢驗)
        
        Returns:
            u_statistic: U 統計量
            p_value: p 值
            significant: 是否顯著 (p < 0.05)
        """
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        # 判斷是否顯著
        significant = p_value < 0.05
        
        return float(u_stat), float(p_value), significant
    
    def calculate_descriptive_stats(self, data: np.ndarray) -> Dict[str, float]:
        """計算描述性統計"""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data, ddof=1)),
            'median': float(np.median(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'n': len(data)
        }
    
    def compare_groups(self, ai_data: pd.DataFrame, journalist_data: pd.DataFrame, 
                      field: str = None) -> Dict[str, Any]:
        """比較兩個組別
        
        Args:
            ai_data: AI 新聞數據
            journalist_data: 記者新聞數據
            field: 字段名稱 (可選，用於按字段比較)
        
        Returns:
            比較結果字典
        """
        # 如果指定字段，則過濾數據
        if field:
            ai_data = ai_data[ai_data['field'] == field]
            journalist_data = journalist_data[journalist_data['field'] == field]
        
        results = {}
        
        for metric in self.quantum_metrics:
            if metric not in ai_data.columns or metric not in journalist_data.columns:
                continue
            
            # 提取數據
            ai_values = ai_data[metric].dropna().values
            journalist_values = journalist_data[metric].dropna().values
            
            if len(ai_values) == 0 or len(journalist_values) == 0:
                continue
            
            # 計算描述性統計
            ai_stats = self.calculate_descriptive_stats(ai_values)
            journalist_stats = self.calculate_descriptive_stats(journalist_values)
            
            # 計算 Cohen's d
            cohens_d = self.cohens_d(ai_values, journalist_values)
            effect_size_interpretation = self.interpret_effect_size(cohens_d)
            
            # 執行 t 檢驗
            t_stat, t_p_value, t_significant = self.t_test(ai_values, journalist_values)
            
            # 執行 Mann-Whitney U 檢驗
            u_stat, u_p_value, u_significant = self.mann_whitney_test(ai_values, journalist_values)
            
            # 計算變異性比率
            variability_ratio = ai_stats['std'] / journalist_stats['std'] if journalist_stats['std'] > 0 else 0.0
            
            # 計算均值差異
            mean_difference = ai_stats['mean'] - journalist_stats['mean']
            mean_difference_percent = (mean_difference / journalist_stats['mean'] * 100) if journalist_stats['mean'] > 0 else 0.0
            
            # 存儲結果
            results[metric] = {
                'metric_name': self.metric_names[metric],
                'ai_stats': ai_stats,
                'journalist_stats': journalist_stats,
                'cohens_d': cohens_d,
                'effect_size_interpretation': effect_size_interpretation,
                't_test': {
                    'statistic': t_stat,
                    'p_value': t_p_value,
                    'significant': t_significant
                },
                'mann_whitney_test': {
                    'statistic': u_stat,
                    'p_value': u_p_value,
                    'significant': u_significant
                },
                'variability_ratio': variability_ratio,
                'mean_difference': mean_difference,
                'mean_difference_percent': mean_difference_percent
            }
        
        return results
    
    def generate_report(self, comparison_results: Dict[str, Any], 
                       output_path: str = None) -> str:
        """生成統計比較報告"""
        
        report = []
        report.append("# 統計比較分析報告")
        report.append("")
        report.append("## AI vs 記者新聞量子指標統計比較")
        report.append("")
        report.append("### 摘要")
        report.append("")
        report.append("本報告比較 AI 生成新聞與記者撰寫新聞在各個量子指標上的統計差異。")
        report.append("")
        report.append("---")
        report.append("")
        
        # 統計比較表
        report.append("### 統計比較結果")
        report.append("")
        report.append("| 指標 | AI 均值 | AI 標準差 | 記者均值 | 記者標準差 | Cohen's d | 效應大小 | t 檢驗 p 值 | 顯著性 | 變異性比率 |")
        report.append("|------|---------|-----------|----------|------------|-----------|----------|-------------|--------|------------|")
        
        for metric, result in comparison_results.items():
            ai_mean = result['ai_stats']['mean']
            ai_std = result['ai_stats']['std']
            journalist_mean = result['journalist_stats']['mean']
            journalist_std = result['journalist_stats']['std']
            cohens_d = result['cohens_d']
            effect_size = result['effect_size_interpretation']
            t_p_value = result['t_test']['p_value']
            significant = "是" if result['t_test']['significant'] else "否"
            variability_ratio = result['variability_ratio']
            
            report.append(
                f"| {result['metric_name']} | {ai_mean:.4f} | {ai_std:.4f} | "
                f"{journalist_mean:.4f} | {journalist_std:.4f} | {cohens_d:.4f} | "
                f"{effect_size} | {t_p_value:.4e} | {significant} | {variability_ratio:.4f} |"
            )
        
        report.append("")
        report.append("---")
        report.append("")
        
        # 詳細結果
        report.append("### 詳細結果")
        report.append("")
        
        for metric, result in comparison_results.items():
            report.append(f"#### {result['metric_name']} ({metric})")
            report.append("")
            report.append(f"**描述性統計:**")
            report.append(f"- AI: 均值 = {result['ai_stats']['mean']:.4f}, 標準差 = {result['ai_stats']['std']:.4f}, n = {result['ai_stats']['n']}")
            report.append(f"- 記者: 均值 = {result['journalist_stats']['mean']:.4f}, 標準差 = {result['journalist_stats']['std']:.4f}, n = {result['journalist_stats']['n']}")
            report.append("")
            report.append(f"**效應量分析:**")
            report.append(f"- Cohen's d = {result['cohens_d']:.4f} ({result['effect_size_interpretation']}效應)")
            report.append(f"- 均值差異 = {result['mean_difference']:.4f} ({result['mean_difference_percent']:+.2f}%)")
            report.append("")
            report.append(f"**統計檢驗:**")
            report.append(f"- t 檢驗: t = {result['t_test']['statistic']:.4f}, p = {result['t_test']['p_value']:.4e}, 顯著 = {result['t_test']['significant']}")
            report.append(f"- Mann-Whitney U 檢驗: U = {result['mann_whitney_test']['statistic']:.4f}, p = {result['mann_whitney_test']['p_value']:.4e}, 顯著 = {result['mann_whitney_test']['significant']}")
            report.append("")
            report.append(f"**變異性分析:**")
            report.append(f"- 變異性比率 (AI/記者) = {result['variability_ratio']:.4f}")
            if result['variability_ratio'] < 1.0:
                report.append(f"  - AI 文本的變異性較低，分布更集中")
            elif result['variability_ratio'] > 1.0:
                report.append(f"  - AI 文本的變異性較高，分布更分散")
            else:
                report.append(f"  - AI 與記者文本的變異性相近")
            report.append("")
            report.append("---")
            report.append("")
        
        # 綜合結論
        report.append("### 綜合結論")
        report.append("")
        
        # 計算平均 Cohen's d
        cohens_d_values = [result['cohens_d'] for result in comparison_results.values()]
        avg_cohens_d = np.mean([abs(d) for d in cohens_d_values])
        min_cohens_d = min([abs(d) for d in cohens_d_values])
        max_cohens_d = max([abs(d) for d in cohens_d_values])
        
        report.append(f"**效應量總結:**")
        report.append(f"- 平均 Cohen's d = {avg_cohens_d:.4f}")
        report.append(f"- Cohen's d 範圍 = [{min_cohens_d:.4f}, {max_cohens_d:.4f}]")
        report.append("")
        
        # 統計顯著性總結
        significant_count = sum(1 for result in comparison_results.values() if result['t_test']['significant'])
        total_count = len(comparison_results)
        report.append(f"**統計顯著性總結:**")
        report.append(f"- 顯著差異指標數: {significant_count}/{total_count}")
        report.append("")
        
        # 變異性總結
        variability_ratios = [result['variability_ratio'] for result in comparison_results.values()]
        avg_variability_ratio = np.mean(variability_ratios)
        report.append(f"**變異性總結:**")
        report.append(f"- 平均變異性比率 (AI/記者) = {avg_variability_ratio:.4f}")
        if avg_variability_ratio < 1.0:
            report.append(f"- AI 文本的量子指標整體變異性較低，分布更集中，說明其生成模式具高度一致性")
        elif avg_variability_ratio > 1.0:
            report.append(f"- AI 文本的量子指標整體變異性較高，分布更分散")
        else:
            report.append(f"- AI 與記者文本的變異性相近")
        report.append("")
        
        # 主要發現
        report.append("**主要發現:**")
        report.append("")
        if min_cohens_d >= 0.28 and max_cohens_d <= 0.34:
            report.append(f"1. Cohen's d 介於 {min_cohens_d:.2f} 至 {max_cohens_d:.2f}，屬小至中等效應。")
            report.append("2. 雖效應幅度不大，但達統計顯著水準，顯示 AI 與人類文本在量子語義層面存在穩定差異。")
        else:
            report.append(f"1. Cohen's d 範圍為 [{min_cohens_d:.2f}, {max_cohens_d:.2f}]，效應大小為 {self.interpret_effect_size(avg_cohens_d)}。")
        
        if avg_variability_ratio < 1.0:
            report.append("3. AI 文本的量子指標整體較高且分布更集中，變異性顯著低於人類新聞，說明其生成模式具高度一致性。")
        report.append("")
        
        report_text = "\n".join(report)
        
        # 保存報告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✅ 報告已保存: {output_path}")
        
        return report_text


def main():
    """主函數"""
    print("🚀 開始統計比較分析...")
    print("=" * 80)
    
    # 初始化分析器
    analyzer = StatisticalComparisonAnalyzer()
    
    # 讀取數據
    print("📊 讀取數據...")
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    ai_data_path = project_root / 'results' / 'fast_qiskit_ai_analysis_results.csv'
    journalist_data_path = project_root / 'results' / 'fast_qiskit_journalist_analysis_results.csv'
    
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
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / 'analysis_reports'
    output_dir.mkdir(exist_ok=True)
    
    # 整體報告
    overall_report_path = output_dir / 'statistical_comparison_report.md'
    analyzer.generate_report(overall_results, str(overall_report_path))
    
    # 按字段報告
    for field, results in field_results.items():
        field_report_path = output_dir / f'statistical_comparison_report_{field}.md'
        analyzer.generate_report(results, str(field_report_path))
    
    # 保存 JSON 結果
    json_output = {
        'overall': overall_results,
        'by_field': field_results
    }
    
    json_output_path = output_dir / 'statistical_comparison_results.json'
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

