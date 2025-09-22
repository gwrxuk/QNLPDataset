#!/usr/bin/env python3
"""
綜合視覺化工具
Comprehensive Visualization Tool for QNLP Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import json
import platform
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_fonts():
    """設置中文字體"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        chinese_fonts = [
            'PingFang SC', 'Heiti SC', 'STHeiti', 
            'Arial Unicode MS', 'Hiragino Sans GB'
        ]
    elif system == "Windows":
        chinese_fonts = [
            'Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong'
        ]
    else:  # Linux
        chinese_fonts = [
            'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans'
        ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_font = None
    
    for font in chinese_fonts:
        if font in available_fonts:
            chinese_font = font
            break
    
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
        print(f"✅ 使用中文字體: {chinese_font}")
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("⚠️  使用默認字體")
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    return chinese_font

class ComprehensiveVisualizer:
    """綜合視覺化類"""
    
    def __init__(self):
        self.chinese_font = setup_chinese_fonts()
        self.colors = {
            'jieba': '#FF6B6B',
            'chatgpt': '#4ECDC4',
            'comparison': '#45B7D1',
            'accent': '#96CEB4',
            'neutral': '#FECA57'
        }
    
    def load_analysis_results(self) -> Optional[Dict]:
        """載入分析結果"""
        try:
            with open('../results/qnlp_comparative_analysis.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ 未找到分析結果檔案，請先運行QNLP分析")
            return None
    
    def create_quantum_metrics_comparison(self, analysis_results: Dict):
        """創建量子指標比較圖"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('jieba vs ChatGPT 量子指標比較分析', fontsize=16, fontweight='bold')
        
        jieba_results = analysis_results['jieba_analysis']['field_results']
        chatgpt_results = analysis_results['chatgpt_analysis']['field_results']
        
        # 準備數據
        fields = list(jieba_results.keys())
        metrics = ['avg_coherence', 'avg_interference', 'avg_entropy', 'avg_superposition']
        metric_names = ['量子連貫性', '量子干涉', '敘事複雜度', '疊加強度']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            jieba_values = [jieba_results[field][metric] for field in fields]
            chatgpt_values = [chatgpt_results[field][metric] for field in fields]
            
            x = np.arange(len(fields))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, jieba_values, width, 
                          label='jieba', color=self.colors['jieba'], alpha=0.8)
            bars2 = ax.bar(x + width/2, chatgpt_values, width, 
                          label='ChatGPT', color=self.colors['chatgpt'], alpha=0.8)
            
            ax.set_xlabel('文本欄位')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name}比較')
            ax.set_xticks(x)
            ax.set_xticklabels(fields)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加數值標籤
            for bar, value in zip(bars1, jieba_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            for bar, value in zip(bars2, chatgpt_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('../visualizations/quantum_metrics_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("💾 量子指標比較圖已保存: ../visualizations/quantum_metrics_comparison.png")
    
    def create_word_count_analysis(self, analysis_results: Dict):
        """創建詞數分析圖"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('斷詞方法詞數統計分析', fontsize=16, fontweight='bold')
        
        jieba_results = analysis_results['jieba_analysis']['field_results']
        chatgpt_results = analysis_results['chatgpt_analysis']['field_results']
        
        fields = list(jieba_results.keys())
        
        # 1. 平均詞數比較
        ax1 = axes[0]
        jieba_word_counts = [jieba_results[field]['avg_word_count'] for field in fields]
        chatgpt_word_counts = [chatgpt_results[field]['avg_word_count'] for field in fields]
        
        x = np.arange(len(fields))
        width = 0.35
        
        ax1.bar(x - width/2, jieba_word_counts, width, 
               label='jieba', color=self.colors['jieba'], alpha=0.8)
        ax1.bar(x + width/2, chatgpt_word_counts, width, 
               label='ChatGPT', color=self.colors['chatgpt'], alpha=0.8)
        
        ax1.set_xlabel('文本欄位')
        ax1.set_ylabel('平均詞數')
        ax1.set_title('平均詞數比較')
        ax1.set_xticks(x)
        ax1.set_xticklabels(fields)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 詞數差異
        ax2 = axes[1]
        word_count_diff = [c - j for c, j in zip(chatgpt_word_counts, jieba_word_counts)]
        colors = [self.colors['chatgpt'] if d > 0 else self.colors['jieba'] for d in word_count_diff]
        
        bars = ax2.bar(fields, word_count_diff, color=colors, alpha=0.8)
        ax2.set_xlabel('文本欄位')
        ax2.set_ylabel('詞數差異 (ChatGPT - jieba)')
        ax2.set_title('詞數差異分析')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, word_count_diff):
            ax2.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (1 if val >= 0 else -2),
                    f'{val:+.1f}', ha='center', 
                    va='bottom' if val >= 0 else 'top', fontsize=10)
        
        # 3. 語義複雜度比較
        ax3 = axes[2]
        jieba_complexity = [jieba_results[field]['avg_semantic_complexity'] for field in fields]
        chatgpt_complexity = [chatgpt_results[field]['avg_semantic_complexity'] for field in fields]
        
        ax3.bar(x - width/2, jieba_complexity, width, 
               label='jieba', color=self.colors['jieba'], alpha=0.8)
        ax3.bar(x + width/2, chatgpt_complexity, width, 
               label='ChatGPT', color=self.colors['chatgpt'], alpha=0.8)
        
        ax3.set_xlabel('文本欄位')
        ax3.set_ylabel('語義複雜度')
        ax3.set_title('語義複雜度比較')
        ax3.set_xticks(x)
        ax3.set_xticklabels(fields)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../visualizations/word_count_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("💾 詞數分析圖已保存: ../visualizations/word_count_analysis.png")
    
    def create_radar_chart_comparison(self, analysis_results: Dict):
        """創建雷達圖比較"""
        fig, axes = plt.subplots(1, len(analysis_results['jieba_analysis']['field_results']), 
                                figsize=(6 * len(analysis_results['jieba_analysis']['field_results']), 6))
        fig.suptitle('各欄位量子指標雷達圖比較', fontsize=16, fontweight='bold')
        
        if len(analysis_results['jieba_analysis']['field_results']) == 1:
            axes = [axes]
        
        jieba_results = analysis_results['jieba_analysis']['field_results']
        chatgpt_results = analysis_results['chatgpt_analysis']['field_results']
        
        metrics = ['avg_coherence', 'avg_interference', 'avg_entropy', 'avg_superposition']
        metric_labels = ['連貫性', '干涉', '複雜度', '疊加']
        
        for idx, field in enumerate(jieba_results.keys()):
            ax = axes[idx]
            
            # 準備雷達圖數據
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 閉合圓圈
            
            jieba_values = [jieba_results[field][metric] for metric in metrics]
            chatgpt_values = [chatgpt_results[field][metric] for metric in metrics]
            
            jieba_values += jieba_values[:1]
            chatgpt_values += chatgpt_values[:1]
            
            # 創建雷達圖
            ax = plt.subplot(1, len(jieba_results), idx + 1, projection='polar')
            
            ax.plot(angles, jieba_values, 'o-', linewidth=2, 
                   label='jieba', color=self.colors['jieba'])
            ax.fill(angles, jieba_values, alpha=0.25, color=self.colors['jieba'])
            
            ax.plot(angles, chatgpt_values, 'o-', linewidth=2, 
                   label='ChatGPT', color=self.colors['chatgpt'])
            ax.fill(angles, chatgpt_values, alpha=0.25, color=self.colors['chatgpt'])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1)
            ax.set_title(f'{field}', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('../visualizations/radar_chart_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("💾 雷達圖比較已保存: ../visualizations/radar_chart_comparison.png")
    
    def create_insights_summary(self, analysis_results: Dict):
        """創建洞察摘要圖"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('QNLP分析洞察摘要', fontsize=16, fontweight='bold')
        
        comparison = analysis_results['comparison']
        
        # 1. 方法差異總覽
        ax1 = axes[0, 0]
        method_comp = comparison['method_comparison']
        metrics = ['coherence_diff', 'interference_diff', 'entropy_diff', 'superposition_diff']
        metric_names = ['連貫性差異', '干涉差異', '複雜度差異', '疊加差異']
        values = [method_comp[metric] for metric in metrics]
        colors = [self.colors['chatgpt'] if v > 0 else self.colors['jieba'] for v in values]
        
        bars = ax1.barh(metric_names, values, color=colors, alpha=0.8)
        ax1.set_xlabel('差異值 (ChatGPT - jieba)')
        ax1.set_title('整體量子指標差異')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, values):
            ax1.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                    f'{val:+.3f}', ha='left' if val >= 0 else 'right', va='center')
        
        # 2. 欄位間變異性
        ax2 = axes[0, 1]
        jieba_results = analysis_results['jieba_analysis']['field_results']
        chatgpt_results = analysis_results['chatgpt_analysis']['field_results']
        
        fields = list(jieba_results.keys())
        jieba_coherences = [jieba_results[field]['avg_coherence'] for field in fields]
        chatgpt_coherences = [chatgpt_results[field]['avg_coherence'] for field in fields]
        
        ax2.scatter(jieba_coherences, chatgpt_coherences, 
                   s=100, alpha=0.7, color=self.colors['comparison'])
        
        # 添加對角線
        min_val = min(min(jieba_coherences), min(chatgpt_coherences))
        max_val = max(max(jieba_coherences), max(chatgpt_coherences))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax2.set_xlabel('jieba 量子連貫性')
        ax2.set_ylabel('ChatGPT 量子連貫性')
        ax2.set_title('量子連貫性散點比較')
        ax2.grid(True, alpha=0.3)
        
        # 標註欄位名稱
        for i, field in enumerate(fields):
            ax2.annotate(field, (jieba_coherences[i], chatgpt_coherences[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 3. 洞察文字摘要
        ax3 = axes[1, :]
        ax3.axis('off')
        
        insights = comparison.get('insights', [])
        if insights:
            insight_text = "🔍 主要發現:\n\n"
            for i, insight in enumerate(insights, 1):
                insight_text += f"{i}. {insight}\n\n"
        else:
            insight_text = "暫無特殊洞察發現"
        
        ax3.text(0.05, 0.95, insight_text, transform=ax3.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['accent'], alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('../visualizations/insights_summary.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("💾 洞察摘要圖已保存: ../visualizations/insights_summary.png")
    
    def create_comprehensive_report(self, analysis_results: Dict):
        """創建綜合報告"""
        print("\n📊 生成綜合視覺化報告...")
        
        # 創建所有視覺化
        self.create_quantum_metrics_comparison(analysis_results)
        self.create_word_count_analysis(analysis_results)
        self.create_radar_chart_comparison(analysis_results)
        self.create_insights_summary(analysis_results)
        
        # 生成統計摘要
        self.generate_statistical_summary(analysis_results)
        
        print("\n🎉 綜合視覺化報告生成完成！")
        print("📁 所有圖表已保存在 ../visualizations/ 目錄下")
    
    def generate_statistical_summary(self, analysis_results: Dict):
        """生成統計摘要"""
        summary = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'methods_compared': ['jieba', 'ChatGPT'],
            'fields_analyzed': list(analysis_results['jieba_analysis']['field_results'].keys()),
            'key_findings': {},
            'recommendations': []
        }
        
        # 提取關鍵發現
        comparison = analysis_results['comparison']
        if 'method_comparison' in comparison:
            mc = comparison['method_comparison']
            summary['key_findings'] = {
                'coherence_advantage': 'ChatGPT' if mc['coherence_diff'] > 0 else 'jieba',
                'complexity_advantage': 'ChatGPT' if mc['entropy_diff'] > 0 else 'jieba',
                'superposition_advantage': 'ChatGPT' if mc['superposition_diff'] > 0 else 'jieba',
                'magnitude_differences': {
                    'coherence': abs(mc['coherence_diff']),
                    'entropy': abs(mc['entropy_diff']),
                    'superposition': abs(mc['superposition_diff'])
                }
            }
        
        # 生成建議
        if summary['key_findings']:
            kf = summary['key_findings']
            if kf['coherence_advantage'] == 'ChatGPT':
                summary['recommendations'].append("ChatGPT在語義連貫性方面表現更佳，適合需要高度語義一致性的分析")
            if kf['complexity_advantage'] == 'ChatGPT':
                summary['recommendations'].append("ChatGPT能捕捉更複雜的敘事結構，適合深度語義分析")
            if kf['superposition_advantage'] == 'ChatGPT':
                summary['recommendations'].append("ChatGPT更能體現多重敘事現象，適合量子語言學研究")
        
        # 保存摘要
        with open('../results/statistical_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print("💾 統計摘要已保存: ../results/statistical_summary.json")

def main():
    """主函數"""
    print("🎨 綜合視覺化分析工具")
    print("=" * 40)
    
    visualizer = ComprehensiveVisualizer()
    
    # 載入分析結果
    analysis_results = visualizer.load_analysis_results()
    if not analysis_results:
        return
    
    # 創建綜合報告
    visualizer.create_comprehensive_report(analysis_results)

if __name__ == "__main__":
    main()
