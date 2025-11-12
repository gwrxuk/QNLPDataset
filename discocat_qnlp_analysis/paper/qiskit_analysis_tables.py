#!/usr/bin/env python3
"""
Qiskit Analysis Tables and Visualizations Generator
Generates comprehensive tables and charts for the 298 articles analyzed by Qiskit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib.font_manager as fm
from collections import Counter
import json

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class QiskitAnalysisTables:
    def __init__(self):
        """Initialize the analysis tables generator"""
        self.ai_data = None
        self.journalist_data = None
        self.load_data()
    
    def load_data(self):
        """Load the Qiskit analysis results"""
        try:
            # Load AI-generated data
            self.ai_data = pd.read_csv('../results/fast_qiskit_ai_analysis_results.csv')
            
            # Load journalist-written data
            self.journalist_data = pd.read_csv('../results/fast_qiskit_journalist_analysis_results.csv')
            
            print(f"✅ Loaded {len(self.ai_data)} AI articles and {len(self.journalist_data)} journalist articles")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def create_overall_statistics_table(self):
        """Create Table 1: Overall Quantum NLP Statistics"""
        
        # Combine all data for overall statistics
        all_data = pd.concat([self.ai_data, self.journalist_data], ignore_index=True)
        
        # Calculate statistics by field
        fields = ['新聞標題', '影片對話', '影片描述', '新聞內容']
        
        table_data = []
        
        for field in fields:
            field_data = all_data[all_data['field'] == field]
            if len(field_data) == 0:
                continue
                
            stats = {
                'Field': field,
                'Count': len(field_data),
                'Multiple Reality (Mean)': f"{field_data['multiple_reality_strength'].mean():.4f}",
                'Multiple Reality (Std)': f"{field_data['multiple_reality_strength'].std():.4f}",
                'Frame Competition (Mean)': f"{field_data['frame_competition'].mean():.4f}",
                'Frame Competition (Std)': f"{field_data['frame_competition'].std():.4f}",
                'Von Neumann Entropy (Mean)': f"{field_data['von_neumann_entropy'].mean():.4f}",
                'Von Neumann Entropy (Std)': f"{field_data['von_neumann_entropy'].std():.4f}",
                'Semantic Interference (Mean)': f"{field_data['semantic_interference'].mean():.4f}",
                'Semantic Interference (Std)': f"{field_data['semantic_interference'].std():.4f}",
                'Word Count (Mean)': f"{field_data['word_count'].mean():.1f}",
                'Word Count (Std)': f"{field_data['word_count'].std():.1f}"
            }
            table_data.append(stats)
        
        # Create DataFrame and save as CSV
        df_table = pd.DataFrame(table_data)
        df_table.to_csv('table1_overall_quantum_statistics.csv', index=False, encoding='utf-8-sig')
        
        # Create LaTeX table
        latex_table = df_table.to_latex(index=False, escape=False)
        with open('table1_overall_quantum_statistics.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print("✅ Created Table 1: Overall Quantum NLP Statistics")
        return df_table
    
    def create_ai_vs_journalist_comparison_table(self):
        """Create Table 2: AI vs Journalist Comparison"""
        
        comparison_data = []
        
        # Calculate statistics for each source
        for source_name, data in [('AI Generated', self.ai_data), ('Journalist Written', self.journalist_data)]:
            stats = {
                'Source': source_name,
                'Total Articles': len(data),
                'Multiple Reality (Mean)': f"{data['multiple_reality_strength'].mean():.4f}",
                'Multiple Reality (Std)': f"{data['multiple_reality_strength'].std():.4f}",
                'Frame Competition (Mean)': f"{data['frame_competition'].mean():.4f}",
                'Frame Competition (Std)': f"{data['frame_competition'].std():.4f}",
                'Frame Conflict (Mean)': f"{data['semantic_interference'].mean():.4f}",
                'Frame Conflict (Std)': f"{data['semantic_interference'].std():.4f}",
                'Quantum Coherence (Mean)': f"{data['quantum_coherence'].mean():.4f}",
                'Quantum Coherence (Std)': f"{data['quantum_coherence'].std():.4f}",
                'Avg Word Count': f"{data['word_count'].mean():.1f}",
                'Avg Unique Words': f"{data['unique_words'].mean():.1f}",
                'Categorical Diversity (Mean)': f"{data['categorical_diversity'].mean():.4f}",
                'Categorical Diversity (Std)': f"{data['categorical_diversity'].std():.4f}"
            }
            comparison_data.append(stats)
        
        # Create DataFrame and save
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv('table2_ai_vs_journalist_comparison.csv', index=False, encoding='utf-8-sig')
        
        # Create LaTeX table
        latex_table = df_comparison.to_latex(index=False, escape=False)
        with open('table2_ai_vs_journalist_comparison.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print("✅ Created Table 2: AI vs Journalist Comparison")
        return df_comparison
    
    def create_frame_competition_analysis_table(self):
        """Create Table 3: Frame Competition Analysis by Section"""
        
        # Analyze frame competition by field and source
        analysis_data = []
        
        for source_name, data in [('AI Generated', self.ai_data), ('Journalist Written', self.journalist_data)]:
            for field in ['新聞標題', '影片對話', '影片描述', '新聞內容']:
                field_data = data[data['field'] == field]
                if len(field_data) == 0:
                    continue
                
                # Calculate frame competition statistics
                frame_comp = field_data['frame_competition']
                semantic_int = field_data['semantic_interference']
                
                stats = {
                    'Source': source_name,
                    'Field': field,
                    'Count': len(field_data),
                    'Frame Competition (Mean)': f"{frame_comp.mean():.4f}",
                    'Frame Competition (Std)': f"{frame_comp.std():.4f}",
                    'Frame Competition (Min)': f"{frame_comp.min():.4f}",
                    'Frame Competition (Max)': f"{frame_comp.max():.4f}",
                    'Semantic Interference (Mean)': f"{semantic_int.mean():.4f}",
                    'Semantic Interference (Std)': f"{semantic_int.std():.4f}",
                    'High Competition (>0.9)': f"{(frame_comp > 0.9).sum()}/{len(field_data)} ({((frame_comp > 0.9).sum()/len(field_data)*100):.1f}%)",
                    'Low Conflict (<0.1)': f"{(semantic_int < 0.1).sum()}/{len(field_data)} ({((semantic_int < 0.1).sum()/len(field_data)*100):.1f}%)"
                }
                analysis_data.append(stats)
        
        # Create DataFrame and save
        df_frame = pd.DataFrame(analysis_data)
        df_frame.to_csv('table3_frame_competition_analysis.csv', index=False, encoding='utf-8-sig')
        
        # Create LaTeX table
        latex_table = df_frame.to_latex(index=False, escape=False)
        with open('table3_frame_competition_analysis.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print("✅ Created Table 3: Frame Competition Analysis")
        return df_frame
    
    def create_emotional_tone_analysis_table(self):
        """Create Table 4: Emotional Tone Analysis"""
        
        # Note: The current data doesn't have explicit emotional tone metrics
        # We'll create a table based on available metrics that relate to emotional intensity
        
        emotional_data = []
        
        for source_name, data in [('AI Generated', self.ai_data), ('Journalist Written', self.journalist_data)]:
            for field in ['新聞標題', '影片對話', '影片描述', '新聞內容']:
                field_data = data[data['field'] == field]
                if len(field_data) == 0:
                    continue
                
                # Use semantic interference as a proxy for emotional intensity
                emotional_intensity = field_data['semantic_interference']
                
                # Categorize emotional intensity
                high_emotion = (emotional_intensity > emotional_intensity.quantile(0.75)).sum()
                medium_emotion = ((emotional_intensity >= emotional_intensity.quantile(0.25)) & 
                                (emotional_intensity <= emotional_intensity.quantile(0.75))).sum()
                low_emotion = (emotional_intensity < emotional_intensity.quantile(0.25)).sum()
                
                stats = {
                    'Source': source_name,
                    'Field': field,
                    'Count': len(field_data),
                    'Emotional Intensity (Mean)': f"{emotional_intensity.mean():.4f}",
                    'Emotional Intensity (Std)': f"{emotional_intensity.std():.4f}",
                    'High Emotion (%)': f"{high_emotion/len(field_data)*100:.1f}%",
                    'Medium Emotion (%)': f"{medium_emotion/len(field_data)*100:.1f}%",
                    'Low Emotion (%)': f"{low_emotion/len(field_data)*100:.1f}%",
                    'Neutral Dominance': f"{(emotional_intensity < emotional_intensity.mean()).sum()}/{len(field_data)} ({((emotional_intensity < emotional_intensity.mean()).sum()/len(field_data)*100):.1f}%)"
                }
                emotional_data.append(stats)
        
        # Create DataFrame and save
        df_emotional = pd.DataFrame(emotional_data)
        df_emotional.to_csv('table4_emotional_tone_analysis.csv', index=False, encoding='utf-8-sig')
        
        # Create LaTeX table
        latex_table = df_emotional.to_latex(index=False, escape=False)
        with open('table4_emotional_tone_analysis.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print("✅ Created Table 4: Emotional Tone Analysis")
        return df_emotional
    
    def create_agenda_breadth_analysis_table(self):
        """Create Table 5: Agenda Breadth and Information Density"""
        
        agenda_data = []
        
        for source_name, data in [('AI Generated', self.ai_data), ('Journalist Written', self.journalist_data)]:
            for field in ['新聞標題', '影片對話', '影片描述', '新聞內容']:
                field_data = data[data['field'] == field]
                if len(field_data) == 0:
                    continue
                
                # Calculate agenda breadth metrics
                von_neumann = field_data['von_neumann_entropy']
                word_count = field_data['word_count']
                unique_words = field_data['unique_words']
                categorical_div = field_data['categorical_diversity']
                
                # Calculate information density
                info_density = unique_words / word_count
                
                stats = {
                    'Source': source_name,
                    'Field': field,
                    'Count': len(field_data),
                    'Von Neumann Entropy (Mean)': f"{von_neumann.mean():.4f}",
                    'Von Neumann Entropy (Std)': f"{von_neumann.std():.4f}",
                    'Information Density (Mean)': f"{info_density.mean():.4f}",
                    'Information Density (Std)': f"{info_density.std():.4f}",
                    'Avg Word Count': f"{word_count.mean():.1f}",
                    'Avg Unique Words': f"{unique_words.mean():.1f}",
                    'Categorical Diversity (Mean)': f"{categorical_div.mean():.4f}",
                    'Categorical Diversity (Std)': f"{categorical_div.std():.4f}",
                    'High Density (>0.5)': f"{(info_density > 0.5).sum()}/{len(field_data)} ({((info_density > 0.5).sum()/len(field_data)*100):.1f}%)",
                    'Broad Agenda (>0.4)': f"{(categorical_div > 0.4).sum()}/{len(field_data)} ({((categorical_div > 0.4).sum()/len(field_data)*100):.1f}%)"
                }
                agenda_data.append(stats)
        
        # Create DataFrame and save
        df_agenda = pd.DataFrame(agenda_data)
        df_agenda.to_csv('table5_agenda_breadth_analysis.csv', index=False, encoding='utf-8-sig')
        
        # Create LaTeX table
        latex_table = df_agenda.to_latex(index=False, escape=False)
        with open('table5_agenda_breadth_analysis.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print("✅ Created Table 5: Agenda Breadth Analysis")
        return df_agenda
    
    def create_multiple_framing_distribution_table(self):
        """Create Table 6: Multiple Framing Distribution Analysis"""
        
        framing_data = []
        
        # Analyze multiple reality strength distribution
        for source_name, data in [('AI Generated', self.ai_data), ('Journalist Written', self.journalist_data)]:
            multiple_reality = data['multiple_reality_strength']
            
            # Create distribution categories
            very_high = (multiple_reality > 1.7).sum()
            high = ((multiple_reality >= 1.65) & (multiple_reality <= 1.7)).sum()
            medium = ((multiple_reality >= 1.6) & (multiple_reality < 1.65)).sum()
            low = (multiple_reality < 1.6).sum()
            
            stats = {
                'Source': source_name,
                'Total Articles': len(data),
                'Mean Multiple Reality': f"{multiple_reality.mean():.4f}",
                'Std Multiple Reality': f"{multiple_reality.std():.4f}",
                'Very High (>1.7)': f"{very_high} ({very_high/len(data)*100:.1f}%)",
                'High (1.65-1.7)': f"{high} ({high/len(data)*100:.1f}%)",
                'Medium (1.6-1.65)': f"{medium} ({medium/len(data)*100:.1f}%)",
                'Low (<1.6)': f"{low} ({low/len(data)*100:.1f}%)",
                'Multi-Framing Prevalence': f"{(multiple_reality > 1.65).sum()}/{len(data)} ({((multiple_reality > 1.65).sum()/len(data)*100):.1f}%)"
            }
            framing_data.append(stats)
        
        # Create DataFrame and save
        df_framing = pd.DataFrame(framing_data)
        df_framing.to_csv('table6_multiple_framing_distribution.csv', index=False, encoding='utf-8-sig')
        
        # Create LaTeX table
        latex_table = df_framing.to_latex(index=False, escape=False)
        with open('table6_multiple_framing_distribution.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print("✅ Created Table 6: Multiple Framing Distribution")
        return df_framing
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig_size = (12, 8)
        
        # 1. Multiple Reality Strength Distribution
        self.plot_multiple_reality_distribution()
        
        # 2. Frame Competition vs Frame Conflict
        self.plot_frame_competition_vs_conflict()
        
        # 3. Emotional Intensity by Field
        self.plot_emotional_intensity_by_field()
        
        # 4. Information Density Analysis
        self.plot_information_density()
        
        # 5. Quantum Metrics Comparison
        self.plot_quantum_metrics_comparison()
        
        print("✅ Created all visualizations")
    
    def plot_multiple_reality_distribution(self):
        """Plot multiple reality strength distribution"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multiple Reality Strength Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Overall distribution
        ax1 = axes[0, 0]
        all_data = pd.concat([self.ai_data, self.journalist_data], ignore_index=True)
        ax1.hist(all_data['multiple_reality_strength'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(all_data['multiple_reality_strength'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {all_data["multiple_reality_strength"].mean():.4f}')
        ax1.set_title('Overall Multiple Reality Distribution')
        ax1.set_xlabel('Multiple Reality Strength')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # By source
        ax2 = axes[0, 1]
        ax2.hist(self.ai_data['multiple_reality_strength'], bins=15, alpha=0.7, label='AI Generated', color='lightcoral')
        ax2.hist(self.journalist_data['multiple_reality_strength'], bins=15, alpha=0.7, label='Journalist Written', color='lightblue')
        ax2.set_title('Multiple Reality by Source')
        ax2.set_xlabel('Multiple Reality Strength')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # By field
        ax3 = axes[1, 0]
        fields = all_data['field'].unique()
        field_means = [all_data[all_data['field'] == field]['multiple_reality_strength'].mean() for field in fields]
        bars = ax3.bar(fields, field_means, color=['gold', 'lightgreen', 'lightcoral', 'lightblue'])
        ax3.set_title('Multiple Reality by Field')
        ax3.set_ylabel('Mean Multiple Reality Strength')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, field_means):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Box plot by field and source
        ax4 = axes[1, 1]
        plot_data = []
        plot_labels = []
        colors = []
        
        for source_name, data in [('AI', self.ai_data), ('Journalist', self.journalist_data)]:
            for field in ['新聞標題', '影片對話', '影片描述', '新聞內容']:
                field_data = data[data['field'] == field]
                if len(field_data) > 0:
                    plot_data.append(field_data['multiple_reality_strength'])
                    plot_labels.append(f'{source_name}\n{field}')
                    colors.append('lightcoral' if source_name == 'AI' else 'lightblue')
        
        bp = ax4.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax4.set_title('Multiple Reality by Field and Source')
        ax4.set_ylabel('Multiple Reality Strength')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('figure1_multiple_reality_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_frame_competition_vs_conflict(self):
        """Plot frame competition vs frame conflict"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Frame Competition vs Frame Conflict Analysis', fontsize=16, fontweight='bold')
        
        # Scatter plot: AI vs Journalist
        ax1 = axes[0, 0]
        ax1.scatter(self.ai_data['frame_competition'], self.ai_data['semantic_interference'], 
                   alpha=0.6, color='lightcoral', label='AI Generated', s=50)
        ax1.scatter(self.journalist_data['frame_competition'], self.journalist_data['semantic_interference'], 
                   alpha=0.6, color='lightblue', label='Journalist Written', s=50)
        ax1.set_xlabel('Frame Competition')
        ax1.set_ylabel('Frame Conflict (Semantic Interference)')
        ax1.set_title('Frame Competition vs Conflict')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add trend lines
        z1 = np.polyfit(self.ai_data['frame_competition'], self.ai_data['semantic_interference'], 1)
        p1 = np.poly1d(z1)
        ax1.plot(self.ai_data['frame_competition'], p1(self.ai_data['frame_competition']), 
                "r--", alpha=0.8, linewidth=2)
        
        z2 = np.polyfit(self.journalist_data['frame_competition'], self.journalist_data['semantic_interference'], 1)
        p2 = np.poly1d(z2)
        ax1.plot(self.journalist_data['frame_competition'], p2(self.journalist_data['frame_competition']), 
                "b--", alpha=0.8, linewidth=2)
        
        # Competition distribution
        ax2 = axes[0, 1]
        ax2.hist(self.ai_data['frame_competition'], bins=10, alpha=0.7, label='AI Generated', color='lightcoral')
        ax2.hist(self.journalist_data['frame_competition'], bins=10, alpha=0.7, label='Journalist Written', color='lightblue')
        ax2.set_title('Frame Competition Distribution')
        ax2.set_xlabel('Frame Competition')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Conflict distribution
        ax3 = axes[1, 0]
        ax3.hist(self.ai_data['semantic_interference'], bins=20, alpha=0.7, label='AI Generated', color='lightcoral')
        ax3.hist(self.journalist_data['semantic_interference'], bins=20, alpha=0.7, label='Journalist Written', color='lightblue')
        ax3.set_title('Frame Conflict Distribution')
        ax3.set_xlabel('Frame Conflict (Semantic Interference)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # High competition, low conflict analysis
        ax4 = axes[1, 1]
        
        # Calculate percentages
        ai_high_comp_low_conf = ((self.ai_data['frame_competition'] > 0.9) & 
                                (self.ai_data['semantic_interference'] < 0.1)).sum() / len(self.ai_data) * 100
        
        journalist_high_comp_low_conf = ((self.journalist_data['frame_competition'] > 0.9) & 
                                       (self.journalist_data['semantic_interference'] < 0.1)).sum() / len(self.journalist_data) * 100
        
        categories = ['AI Generated', 'Journalist Written']
        percentages = [ai_high_comp_low_conf, journalist_high_comp_low_conf]
        colors = ['lightcoral', 'lightblue']
        
        bars = ax4.bar(categories, percentages, color=colors)
        ax4.set_title('High Competition, Low Conflict Style')
        ax4.set_ylabel('Percentage of Articles (%)')
        ax4.set_ylim(0, 100)
        
        # Add value labels
        for bar, value in zip(bars, percentages):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('figure2_frame_competition_vs_conflict.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_emotional_intensity_by_field(self):
        """Plot emotional intensity analysis by field"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Emotional Intensity Analysis by Field', fontsize=16, fontweight='bold')
        
        # Combine data for field analysis
        all_data = pd.concat([self.ai_data, self.journalist_data], ignore_index=True)
        
        # Emotional intensity by field
        ax1 = axes[0, 0]
        fields = all_data['field'].unique()
        field_means = [all_data[all_data['field'] == field]['semantic_interference'].mean() for field in fields]
        field_stds = [all_data[all_data['field'] == field]['semantic_interference'].std() for field in fields]
        
        bars = ax1.bar(fields, field_means, yerr=field_stds, capsize=5, 
                      color=['gold', 'lightgreen', 'lightcoral', 'lightblue'])
        ax1.set_title('Emotional Intensity by Field')
        ax1.set_ylabel('Mean Emotional Intensity')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, field_means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Emotional intensity distribution by field
        ax2 = axes[0, 1]
        for i, field in enumerate(fields):
            field_data = all_data[all_data['field'] == field]['semantic_interference']
            ax2.hist(field_data, bins=15, alpha=0.6, label=field, 
                    color=['gold', 'lightgreen', 'lightcoral', 'lightblue'][i])
        ax2.set_title('Emotional Intensity Distribution by Field')
        ax2.set_xlabel('Emotional Intensity')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Neutrality dominance analysis
        ax3 = axes[1, 0]
        neutrality_data = []
        field_labels = []
        
        for field in fields:
            field_data = all_data[all_data['field'] == field]['semantic_interference']
            neutral_count = (field_data < field_data.mean()).sum()
            total_count = len(field_data)
            neutrality_data.append(neutral_count / total_count * 100)
            field_labels.append(field)
        
        bars = ax3.bar(field_labels, neutrality_data, 
                      color=['gold', 'lightgreen', 'lightcoral', 'lightblue'])
        ax3.set_title('Neutrality Dominance by Field')
        ax3.set_ylabel('Percentage Below Mean (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 100)
        
        # Add value labels
        for bar, value in zip(bars, neutrality_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # AI vs Journalist emotional intensity
        ax4 = axes[1, 1]
        ai_means = [self.ai_data[self.ai_data['field'] == field]['semantic_interference'].mean() 
                   for field in fields if len(self.ai_data[self.ai_data['field'] == field]) > 0]
        journalist_means = [self.journalist_data[self.journalist_data['field'] == field]['semantic_interference'].mean() 
                           for field in fields if len(self.journalist_data[self.journalist_data['field'] == field]) > 0]
        
        available_fields = [field for field in fields if len(self.ai_data[self.ai_data['field'] == field]) > 0]
        
        x = np.arange(len(available_fields))
        width = 0.35
        
        ax4.bar(x - width/2, ai_means, width, label='AI Generated', color='lightcoral')
        ax4.bar(x + width/2, journalist_means, width, label='Journalist Written', color='lightblue')
        
        ax4.set_title('Emotional Intensity: AI vs Journalist')
        ax4.set_ylabel('Mean Emotional Intensity')
        ax4.set_xticks(x)
        ax4.set_xticklabels(available_fields, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('figure3_emotional_intensity_by_field.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_information_density(self):
        """Plot information density analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Information Density and Agenda Breadth Analysis', fontsize=16, fontweight='bold')
        
        # Calculate information density
        self.ai_data['info_density'] = self.ai_data['unique_words'] / self.ai_data['word_count']
        self.journalist_data['info_density'] = self.journalist_data['unique_words'] / self.journalist_data['word_count']
        
        # Von Neumann Entropy by field
        ax1 = axes[0, 0]
        all_data = pd.concat([self.ai_data, self.journalist_data], ignore_index=True)
        fields = all_data['field'].unique()
        
        field_entropy = [all_data[all_data['field'] == field]['von_neumann_entropy'].mean() for field in fields]
        bars = ax1.bar(fields, field_entropy, color=['gold', 'lightgreen', 'lightcoral', 'lightblue'])
        ax1.set_title('Information Density (Von Neumann Entropy) by Field')
        ax1.set_ylabel('Mean Von Neumann Entropy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, field_entropy):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Information density distribution
        ax2 = axes[0, 1]
        ax2.hist(self.ai_data['info_density'], bins=20, alpha=0.7, label='AI Generated', color='lightcoral')
        ax2.hist(self.journalist_data['info_density'], bins=20, alpha=0.7, label='Journalist Written', color='lightblue')
        ax2.set_title('Information Density Distribution')
        ax2.set_xlabel('Information Density (Unique Words / Total Words)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Word count vs unique words
        ax3 = axes[1, 0]
        ax3.scatter(self.ai_data['word_count'], self.ai_data['unique_words'], 
                   alpha=0.6, color='lightcoral', label='AI Generated', s=50)
        ax3.scatter(self.journalist_data['word_count'], self.journalist_data['unique_words'], 
                   alpha=0.6, color='lightblue', label='Journalist Written', s=50)
        ax3.set_xlabel('Total Word Count')
        ax3.set_ylabel('Unique Words')
        ax3.set_title('Word Count vs Unique Words')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Categorical diversity analysis
        ax4 = axes[1, 1]
        ai_diversity = [self.ai_data[self.ai_data['field'] == field]['categorical_diversity'].mean() 
                       for field in fields if len(self.ai_data[self.ai_data['field'] == field]) > 0]
        journalist_diversity = [self.journalist_data[self.journalist_data['field'] == field]['categorical_diversity'].mean() 
                               for field in fields if len(self.journalist_data[self.journalist_data['field'] == field]) > 0]
        
        available_fields = [field for field in fields if len(self.ai_data[self.ai_data['field'] == field]) > 0]
        
        x = np.arange(len(available_fields))
        width = 0.35
        
        ax4.bar(x - width/2, ai_diversity, width, label='AI Generated', color='lightcoral')
        ax4.bar(x + width/2, journalist_diversity, width, label='Journalist Written', color='lightblue')
        
        ax4.set_title('Categorical Diversity by Field')
        ax4.set_ylabel('Mean Categorical Diversity')
        ax4.set_xticks(x)
        ax4.set_xticklabels(available_fields, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('figure4_information_density_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quantum_metrics_comparison(self):
        """Plot comprehensive quantum metrics comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Quantum Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Quantum coherence comparison
        ax1 = axes[0, 0]
        coherence_data = [self.ai_data['quantum_coherence'], self.journalist_data['quantum_coherence']]
        labels = ['AI Generated', 'Journalist Written']
        colors = ['lightcoral', 'lightblue']
        
        bp1 = ax1.boxplot(coherence_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_title('Quantum Coherence Comparison')
        ax1.set_ylabel('Quantum Coherence')
        ax1.grid(True, alpha=0.3)
        
        # Superposition strength comparison
        ax2 = axes[0, 1]
        superposition_data = [self.ai_data['superposition_strength'], self.journalist_data['superposition_strength']]
        
        bp2 = ax2.boxplot(superposition_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_title('Superposition Strength Comparison')
        ax2.set_ylabel('Superposition Strength')
        ax2.grid(True, alpha=0.3)
        
        # Multiple reality strength comparison
        ax3 = axes[1, 0]
        reality_data = [self.ai_data['multiple_reality_strength'], self.journalist_data['multiple_reality_strength']]
        
        bp3 = ax3.boxplot(reality_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp3['boxes'], colors):
            patch.set_facecolor(color)
        
        ax3.set_title('Multiple Reality Strength Comparison')
        ax3.set_ylabel('Multiple Reality Strength')
        ax3.grid(True, alpha=0.3)
        
        # Correlation heatmap
        ax4 = axes[1, 1]
        
        # Select key metrics for correlation
        metrics = ['multiple_reality_strength', 'frame_competition', 'semantic_interference', 
                  'quantum_coherence', 'von_neumann_entropy', 'categorical_diversity']
        
        # Combine data for correlation analysis
        combined_data = pd.concat([self.ai_data[metrics], self.journalist_data[metrics]], ignore_index=True)
        
        correlation_matrix = combined_data.corr()
        
        im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(metrics)))
        ax4.set_yticks(range(len(metrics)))
        ax4.set_xticklabels(metrics, rotation=45, ha='right')
        ax4.set_yticklabels(metrics)
        ax4.set_title('Quantum Metrics Correlation Matrix')
        
        # Add correlation values to the heatmap
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig('figure5_quantum_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_tables_and_figures(self):
        """Generate all tables and figures"""
        
        print("🚀 Starting comprehensive Qiskit analysis table and figure generation...")
        
        # Generate all tables
        table1 = self.create_overall_statistics_table()
        table2 = self.create_ai_vs_journalist_comparison_table()
        table3 = self.create_frame_competition_analysis_table()
        table4 = self.create_emotional_tone_analysis_table()
        table5 = self.create_agenda_breadth_analysis_table()
        table6 = self.create_multiple_framing_distribution_table()
        
        # Generate all visualizations
        self.create_visualizations()
        
        print("\n🎉 All tables and figures generated successfully!")
        print("\n📊 Generated Files:")
        print("Tables (CSV and LaTeX):")
        print("  - table1_overall_quantum_statistics")
        print("  - table2_ai_vs_journalist_comparison")
        print("  - table3_frame_competition_analysis")
        print("  - table4_emotional_tone_analysis")
        print("  - table5_agenda_breadth_analysis")
        print("  - table6_multiple_framing_distribution")
        print("\nFigures (PNG):")
        print("  - figure1_multiple_reality_distribution")
        print("  - figure2_frame_competition_vs_conflict")
        print("  - figure3_emotional_intensity_by_field")
        print("  - figure4_information_density_analysis")
        print("  - figure5_quantum_metrics_comparison")

def main():
    """Main function to generate all analysis tables and figures"""
    
    # Create the analysis generator
    analyzer = QiskitAnalysisTables()
    
    # Generate all tables and figures
    analyzer.generate_all_tables_and_figures()

if __name__ == "__main__":
    main()
