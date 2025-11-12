#!/usr/bin/env python3
"""
Simple Qiskit Analysis Tables Generator
Generates comprehensive tables for the 298 articles analyzed by Qiskit
"""

import pandas as pd
import numpy as np
import json

class SimpleQiskitTables:
    def __init__(self):
        """Initialize the simple tables generator"""
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
    
    def create_summary_statistics(self):
        """Create comprehensive summary statistics"""
        
        print("\n📊 QISKIT ANALYSIS SUMMARY STATISTICS")
        print("=" * 60)
        
        # Overall statistics
        all_data = pd.concat([self.ai_data, self.journalist_data], ignore_index=True)
        
        print(f"\n🔍 DATASET OVERVIEW:")
        print(f"   Total Articles Analyzed: {len(all_data)}")
        print(f"   AI-Generated Articles: {len(self.ai_data)}")
        print(f"   Journalist-Written Articles: {len(self.journalist_data)}")
        
        # Multiple Reality Analysis
        print(f"\n🎯 MULTIPLE REALITY ANALYSIS:")
        ai_mr_mean = self.ai_data['multiple_reality_strength'].mean()
        journalist_mr_mean = self.journalist_data['multiple_reality_strength'].mean()
        
        print(f"   AI Generated - Mean: {ai_mr_mean:.4f} ± {self.ai_data['multiple_reality_strength'].std():.4f}")
        print(f"   Journalist Written - Mean: {journalist_mr_mean:.4f} ± {self.journalist_data['multiple_reality_strength'].std():.4f}")
        
        # High multiple reality prevalence
        ai_high_mr = (self.ai_data['multiple_reality_strength'] > 1.7).sum()
        journalist_high_mr = (self.journalist_data['multiple_reality_strength'] > 1.7).sum()
        
        print(f"   AI High Multi-Reality (>1.7): {ai_high_mr}/{len(self.ai_data)} ({ai_high_mr/len(self.ai_data)*100:.1f}%)")
        print(f"   Journalist High Multi-Reality (>1.7): {journalist_high_mr}/{len(self.journalist_data)} ({journalist_high_mr/len(self.journalist_data)*100:.1f}%)")
        
        # Frame Competition Analysis
        print(f"\n🏆 FRAME COMPETITION ANALYSIS:")
        ai_fc_mean = self.ai_data['frame_competition'].mean()
        journalist_fc_mean = self.journalist_data['frame_competition'].mean()
        
        print(f"   AI Generated - Mean: {ai_fc_mean:.4f} ± {self.ai_data['frame_competition'].std():.4f}")
        print(f"   Journalist Written - Mean: {journalist_fc_mean:.4f} ± {self.journalist_data['frame_competition'].std():.4f}")
        
        # High competition, low conflict analysis
        ai_high_comp = (self.ai_data['frame_competition'] > 0.9).sum()
        ai_low_conflict = (self.ai_data['semantic_interference'] < 0.1).sum()
        ai_high_comp_low_conflict = ((self.ai_data['frame_competition'] > 0.9) & 
                                   (self.ai_data['semantic_interference'] < 0.1)).sum()
        
        journalist_high_comp = (self.journalist_data['frame_competition'] > 0.9).sum()
        journalist_low_conflict = (self.journalist_data['semantic_interference'] < 0.1).sum()
        journalist_high_comp_low_conflict = ((self.journalist_data['frame_competition'] > 0.9) & 
                                           (self.journalist_data['semantic_interference'] < 0.1)).sum()
        
        print(f"   AI High Competition (>0.9): {ai_high_comp}/{len(self.ai_data)} ({ai_high_comp/len(self.ai_data)*100:.1f}%)")
        print(f"   AI Low Conflict (<0.1): {ai_low_conflict}/{len(self.ai_data)} ({ai_low_conflict/len(self.ai_data)*100:.1f}%)")
        print(f"   AI High Competition + Low Conflict: {ai_high_comp_low_conflict}/{len(self.ai_data)} ({ai_high_comp_low_conflict/len(self.ai_data)*100:.1f}%)")
        
        print(f"   Journalist High Competition (>0.9): {journalist_high_comp}/{len(self.journalist_data)} ({journalist_high_comp/len(self.journalist_data)*100:.1f}%)")
        print(f"   Journalist Low Conflict (<0.1): {journalist_low_conflict}/{len(self.journalist_data)} ({journalist_low_conflict/len(self.journalist_data)*100:.1f}%)")
        print(f"   Journalist High Competition + Low Conflict: {journalist_high_comp_low_conflict}/{len(self.journalist_data)} ({journalist_high_comp_low_conflict/len(self.journalist_data)*100:.1f}%)")
        
        # Emotional Tone Analysis
        print(f"\n😊 EMOTIONAL TONE ANALYSIS:")
        ai_emotion_mean = self.ai_data['semantic_interference'].mean()
        journalist_emotion_mean = self.journalist_data['semantic_interference'].mean()
        
        print(f"   AI Generated - Mean Emotional Intensity: {ai_emotion_mean:.4f} ± {self.ai_data['semantic_interference'].std():.4f}")
        print(f"   Journalist Written - Mean Emotional Intensity: {journalist_emotion_mean:.4f} ± {self.journalist_data['semantic_interference'].std():.4f}")
        
        # Neutrality analysis
        ai_neutral = (self.ai_data['semantic_interference'] < ai_emotion_mean).sum()
        journalist_neutral = (self.journalist_data['semantic_interference'] < journalist_emotion_mean).sum()
        
        print(f"   AI Neutral Dominance: {ai_neutral}/{len(self.ai_data)} ({ai_neutral/len(self.ai_data)*100:.1f}%)")
        print(f"   Journalist Neutral Dominance: {journalist_neutral}/{len(self.journalist_data)} ({journalist_neutral/len(self.journalist_data)*100:.1f}%)")
        
        # Information Density Analysis
        print(f"\n📚 INFORMATION DENSITY ANALYSIS:")
        ai_entropy_mean = self.ai_data['von_neumann_entropy'].mean()
        journalist_entropy_mean = self.journalist_data['von_neumann_entropy'].mean()
        
        print(f"   AI Generated - Mean Von Neumann Entropy: {ai_entropy_mean:.4f} ± {self.ai_data['von_neumann_entropy'].std():.4f}")
        print(f"   Journalist Written - Mean Von Neumann Entropy: {journalist_entropy_mean:.4f} ± {self.journalist_data['von_neumann_entropy'].std():.4f}")
        
        # Word count analysis
        ai_words_mean = self.ai_data['word_count'].mean()
        journalist_words_mean = self.journalist_data['word_count'].mean()
        
        print(f"   AI Generated - Mean Word Count: {ai_words_mean:.1f} ± {self.ai_data['word_count'].std():.1f}")
        print(f"   Journalist Written - Mean Word Count: {journalist_words_mean:.1f} ± {self.journalist_data['word_count'].std():.1f}")
        
        # Categorical diversity
        ai_div_mean = self.ai_data['categorical_diversity'].mean()
        journalist_div_mean = self.journalist_data['categorical_diversity'].mean()
        
        print(f"   AI Generated - Mean Categorical Diversity: {ai_div_mean:.4f} ± {self.ai_data['categorical_diversity'].std():.4f}")
        print(f"   Journalist Written - Mean Categorical Diversity: {journalist_div_mean:.4f} ± {self.journalist_data['categorical_diversity'].std():.4f}")
    
    def create_detailed_field_analysis(self):
        """Create detailed analysis by field"""
        
        print(f"\n📋 DETAILED FIELD ANALYSIS")
        print("=" * 60)
        
        fields = ['新聞標題', '影片對話', '影片描述', '新聞內容']
        
        for field in fields:
            print(f"\n🔍 {field.upper()}:")
            
            # Get data for this field
            ai_field_data = self.ai_data[self.ai_data['field'] == field]
            journalist_field_data = self.journalist_data[self.journalist_data['field'] == field]
            
            if len(ai_field_data) == 0 and len(journalist_field_data) == 0:
                print("   No data available for this field")
                continue
            
            # AI data analysis
            if len(ai_field_data) > 0:
                print(f"   AI Generated ({len(ai_field_data)} articles):")
                print(f"     Multiple Reality: {ai_field_data['multiple_reality_strength'].mean():.4f} ± {ai_field_data['multiple_reality_strength'].std():.4f}")
                print(f"     Frame Competition: {ai_field_data['frame_competition'].mean():.4f} ± {ai_field_data['frame_competition'].std():.4f}")
                print(f"     Frame Conflict: {ai_field_data['semantic_interference'].mean():.4f} ± {ai_field_data['semantic_interference'].std():.4f}")
                print(f"     Von Neumann Entropy: {ai_field_data['von_neumann_entropy'].mean():.4f} ± {ai_field_data['von_neumann_entropy'].std():.4f}")
                print(f"     Word Count: {ai_field_data['word_count'].mean():.1f} ± {ai_field_data['word_count'].std():.1f}")
            
            # Journalist data analysis
            if len(journalist_field_data) > 0:
                print(f"   Journalist Written ({len(journalist_field_data)} articles):")
                print(f"     Multiple Reality: {journalist_field_data['multiple_reality_strength'].mean():.4f} ± {journalist_field_data['multiple_reality_strength'].std():.4f}")
                print(f"     Frame Competition: {journalist_field_data['frame_competition'].mean():.4f} ± {journalist_field_data['frame_competition'].std():.4f}")
                print(f"     Frame Conflict: {journalist_field_data['semantic_interference'].mean():.4f} ± {journalist_field_data['semantic_interference'].std():.4f}")
                print(f"     Von Neumann Entropy: {journalist_field_data['von_neumann_entropy'].mean():.4f} ± {journalist_field_data['von_neumann_entropy'].std():.4f}")
                print(f"     Word Count: {journalist_field_data['word_count'].mean():.1f} ± {journalist_field_data['word_count'].std():.1f}")
    
    def create_csv_tables(self):
        """Create CSV tables for the paper"""
        
        print(f"\n📊 CREATING CSV TABLES FOR PAPER")
        print("=" * 60)
        
        # Table 1: Overall Statistics
        all_data = pd.concat([self.ai_data, self.journalist_data], ignore_index=True)
        fields = ['新聞標題', '影片對話', '影片描述', '新聞內容']
        
        table1_data = []
        for field in fields:
            field_data = all_data[all_data['field'] == field]
            if len(field_data) == 0:
                continue
                
            stats = {
                'Field': field,
                'Count': len(field_data),
                'Multiple_Reality_Mean': f"{field_data['multiple_reality_strength'].mean():.4f}",
                'Multiple_Reality_Std': f"{field_data['multiple_reality_strength'].std():.4f}",
                'Frame_Competition_Mean': f"{field_data['frame_competition'].mean():.4f}",
                'Frame_Competition_Std': f"{field_data['frame_competition'].std():.4f}",
                'Von_Neumann_Entropy_Mean': f"{field_data['von_neumann_entropy'].mean():.4f}",
                'Von_Neumann_Entropy_Std': f"{field_data['von_neumann_entropy'].std():.4f}",
                'Semantic_Interference_Mean': f"{field_data['semantic_interference'].mean():.4f}",
                'Semantic_Interference_Std': f"{field_data['semantic_interference'].std():.4f}",
                'Word_Count_Mean': f"{field_data['word_count'].mean():.1f}",
                'Word_Count_Std': f"{field_data['word_count'].std():.1f}"
            }
            table1_data.append(stats)
        
        df_table1 = pd.DataFrame(table1_data)
        df_table1.to_csv('table1_overall_quantum_statistics.csv', index=False, encoding='utf-8-sig')
        print("✅ Created table1_overall_quantum_statistics.csv")
        
        # Table 2: AI vs Journalist Comparison
        comparison_data = []
        for source_name, data in [('AI Generated', self.ai_data), ('Journalist Written', self.journalist_data)]:
            stats = {
                'Source': source_name,
                'Total_Articles': len(data),
                'Multiple_Reality_Mean': f"{data['multiple_reality_strength'].mean():.4f}",
                'Multiple_Reality_Std': f"{data['multiple_reality_strength'].std():.4f}",
                'Frame_Competition_Mean': f"{data['frame_competition'].mean():.4f}",
                'Frame_Competition_Std': f"{data['frame_competition'].std():.4f}",
                'Frame_Conflict_Mean': f"{data['semantic_interference'].mean():.4f}",
                'Frame_Conflict_Std': f"{data['semantic_interference'].std():.4f}",
                'Quantum_Coherence_Mean': f"{data['quantum_coherence'].mean():.4f}",
                'Quantum_Coherence_Std': f"{data['quantum_coherence'].std():.4f}",
                'Avg_Word_Count': f"{data['word_count'].mean():.1f}",
                'Avg_Unique_Words': f"{data['unique_words'].mean():.1f}",
                'Categorical_Diversity_Mean': f"{data['categorical_diversity'].mean():.4f}",
                'Categorical_Diversity_Std': f"{data['categorical_diversity'].std():.4f}"
            }
            comparison_data.append(stats)
        
        df_table2 = pd.DataFrame(comparison_data)
        df_table2.to_csv('table2_ai_vs_journalist_comparison.csv', index=False, encoding='utf-8-sig')
        print("✅ Created table2_ai_vs_journalist_comparison.csv")
        
        # Table 3: Frame Competition Analysis
        frame_data = []
        for source_name, data in [('AI Generated', self.ai_data), ('Journalist Written', self.journalist_data)]:
            for field in fields:
                field_data = data[data['field'] == field]
                if len(field_data) == 0:
                    continue
                
                frame_comp = field_data['frame_competition']
                semantic_int = field_data['semantic_interference']
                
                stats = {
                    'Source': source_name,
                    'Field': field,
                    'Count': len(field_data),
                    'Frame_Competition_Mean': f"{frame_comp.mean():.4f}",
                    'Frame_Competition_Std': f"{frame_comp.std():.4f}",
                    'Frame_Competition_Min': f"{frame_comp.min():.4f}",
                    'Frame_Competition_Max': f"{frame_comp.max():.4f}",
                    'Semantic_Interference_Mean': f"{semantic_int.mean():.4f}",
                    'Semantic_Interference_Std': f"{semantic_int.std():.4f}",
                    'High_Competition_gt_09': f"{(frame_comp > 0.9).sum()}/{len(field_data)} ({((frame_comp > 0.9).sum()/len(field_data)*100):.1f}%)",
                    'Low_Conflict_lt_01': f"{(semantic_int < 0.1).sum()}/{len(field_data)} ({((semantic_int < 0.1).sum()/len(field_data)*100):.1f}%)"
                }
                frame_data.append(stats)
        
        df_table3 = pd.DataFrame(frame_data)
        df_table3.to_csv('table3_frame_competition_analysis.csv', index=False, encoding='utf-8-sig')
        print("✅ Created table3_frame_competition_analysis.csv")
        
        # Table 4: Multiple Framing Distribution
        framing_data = []
        for source_name, data in [('AI Generated', self.ai_data), ('Journalist Written', self.journalist_data)]:
            multiple_reality = data['multiple_reality_strength']
            
            # Create distribution categories
            very_high = (multiple_reality > 1.7).sum()
            high = ((multiple_reality >= 1.65) & (multiple_reality <= 1.7)).sum()
            medium = ((multiple_reality >= 1.6) & (multiple_reality < 1.65)).sum()
            low = (multiple_reality < 1.6).sum()
            
            stats = {
                'Source': source_name,
                'Total_Articles': len(data),
                'Mean_Multiple_Reality': f"{multiple_reality.mean():.4f}",
                'Std_Multiple_Reality': f"{multiple_reality.std():.4f}",
                'Very_High_gt_17': f"{very_high} ({very_high/len(data)*100:.1f}%)",
                'High_165_to_17': f"{high} ({high/len(data)*100:.1f}%)",
                'Medium_16_to_165': f"{medium} ({medium/len(data)*100:.1f}%)",
                'Low_lt_16': f"{low} ({low/len(data)*100:.1f}%)",
                'Multi_Framing_Prevalence': f"{(multiple_reality > 1.65).sum()}/{len(data)} ({((multiple_reality > 1.65).sum()/len(data)*100):.1f}%)"
            }
            framing_data.append(stats)
        
        df_table4 = pd.DataFrame(framing_data)
        df_table4.to_csv('table4_multiple_framing_distribution.csv', index=False, encoding='utf-8-sig')
        print("✅ Created table4_multiple_framing_distribution.csv")
        
        # Table 5: Information Density Analysis
        info_data = []
        for source_name, data in [('AI Generated', self.ai_data), ('Journalist Written', self.journalist_data)]:
            for field in fields:
                field_data = data[data['field'] == field]
                if len(field_data) == 0:
                    continue
                
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
                    'Von_Neumann_Entropy_Mean': f"{von_neumann.mean():.4f}",
                    'Von_Neumann_Entropy_Std': f"{von_neumann.std():.4f}",
                    'Information_Density_Mean': f"{info_density.mean():.4f}",
                    'Information_Density_Std': f"{info_density.std():.4f}",
                    'Avg_Word_Count': f"{word_count.mean():.1f}",
                    'Avg_Unique_Words': f"{unique_words.mean():.1f}",
                    'Categorical_Diversity_Mean': f"{categorical_div.mean():.4f}",
                    'Categorical_Diversity_Std': f"{categorical_div.std():.4f}",
                    'High_Density_gt_05': f"{(info_density > 0.5).sum()}/{len(field_data)} ({((info_density > 0.5).sum()/len(field_data)*100):.1f}%)",
                    'Broad_Agenda_gt_04': f"{(categorical_div > 0.4).sum()}/{len(field_data)} ({((categorical_div > 0.4).sum()/len(field_data)*100):.1f}%)"
                }
                info_data.append(stats)
        
        df_table5 = pd.DataFrame(info_data)
        df_table5.to_csv('table5_information_density_analysis.csv', index=False, encoding='utf-8-sig')
        print("✅ Created table5_information_density_analysis.csv")
    
    def create_latex_tables(self):
        """Create LaTeX tables for academic papers"""
        
        print(f"\n📝 CREATING LATEX TABLES")
        print("=" * 60)
        
        # Read the CSV files and convert to LaTeX
        tables = [
            ('table1_overall_quantum_statistics.csv', 'table1_overall_quantum_statistics.tex'),
            ('table2_ai_vs_journalist_comparison.csv', 'table2_ai_vs_journalist_comparison.tex'),
            ('table3_frame_competition_analysis.csv', 'table3_frame_competition_analysis.tex'),
            ('table4_multiple_framing_distribution.csv', 'table4_multiple_framing_distribution.tex'),
            ('table5_information_density_analysis.csv', 'table5_information_density_analysis.tex')
        ]
        
        for csv_file, tex_file in tables:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                latex_table = df.to_latex(index=False, escape=False)
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(latex_table)
                print(f"✅ Created {tex_file}")
            except Exception as e:
                print(f"❌ Error creating {tex_file}: {e}")
    
    def generate_all_tables(self):
        """Generate all tables and analysis"""
        
        print("🚀 Starting comprehensive Qiskit analysis table generation...")
        
        # Generate summary statistics
        self.create_summary_statistics()
        
        # Generate detailed field analysis
        self.create_detailed_field_analysis()
        
        # Generate CSV tables
        self.create_csv_tables()
        
        # Generate LaTeX tables
        self.create_latex_tables()
        
        print("\n🎉 All tables generated successfully!")
        print("\n📊 Generated Files:")
        print("CSV Tables:")
        print("  - table1_overall_quantum_statistics.csv")
        print("  - table2_ai_vs_journalist_comparison.csv")
        print("  - table3_frame_competition_analysis.csv")
        print("  - table4_multiple_framing_distribution.csv")
        print("  - table5_information_density_analysis.csv")
        print("\nLaTeX Tables:")
        print("  - table1_overall_quantum_statistics.tex")
        print("  - table2_ai_vs_journalist_comparison.tex")
        print("  - table3_frame_competition_analysis.tex")
        print("  - table4_multiple_framing_distribution.tex")
        print("  - table5_information_density_analysis.tex")

def main():
    """Main function to generate all analysis tables"""
    
    # Create the analysis generator
    analyzer = SimpleQiskitTables()
    
    # Generate all tables
    analyzer.generate_all_tables()

if __name__ == "__main__":
    main()
