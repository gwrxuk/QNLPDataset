#!/usr/bin/env python3
"""
Simple Visualizations for Qiskit Analysis
Creates basic charts without Chinese font dependencies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_simple_visualizations():
    """Create simple visualizations for the paper"""
    
    # Load data
    ai_data = pd.read_csv('../results/fast_qiskit_ai_analysis_results.csv')
    journalist_data = pd.read_csv('../results/fast_qiskit_journalist_analysis_results.csv')
    
    # Combine data
    all_data = pd.concat([ai_data, journalist_data], ignore_index=True)
    
    # 1. Multiple Reality Strength Distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(ai_data['multiple_reality_strength'], bins=20, alpha=0.7, 
             label='AI Generated', color='lightcoral', edgecolor='black')
    plt.hist(journalist_data['multiple_reality_strength'], bins=20, alpha=0.7, 
             label='Journalist Written', color='lightblue', edgecolor='black')
    plt.axvline(ai_data['multiple_reality_strength'].mean(), color='red', linestyle='--', 
               label=f'AI Mean: {ai_data["multiple_reality_strength"].mean():.4f}')
    plt.axvline(journalist_data['multiple_reality_strength'].mean(), color='blue', linestyle='--', 
               label=f'Journalist Mean: {journalist_data["multiple_reality_strength"].mean():.4f}')
    plt.title('Multiple Reality Strength Distribution')
    plt.xlabel('Multiple Reality Strength')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Frame Competition vs Frame Conflict
    plt.subplot(2, 2, 2)
    plt.scatter(ai_data['frame_competition'], ai_data['semantic_interference'], 
               alpha=0.6, color='lightcoral', label='AI Generated', s=50)
    plt.scatter(journalist_data['frame_competition'], journalist_data['semantic_interference'], 
               alpha=0.6, color='lightblue', label='Journalist Written', s=50)
    plt.xlabel('Frame Competition')
    plt.ylabel('Frame Conflict (Semantic Interference)')
    plt.title('High Competition, Low Conflict Pattern')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotations
    plt.text(0.5, 0.02, 'High Competition\nLow Conflict', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
             fontsize=10, ha='center')
    
    # 3. Information Density by Field
    plt.subplot(2, 2, 3)
    fields = all_data['field'].unique()
    field_entropy = [all_data[all_data['field'] == field]['von_neumann_entropy'].mean() for field in fields]
    
    bars = plt.bar(range(len(fields)), field_entropy, 
                  color=['gold', 'lightgreen', 'lightcoral', 'lightblue'])
    plt.title('Information Density by Field')
    plt.ylabel('Mean Von Neumann Entropy')
    plt.xticks(range(len(fields)), ['News Titles', 'Video Dialogues', 'Video Descriptions', 'News Content'], 
               rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, field_entropy)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    # 4. Emotional Intensity Analysis
    plt.subplot(2, 2, 4)
    ai_emotion = ai_data['semantic_interference']
    journalist_emotion = journalist_data['semantic_interference']
    
    plt.hist(ai_emotion, bins=15, alpha=0.7, label='AI Generated', color='lightcoral')
    plt.hist(journalist_emotion, bins=15, alpha=0.7, label='Journalist Written', color='lightblue')
    plt.axvline(ai_emotion.mean(), color='red', linestyle='--', 
               label=f'AI Mean: {ai_emotion.mean():.4f}')
    plt.axvline(journalist_emotion.mean(), color='blue', linestyle='--', 
               label=f'Journalist Mean: {journalist_emotion.mean():.4f}')
    plt.title('Emotional Intensity Distribution')
    plt.xlabel('Emotional Intensity (Semantic Interference)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Quantum Metrics Comparison
    plt.figure(figsize=(15, 10))
    
    # Metrics to compare
    metrics = ['multiple_reality_strength', 'frame_competition', 'semantic_interference', 
              'quantum_coherence', 'von_neumann_entropy']
    metric_names = ['Multiple Reality', 'Frame Competition', 'Frame Conflict', 
                   'Quantum Coherence', 'Information Density']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 3, i+1)
        
        ai_values = ai_data[metric]
        journalist_values = journalist_data[metric]
        
        plt.boxplot([ai_values, journalist_values], 
                   labels=['AI Generated', 'Journalist Written'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        
        # Color the second box differently
        plt.gca().artists[1].set_facecolor('lightblue')
        
        plt.title(f'{name} Comparison')
        plt.ylabel(name)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        ai_mean = ai_values.mean()
        ai_std = ai_values.std()
        j_mean = journalist_values.mean()
        j_std = journalist_values.std()
        
        plt.text(0.5, 0.95, f'AI: {ai_mean:.4f}±{ai_std:.4f}\nJ: {j_mean:.4f}±{j_std:.4f}', 
                transform=plt.gca().transAxes, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Calculate key statistics
    ai_high_mr = (ai_data['multiple_reality_strength'] > 1.7).sum()
    j_high_mr = (journalist_data['multiple_reality_strength'] > 1.7).sum()
    
    ai_high_comp = (ai_data['frame_competition'] > 0.9).sum()
    j_high_comp = (journalist_data['frame_competition'] > 0.9).sum()
    
    ai_low_conflict = (ai_data['semantic_interference'] < 0.1).sum()
    j_low_conflict = (journalist_data['semantic_interference'] < 0.1).sum()
    
    summary_text = f"""
KEY FINDINGS SUMMARY:

Multiple Reality (>1.7):
• AI: {ai_high_mr}/{len(ai_data)} ({ai_high_mr/len(ai_data)*100:.1f}%)
• Journalist: {j_high_mr}/{len(journalist_data)} ({j_high_mr/len(journalist_data)*100:.1f}%)

High Competition (>0.9):
• AI: {ai_high_comp}/{len(ai_data)} ({ai_high_comp/len(ai_data)*100:.1f}%)
• Journalist: {j_high_comp}/{len(journalist_data)} ({j_high_comp/len(journalist_data)*100:.1f}%)

Low Conflict (<0.1):
• AI: {ai_low_conflict}/{len(ai_data)} ({ai_low_conflict/len(ai_data)*100:.1f}%)
• Journalist: {j_low_conflict}/{len(journalist_data)} ({j_low_conflict/len(journalist_data)*100:.1f}%)

Total Articles: {len(all_data)}
• AI Generated: {len(ai_data)} ({len(ai_data)/len(all_data)*100:.1f}%)
• Journalist Written: {len(journalist_data)} ({len(journalist_data)/len(all_data)*100:.1f}%)
"""
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figure_quantum_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Frame Competition Analysis by Field
    plt.figure(figsize=(12, 8))
    
    # Create a comprehensive frame analysis
    plt.subplot(2, 2, 1)
    
    # Frame competition by field
    field_data = []
    field_labels = []
    colors = []
    
    for source_name, data, color in [('AI', ai_data, 'lightcoral'), ('Journalist', journalist_data, 'lightblue')]:
        for field in ['新聞標題', '影片對話', '影片描述', '新聞內容']:
            field_subset = data[data['field'] == field]
            if len(field_subset) > 0:
                field_data.append(field_subset['frame_competition'])
                field_labels.append(f'{source_name}\n{field}')
                colors.append(color)
    
    bp = plt.boxplot(field_data, labels=field_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Frame Competition by Field and Source')
    plt.ylabel('Frame Competition')
    plt.xticks(rotation=45, ha='right')
    
    # Multiple reality by field
    plt.subplot(2, 2, 2)
    field_mr_data = []
    field_mr_labels = []
    colors_mr = []
    
    for source_name, data, color in [('AI', ai_data, 'lightcoral'), ('Journalist', journalist_data, 'lightblue')]:
        for field in ['新聞標題', '影片對話', '影片描述', '新聞內容']:
            field_subset = data[data['field'] == field]
            if len(field_subset) > 0:
                field_mr_data.append(field_subset['multiple_reality_strength'])
                field_mr_labels.append(f'{source_name}\n{field}')
                colors_mr.append(color)
    
    bp2 = plt.boxplot(field_mr_data, labels=field_mr_labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors_mr):
        patch.set_facecolor(color)
    
    plt.title('Multiple Reality Strength by Field and Source')
    plt.ylabel('Multiple Reality Strength')
    plt.xticks(rotation=45, ha='right')
    
    # Information density scatter
    plt.subplot(2, 2, 3)
    ai_density = ai_data['unique_words'] / ai_data['word_count']
    j_density = journalist_data['unique_words'] / journalist_data['word_count']
    
    plt.scatter(ai_data['word_count'], ai_density, alpha=0.6, color='lightcoral', 
               label='AI Generated', s=50)
    plt.scatter(journalist_data['word_count'], j_density, alpha=0.6, color='lightblue', 
               label='Journalist Written', s=50)
    plt.xlabel('Total Word Count')
    plt.ylabel('Information Density (Unique Words / Total Words)')
    plt.title('Information Density vs Word Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Categorical diversity
    plt.subplot(2, 2, 4)
    plt.hist(ai_data['categorical_diversity'], bins=15, alpha=0.7, 
             label='AI Generated', color='lightcoral')
    plt.hist(journalist_data['categorical_diversity'], bins=15, alpha=0.7, 
             label='Journalist Written', color='lightblue')
    plt.axvline(ai_data['categorical_diversity'].mean(), color='red', linestyle='--', 
               label=f'AI Mean: {ai_data["categorical_diversity"].mean():.1f}')
    plt.axvline(journalist_data['categorical_diversity'].mean(), color='blue', linestyle='--', 
               label=f'Journalist Mean: {journalist_data["categorical_diversity"].mean():.1f}')
    plt.title('Categorical Diversity Distribution')
    plt.xlabel('Categorical Diversity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_detailed_field_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created all visualizations:")
    print("  - figure_comprehensive_analysis.png")
    print("  - figure_quantum_metrics_comparison.png")
    print("  - figure_detailed_field_analysis.png")

if __name__ == "__main__":
    create_simple_visualizations()
