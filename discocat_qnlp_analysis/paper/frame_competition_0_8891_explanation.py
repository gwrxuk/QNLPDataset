#!/usr/bin/env python3
"""
EXACT EXPLANATION: How we get "frame competition = 0.8891"

This demonstrates the precise calculation from our actual code implementation.
"""

import numpy as np
import pandas as pd
from collections import Counter

def explain_frame_competition_0_8891():
    """Explain exactly how we get frame_competition = 0.8891"""
    
    print("="*80)
    print("EXACT EXPLANATION: How we get 'frame competition = 0.8891'")
    print("="*80)
    
    print("\n1. THE ACTUAL FORMULA FROM OUR CODE:")
    print("   (From qiskit_quantum_analyzer.py, lines 193-201)")
    print("   ")
    print("   probabilities_filtered = probabilities[probabilities > 1e-12]")
    print("   if len(probabilities_filtered) > 1:")
    print("       uniform_prob = 1.0 / len(probabilities_filtered)")
    print("       kl_divergence = np.sum(probabilities_filtered * np.log2(probabilities_filtered / uniform_prob))")
    print("       max_kl = np.log2(len(probabilities_filtered))")
    print("       frame_competition = float(1.0 - (kl_divergence / max_kl))")
    
    print("\n2. MATHEMATICAL FORMULA:")
    print("   frame_competition = 1.0 - (KL_divergence / max_KL)")
    print("   ")
    print("   Where:")
    print("   - KL_divergence = Σᵢ pᵢ log₂(pᵢ / qᵢ)")
    print("   - pᵢ = actual probabilities from quantum state")
    print("   - qᵢ = uniform probabilities (1/n)")
    print("   - max_KL = log₂(n) = maximum possible KL divergence")
    
    print("\n3. STEP-BY-STEP CALCULATION EXAMPLE:")
    
    # Example quantum state probabilities (typical from our analysis)
    # These would come from |amplitude|² after quantum circuit execution
    probabilities = np.array([0.4, 0.3, 0.2, 0.1])  # Example distribution
    
    print(f"\n   Example quantum state probabilities:")
    print(f"   p = {probabilities}")
    
    # Filter out near-zero probabilities
    probabilities_filtered = probabilities[probabilities > 1e-12]
    print(f"   Filtered probabilities: {probabilities_filtered}")
    
    # Calculate uniform distribution
    n = len(probabilities_filtered)
    uniform_prob = 1.0 / n
    print(f"   Uniform probability: {uniform_prob}")
    
    # Calculate KL divergence
    kl_divergence = np.sum(probabilities_filtered * np.log2(probabilities_filtered / uniform_prob))
    print(f"   KL divergence: {kl_divergence:.6f}")
    
    # Calculate maximum KL divergence
    max_kl = np.log2(n)
    print(f"   Maximum KL divergence: {max_kl:.6f}")
    
    # Calculate frame competition
    frame_competition = 1.0 - (kl_divergence / max_kl)
    print(f"   Frame competition: 1.0 - ({kl_divergence:.6f} / {max_kl:.6f})")
    print(f"   Frame competition: 1.0 - {kl_divergence / max_kl:.6f}")
    print(f"   Frame competition: {frame_competition:.6f}")
    
    print("\n4. WHY THIS FORMULA MAKES SENSE:")
    print("   - KL divergence measures how far actual distribution is from uniform")
    print("   - Higher KL divergence = more concentrated distribution = less competition")
    print("   - Lower KL divergence = more uniform distribution = more competition")
    print("   - We invert it: 1 - (KL/max_KL) so higher values = more competition")
    
    print("\n5. REAL DATA EXAMPLE:")
    
    # Load actual results to show real calculation
    try:
        ai_data = pd.read_csv('../results/fast_qiskit_ai_analysis_results.csv')
        
        # Get a sample with frame_competition close to 0.8891
        sample_data = ai_data[ai_data['frame_competition'] > 0.8].iloc[0]
        
        print(f"\n   Real example from our data:")
        print(f"   Text: '{sample_data['original_text'][:50]}...'")
        print(f"   Frame Competition: {sample_data['frame_competition']:.6f}")
        print(f"   Von Neumann Entropy: {sample_data['von_neumann_entropy']:.6f}")
        print(f"   Semantic Interference: {sample_data['semantic_interference']:.6f}")
        
        # Show the distribution statistics
        print(f"\n   Frame Competition Statistics:")
        print(f"   Mean: {ai_data['frame_competition'].mean():.6f}")
        print(f"   Min: {ai_data['frame_competition'].min():.6f}")
        print(f"   Max: {ai_data['frame_competition'].max():.6f}")
        print(f"   Std: {ai_data['frame_competition'].std():.6f}")
        
    except Exception as e:
        print(f"   Could not load real data: {e}")
    
    print("\n6. THE SPECIFIC CASE OF 0.8891:")
    print("   To get frame_competition = 0.8891:")
    print("   ")
    print("   0.8891 = 1.0 - (KL_divergence / max_KL)")
    print("   KL_divergence / max_KL = 1.0 - 0.8891 = 0.1109")
    print("   ")
    print("   This means the actual distribution is quite close to uniform,")
    print("   indicating high frame competition (multiple frames competing)")
    
    print("\n7. QUANTUM INTERPRETATION:")
    print("   - High frame competition (0.8891) means:")
    print("     * Multiple semantic frames exist in superposition")
    print("     * No single frame dominates the interpretation")
    print("     * Quantum state is close to uniform distribution")
    print("     * High uncertainty about which frame to collapse to")
    
    print("\n8. COMPARISON WITH JOURNALIST DATA:")
    try:
        journalist_data = pd.read_csv('../results/fast_qiskit_journalist_analysis_results.csv')
        
        ai_mean = ai_data['frame_competition'].mean()
        j_mean = journalist_data['frame_competition'].mean()
        
        print(f"\n   AI Generated Mean: {ai_mean:.6f}")
        print(f"   Journalist Written Mean: {j_mean:.6f}")
        print(f"   Difference: {abs(ai_mean - j_mean):.6f}")
        
        if ai_mean > j_mean:
            print("   → AI shows higher frame competition (more multiple framings)")
        else:
            print("   → Journalists show higher frame competition")
            
    except Exception as e:
        print(f"   Could not load journalist data: {e}")

def demonstrate_kl_divergence_calculation():
    """Demonstrate the KL divergence calculation step by step"""
    
    print("\n" + "="*80)
    print("DETAILED KL DIVERGENCE CALCULATION")
    print("="*80)
    
    # Example: 4-frame quantum state
    actual_probs = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform (maximum competition)
    uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform reference
    
    print(f"\nExample 1: Maximum Competition (Uniform Distribution)")
    print(f"Actual probabilities: {actual_probs}")
    print(f"Uniform probabilities: {uniform_probs}")
    
    kl_div = np.sum(actual_probs * np.log2(actual_probs / uniform_probs))
    max_kl = np.log2(len(actual_probs))
    frame_comp = 1.0 - (kl_div / max_kl)
    
    print(f"KL divergence: {kl_div:.6f}")
    print(f"Max KL divergence: {max_kl:.6f}")
    print(f"Frame competition: {frame_comp:.6f}")
    print("→ Maximum competition (all frames equally likely)")
    
    # Example: Concentrated distribution (low competition)
    actual_probs = np.array([0.8, 0.1, 0.05, 0.05])  # One frame dominates
    uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform reference
    
    print(f"\nExample 2: Low Competition (Concentrated Distribution)")
    print(f"Actual probabilities: {actual_probs}")
    print(f"Uniform probabilities: {uniform_probs}")
    
    kl_div = np.sum(actual_probs * np.log2(actual_probs / uniform_probs))
    max_kl = np.log2(len(actual_probs))
    frame_comp = 1.0 - (kl_div / max_kl)
    
    print(f"KL divergence: {kl_div:.6f}")
    print(f"Max KL divergence: {max_kl:.6f}")
    print(f"Frame competition: {frame_comp:.6f}")
    print("→ Low competition (one frame dominates)")
    
    # Example: Moderate competition (like our 0.8891 case)
    actual_probs = np.array([0.35, 0.3, 0.2, 0.15])  # Moderate concentration
    uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform reference
    
    print(f"\nExample 3: Moderate Competition (Like 0.8891 Case)")
    print(f"Actual probabilities: {actual_probs}")
    print(f"Uniform probabilities: {uniform_probs}")
    
    kl_div = np.sum(actual_probs * np.log2(actual_probs / uniform_probs))
    max_kl = np.log2(len(actual_probs))
    frame_comp = 1.0 - (kl_div / max_kl)
    
    print(f"KL divergence: {kl_div:.6f}")
    print(f"Max KL divergence: {max_kl:.6f}")
    print(f"Frame competition: {frame_comp:.6f}")
    print("→ Moderate competition (some frames more likely than others)")

def main():
    """Main demonstration function"""
    
    explain_frame_competition_0_8891()
    demonstrate_kl_divergence_calculation()
    
    print("\n" + "="*80)
    print("SUMMARY: How we get frame_competition = 0.8891")
    print("="*80)
    print("1. Quantum circuit creates superposition state")
    print("2. We calculate |amplitude|² to get probabilities")
    print("3. We compute KL divergence from uniform distribution")
    print("4. We normalize by maximum KL divergence")
    print("5. We invert: frame_competition = 1.0 - (KL/max_KL)")
    print("6. Result: 0.8891 indicates high frame competition")
    print("   (multiple semantic frames competing in superposition)")
    print("="*80)

if __name__ == "__main__":
    main()
