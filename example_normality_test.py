#!/usr/bin/env python3
"""
Example script showing how to test normality of hamming distance datapoints.

This demonstrates the key datapoints collected in the hamming distance experiment
and how to test them for normality.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

def create_sample_data():
    """Create sample data that mimics the structure of hamming distance results."""
    np.random.seed(42)
    
    # Simulate data for both methods
    n_patterns = 50
    
    # Backprop method data
    bp_data = {
        'pattern_idx': range(n_patterns),
        'method': ['bp'] * n_patterns,
        'overall_bitwise_fraction_diff': np.random.beta(2, 5, n_patterns),  # Skewed distribution
        'per_gate_mean_hamming': np.random.normal(0.3, 0.1, n_patterns),   # Normal distribution
        'final_hard_accuracy': np.random.beta(8, 2, n_patterns),           # Skewed distribution
        'final_hard_loss': np.random.exponential(0.5, n_patterns),         # Exponential distribution
    }
    
    # GNN method data
    gnn_data = {
        'pattern_idx': range(n_patterns),
        'method': ['gnn'] * n_patterns,
        'overall_bitwise_fraction_diff': np.random.beta(3, 4, n_patterns),  # Different beta parameters
        'per_gate_mean_hamming': np.random.normal(0.25, 0.08, n_patterns),  # Normal distribution
        'final_hard_accuracy': np.random.beta(9, 1, n_patterns),            # Different beta parameters
        'final_hard_loss': np.random.exponential(0.3, n_patterns),          # Different exponential rate
    }
    
    # Combine data
    df = pd.DataFrame(bp_data)
    df = pd.concat([df, pd.DataFrame(gnn_data)], ignore_index=True)
    
    return df

def quick_normality_test(data, name):
    """Quick normality test using Shapiro-Wilk and visual inspection."""
    print(f"\n--- Testing {name} ---")
    print(f"Sample size: {len(data)}")
    print(f"Mean: {data.mean():.4f}, Std: {data.std():.4f}")
    print(f"Skewness: {stats.skew(data):.4f}, Kurtosis: {stats.kurtosis(data):.4f}")
    
    # Shapiro-Wilk test
    if len(data) <= 5000:  # Shapiro-Wilk works best for smaller samples
        stat, p_value = stats.shapiro(data)
        print(f"Shapiro-Wilk: statistic={stat:.4f}, p-value={p_value:.6f}")
        print(f"Normality: {'Likely normal' if p_value > 0.05 else 'Likely not normal'}")
    else:
        print("Sample too large for Shapiro-Wilk test")
    
    # D'Agostino and Pearson's test (good for larger samples)
    stat, p_value = stats.normaltest(data)
    print(f"D'Agostino-Pearson: statistic={stat:.4f}, p-value={p_value:.6f}")
    print(f"Normality: {'Likely normal' if p_value > 0.05 else 'Likely not normal'}")

def main():
    print("Hamming Distance Experiment - Normality Testing Example")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample dataset with {len(df)} rows")
    print(f"Methods: {df['method'].unique()}")
    print(f"Variables: {[col for col in df.columns if col not in ['pattern_idx', 'method']]}")
    
    # Test each method and variable
    variables_to_test = [
        'overall_bitwise_fraction_diff',
        'per_gate_mean_hamming', 
        'final_hard_accuracy',
        'final_hard_loss'
    ]
    
    for method in df['method'].unique():
        print(f"\n{'='*40}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*40}")
        
        method_data = df[df['method'] == method]
        
        for var in variables_to_test:
            if var in method_data.columns:
                quick_normality_test(method_data[var], f"{method.upper()} - {var}")
    
    # Save sample data for testing
    output_file = "sample_hamming_distance_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSample data saved to: {output_file}")
    print(f"\nTo test with the full normality analysis script:")
    print(f"python test_normality_hamming_distance.py --csv-path {output_file}")

if __name__ == "__main__":
    main()
