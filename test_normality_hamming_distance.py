#!/usr/bin/env python3
"""
Normality testing for hamming distance experiment datapoints.

This script loads the CSV output from hamming_distance.py and tests the collected
datapoints for both backprop (BP) and GNN methods for normality using multiple
statistical tests.

Key datapoints collected:
- overall_bitwise_fraction_diff: fraction of differing bits across all counted entries
- per_gate_mean_hamming: mean over gates of per-table Hamming normalized by table size
- final_hard_accuracy: final hard accuracy after training/evaluation
- final_hard_loss: final hard loss after training/evaluation
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, normaltest, jarque_bera, anderson
import warnings
warnings.filterwarnings('ignore')

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib for box plots")

def load_hamming_distance_data(csv_path):
    """Load the CSV data from hamming distance experiment."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Methods present: {df['method'].unique()}")
    return df

def test_normality_comprehensive(data, method_name, variable_name, alpha=0.05):
    """
    Comprehensive normality testing using multiple statistical tests.
    
    Returns dict with test results and recommendations.
    """
    if len(data) < 3:
        return {"error": f"Insufficient data points ({len(data)}) for normality testing"}
    
    results = {
        "method": method_name,
        "variable": variable_name,
        "n_samples": len(data),
        "mean": np.mean(data),
        "std": np.std(data),
        "skewness": stats.skew(data),
        "kurtosis": stats.kurtosis(data),
        "tests": {}
    }
    
    # Remove any NaN values
    clean_data = data.dropna()
    if len(clean_data) < 3:
        return {"error": f"Insufficient clean data points ({len(clean_data)}) for normality testing"}
    
    # 1. Shapiro-Wilk test (best for small samples < 50)
    if len(clean_data) <= 5000:  # Shapiro-Wilk has computational limits
        try:
            shapiro_stat, shapiro_p = shapiro(clean_data)
            results["tests"]["shapiro_wilk"] = {
                "statistic": shapiro_stat,
                "p_value": shapiro_p,
                "is_normal": shapiro_p > alpha,
                "interpretation": "Normal" if shapiro_p > alpha else "Not normal"
            }
        except Exception as e:
            results["tests"]["shapiro_wilk"] = {"error": str(e)}
    
    # 2. D'Agostino and Pearson's test
    try:
        dagostino_stat, dagostino_p = normaltest(clean_data)
        results["tests"]["dagostino_pearson"] = {
            "statistic": dagostino_stat,
            "p_value": dagostino_p,
            "is_normal": dagostino_p > alpha,
            "interpretation": "Normal" if dagostino_p > alpha else "Not normal"
        }
    except Exception as e:
        results["tests"]["dagostino_pearson"] = {"error": str(e)}
    
    # 3. Jarque-Bera test
    try:
        jb_stat, jb_p = jarque_bera(clean_data)
        results["tests"]["jarque_bera"] = {
            "statistic": jb_stat,
            "p_value": jb_p,
            "is_normal": jb_p > alpha,
            "interpretation": "Normal" if jb_p > alpha else "Not normal"
        }
    except Exception as e:
        results["tests"]["jarque_bera"] = {"error": str(e)}
    
    # 4. Anderson-Darling test
    try:
        ad_stat, ad_critical, ad_significance = anderson(clean_data, dist='norm')
        # Use 5% significance level (index 2 in the critical values)
        ad_is_normal = ad_stat < ad_critical[2]
        results["tests"]["anderson_darling"] = {
            "statistic": ad_stat,
            "critical_5pct": ad_critical[2],
            "is_normal": ad_is_normal,
            "interpretation": "Normal" if ad_is_normal else "Not normal"
        }
    except Exception as e:
        results["tests"]["anderson_darling"] = {"error": str(e)}
    
    # Overall assessment
    normal_tests = []
    for test_name, test_result in results["tests"].items():
        if "is_normal" in test_result:
            normal_tests.append(test_result["is_normal"])
    
    if normal_tests:
        results["overall_assessment"] = {
            "normal_count": sum(normal_tests),
            "total_tests": len(normal_tests),
            "majority_normal": sum(normal_tests) > len(normal_tests) / 2,
            "consensus": "Likely normal" if sum(normal_tests) > len(normal_tests) / 2 else "Likely not normal"
        }
    else:
        results["overall_assessment"] = {"error": "No valid normality tests could be performed"}
    
    return results

def create_visualizations(df, output_dir):
    """Create comprehensive visualizations for normality assessment."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Key variables to analyze
    variables = [
        'overall_bitwise_fraction_diff',
        'per_gate_mean_hamming', 
        'final_hard_accuracy',
        'final_hard_loss'
    ]
    
    methods = df['method'].unique()
    
    # Create subplots for each variable
    for var in variables:
        if var not in df.columns:
            print(f"Warning: Variable {var} not found in data")
            continue
            
        fig, axes = plt.subplots(2, len(methods), figsize=(5*len(methods), 10))
        if len(methods) == 1:
            axes = axes.reshape(2, 1)
        
        for i, method in enumerate(methods):
            method_data = df[df['method'] == method][var].dropna()
            
            if len(method_data) == 0:
                continue
            
            # Histogram with normal overlay
            axes[0, i].hist(method_data, bins=min(30, len(method_data)//2), density=True, alpha=0.7, 
                           color='skyblue', edgecolor='black')
            
            # Overlay normal distribution
            mu, sigma = method_data.mean(), method_data.std()
            x = np.linspace(method_data.min(), method_data.max(), 100)
            normal_curve = stats.norm.pdf(x, mu, sigma)
            axes[0, i].plot(x, normal_curve, 'r-', linewidth=2, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
            axes[0, i].set_title(f'{method.upper()} - {var}\nHistogram')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(method_data, dist="norm", plot=axes[1, i])
            axes[1, i].set_title(f'{method.upper()} - {var}\nQ-Q Plot')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'normality_plots_{var}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Box plots comparing methods
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, var in enumerate(variables):
        if var not in df.columns:
            continue
            
        if HAS_SEABORN:
            sns.boxplot(data=df, x='method', y=var, ax=axes[i])
        else:
            # Fallback to matplotlib boxplot
            methods = df['method'].unique()
            box_data = [df[df['method'] == method][var].dropna() for method in methods]
            axes[i].boxplot(box_data, labels=methods)
        axes[i].set_title(f'Distribution Comparison: {var}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test normality of hamming distance experiment datapoints")
    parser.add_argument("--csv-path", type=str, required=True, 
                       help="Path to the summary.csv file from hamming_distance.py")
    parser.add_argument("--output-dir", type=str, default="normality_analysis",
                       help="Output directory for results and plots")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Significance level for normality tests")
    args = parser.parse_args()
    
    # Load data
    df = load_hamming_distance_data(args.csv_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Key variables to test
    variables = [
        'overall_bitwise_fraction_diff',
        'per_gate_mean_hamming', 
        'final_hard_accuracy',
        'final_hard_loss'
    ]
    
    # Test normality for each method and variable
    all_results = []
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        print(f"\n{'='*60}")
        print(f"TESTING NORMALITY FOR METHOD: {method.upper()}")
        print(f"{'='*60}")
        
        for var in variables:
            if var not in df.columns:
                print(f"Warning: Variable {var} not found in data")
                continue
                
            var_data = method_df[var].dropna()
            if len(var_data) == 0:
                print(f"Warning: No data for {var} in method {method}")
                continue
            
            print(f"\n--- Testing {var} ---")
            print(f"Sample size: {len(var_data)}")
            print(f"Mean: {var_data.mean():.6f}")
            print(f"Std: {var_data.std():.6f}")
            print(f"Min: {var_data.min():.6f}")
            print(f"Max: {var_data.max():.6f}")
            
            # Run normality tests
            results = test_normality_comprehensive(var_data, method, var, args.alpha)
            all_results.append(results)
            
            if "error" in results:
                print(f"Error: {results['error']}")
                continue
            
            # Print test results
            print("\nNormality Test Results:")
            for test_name, test_result in results["tests"].items():
                if "error" in test_result:
                    print(f"  {test_name}: ERROR - {test_result['error']}")
                else:
                    status = "✓ NORMAL" if test_result["is_normal"] else "✗ NOT NORMAL"
                    if "p_value" in test_result:
                        print(f"  {test_name}: {status} (p={test_result['p_value']:.6f})")
                    else:
                        # Anderson-Darling doesn't have p-value, show statistic instead
                        stat_key = "statistic" if "statistic" in test_result else "critical_5pct"
                        print(f"  {test_name}: {status} (stat={test_result.get(stat_key, 'N/A'):.6f})")
            
            # Overall assessment
            if "overall_assessment" in results:
                assessment = results["overall_assessment"]
                if "error" not in assessment:
                    print(f"\nOverall Assessment: {assessment['consensus']}")
                    print(f"  ({assessment['normal_count']}/{assessment['total_tests']} tests suggest normality)")
    
    # Create visualizations
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    create_visualizations(df, args.output_dir)
    print(f"Visualizations saved to: {args.output_dir}")
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, "normality_test_results.json")
    import json
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Detailed results saved to: {results_file}")
    
    # Summary report
    print(f"\n{'='*60}")
    print("SUMMARY REPORT")
    print(f"{'='*60}")
    
    summary_data = []
    for result in all_results:
        if "error" not in result and "overall_assessment" in result:
            assessment = result["overall_assessment"]
            if "error" not in assessment:
                summary_data.append({
                    "method": result["method"],
                    "variable": result["variable"],
                    "n_samples": result["n_samples"],
                    "consensus": assessment["consensus"],
                    "normal_tests": assessment["normal_count"],
                    "total_tests": assessment["total_tests"]
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_file = os.path.join(args.output_dir, "normality_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
    
    print(f"\nAnalysis complete. Check {args.output_dir} for all results and visualizations.")

if __name__ == "__main__":
    main()
