#!/usr/bin/env python3
"""
Test script for colorblind-friendly color cycles in figure 3 background shading.
Tests different slices of the CB_color_cycle to find the best combination.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Colorblind-friendly color cycle
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def create_test_plot(colors, title_suffix, alpha_val, save_path=None):
    """Create a test plot with the given colors for background regions."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create some sample data to show the colors in context
    x = np.linspace(0, 20, 100)
    y = 0.7 + 0.2 * np.sin(x * 0.5) + 0.1 * np.random.randn(100)
    
    # Plot sample data
    ax.plot(x, y, 'k-', linewidth=2, alpha=0.8, label='Sample Performance')
    
    # Add the three background regions with the test colors
    ax.axvspan(0, 2, alpha=alpha_val, color=colors[0], label='Pre-damage')  # First color
    ax.axvspan(2, 3, alpha=alpha_val, color=colors[1], label='Damage')      # Second color  
    ax.axvspan(3, 17, alpha=alpha_val, color=colors[2], label='Recovery')   # Third color
    
    # Add damage line
    ax.axvline(x=2, color=colors[1], linestyle='--', alpha=0.8, linewidth=2, label='Damage Point')
    
    # Styling
    ax.set_xlabel('Message Steps', fontsize=16)
    ax.set_ylabel('Hard Accuracy', fontsize=16)
    ax.set_title(f'Colorblind-Friendly Colors Test - {title_suffix} (α={alpha_val})', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlim(0, 20)
    
    # Create legend
    legend = ax.legend(loc='lower right', fontsize=12)
    legend.get_frame().set_alpha(1.0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig

def test_color_cycles():
    """Test different slices of the color cycle with different alpha values."""
    
    # Define the four test slices
    test_slices = [
        {
            'name': 'Slice 0,1,2',
            'colors': [CB_color_cycle[0], CB_color_cycle[1], CB_color_cycle[2]],
            'description': 'Blue, Orange, Green'
        },
        {
            'name': 'Slice 1,2,3', 
            'colors': [CB_color_cycle[1], CB_color_cycle[2], CB_color_cycle[3]],
            'description': 'Orange, Green, Pink'
        },
        {
            'name': 'Slice 4,5,6',
            'colors': [CB_color_cycle[4], CB_color_cycle[5], CB_color_cycle[6]],
            'description': 'Brown, Purple, Gray'
        },
        {
            'name': 'Slice 7,8,0',
            'colors': [CB_color_cycle[7], CB_color_cycle[8], CB_color_cycle[0]],
            'description': 'Red, Yellow, Blue'
        }
    ]
    
    # Test all alpha values
    alpha_values = [0.1, 0.15, 0.2]
    
    # Create plots for each slice and alpha combination
    for alpha_val in alpha_values:
        print(f"\n=== Testing Alpha = {alpha_val} ===")
        for i, test in enumerate(test_slices):
            print(f"\nTesting {test['name']}: {test['description']}")
            print(f"Colors: {test['colors']}")
            
            fig = create_test_plot(
                colors=test['colors'],
                title_suffix=test['name'],
                alpha_val=alpha_val,
                save_path=f'/home/marcello/workspace/boolean_nca_cc/color_test_alpha{alpha_val}_{i+1}_{test["name"].replace(",", "_").replace(" ", "")}.png'
            )
            
            # Show the plot
            plt.show()
    
    # Create comparison plots for each alpha value
    for alpha_val in alpha_values:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, test in enumerate(test_slices):
            ax = axes[i]
            
            # Sample data
            x = np.linspace(0, 20, 100)
            y = 0.7 + 0.2 * np.sin(x * 0.5) + 0.1 * np.random.randn(100)
            
            # Plot sample data
            ax.plot(x, y, 'k-', linewidth=2, alpha=0.8)
            
            # Add background regions
            ax.axvspan(0, 2, alpha=alpha_val, color=test['colors'][0])
            ax.axvspan(2, 3, alpha=alpha_val, color=test['colors'][1])
            ax.axvspan(3, 17, alpha=alpha_val, color=test['colors'][2])
            ax.axvline(x=2, color=test['colors'][1], linestyle='--', alpha=0.8, linewidth=2)
            
            # Styling
            ax.set_xlabel('Message Steps', fontsize=12)
            ax.set_ylabel('Hard Accuracy', fontsize=12)
            ax.set_title(f'{test["name"]}\n{test["description"]} (α={alpha_val})', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.6, 1.0)
            ax.set_xlim(0, 20)
        
        plt.tight_layout()
        plt.savefig(f'/home/marcello/workspace/boolean_nca_cc/color_comparison_alpha{alpha_val}.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison plot to color_comparison_alpha{alpha_val}.png")
        plt.show()

def print_color_info():
    """Print detailed information about the color cycle."""
    print("Colorblind-Friendly Color Cycle:")
    print("=" * 50)
    for i, color in enumerate(CB_color_cycle):
        print(f"{i:2d}: {color:8s} - {color}")
    
    print("\nTest Slices:")
    print("-" * 30)
    print("Slice 0,1,2: Blue, Orange, Green")
    print("Slice 1,2,3: Orange, Green, Pink") 
    print("Slice -1,0,1: Yellow, Blue, Orange")

if __name__ == "__main__":
    print_color_info()
    test_color_cycles()
