"""
Generate K vs Power and K vs mAP plots from power_map_per5images.csv
This script works with any number of images (3, 5, etc.)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

try:
    # Read the CSV file
    df = pd.read_csv('power_map_per5images.csv')
    
    # Extract unique K values and aggregate data
    k_values = df['K'].unique()
    
    results = []
    for k in k_values:
        k_data = df[df['K'] == k]
        results.append({
            'K': k,
            'threads': k_data['Threads'].iloc[0],
            'mAP': k_data['mAP'].iloc[0],
            'mAP_50': k_data['mAP_50'].iloc[0],
            'power': k_data['Avg_Power_W'].mean(),
            'energy': k_data['Avg_Energy_J'].mean(),
            'time': k_data['Avg_Time_s'].mean()
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: K vs mAP
    ax1.plot(results_df['K'], results_df['mAP'], 'go-', linewidth=3, markersize=10, label='mAP @ IoU=0.50:0.95')
    ax1.plot(results_df['K'], results_df['mAP_50'], 'b^--', linewidth=2.5, markersize=8, label='mAP @ IoU=0.50')
    ax1.set_xlabel('K (Number of Queries)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Average Precision (mAP)', fontsize=14, fontweight='bold')
    ax1.set_title('mAP vs K (Optimal CPU Configuration)', fontsize=16, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=11)
    
    # Annotate max mAP
    max_idx = results_df['mAP'].idxmax()
    ax1.annotate(f'Max: {results_df.loc[max_idx, "mAP"]:.4f}\nK={results_df.loc[max_idx, "K"]:.0f}',
                xy=(results_df.loc[max_idx, 'K'], results_df.loc[max_idx, 'mAP']),
                xytext=(results_df.loc[max_idx, 'K']*0.3, results_df.loc[max_idx, 'mAP']-0.05),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # Plot 2: K vs Power
    ax2.plot(results_df['K'], results_df['power'], 'ro-', linewidth=3, markersize=10)
    ax2.set_xlabel('K (Number of Queries)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Power (W)', fontsize=14, fontweight='bold')
    ax2.set_title('Power vs K (Optimal CPU Configuration)', fontsize=16, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=11)
    
    # Add power range annotations based on data
    low_k_power = results_df[results_df['K'] <= 20]['power'].mean()
    mid_k_power = results_df[(results_df['K'] > 50) & (results_df['K'] <= 300)]['power'].mean()
    high_k_power = results_df[results_df['K'] > 300]['power'].mean()
    
    # Color regions based on power levels
    if len(results_df[results_df['K'] <= 20]) > 0:
        ax2.axhspan(low_k_power-5, low_k_power+5, alpha=0.2, color='green', label=f'Low K: ~{low_k_power:.1f}W')
    if len(results_df[(results_df['K'] > 50) & (results_df['K'] <= 300)]) > 0:
        ax2.axhspan(mid_k_power-10, mid_k_power+10, alpha=0.2, color='yellow', label=f'Mid K: ~{mid_k_power:.1f}W')
    if len(results_df[results_df['K'] > 300]) > 0:
        ax2.axhspan(high_k_power-15, high_k_power+15, alpha=0.2, color='red', label=f'High K: ~{high_k_power:.1f}W')
    
    ax2.legend(fontsize=10, loc='upper left')
    
    num_images = df['K'].value_counts().iloc[0]  # Count how many rows per K
    plt.suptitle(f'RT-DETR: Power and Accuracy Analysis ({num_images} Images, Optimal CPU)', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('k_vs_power_map_3images.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: k_vs_power_map_3images.png")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'K':>6} | {'Thr':>4} | {'mAP':>8} | {'Power(W)':>9} | {'Energy(J)':>10} | {'Time(s)':>8}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['K']:>6.0f} | {row['threads']:>4.0f} | {row['mAP']:>8.4f} | "
              f"{row['power']:>9.2f} | {row['energy']:>10.2f} | {row['time']:>8.4f}")
    print("="*70)
    
    print(f"\nKey Results:")
    print(f"  Best mAP: {results_df['mAP'].max():.4f} @ K={results_df.loc[results_df['mAP'].idxmax(), 'K']:.0f}")
    print(f"  Power Range: {results_df['power'].min():.2f}W - {results_df['power'].max():.2f}W")
    print(f"  Power increases with K: {results_df['power'].max() - results_df['power'].min():.2f}W increase")
    print()

except FileNotFoundError:
    print("Error: power_map_per5images.csv not found. Script still running?")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
