"""
Generate Power vs K and mAP vs K plots from CSV data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the minimum power results
df = pd.read_csv('min_power_map_results_5images.csv')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: mAP vs K
k_vals = df['K'].values
map_vals = df['Overall_mAP'].values
map50_vals = df['mAP_50'].values
map75_vals = df['mAP_75'].values

ax1.plot(k_vals, map_vals, 'go-', linewidth=3, markersize=10, label='mAP @ IoU=0.50:0.95')
ax1.plot(k_vals, map50_vals, 'b^--', linewidth=2.5, markersize=8, label='mAP @ IoU=0.50')
ax1.plot(k_vals, map75_vals, 'rs--', linewidth=2, markersize=7, label='mAP @ IoU=0.75')

ax1.set_xlabel('K (Number of Queries)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Mean Average Precision (mAP)', fontsize=14, fontweight='bold')
ax1.set_title('mAP vs K (Minimum Power Configuration)', fontsize=16, fontweight='bold')
ax1.set_xscale('log')
ax1.legend(fontsize=12, loc='lower right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=11)

# Add annotations for key points
max_map_idx = np.argmax(map_vals)
ax1.annotate(f'Max mAP: {map_vals[max_map_idx]:.4f}\nK={k_vals[max_map_idx]}',
             xy=(k_vals[max_map_idx], map_vals[max_map_idx]),
             xytext=(k_vals[max_map_idx]*0.3, map_vals[max_map_idx]-0.05),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', lw=2))

# Plot 2: Power vs K
power_vals = df['Avg_Power_W'].values

ax2.plot(k_vals, power_vals, 'ro-', linewidth=3, markersize=10)
ax2.set_xlabel('K (Number of Queries)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Average Power (W)', fontsize=14, fontweight='bold')
ax2.set_title('Power vs K (Minimum Power Configuration)', fontsize=16, fontweight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(labelsize=11)

# Add power stability annotations
ax2.axhspan(14.5, 15.5, alpha=0.3, color='green', label='Stable Low Power')
ax2.legend(fontsize=12, loc='upper left')

# Add text annotations for power range
ax2.text(10, 15.5, f'~{power_vals[0]:.1f}W\n(1 thread)', 
         fontsize=11, ha='center', va='bottom', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))

plt.suptitle('RT-DETR: Power and Accuracy Analysis (5 Images, Minimum Power)', 
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('power_map_vs_k_plots.png', dpi=150, bbox_inches='tight')
print("✓ Saved: power_map_vs_k_plots.png")

# Also create individual plots for clarity
fig1, ax = plt.subplots(figsize=(10, 7))
ax.plot(k_vals, map_vals, 'go-', linewidth=3.5, markersize=12, label='mAP @ IoU=0.50:0.95')
ax.plot(k_vals, map50_vals, 'b^--', linewidth=3, markersize=10, label='mAP @ IoU=0.50')
ax.set_xlabel('K (Number of Queries)', fontsize=16, fontweight='bold')
ax.set_ylabel('Mean Average Precision (mAP)', fontsize=16, fontweight='bold')
ax.set_title('mAP vs K (Minimum Power Configuration - 1 Thread)', fontsize=18, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=14, loc='lower right')
ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
ax.tick_params(labelsize=13)

# Enhanced annotations
max_map_idx = np.argmax(map_vals)
ax.annotate(f'Maximum: {map_vals[max_map_idx]:.4f}\nK={k_vals[max_map_idx]}',
            xy=(k_vals[max_map_idx], map_vals[max_map_idx]),
            xytext=(k_vals[max_map_idx]*0.2, map_vals[max_map_idx]-0.08),
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='yellow', alpha=0.9),
            arrowprops=dict(arrowstyle='->', lw=3, color='red'))

plt.tight_layout()
plt.savefig('map_vs_k_single.png', dpi=150, bbox_inches='tight')
print("✓ Saved: map_vs_k_single.png")

fig2, ax = plt.subplots(figsize=(10, 7))
ax.plot(k_vals, power_vals, 'ro-', linewidth=3.5, markersize=12)
ax.set_xlabel('K (Number of Queries)', fontsize=16, fontweight='bold')
ax.set_ylabel('Average Power (W)', fontsize=16, fontweight='bold')
ax.set_title('Power vs K (Minimum Power: 1 Thread)', fontsize=18, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
ax.tick_params(labelsize=13)

# Power stability region
ax.axhspan(14.5, 15.5, alpha=0.3, color='green', label='Stable Power Zone (±2%)')
ax.axhline(y=15.0, color='blue', linestyle='--', linewidth=2, alpha=0.6, label='Target: 15W')
ax.legend(fontsize=13, loc='upper right')

# Add statistics
mean_power = np.mean(power_vals)
std_power = np.std(power_vals)
ax.text(0.02, 0.98, f'Mean Power: {mean_power:.2f}W\nStd Dev: {std_power:.2f}W\nCV: {(std_power/mean_power)*100:.2f}%',
        transform=ax.transAxes, fontsize=13, fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='lightblue', alpha=0.9))

plt.tight_layout()
plt.savefig('power_vs_k_single.png', dpi=150, bbox_inches='tight')
print("✓ Saved: power_vs_k_single.png")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nmAP Statistics:")
print(f"  Maximum mAP:     {max(map_vals):.4f} @ K={k_vals[np.argmax(map_vals)]}")
print(f"  Minimum mAP:     {min(map_vals):.4f} @ K={k_vals[np.argmin(map_vals)]}")
print(f"  Mean mAP:        {np.mean(map_vals):.4f}")
print(f"  Std Dev:         {np.std(map_vals):.4f}")

print(f"\nPower Statistics:")
print(f"  Maximum Power:   {max(power_vals):.2f}W @ K={k_vals[np.argmax(power_vals)]}")
print(f"  Minimum Power:   {min(power_vals):.2f}W @ K={k_vals[np.argmin(power_vals)]}")
print(f"  Mean Power:      {mean_power:.2f}W")
print(f"  Std Dev:         {std_power:.2f}W")
print(f"  Coefficient of Variation: {(std_power/mean_power)*100:.3f}%")

print(f"\nEnergy Statistics:")
energy_vals = df['Avg_Energy_per_Image_J'].values
print(f"  Maximum Energy:  {max(energy_vals):.2f}J @ K={k_vals[np.argmax(energy_vals)]}")
print(f"  Minimum Energy:  {min(energy_vals):.2f}J @ K={k_vals[np.argmin(energy_vals)]}")
print(f"  Mean Energy:     {np.mean(energy_vals):.2f}J")

print(f"\nTime Statistics:")
time_vals = df['Avg_Time_per_Image_s'].values
print(f"  Maximum Time:    {max(time_vals):.3f}s @ K={k_vals[np.argmax(time_vals)]}")
print(f"  Minimum Time:    {min(time_vals):.3f}s @ K={k_vals[np.argmin(time_vals)]}")
print(f"  Mean Time:       {np.mean(time_vals):.3f}s")

print("\n" + "="*60)
print("PLOTS GENERATED SUCCESSFULLY!")
print("="*60)
print("\nFiles created:")
print("  1. power_map_vs_k_plots.png   (Combined: mAP + Power)")
print("  2. map_vs_k_single.png        (mAP only, large)")
print("  3. power_vs_k_single.png      (Power only, large)")
print("\n")
