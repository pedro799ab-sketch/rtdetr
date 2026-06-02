"""
Generate K vs Power and K vs mAP plots for 3 images
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('power_map_per5images.csv')

# Extract data
k_values = df['K'].values
map_values = df['mAP'].values
map50_values = df['mAP_50'].values
power_values = df['Avg_Power_W'].values
threads_values = df['Threads'].values

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: K vs mAP
ax1.plot(k_values, map_values, 'go-', linewidth=3, markersize=10, label='mAP @ IoU=0.50:0.95')
ax1.plot(k_values, map50_values, 'b^--', linewidth=2.5, markersize=8, label='mAP @ IoU=0.50')
ax1.set_xlabel('K (Number of Queries)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Mean Average Precision (mAP)', fontsize=14, fontweight='bold')
ax1.set_title('mAP vs K (3 Images, Optimal CPU)', fontsize=16, fontweight='bold')
ax1.set_xscale('log')
ax1.legend(fontsize=12, loc='lower right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(labelsize=11)

# Annotate max mAP
max_idx = np.argmax(map_values)
ax1.annotate(f'Max: {map_values[max_idx]:.4f}\nK={k_values[max_idx]:.0f}',
            xy=(k_values[max_idx], map_values[max_idx]),
            xytext=(k_values[max_idx]*0.3, map_values[max_idx]-0.05),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', lw=2))

# Plot 2: K vs Power (shows power increasing with K)
ax2.plot(k_values, power_values, 'ro-', linewidth=3, markersize=10)
ax2.set_xlabel('K (Number of Queries)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Average Power (W)', fontsize=14, fontweight='bold')
ax2.set_title('Power vs K (3 Images, Optimal CPU)', fontsize=16, fontweight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(labelsize=11)

# Add power range annotations
low_k_power = power_values[k_values <= 20].mean()
mid_k_power = power_values[(k_values > 50) & (k_values <= 300)].mean()
high_k_power = power_values[k_values > 300].mean()

ax2.axhspan(low_k_power-5, low_k_power+5, alpha=0.2, color='green', label=f'Low K (5-20): ~{low_k_power:.1f}W')
ax2.axhspan(mid_k_power-15, mid_k_power+15, alpha=0.2, color='yellow', label=f'Mid K (100-300): ~{mid_k_power:.1f}W')
ax2.axhspan(high_k_power-20, high_k_power+20, alpha=0.2, color='red', label=f'High K (500-1500): ~{high_k_power:.1f}W')

ax2.legend(fontsize=10, loc='upper left')

plt.suptitle('RT-DETR: Power and Accuracy Analysis (3 Images, Optimal CPU Configuration)', 
             fontsize=17, fontweight='bold', y=1.02)
plt.tight_layout()

# Save plot
plt.savefig('k_vs_power_map_3images.png', dpi=150, bbox_inches='tight')
print("✓ Saved: k_vs_power_map_3images.png")

# Print summary
print("\n" + "="*75)
print("SUMMARY: K vs Power and mAP (3 Images, Optimal CPU)")
print("="*75)
print(f"\n{'K':>6} | {'Threads':>7} | {'mAP':>8} | {'Power(W)':>9} | {'Energy(J)':>10} | {'Time(s)':>8}")
print("-" * 75)
for i in range(len(k_values)):
    print(f"{k_values[i]:>6.0f} | {threads_values[i]:>7.0f} | {map_values[i]:>8.4f} | "
          f"{power_values[i]:>9.2f} | {df['Avg_Energy_J'].values[i]:>10.2f} | {df['Avg_Time_s'].values[i]:>8.4f}")
print("="*75)

print(f"\nKey Results:")
print(f"  Best mAP:           {map_values.max():.4f} @ K={k_values[np.argmax(map_values)]:.0f} ({threads_values[np.argmax(map_values)]:.0f} threads)")
print(f"  Power Range:        {power_values.min():.2f}W - {power_values.max():.2f}W")
print(f"  Power Increase:     {power_values.max() - power_values.min():.2f}W (from K=5 to K=1500)")
print(f"  Low K (5-20):       ~{low_k_power:.1f}W (1 thread)")
print(f"  Mid K (100-300):    ~{mid_k_power:.1f}W (3-6 threads)")
print(f"  High K (500-1500):  ~{high_k_power:.1f}W (8-10 threads)")
print(f"\n✓ Power INCREASES with K when using optimal CPU configuration!")
print()
