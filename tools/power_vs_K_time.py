"""
Power vs K at Different Execution Times

This script measures decoder power consumption across K values (5-1500)
while controlling execution time by adjusting the number of CPU threads.

For each target execution time (e.g., 1s, 2s, 3s, etc.):
- Dynamically adjust thread count to achieve target time
- Measure power at each K value (5, 10, 15, ..., 1500)
- Generate graphs showing Power vs K for each time target

Strategy:
- Low K + target time → fewer threads → lower power
- High K + target time → more threads → higher power
- This shows the power-time tradeoff across different K values
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
import psutil
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from src.core import YAMLConfig
from src.solver import TASKS


# ============================================================
# PROCESS CPU MONITOR
# ============================================================

class ProcessCPUMonitor:
    """Measures power using CPU utilization and frequency scaling."""
    
    def __init__(self, cpu_tdp=15.0):
        self.cpu_tdp = cpu_tdp
        self.num_cores = psutil.cpu_count(logical=False) or 4
        self.process = psutil.Process()
        
        try:
            freq_info = psutil.cpu_freq()
            self.freq_max = freq_info.max if freq_info and freq_info.max > 0 else None
        except Exception:
            self.freq_max = None
        
    def measure(self, func):
        """Measure time and power for a function."""
        cpu_before = self.process.cpu_times()
        cpu_start = cpu_before.user + cpu_before.system
        
        try:
            freq_start = psutil.cpu_freq()
        except Exception:
            freq_start = None
        
        wall_start = time.perf_counter()
        result = func()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        wall_end = time.perf_counter()
        cpu_after = self.process.cpu_times()
        cpu_end = cpu_after.user + cpu_after.system
        
        try:
            freq_end = psutil.cpu_freq()
            if freq_start and freq_end and self.freq_max and self.freq_max > 0:
                avg_freq = (freq_start.current + freq_end.current) / 2
                freq_ratio = avg_freq / self.freq_max
            else:
                freq_ratio = 1.0
        except Exception:
            freq_ratio = 1.0
        
        wall_time = wall_end - wall_start
        cpu_time = cpu_end - cpu_start
        
        raw_cpu_utilization = cpu_time / wall_time if wall_time > 0 else 0.0
        cpu_utilization = min(raw_cpu_utilization / self.num_cores, 1.0)
        
        total_tdp = self.cpu_tdp * self.num_cores
        power = total_tdp * cpu_utilization * freq_ratio
        power = min(power, total_tdp)
        
        return {
            "result": result,
            "time_s": wall_time,
            "cpu_time_s": cpu_time,
            "power_W": power,
            "cpu_utilization": cpu_utilization,
            "freq_ratio": freq_ratio,
        }


# ============================================================
# THREAD CALIBRATION WITH ITERATIVE ADJUSTMENT
# ============================================================

def calibrate_threads_for_target_time(k_value, target_time, max_cores, decoder, encoder_output, batch, pm, tolerance=0.20, max_iterations=3):
    """
    Iteratively calibrate thread count to achieve target execution time within tolerance.
    
    Args:
        k_value: Number of queries
        target_time: Target execution time in seconds
        max_cores: Maximum CPU cores available
        decoder: The decoder model to measure
        encoder_output: Pre-computed encoder output
        batch: Input batch for decoder
        pm: ProcessCPUMonitor instance
        tolerance: Acceptable deviation from target (default: ±20%)
        max_iterations: Maximum calibration attempts (default: 3)
    
    Returns:
        num_threads: Calibrated thread count
        actual_time: Measured execution time with calibrated threads
    """
    # Start with heuristic estimate
    k_scaling = k_value / 100.0
    time_scaling = 1.0 / target_time
    thread_ratio = (k_scaling * time_scaling) / 3.0
    num_threads = max(1, min(max_cores, int(thread_ratio * max_cores)))
    
    prev_threads = -1
    
    # Iterative calibration
    for iteration in range(max_iterations):
        # Set threads
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        torch.set_num_threads(num_threads)
        
        # Quick warm-up (only 1 iteration)
        with torch.no_grad():
            _ = decoder(encoder_output, batch)
        
        # Measure actual time
        def run_decoder():
            with torch.no_grad():
                _ = decoder(encoder_output, batch)
            return None
        
        stats = pm.measure(run_decoder)
        actual_time = stats['time_s']
        
        # Check if within tolerance
        time_diff = actual_time - target_time
        relative_error = abs(time_diff) / target_time
        
        if relative_error <= tolerance:
            # Within acceptable range
            return num_threads, actual_time
        
        # Adjust threads based on how far off we are
        if actual_time > target_time:
            # Too slow, need more threads
            adjustment = min(1.5, target_time / actual_time)
            num_threads = min(max_cores, int(num_threads * adjustment) + 1)
        else:
            # Too fast, need fewer threads  
            adjustment = actual_time / target_time
            num_threads = max(1, int(num_threads * adjustment))
        
        # Safety check: if threads didn't change, break
        if num_threads == prev_threads:
            break
        prev_threads = num_threads
    
    # Return best result after max iterations
    return num_threads, actual_time


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment_for_time_target(args, batches, pm, k_values, target_time):
    """
    Measure power for decoder across K values with fixed target execution time.
    Thread count is dynamically adjusted per K to maintain target time.
    
    Returns: list of dicts with results for each K
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build base model for encoder measurement
    cfg = YAMLConfig(args.config, resume=args.resume)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    base_model = solver.model.to(device).eval()
    
    first_batch = batches[0]
    
    # Warm-up
    print(f"    [Warm-up for target time {target_time}s...]", flush=True)
    with torch.no_grad():
        for _ in range(3):
            _ = base_model(first_batch)
    
    # ========================================
    # MEASURE ENCODER (once for this target time)
    # ========================================
    def run_encoder_all_batches():
        encoder_outputs = []
        with torch.no_grad():
            for batch in batches:
                feats = base_model.backbone(batch)
                enc_out = base_model.encoder(feats)
                encoder_outputs.append(enc_out)
        return encoder_outputs
    
    encoder_stats = pm.measure(run_encoder_all_batches)
    encoder_outputs = encoder_stats["result"]
    
    print(f"    Encoder: Time={encoder_stats['time_s']:.4f}s, Power={encoder_stats['power_W']:.2f}W")
    
    # ========================================
    # MEASURE DECODER for each K with calibrated threads
    # ========================================
    results = []
    
    # Store original thread setting
    original_threads = os.environ.get('OMP_NUM_THREADS', str(pm.num_cores))
    
    # Build a single decoder that we'll measure with different K values
    # We only measure decoder time/power, not model construction time
    decoder = base_model.decoder
    
    print(f"    [Calibrating threads for each K value...]")
    
    for k in k_values:
        # Iteratively calibrate thread count for this K and target time
        num_threads, calibration_time = calibrate_threads_for_target_time(
            k, target_time, pm.num_cores, decoder, encoder_outputs[0], batches[0], pm
        )
        
        # Final measurement with calibrated threads
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        torch.set_num_threads(num_threads)
        
        # Measure decoder across all batches
        def run_decoder_batches():
            with torch.no_grad():
                for i, batch in enumerate(batches):
                    _ = decoder(encoder_outputs[i], batch)
            return None
        
        times = []
        powers = []
        for _ in range(args.num_iterations):
            stats = pm.measure(run_decoder_batches)
            times.append(stats['time_s'])
            powers.append(stats['power_W'])
        
        avg_time = sum(times) / len(times)
        avg_power = sum(powers) / len(powers)
        std_power = (sum((p - avg_power)**2 for p in powers) / len(powers)) ** 0.5
        
        # Calculate energy and deviation from target
        dec_energy = avg_power * avg_time
        time_error = abs(avg_time - target_time) / target_time * 100  # percentage
        
        results.append({
            "K": k,
            "target_time": target_time,
            "num_threads": num_threads,
            "dec_time": avg_time,
            "dec_power": avg_power,
            "dec_power_std": std_power,
            "dec_energy": dec_energy,
            "cpu_util": stats["cpu_utilization"],
            "freq_ratio": stats["freq_ratio"],
            "time_error_pct": time_error,
        })
        
        # Print progress
        if k % 100 == 0 or k == k_values[0] or k == k_values[-1]:
            print(f"      K={k:>4}: Power={avg_power:.2f}W ±{std_power:.1f}W, "
                  f"Time={avg_time:.4f}s (target={target_time:.1f}s, error={time_error:.1f}%), "
                  f"{num_threads}/{pm.num_cores} threads", flush=True)
    
    # Restore original thread setting
    os.environ['OMP_NUM_THREADS'] = original_threads
    torch.set_num_threads(int(original_threads))
    
    return results


# ============================================================
# MAIN
# ============================================================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device}")
    print(f"[INFO] Number of images: {args.num_images}")
    
    pm = ProcessCPUMonitor(cpu_tdp=args.cpu_tdp)
    total_tdp = pm.cpu_tdp * pm.num_cores
    print(f"[ProcessCPUMonitor] TDP/core={pm.cpu_tdp}W, Cores={pm.num_cores}, TDP_total={total_tdp}W")
    
    # -------------------------------
    # Load COCO + images
    # -------------------------------
    coco = COCO(args.gt_json)
    image_ids = coco.getImgIds()[:args.num_images]
    
    all_tensors = []
    for img_id in image_ids:
        info = coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(args.image_dir, info["file_name"])).convert("RGB")
        img = img.resize((640, 640))
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        all_tensors.append(t)
    
    batch_size = min(args.batch_size, len(all_tensors))
    batches = []
    for i in range(0, len(all_tensors), batch_size):
        batch = torch.stack(all_tensors[i:i+batch_size]).to(device)
        batches.append(batch)
    
    print(f"[INFO] Loaded {len(all_tensors)} images in {len(batches)} batches (batch_size={batch_size})")
    
    # K values to test
    k_values = [5, 10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 500, 1000, 1500]
    print(f"[INFO] K values to test: {k_values}")
    
    # Parse target times from command line
    target_times = [float(t) for t in args.target_times.split(",")]
    print(f"[INFO] Target execution times: {target_times}")
    
    # -------------------------------
    # Run experiments for each target time
    # -------------------------------
    all_results = {}  # {target_time: [results]}
    
    print(f"\n{'='*80}")
    print(f"POWER vs K AT DIFFERENT TARGET TIMES")
    print(f"{'='*80}\n")
    
    for target_time in target_times:
        print(f"\n[Target Time: {target_time}s]")
        print(f"{'-'*80}")
        
        results = run_experiment_for_time_target(args, batches, pm, k_values, target_time)
        all_results[target_time] = results
        
        # Summary for this target time
        avg_power = sum(r["dec_power"] for r in results) / len(results)
        min_power = min(r["dec_power"] for r in results)
        max_power = max(r["dec_power"] for r in results)
        
        print(f"\n  [Summary for {target_time}s target]")
        print(f"    Average Power: {avg_power:.2f}W")
        print(f"    Min Power:     {min_power:.2f}W (K={[r['K'] for r in results if r['dec_power'] == min_power][0]})")
        print(f"    Max Power:     {max_power:.2f}W (K={[r['K'] for r in results if r['dec_power'] == max_power][0]})")
    
    # -------------------------------
    # Save results to CSV
    # -------------------------------
    csv_path = "power_vs_K_different_times.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "K", "Target_Time_s", "Actual_Decoder_Time_s", "Decoder_Power_W"
        ])
        
        for target_time, results in sorted(all_results.items()):
            for r in results:
                writer.writerow([
                    r["K"], target_time, f"{r['dec_time']:.4f}", f"{r['dec_power']:.2f}"
                ])
    
    print(f"\n[INFO] Results saved to: {csv_path}")
    
    # -------------------------------
    # Generate Visualization: Power vs K for Different Times
    # -------------------------------
    print(f"\n[INFO] Generating visualization...")
    
    # Create main figure: one subplot for each target time
    num_times = len(target_times)
    cols = min(3, num_times)
    rows = (num_times + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 5*rows))
    if num_times == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_times > 1 else [axes]
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, num_times))
    
    for idx, target_time in enumerate(sorted(target_times)):
        results = all_results[target_time]
        k_vals = [r["K"] for r in results]
        powers = [r["dec_power"] for r in results]
        
        ax = axes[idx]
        ax.plot(k_vals, powers, 'o-', color=colors[idx], linewidth=2, markersize=5)
        ax.set_xlabel('K (Number of Queries)', fontsize=12)
        ax.set_ylabel('Decoder Power (W)', fontsize=12)
        ax.set_title(f'Target Time: {target_time}s', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics annotation
        avg_power = sum(powers) / len(powers)
        min_power = min(powers)
        max_power = max(powers)
        ax.text(0.05, 0.95, f'Avg: {avg_power:.1f}W\nMin: {min_power:.1f}W\nMax: {max_power:.1f}W',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Hide extra subplots if any
    for idx in range(num_times, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    fig_path = "power_vs_K_different_times.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {fig_path}")
    plt.close()
    
    # Create combined plot: all times on same graph
    fig2, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    for idx, target_time in enumerate(sorted(target_times)):
        results = all_results[target_time]
        k_vals = [r["K"] for r in results]
        powers = [r["dec_power"] for r in results]
        
        ax.plot(k_vals, powers, 'o-', color=colors[idx], 
                linewidth=2, markersize=5, label=f'{target_time}s target', alpha=0.8)
    
    ax.set_xlabel('K (Number of Queries)', fontsize=14)
    ax.set_ylabel('Decoder Power (W)', fontsize=14)
    ax.set_title('Power vs K at Different Target Execution Times', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig2_path = "power_vs_K_all_times_combined.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Combined figure saved to: {fig2_path}")
    plt.close()
    
    # Create heatmap: Time vs K vs Power
    fig3, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Prepare data for heatmap
    k_values_sorted = sorted(set(k_vals))
    times_sorted = sorted(target_times)
    power_matrix = np.zeros((len(times_sorted), len(k_values_sorted)))
    
    for i, target_time in enumerate(times_sorted):
        results = all_results[target_time]
        for r in results:
            j = k_values_sorted.index(r["K"])
            power_matrix[i, j] = r["dec_power"]
    
    im = ax.imshow(power_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(0, len(k_values_sorted), max(1, len(k_values_sorted)//10)))
    ax.set_xticklabels([k_values_sorted[i] for i in range(0, len(k_values_sorted), max(1, len(k_values_sorted)//10))])
    ax.set_yticks(range(len(times_sorted)))
    ax.set_yticklabels([f'{t}s' for t in times_sorted])
    
    ax.set_xlabel('K (Number of Queries)', fontsize=14)
    ax.set_ylabel('Target Execution Time', fontsize=14)
    ax.set_title('Decoder Power Heatmap: Time × K', fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (W)', fontsize=12)
    
    plt.tight_layout()
    
    fig3_path = "power_heatmap_time_vs_K.png"
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Heatmap saved to: {fig3_path}")
    plt.close()
    
    # Create decoder time plot: Time vs K for different targets
    fig4, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    for idx, target_time in enumerate(sorted(target_times)):
        results = all_results[target_time]
        k_vals = [r["K"] for r in results]
        dec_times = [r["dec_time"] for r in results]
        
        ax.plot(k_vals, dec_times, 'o-', color=colors[idx], 
                linewidth=2, markersize=5, label=f'{target_time}s target', alpha=0.8)
    
    ax.set_xlabel('K (Number of Queries)', fontsize=14)
    ax.set_ylabel('Decoder Time (seconds)', fontsize=14)
    ax.set_title('Decoder Time vs K at Different Target Execution Times', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal reference lines for target times
    for target_time in target_times:
        ax.axhline(y=target_time, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    
    fig4_path = "decoder_time_vs_K_different_targets.png"
    plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Decoder time plot saved to: {fig4_path}")
    plt.close()
    
    # Create decoder time heatmap
    fig5, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Prepare data for heatmap
    time_matrix = np.zeros((len(times_sorted), len(k_values_sorted)))
    
    for i, target_time in enumerate(times_sorted):
        results = all_results[target_time]
        for r in results:
            j = k_values_sorted.index(r["K"])
            time_matrix[i, j] = r["dec_time"]
    
    im = ax.imshow(time_matrix, aspect='auto', cmap='plasma', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(0, len(k_values_sorted), max(1, len(k_values_sorted)//10)))
    ax.set_xticklabels([k_values_sorted[i] for i in range(0, len(k_values_sorted), max(1, len(k_values_sorted)//10))])
    ax.set_yticks(range(len(times_sorted)))
    ax.set_yticklabels([f'{t}s' for t in times_sorted])
    
    ax.set_xlabel('K (Number of Queries)', fontsize=14)
    ax.set_ylabel('Target Execution Time', fontsize=14)
    ax.set_title('Decoder Execution Time Heatmap: Time × K', fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Actual Time (s)', fontsize=12)
    
    plt.tight_layout()
    
    fig5_path = "decoder_time_heatmap_time_vs_K.png"
    plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Decoder time heatmap saved to: {fig5_path}")
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"COMPLETE! Generated 5 visualizations:")
    print(f"  1. {fig_path} - Individual power plots for each time")
    print(f"  2. {fig2_path} - Combined power plot with all times")
    print(f"  3. {fig3_path} - Power heatmap showing Time × K × Power")
    print(f"  4. {fig4_path} - Decoder time vs K for different targets")
    print(f"  5. {fig5_path} - Decoder time heatmap showing Time × K × Actual Time")
    print(f"{'='*80}\n")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("RT-DETR Power vs K at Different Target Times")
    parser.add_argument("-c", "--config",
                        default="./configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
    parser.add_argument("-r", "--resume",
                        default="rtdetr_r50vd_6x_coco_from_paddle.pth")
    parser.add_argument("--image-dir",
                        default="./dataset/coco/subset_10/images")
    parser.add_argument("--gt-json",
                        default="./dataset/coco/subset_10/instances_train2017.json")
    parser.add_argument("--num-images", type=int, default=10,
                        help="Number of images to process (default: 10)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size for processing (default: 10)")
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of iterations to average (default: 3)")
    parser.add_argument("--cpu-tdp", type=float, default=15.0,
                        help="CPU TDP in Watts per core")
    parser.add_argument("--target-times", type=str, default="1.0,2.0,3.0,4.0,5.0",
                        help="Comma-separated target execution times in seconds (default: 1.0,2.0,3.0,4.0,5.0)")

    args = parser.parse_args()
    main(args)
