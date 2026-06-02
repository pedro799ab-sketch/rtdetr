"""
Power Minimization for RT-DETR with Dynamic Core Allocation

This script finds the optimal number of CPU cores to minimize power consumption
while measuring decoder performance across K values from 5 to 1500.

Strategy:
- Test multiple core configurations (1, 2, 4, 8, 16, etc.)
- For each configuration, measure power across all K values
- Find the configuration that minimizes total power consumption
- Generate comprehensive results showing power at each K value for each configuration
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
    """
    Measures power using CPU utilization and frequency scaling.
    Power = TDP × CPU_Utilization × (freq_current / freq_max)
    """
    
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
        """
        Measure time and power for a function.
        Returns dict with time_s, cpu_time_s, power_W, cpu_utilization, freq_ratio
        """
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
# CORE ALLOCATION STRATEGIES
# ============================================================

def generate_core_configs(max_cores):
    """
    Generate different core configurations to test.
    Returns list of thread counts to test.
    """
    configs = [1]  # Always test single-threaded
    
    # Powers of 2 up to max_cores
    power = 2
    while power <= max_cores:
        configs.append(power)
        power *= 2
    
    # Add max_cores if not already included
    if max_cores not in configs:
        configs.append(max_cores)
    
    # Add some intermediate values for better granularity
    for threads in [3, 6, 12]:
        if threads < max_cores and threads not in configs:
            configs.append(threads)
    
    return sorted(configs)


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment_for_config(args, batches, pm, k_values, num_threads):
    """
    Run experiment with a specific number of threads.
    Measure encoder once, then decoder for all K values.
    
    Returns: list of dicts with results for each K
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set thread count
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    torch.set_num_threads(num_threads)
    
    # Build base model
    cfg = YAMLConfig(args.config, resume=args.resume)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    base_model = solver.model.to(device).eval()
    
    first_batch = batches[0]
    
    # Warm-up
    print(f"    [Warm-up with {num_threads} threads...]", flush=True)
    with torch.no_grad():
        for _ in range(3):
            _ = base_model(first_batch)
    
    # ========================================
    # MEASURE ENCODER (once for this config)
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
    
    # ========================================
    # MEASURE DECODER for each K
    # ========================================
    results = []
    
    for k in k_values:
        cfg_k = YAMLConfig(args.config, resume=args.resume)
        
        # Override num_queries
        if "RTDETRTransformer" in cfg_k.yaml_cfg:
            cfg_k.yaml_cfg["RTDETRTransformer"]["num_queries"] = k
        if "RTDETRPostProcessor" in cfg_k.yaml_cfg:
            cfg_k.yaml_cfg["RTDETRPostProcessor"]["num_top_queries"] = k
        if "RTDETRTransformerv2" in cfg_k.yaml_cfg:
            cfg_k.yaml_cfg["RTDETRTransformerv2"]["num_queries"] = k
        
        solver_k = TASKS[cfg_k.yaml_cfg["task"]](cfg_k)
        solver_k._setup()
        model_k = solver_k.model.to(device).eval()
        
        # Warm-up decoder
        with torch.no_grad():
            for _ in range(2):
                _ = model_k.decoder(encoder_outputs[0], batches[0])
        
        # Measure decoder
        def run_decoder_batches():
            with torch.no_grad():
                for i, batch in enumerate(batches):
                    _ = model_k.decoder(encoder_outputs[i], batch)
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
        
        # Calculate energy
        enc_energy = encoder_stats["power_W"] * encoder_stats["time_s"]
        dec_energy = avg_power * avg_time
        total_energy = enc_energy + dec_energy
        total_time = encoder_stats["time_s"] + avg_time
        total_power = total_energy / total_time if total_time > 0 else 0
        
        results.append({
            "K": k,
            "num_threads": num_threads,
            "enc_time": encoder_stats["time_s"],
            "enc_power": encoder_stats["power_W"],
            "enc_energy": enc_energy,
            "dec_time": avg_time,
            "dec_power": avg_power,
            "dec_power_std": std_power,
            "dec_energy": dec_energy,
            "total_time": total_time,
            "total_power": total_power,
            "total_energy": total_energy,
            "cpu_util": stats["cpu_utilization"],
            "freq_ratio": stats["freq_ratio"],
        })
        
        # Print progress every 10 K values
        if k % 50 == 0 or k == k_values[0] or k == k_values[-1]:
            print(f"      K={k:>4}: Power={avg_power:.2f}W, Time={avg_time:.4f}s, "
                  f"CPU={stats['cpu_utilization']:.1%}", flush=True)
    
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
    
    # Generate K values from 5 to 1500
    k_values = list(range(5, 101, 5))  # 5, 10, 15, ..., 100
    k_values.extend(range(150, 501, 50))  # 150, 200, 250, ..., 500
    k_values.extend(range(600, 1501, 100))  # 600, 700, ..., 1500
    print(f"[INFO] K values to test: {len(k_values)} values from {k_values[0]} to {k_values[-1]}")
    
    # Generate core configurations
    core_configs = generate_core_configs(pm.num_cores)
    print(f"[INFO] Core configurations to test: {core_configs}")
    
    # -------------------------------
    # Run experiments for each core configuration
    # -------------------------------
    all_results = {}  # {num_threads: [results]}
    
    print(f"\n{'='*80}")
    print(f"POWER MINIMIZATION: Testing {len(core_configs)} core configurations")
    print(f"{'='*80}\n")
    
    for num_threads in core_configs:
        print(f"\n[Testing {num_threads}/{pm.num_cores} threads]")
        print(f"{'-'*80}")
        
        results = run_experiment_for_config(args, batches, pm, k_values, num_threads)
        all_results[num_threads] = results
        
        # Calculate summary statistics
        avg_power = sum(r["total_power"] for r in results) / len(results)
        total_energy = sum(r["total_energy"] for r in results)
        total_time = sum(r["total_time"] for r in results)
        
        print(f"\n  [Summary for {num_threads} threads]")
        print(f"    Average Power: {avg_power:.2f}W")
        print(f"    Total Energy:  {total_energy:.2f}J")
        print(f"    Total Time:    {total_time:.2f}s")
    
    # -------------------------------
    # Find optimal configuration
    # -------------------------------
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION RESULTS")
    print(f"{'='*80}\n")
    
    # Calculate average metrics for each configuration
    config_summary = []
    for num_threads, results in all_results.items():
        avg_power = sum(r["total_power"] for r in results) / len(results)
        total_energy = sum(r["total_energy"] for r in results)
        total_time = sum(r["total_time"] for r in results)
        
        config_summary.append({
            "threads": num_threads,
            "avg_power": avg_power,
            "total_energy": total_energy,
            "total_time": total_time,
        })
    
    # Sort by average power (ascending)
    config_summary_sorted = sorted(config_summary, key=lambda x: x["avg_power"])
    
    print(f"{'Threads':>8} | {'Avg Power (W)':>14} | {'Total Energy (J)':>17} | {'Total Time (s)':>15}")
    print(f"{'-'*80}")
    for cfg in config_summary_sorted:
        print(f"{cfg['threads']:>8} | {cfg['avg_power']:>14.2f} | {cfg['total_energy']:>17.2f} | {cfg['total_time']:>15.2f}")
    
    optimal_config = config_summary_sorted[0]
    print(f"\n[OPTIMAL] {optimal_config['threads']} threads → Avg Power: {optimal_config['avg_power']:.2f}W")
    
    # -------------------------------
    # Save detailed results to CSV
    # -------------------------------
    csv_path = "power_minimization_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Num_Threads", "K",
            "Encoder_Time_s", "Encoder_Power_W", "Encoder_Energy_J",
            "Decoder_Time_s", "Decoder_Power_W", "Decoder_Power_Std_W", "Decoder_Energy_J",
            "Total_Time_s", "Total_Power_W", "Total_Energy_J",
            "CPU_Utilization", "Freq_Ratio"
        ])
        
        for num_threads, results in sorted(all_results.items()):
            for r in results:
                writer.writerow([
                    r["num_threads"], r["K"],
                    f"{r['enc_time']:.4f}", f"{r['enc_power']:.2f}", f"{r['enc_energy']:.2f}",
                    f"{r['dec_time']:.4f}", f"{r['dec_power']:.2f}", f"{r['dec_power_std']:.2f}", f"{r['dec_energy']:.2f}",
                    f"{r['total_time']:.4f}", f"{r['total_power']:.2f}", f"{r['total_energy']:.2f}",
                    f"{r['cpu_util']:.4f}", f"{r['freq_ratio']:.4f}"
                ])
    
    print(f"\n[INFO] Detailed results saved to: {csv_path}")
    
    # Save summary to separate CSV
    summary_csv_path = "power_minimization_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Num_Threads", "Avg_Power_W", "Total_Energy_J", "Total_Time_s"])
        for cfg in config_summary_sorted:
            writer.writerow([
                cfg["threads"],
                f"{cfg['avg_power']:.2f}",
                f"{cfg['total_energy']:.2f}",
                f"{cfg['total_time']:.2f}"
            ])
    
    print(f"[INFO] Summary saved to: {summary_csv_path}")
    
    # -------------------------------
    # Generate visualization
    # -------------------------------
    print(f"\n[INFO] Generating visualization...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Power vs K for each thread configuration
    ax1 = plt.subplot(2, 3, 1)
    for num_threads in sorted(all_results.keys()):
        results = all_results[num_threads]
        k_vals = [r["K"] for r in results]
        powers = [r["dec_power"] for r in results]
        ax1.plot(k_vals, powers, 'o-', label=f'{num_threads} threads', alpha=0.7, markersize=3)
    ax1.set_xlabel('K (Number of Queries)')
    ax1.set_ylabel('Decoder Power (W)')
    ax1.set_title('Decoder Power vs K (Different Thread Configs)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Time vs K for each thread configuration
    ax2 = plt.subplot(2, 3, 2)
    for num_threads in sorted(all_results.keys()):
        results = all_results[num_threads]
        k_vals = [r["K"] for r in results]
        times = [r["dec_time"] for r in results]
        ax2.plot(k_vals, times, 'o-', label=f'{num_threads} threads', alpha=0.7, markersize=3)
    ax2.set_xlabel('K (Number of Queries)')
    ax2.set_ylabel('Decoder Time (s)')
    ax2.set_title('Decoder Time vs K (Different Thread Configs)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Energy vs K for each thread configuration
    ax3 = plt.subplot(2, 3, 3)
    for num_threads in sorted(all_results.keys()):
        results = all_results[num_threads]
        k_vals = [r["K"] for r in results]
        energies = [r["dec_energy"] for r in results]
        ax3.plot(k_vals, energies, 'o-', label=f'{num_threads} threads', alpha=0.7, markersize=3)
    ax3.set_xlabel('K (Number of Queries)')
    ax3.set_ylabel('Decoder Energy (J)')
    ax3.set_title('Decoder Energy vs K (Different Thread Configs)')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Average Power by Thread Count
    ax4 = plt.subplot(2, 3, 4)
    threads_list = [cfg["threads"] for cfg in config_summary_sorted]
    avg_powers = [cfg["avg_power"] for cfg in config_summary_sorted]
    colors = ['green' if cfg["threads"] == optimal_config["threads"] else 'steelblue' 
              for cfg in config_summary_sorted]
    ax4.bar(range(len(threads_list)), avg_powers, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(threads_list)))
    ax4.set_xticklabels(threads_list)
    ax4.set_xlabel('Number of Threads')
    ax4.set_ylabel('Average Power (W)')
    ax4.set_title('Average Power by Thread Configuration')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Total Energy by Thread Count
    ax5 = plt.subplot(2, 3, 5)
    total_energies = [cfg["total_energy"] for cfg in config_summary_sorted]
    ax5.bar(range(len(threads_list)), total_energies, color=colors, alpha=0.7)
    ax5.set_xticks(range(len(threads_list)))
    ax5.set_xticklabels(threads_list)
    ax5.set_xlabel('Number of Threads')
    ax5.set_ylabel('Total Energy (J)')
    ax5.set_title('Total Energy by Thread Configuration')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Total Time by Thread Count
    ax6 = plt.subplot(2, 3, 6)
    total_times = [cfg["total_time"] for cfg in config_summary_sorted]
    ax6.bar(range(len(threads_list)), total_times, color=colors, alpha=0.7)
    ax6.set_xticks(range(len(threads_list)))
    ax6.set_xticklabels(threads_list)
    ax6.set_xlabel('Number of Threads')
    ax6.set_ylabel('Total Time (s)')
    ax6.set_title('Total Time by Thread Configuration')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    fig_path = "power_minimization_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Visualization saved to: {fig_path}")
    plt.close()
    
    # Create a focused plot for the optimal configuration
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    optimal_results = all_results[optimal_config["threads"]]
    k_vals = [r["K"] for r in optimal_results]
    
    # Power vs K
    powers = [r["dec_power"] for r in optimal_results]
    axes[0].plot(k_vals, powers, 'o-', color='green', linewidth=2, markersize=4)
    axes[0].set_xlabel('K (Number of Queries)', fontsize=12)
    axes[0].set_ylabel('Decoder Power (W)', fontsize=12)
    axes[0].set_title(f'Optimal Config: {optimal_config["threads"]} Threads - Power vs K', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Time vs K
    times = [r["dec_time"] for r in optimal_results]
    axes[1].plot(k_vals, times, 'o-', color='blue', linewidth=2, markersize=4)
    axes[1].set_xlabel('K (Number of Queries)', fontsize=12)
    axes[1].set_ylabel('Decoder Time (s)', fontsize=12)
    axes[1].set_title(f'Optimal Config: {optimal_config["threads"]} Threads - Time vs K', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Energy vs K
    energies = [r["dec_energy"] for r in optimal_results]
    axes[2].plot(k_vals, energies, 'o-', color='red', linewidth=2, markersize=4)
    axes[2].set_xlabel('K (Number of Queries)', fontsize=12)
    axes[2].set_ylabel('Decoder Energy (J)', fontsize=12)
    axes[2].set_title(f'Optimal Config: {optimal_config["threads"]} Threads - Energy vs K', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig2_path = "power_minimization_optimal.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Optimal configuration plot saved to: {fig2_path}")
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"COMPLETE! Use {optimal_config['threads']} threads for minimum power consumption.")
    print(f"{'='*80}\n")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("RT-DETR Power Minimization")
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
    
    args = parser.parse_args()
    main(args)
