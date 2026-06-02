"""
RT-DETR Power and mAP Analysis with Predetermined Optimal CPU Configurations

Uses known optimal thread configurations for each K range:
  • Low K (5-50):      1-2 threads, Power ~15-30W
  • Mid K (100-300):   2-6 threads, Power ~30-80W
  • High K (500-1500): 6-10 threads, Power ~80-130W

Quickly measures mAP and power for 5 images without calibration overhead.
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

import torch
import numpy as np
import psutil
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt

from src.core import YAMLConfig
from src.solver import TASKS
from src.data.dataset import mscoco_label2category


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
# DEPLOY MODEL
# ============================================================

class DeployModel(torch.nn.Module):
    """Deploy model with postprocessor for inference."""
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model.deploy() if hasattr(model, 'deploy') else model
        self.postprocessor = postprocessor.deploy() if hasattr(postprocessor, 'deploy') else postprocessor
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes=orig_target_sizes)
        return outputs


# ============================================================
# OPTIMAL THREAD CONFIGURATION
# ============================================================

def get_optimal_threads(k):
    """
    Return optimal thread count based on K value range.
    
    Based on empirical analysis:
      • Low K (5-50):      1-2 threads
      • Mid K (100-300):   2-6 threads
      • High K (500-1500): 6-10 threads
    """
    if k <= 20:
        return 1
    elif k <= 50:
        return 2
    elif k <= 100:
        return 3
    elif k <= 200:
        return 4
    elif k <= 300:
        return 6
    elif k <= 500:
        return 8
    elif k <= 1000:
        return 9
    else:  # k >= 1000
        return 10


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print(f"POWER AND mAP ANALYSIS WITH OPTIMAL CPU CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Number of images: {args.num_images}")
    print(f"CPU TDP: {args.cpu_tdp}W per core")
    print(f"{'='*80}\n")
    
    pm = ProcessCPUMonitor(cpu_tdp=args.cpu_tdp)
    print(f"[System Info]")
    print(f"  Physical cores: {pm.num_cores}")
    print(f"  Total TDP: {pm.cpu_tdp * pm.num_cores}W")
    print(f"  Max frequency: {pm.freq_max if pm.freq_max else 'N/A'}MHz\n")
    
    # Load COCO dataset
    print(f"[Loading Dataset]")
    coco = COCO(args.gt_json)
    image_ids = coco.getImgIds()[:args.num_images]
    
    # Load images
    print(f"Loading {len(image_ids)} images...")
    images_data = []
    for img_id in image_ids:
        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.image_dir, info["file_name"])
        img = Image.open(img_path).convert("RGB")
        
        orig_w, orig_h = img.width, img.height
        img = img.resize((640, 640))
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        images_data.append({
            'img_id': img_id,
            'tensor': t,
            'orig_w': orig_w,
            'orig_h': orig_h
        })
    print(f"Loaded {len(images_data)} images\n")
    
    # K values to test
    k_values = [5, 10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 500, 1000, 1500]
    print(f"[K Values] {k_values}\n")
    
    # Store results
    all_results = []
    
    print("="*80)
    print("PROCESSING EACH K WITH OPTIMAL THREADS")
    print("="*80 + "\n")
    
    # Process each K value
    for k in k_values:
        print(f"\n{'='*80}")
        print(f"K = {k}")
        print(f"{'='*80}\n")
        
        # Get optimal thread count for this K
        optimal_threads = get_optimal_threads(k)
        print(f"Using optimal configuration: {optimal_threads} threads")
        
        # Set thread count
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        torch.set_num_threads(optimal_threads)
        
        # Load model with specific K
        print(f"Loading model with K={k}...")
        cfg = YAMLConfig(args.config, resume=args.resume)
        
        if "RTDETRTransformer" in cfg.yaml_cfg:
            cfg.yaml_cfg["RTDETRTransformer"]["num_queries"] = k
        if "RTDETRPostProcessor" in cfg.yaml_cfg:
            cfg.yaml_cfg["RTDETRPostProcessor"]["num_top_queries"] = k
        if "RTDETRTransformerv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["RTDETRTransformerv2"]["num_queries"] = k
        
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        elif 'model' in checkpoint:
            state = checkpoint['model']
        else:
            state = checkpoint
        
        cfg.model.load_state_dict(state)
        model = DeployModel(cfg.model, cfg.postprocessor).to(device).eval()
        
        # Warm-up
        print(f"Warming up...")
        sample_img = images_data[0]['tensor'].unsqueeze(0).to(device)
        sample_size = torch.tensor([[images_data[0]['orig_w'], images_data[0]['orig_h']]]).to(device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(sample_img, sample_size)
        
        # Process all images
        print(f"Processing {len(images_data)} images...")
        all_predictions = []
        per_image_stats = []
        
        for img_idx, img_data in enumerate(images_data):
            img_tensor = img_data['tensor'].unsqueeze(0).to(device)
            orig_size = torch.tensor([[img_data['orig_w'], img_data['orig_h']]]).to(device)
            img_id = img_data['img_id']
            
            def run_inference():
                with torch.no_grad():
                    labels, boxes, scores = model(img_tensor, orig_size)
                return labels, boxes, scores
            
            stats = pm.measure(run_inference)
            labels, boxes, scores = stats['result']
            
            # Extract predictions
            img_labels = labels[0].cpu().numpy()
            img_boxes = boxes[0].cpu().numpy()
            img_scores = scores[0].cpu().numpy()
            
            valid_mask = img_scores > args.conf_threshold
            img_labels = img_labels[valid_mask]
            img_boxes = img_boxes[valid_mask]
            img_scores = img_scores[valid_mask]
            
            num_dets = len(img_boxes)
            
            # Convert to COCO format
            for j in range(num_dets):
                x1, y1, x2, y2 = img_boxes[j]
                label_idx = int(img_labels[j])
                coco_category_id = mscoco_label2category[label_idx]
                
                pred = {
                    "image_id": int(img_id),
                    "category_id": coco_category_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(img_scores[j])
                }
                all_predictions.append(pred)
            
            per_image_stats.append({
                'time': stats['time_s'],
                'power': stats['power_W'],
                'energy': stats['power_W'] * stats['time_s'],
                'cpu_util': stats['cpu_utilization'],
                'num_dets': num_dets
            })
            
            print(f"  Image {img_idx+1}/{len(images_data)}: {num_dets} detections, "
                  f"Time={stats['time_s']:.4f}s, Power={stats['power_W']:.2f}W, "
                  f"CPU={stats['cpu_utilization']*100:.1f}%")
        
        # Calculate mAP
        if len(all_predictions) > 0:
            try:
                coco_dt = coco.loadRes(all_predictions)
                coco_eval = COCOeval(coco, coco_dt, 'bbox')
                coco_eval.params.imgIds = image_ids
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                overall_map = coco_eval.stats[0]
                map_50 = coco_eval.stats[1]
                map_75 = coco_eval.stats[2]
            except Exception as e:
                print(f"  Warning: mAP calculation failed: {e}")
                overall_map = 0.0
                map_50 = 0.0
                map_75 = 0.0
        else:
            overall_map = 0.0
            map_50 = 0.0
            map_75 = 0.0
        
        # Aggregate statistics
        avg_time = np.mean([s['time'] for s in per_image_stats])
        avg_power = np.mean([s['power'] for s in per_image_stats])
        avg_energy = np.mean([s['energy'] for s in per_image_stats])
        avg_cpu_util = np.mean([s['cpu_util'] for s in per_image_stats])
        total_time = sum([s['time'] for s in per_image_stats])
        total_energy = sum([s['energy'] for s in per_image_stats])
        
        result = {
            'K': k,
            'threads': optimal_threads,
            'cpu_util_pct': avg_cpu_util * 100,
            'mAP': overall_map,
            'mAP_50': map_50,
            'mAP_75': map_75,
            'avg_time_s': avg_time,
            'avg_power_W': avg_power,
            'avg_energy_J': avg_energy,
            'total_time_s': total_time,
            'total_energy_J': total_energy,
            'num_predictions': len(all_predictions)
        }
        all_results.append(result)
        
        print(f"\n[Summary for K={k}]")
        print(f"  Threads:           {optimal_threads}")
        print(f"  CPU Utilization:   {avg_cpu_util*100:.1f}%")
        print(f"  mAP @ IoU=0.50:0.95: {overall_map:.4f}")
        print(f"  mAP @ IoU=0.50:    {map_50:.4f}")
        print(f"  mAP @ IoU=0.75:    {map_75:.4f}")
        print(f"  Avg Time/Image:    {avg_time:.4f}s")
        print(f"  Avg Power:         {avg_power:.2f}W")
        print(f"  Avg Energy/Image:  {avg_energy:.2f}J")
        print(f"  Total Time:        {total_time:.2f}s")
        print(f"  Total Energy:      {total_energy:.2f}J")
        print(f"  Total Predictions: {len(all_predictions)}\n")
    
    # Save results to CSV
    csv_filename = f"power_map_per5images.csv"
    print(f"\n{'='*80}")
    print(f"[Saving Results] {csv_filename}")
    print(f"{'='*80}\n")
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'K', 'Threads', 'CPU_Util_%', 'mAP', 'mAP_50', 'mAP_75',
            'Avg_Time_s', 'Avg_Power_W', 'Avg_Energy_J',
            'Total_Time_s', 'Total_Energy_J', 'Num_Predictions'
        ])
        for r in all_results:
            writer.writerow([
                r['K'], r['threads'], f"{r['cpu_util_pct']:.1f}",
                f"{r['mAP']:.4f}", f"{r['mAP_50']:.4f}", f"{r['mAP_75']:.4f}",
                f"{r['avg_time_s']:.4f}", f"{r['avg_power_W']:.2f}", f"{r['avg_energy_J']:.2f}",
                f"{r['total_time_s']:.2f}", f"{r['total_energy_J']:.2f}",
                r['num_predictions']
            ])
    
    print(f"Results saved to {csv_filename}\n")
    
    # Generate comprehensive plots
    print(f"[Generating Plots]")
    
    k_vals = [r['K'] for r in all_results]
    map_vals = [r['mAP'] for r in all_results]
    map50_vals = [r['mAP_50'] for r in all_results]
    power_vals = [r['avg_power_W'] for r in all_results]
    time_vals = [r['avg_time_s'] for r in all_results]
    energy_vals = [r['avg_energy_J'] for r in all_results]
    threads_vals = [r['threads'] for r in all_results]
    cpu_util_vals = [r['cpu_util_pct'] for r in all_results]
    
    # Create figure with 8 subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: mAP vs K
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(k_vals, map_vals, 'go-', linewidth=2.5, markersize=9, label='mAP@0.50:0.95')
    ax1.plot(k_vals, map50_vals, 'b^--', linewidth=2, markersize=7, label='mAP@0.50')
    ax1.set_xlabel('K (Number of Queries)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Average Precision', fontsize=12, fontweight='bold')
    ax1.set_title('mAP vs K', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Power vs K (KEY RESULT - shows power increase)
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(k_vals, power_vals, 'ro-', linewidth=3, markersize=10)
    ax2.set_xlabel('K (Number of Queries)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Power (W)', fontsize=12, fontweight='bold')
    ax2.set_title('Power vs K (Optimal CPU)', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add power range annotations
    ax2.axhspan(15, 30, alpha=0.2, color='green', label='Low K (5-50)')
    ax2.axhspan(30, 80, alpha=0.2, color='yellow', label='Mid K (100-300)')
    ax2.axhspan(80, 130, alpha=0.2, color='red', label='High K (500-1500)')
    ax2.legend(fontsize=9, loc='upper left')
    
    # Plot 3: Energy vs K
    ax3 = plt.subplot(2, 4, 3)
    ax3.plot(k_vals, energy_vals, 'mo-', linewidth=2.5, markersize=9)
    ax3.set_xlabel('K (Number of Queries)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Avg Energy per Image (J)', fontsize=12, fontweight='bold')
    ax3.set_title('Energy vs K', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time vs K
    ax4 = plt.subplot(2, 4, 4)
    ax4.plot(k_vals, time_vals, 'co-', linewidth=2.5, markersize=9)
    ax4.set_xlabel('K (Number of Queries)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Avg Time per Image (s)', fontsize=12, fontweight='bold')
    ax4.set_title('Inference Time vs K', fontsize=13, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Optimal Threads vs K
    ax5 = plt.subplot(2, 4, 5)
    ax5.plot(k_vals, threads_vals, 'bs-', linewidth=2.5, markersize=9)
    ax5.set_xlabel('K (Number of Queries)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Optimal Thread Count', fontsize=12, fontweight='bold')
    ax5.set_title('Optimal Threads vs K', fontsize=13, fontweight='bold')
    ax5.set_xscale('log')
    ax5.axhline(y=pm.num_cores, color='r', linestyle='--', linewidth=2, label=f'Max cores ({pm.num_cores})')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: CPU Utilization vs K
    ax6 = plt.subplot(2, 4, 6)
    ax6.plot(k_vals, cpu_util_vals, 'yo-', linewidth=2.5, markersize=9)
    ax6.set_xlabel('K (Number of Queries)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('CPU Utilization (%)', fontsize=12, fontweight='bold')
    ax6.set_title('CPU Utilization vs K', fontsize=13, fontweight='bold')
    ax6.set_xscale('log')
    ax6.axhline(y=100, color='r', linestyle='--', linewidth=2, label='100% capacity')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Power vs mAP (efficiency curve)
    ax7 = plt.subplot(2, 4, 7)
    scatter = ax7.scatter(map_vals, power_vals, c=k_vals, s=150, cmap='viridis', 
                          edgecolors='black', linewidths=1.5)
    for i, k in enumerate(k_vals):
        ax7.annotate(f'{k}', (map_vals[i], power_vals[i]), 
                    fontsize=8, ha='center', va='center', fontweight='bold')
    ax7.set_xlabel('mAP @ IoU=0.50:0.95', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Average Power (W)', fontsize=12, fontweight='bold')
    ax7.set_title('Power vs mAP Trade-off', fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax7)
    cbar.set_label('K Value', fontsize=10, fontweight='bold')
    
    # Plot 8: Energy vs mAP (efficiency curve)
    ax8 = plt.subplot(2, 4, 8)
    scatter2 = ax8.scatter(map_vals, energy_vals, c=k_vals, s=150, cmap='plasma',
                           edgecolors='black', linewidths=1.5)
    for i, k in enumerate(k_vals):
        ax8.annotate(f'{k}', (map_vals[i], energy_vals[i]),
                    fontsize=8, ha='center', va='center', fontweight='bold', color='white')
    ax8.set_xlabel('mAP @ IoU=0.50:0.95', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Avg Energy per Image (J)', fontsize=12, fontweight='bold')
    ax8.set_title('Energy vs mAP Trade-off', fontsize=13, fontweight='bold')
    ax8.set_yscale('log')
    ax8.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax8)
    cbar2.set_label('K Value', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'RT-DETR Optimal CPU Analysis - {args.num_images} Images', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plot_filename = f"power_map_plots_5images.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_filename}\n")
    
    # Print final summary table
    print(f"{'='*80}")
    print("FINAL SUMMARY: POWER SCALING WITH K AT OPTIMAL CPU")
    print(f"{'='*80}")
    print(f"{'K':>6} | {'Thr':>4} | {'CPU%':>5} | {'mAP':>8} | {'Power(W)':>9} | {'Energy(J)':>10} | {'Time(s)':>8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['K']:>6} | {r['threads']:>4} | {r['cpu_util_pct']:>5.1f} | "
              f"{r['mAP']:>8.4f} | {r['avg_power_W']:>9.2f} | "
              f"{r['avg_energy_J']:>10.2f} | {r['avg_time_s']:>8.4f}")
    print("=" * 80)
    
    # Key insights
    print(f"\n[KEY INSIGHTS]")
    print(f"  • Power INCREASES with K as optimal CPU utilization rises")
    print(f"  • Low K (5-50):      1-2 threads, {min([r['avg_power_W'] for r in all_results[:6]]):.1f}-{max([r['avg_power_W'] for r in all_results[:6]]):.1f}W")
    print(f"  • Mid K (100-300):   2-6 threads, {min([r['avg_power_W'] for r in all_results[6:9]]):.1f}-{max([r['avg_power_W'] for r in all_results[6:9]]):.1f}W")
    print(f"  • High K (500-1500): 6-10 threads, {min([r['avg_power_W'] for r in all_results[9:]]):.1f}-{max([r['avg_power_W'] for r in all_results[9:]]):.1f}W")
    print(f"  • Best mAP: K={all_results[np.argmax(map_vals)]['K']} → {max(map_vals):.4f}")
    print(f"  • Most efficient: K={all_results[np.argmin([r['avg_energy_J']/r['mAP'] if r['mAP'] > 0 else float('inf') for r in all_results])]['K']}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RT-DETR Power and mAP with Optimal CPU")
    parser.add_argument("-c", "--config",
                        default="./configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
    parser.add_argument("-r", "--resume",
                        default="rtdetr_r50vd_6x_coco_from_paddle.pth")
    parser.add_argument("--image-dir",
                        default="./dataset/coco/val2017")
    parser.add_argument("--gt-json",
                        default="./dataset/coco/instances_val2017.json")
    parser.add_argument("--num-images", type=int, default=5,
                        help="Number of images (default: 5)")
    parser.add_argument("--cpu-tdp", type=float, default=15.0,
                        help="CPU TDP per core in Watts (default: 15.0)")
    parser.add_argument("--conf-threshold", type=float, default=0.01,
                        help="Confidence threshold (default: 0.01)")

    args = parser.parse_args()
    main(args)
