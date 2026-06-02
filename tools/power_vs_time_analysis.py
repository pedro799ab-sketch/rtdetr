#!/usr/bin/env python3
"""
Power and mAP Analysis across Different Target Times
Generates graphs: Power vs K, mAP vs K, Power vs Time, mAP vs Time
for target times: 1, 2, 3, 4, 5 seconds
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
import psutil
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt

from src.core import YAMLConfig
from src.solver import TASKS


# ============================================================
# DEPLOY MODEL WITH POSTPROCESSOR
# ============================================================

class DeployModel(torch.nn.Module):
    """Deploy model with postprocessor for proper inference."""
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model.deploy() if hasattr(model, 'deploy') else model
        self.postprocessor = postprocessor.deploy() if hasattr(postprocessor, 'deploy') else postprocessor
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes=orig_target_sizes)
        return outputs


# ============================================================
# PROCESS CPU MONITOR
# ============================================================

class ProcessCPUMonitor:
    """Measures power directly using CPU utilization and frequency scaling."""
    
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
        wall_end = time.perf_counter()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
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
# mAP CALCULATION
# ============================================================

def calculate_map(coco_gt, predictions, image_ids, conf_threshold=0.01):
    """Calculate mAP using COCO evaluation."""
    if predictions is None or len(predictions) == 0:
        return {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
        }

    coco_results = []
    
    for pred, img_id in zip(predictions, image_ids):
        if pred is None:
            continue
        
        if isinstance(pred, list):
            pred = pred[0] if len(pred) > 0 else None
        
        if pred is None or not isinstance(pred, dict):
            continue
        
        labels = pred.get('labels')
        boxes = pred.get('boxes')
        scores = pred.get('scores')
        
        if labels is None or boxes is None or scores is None:
            continue
        
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        
        keep = scores > conf_threshold
        filtered_boxes = boxes[keep]
        filtered_scores = scores[keep]
        filtered_labels = labels[keep]
        
        for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
            if len(box) == 4:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                if width <= 0 or height <= 0:
                    continue
                
                coco_results.append({
                    'image_id': int(img_id),
                    'category_id': int(label) + 1,
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'score': float(score)
                })
    
    if len(coco_results) == 0:
        return {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
        }
    
    try:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return {
            'mAP': coco_eval.stats[0],
            'mAP_50': coco_eval.stats[1],
            'mAP_75': coco_eval.stats[2],
        }
    except Exception as e:
        print(f"    Warning: mAP calculation failed: {e}")
        return {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
        }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_calibrated_experiment(args, batches, pm, k_values, target_time, coco_gt, image_ids):
    """
    Run experiment calibrated to a specific target time.
    For each K value, calibrate thread count to hit the target time.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model with checkpoint
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu')
    
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint
    
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    solver.model.load_state_dict(state)
    
    deploy_model = DeployModel(solver.model, solver.postprocessor).to(device).eval()
    
    results = []
    print(f"\n  Target Time: {target_time}s")
    print(f"  {'K':>6} | {'Threads':>8} | {'Time (s)':>10} | {'Power (W)':>10} | {'Energy (J)':>11} | {'CPU %':>7} | {'mAP':>8}")
    print(f"  {'-'*80}")
    
    for k in k_values:
        # Calibrate threads for this K
        best_threads = 1
        best_time_diff = float('inf')
        thread_times = {}
        
        # Try different thread counts
        for num_threads in [1, 2, 4, 8, pm.num_cores]:
            if num_threads > pm.num_cores:
                continue
            
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            torch.set_num_threads(num_threads)
            
            # Warm-up
            with torch.no_grad():
                for batch in batches[:1]:
                    _ = deploy_model(batch, torch.tensor([[640, 640]]))
            
            # Measure
            def run_inference():
                all_outputs = []
                with torch.no_grad():
                    for batch in batches:
                        output = deploy_model(batch, torch.tensor([[640, 640]] * batch.shape[0]))
                        all_outputs.append(output)
                return all_outputs
            
            stats = pm.measure(run_inference)
            thread_times[num_threads] = (stats['time_s'], stats['power_W'], stats['cpu_utilization'])
            
            # Check if this is closer to target
            time_diff = abs(stats['time_s'] - target_time)
            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_threads = num_threads
        
        # Use best thread count for this K
        os.environ['OMP_NUM_THREADS'] = str(best_threads)
        torch.set_num_threads(best_threads)
        
        # Final measurement
        def run_inference():
            all_outputs = []
            with torch.no_grad():
                for batch in batches:
                    output = deploy_model(batch, torch.tensor([[640, 640]] * batch.shape[0]))
                    all_outputs.append(output)
            return all_outputs
        
        stats = pm.measure(run_inference)
        outputs = stats['result']
        
        # Calculate metrics
        meas_time = stats['time_s']
        meas_power = stats['power_W']
        meas_energy = meas_power * meas_time
        meas_cpu = stats['cpu_utilization'] * 100
        
        # Calculate mAP
        flat_predictions = []
        if isinstance(outputs, list):
            for output in outputs:
                if isinstance(output, list):
                    flat_predictions.extend(output)
                else:
                    flat_predictions.append(output)
        
        map_stats = calculate_map(coco_gt, flat_predictions, image_ids)
        
        result = {
            'K': k,
            'target_time': target_time,
            'threads': best_threads,
            'time_s': meas_time,
            'power_W': meas_power,
            'energy_J': meas_energy,
            'cpu_util': meas_cpu,
            'mAP': map_stats['mAP'],
            'mAP_50': map_stats['mAP_50'],
            'mAP_75': map_stats['mAP_75'],
        }
        results.append(result)
        
        print(f"  {k:>6} | {best_threads:>8} | {meas_time:>10.4f} | {meas_power:>10.2f} | {meas_energy:>11.2f} | {meas_cpu:>7.1f} | {map_stats['mAP']:>8.4f}")
    
    return results


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print(f"Power & mAP Analysis across Different Target Times")
    print(f"{'='*80}")
    print(f"Device: {device}")
    
    pm = ProcessCPUMonitor(cpu_tdp=args.cpu_tdp)
    total_tdp = pm.cpu_tdp * pm.num_cores
    print(f"CPU TDP: {pm.cpu_tdp}W/core × {pm.num_cores} cores = {total_tdp}W total\n")
    
    # Load images
    print(f"Loading COCO dataset...")
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
    
    print(f"Loaded {len(all_tensors)} images in {len(batches)} batches\n")
    
    # K values
    k_values = [5, 10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 500, 1000, 1500]
    
    # Target times
    target_times = [1, 2, 3, 4, 5]
    
    # Run experiments for each target time
    all_results = {}
    
    for target_time in target_times:
        print(f"\n{'='*80}")
        print(f"Calibrating for Target Time: {target_time}s")
        print(f"{'='*80}")
        
        results = run_calibrated_experiment(
            args, batches, pm, k_values, target_time, coco, image_ids
        )
        all_results[target_time] = results
    
    # Save to CSV
    csv_path = "power_vs_time_analysis.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Target_Time_s', 'K', 'Threads', 'Time_s', 'Power_W', 'Energy_J', 'CPU_Util_Pct', 'mAP', 'mAP_50', 'mAP_75'])
        
        for target_time in target_times:
            for result in all_results[target_time]:
                writer.writerow([
                    result['target_time'],
                    result['K'],
                    result['threads'],
                    f"{result['time_s']:.4f}",
                    f"{result['power_W']:.2f}",
                    f"{result['energy_J']:.2f}",
                    f"{result['cpu_util']:.1f}",
                    f"{result['mAP']:.4f}",
                    f"{result['mAP_50']:.4f}",
                    f"{result['mAP_75']:.4f}",
                ])
    
    print(f"\n[INFO] CSV saved to: {csv_path}")
    
    # Generate visualization plots
    print(f"\n[INFO] Generating comparison plots...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Define colors for each target time
    colors = {1: 'red', 2: 'orange', 3: 'green', 4: 'blue', 5: 'purple'}
    markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
    
    # 1. Power vs K (different target times)
    ax1 = fig.add_subplot(gs[0, 0])
    for target_time in target_times:
        results = all_results[target_time]
        k_vals = [r['K'] for r in results]
        powers = [r['power_W'] for r in results]
        ax1.plot(k_vals, powers, marker=markers[target_time], label=f'Target: {target_time}s',
                color=colors[target_time], linewidth=2, markersize=6)
    ax1.set_xlabel('K (Number of Queries)', fontsize=11)
    ax1.set_ylabel('Power (W)', fontsize=11)
    ax1.set_title('Power vs K (Different Target Times)', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 2. mAP vs K (different target times)
    ax2 = fig.add_subplot(gs[0, 1])
    for target_time in target_times:
        results = all_results[target_time]
        k_vals = [r['K'] for r in results]
        maps = [r['mAP'] for r in results]
        ax2.plot(k_vals, maps, marker=markers[target_time], label=f'Target: {target_time}s',
                color=colors[target_time], linewidth=2, markersize=6)
    ax2.set_xlabel('K (Number of Queries)', fontsize=11)
    ax2.set_ylabel('mAP', fontsize=11)
    ax2.set_title('mAP vs K (Different Target Times)', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 3. Power vs Time (for each K)
    ax3 = fig.add_subplot(gs[1, 0])
    for target_time in target_times:
        results = all_results[target_time]
        times = [r['time_s'] for r in results]
        powers = [r['power_W'] for r in results]
        ax3.plot(times, powers, marker=markers[target_time], label=f'Target: {target_time}s',
                color=colors[target_time], linewidth=2, markersize=6)
    ax3.set_xlabel('Actual Execution Time (s)', fontsize=11)
    ax3.set_ylabel('Power (W)', fontsize=11)
    ax3.set_title('Power vs Execution Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # 4. mAP vs Time (for each K)
    ax4 = fig.add_subplot(gs[1, 1])
    for target_time in target_times:
        results = all_results[target_time]
        times = [r['time_s'] for r in results]
        maps = [r['mAP'] for r in results]
        ax4.plot(times, maps, marker=markers[target_time], label=f'Target: {target_time}s',
                color=colors[target_time], linewidth=2, markersize=6)
    ax4.set_xlabel('Actual Execution Time (s)', fontsize=11)
    ax4.set_ylabel('mAP', fontsize=11)
    ax4.set_title('mAP vs Execution Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    # 5. Energy vs K
    ax5 = fig.add_subplot(gs[2, 0])
    for target_time in target_times:
        results = all_results[target_time]
        k_vals = [r['K'] for r in results]
        energies = [r['energy_J'] for r in results]
        ax5.plot(k_vals, energies, marker=markers[target_time], label=f'Target: {target_time}s',
                color=colors[target_time], linewidth=2, markersize=6)
    ax5.set_xlabel('K (Number of Queries)', fontsize=11)
    ax5.set_ylabel('Energy (J)', fontsize=11)
    ax5.set_title('Energy vs K (Different Target Times)', fontsize=12, fontweight='bold')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)
    
    # 6. Time vs K (showing actual vs target)
    ax6 = fig.add_subplot(gs[2, 1])
    for target_time in target_times:
        results = all_results[target_time]
        k_vals = [r['K'] for r in results]
        times = [r['time_s'] for r in results]
        ax6.plot(k_vals, times, marker=markers[target_time], label=f'Target: {target_time}s',
                color=colors[target_time], linewidth=2, markersize=6)
        ax6.axhline(y=target_time, color=colors[target_time], linestyle='--', alpha=0.5, linewidth=1)
    ax6.set_xlabel('K (Number of Queries)', fontsize=11)
    ax6.set_ylabel('Actual Execution Time (s)', fontsize=11)
    ax6.set_title('Actual Time vs K (dashed = target)', fontsize=12, fontweight='bold')
    ax6.set_xscale('log')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=10)
    
    fig.suptitle('Power & mAP Analysis: Target Time 1-5 seconds', fontsize=14, fontweight='bold', y=0.995)
    
    plot_path = "power_mAP_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Visualization saved to: {plot_path}")
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Power & mAP Analysis across Target Times")
    parser.add_argument("-c", "--config", default="./configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
    parser.add_argument("-r", "--resume", default="rtdetr_r50vd_6x_coco_from_paddle.pth")
    parser.add_argument("--image-dir", default="./dataset/coco/subset_10/images")
    parser.add_argument("--gt-json", default="./dataset/coco/subset_10/instances_train2017.json")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--cpu-tdp", type=float, default=15.0)
    
    args = parser.parse_args()
    main(args)
