"""
RT-DETR mAP Calculation with Minimum Power Consumption

Measures mAP for different K values while minimizing CPU power consumption
by using minimal CPU utilization (single-threaded execution).

Strategy:
- Use K values: 5, 10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 500, 1000, 1500
- Run with 1 thread (minimal CPU utilization) to minimize power
- Measure mAP, power, energy, and time for each K value
- Generate comprehensive results and plots
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
    """
    Measures power directly using CPU utilization and frequency scaling.
    
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
# MAIN EXPERIMENT
# ============================================================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device}")
    print(f"[INFO] Number of images: {args.num_images}")
    print(f"[INFO] CPU TDP: {args.cpu_tdp}W per core")
    
    # Set to single-threaded for minimum power consumption
    num_threads = 1
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    torch.set_num_threads(num_threads)
    print(f"[INFO] Using {num_threads} thread (minimum CPU utilization for power minimization)")
    
    pm = ProcessCPUMonitor(cpu_tdp=args.cpu_tdp)
    print(f"[INFO] System: {pm.num_cores} physical cores, {pm.cpu_tdp * pm.num_cores}W total TDP\n")
    
    # Load COCO dataset
    print(f"[INFO] Loading COCO annotations...")
    coco = COCO(args.gt_json)
    image_ids = coco.getImgIds()[:args.num_images]
    
    # Load images
    print(f"[INFO] Loading {len(image_ids)} images...")
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
            'orig_h': orig_h,
            'filename': info['file_name']
        })
        print(f"  [{len(images_data)}/{len(image_ids)}] {info['file_name']} ({orig_w}x{orig_h})")
    
    # K values to test
    k_values = [5, 10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 500, 1000, 1500]
    print(f"\n[INFO] K values to test: {k_values}\n")
    
    # Store results for each K
    all_results = []
    
    print("="*80)
    print("RUNNING EXPERIMENTS WITH MINIMUM POWER (1 THREAD)")
    print("="*80 + "\n")
    
    # Process each K value
    for k in k_values:
        print(f"\n{'='*80}")
        print(f"K = {k}")
        print(f"{'='*80}")
        
        # Load model with specific K configuration
        print(f"  Loading model with K={k}...")
        cfg = YAMLConfig(args.config, resume=args.resume)
        
        # Update K in config
        if "RTDETRTransformer" in cfg.yaml_cfg:
            cfg.yaml_cfg["RTDETRTransformer"]["num_queries"] = k
        if "RTDETRPostProcessor" in cfg.yaml_cfg:
            cfg.yaml_cfg["RTDETRPostProcessor"]["num_top_queries"] = k
        if "RTDETRTransformerv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["RTDETRTransformerv2"]["num_queries"] = k
        
        # Load checkpoint
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
        print(f"  Warming up...")
        sample_img = images_data[0]['tensor'].unsqueeze(0).to(device)
        sample_size = torch.tensor([[images_data[0]['orig_w'], images_data[0]['orig_h']]]).to(device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(sample_img, sample_size)
        
        # Process all images and collect predictions
        print(f"  Processing {len(images_data)} images...")
        all_predictions = []
        total_time = 0
        total_power = 0
        total_energy = 0
        
        for img_idx, img_data in enumerate(images_data):
            img_tensor = img_data['tensor'].unsqueeze(0).to(device)
            orig_size = torch.tensor([[img_data['orig_w'], img_data['orig_h']]]).to(device)
            img_id = img_data['img_id']
            
            # Measure inference
            def run_inference():
                with torch.no_grad():
                    labels, boxes, scores = model(img_tensor, orig_size)
                return labels, boxes, scores
            
            stats = pm.measure(run_inference)
            labels, boxes, scores = stats['result']
            
            # Extract predictions for this image
            img_labels = labels[0].cpu().numpy()
            img_boxes = boxes[0].cpu().numpy()
            img_scores = scores[0].cpu().numpy()
            
            # Filter by score threshold
            valid_mask = img_scores > args.conf_threshold
            img_labels = img_labels[valid_mask]
            img_boxes = img_boxes[valid_mask]
            img_scores = img_scores[valid_mask]
            
            # Convert to COCO format
            num_detections = 0
            for j in range(len(img_boxes)):
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
                num_detections += 1
            
            # Accumulate metrics
            img_time = stats['time_s']
            img_power = stats['power_W']
            img_energy = img_power * img_time
            
            total_time += img_time
            total_power += img_power
            total_energy += img_energy
            
            print(f"    Image {img_idx+1}/{len(images_data)}: {num_detections} detections, "
                  f"Time={img_time:.3f}s, Power={img_power:.2f}W, Energy={img_energy:.2f}J, "
                  f"CPU={stats['cpu_utilization']*100:.1f}%")
        
        # Calculate overall mAP
        print(f"\n  Calculating mAP...")
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
            print(f"  Warning: No predictions generated")
            overall_map = 0.0
            map_50 = 0.0
            map_75 = 0.0
        
        # Calculate averages
        avg_time = total_time / len(images_data)
        avg_power = total_power / len(images_data)
        avg_energy = total_energy / len(images_data)
        
        # Store results
        result = {
            'K': k,
            'num_threads': num_threads,
            'num_images': len(images_data),
            'overall_mAP': overall_map,
            'mAP_50': map_50,
            'mAP_75': map_75,
            'total_time_s': total_time,
            'total_energy_J': total_energy,
            'avg_time_s': avg_time,
            'avg_power_W': avg_power,
            'avg_energy_J': avg_energy,
            'num_predictions': len(all_predictions)
        }
        all_results.append(result)
        
        # Print summary
        print(f"\n  {'='*70}")
        print(f"  RESULTS FOR K={k}:")
        print(f"  {'='*70}")
        print(f"    Overall mAP:       {overall_map:.4f}")
        print(f"    mAP @ IoU=0.50:    {map_50:.4f}")
        print(f"    mAP @ IoU=0.75:    {map_75:.4f}")
        print(f"    Total Time:        {total_time:.3f}s")
        print(f"    Total Energy:      {total_energy:.2f}J")
        print(f"    Avg Time/Image:    {avg_time:.3f}s")
        print(f"    Avg Power:         {avg_power:.2f}W")
        print(f"    Avg Energy/Image:  {avg_energy:.2f}J")
        print(f"    Total Predictions: {len(all_predictions)}")
        print(f"  {'='*70}\n")
    
    # Save results to CSV
    csv_filename = f"min_power_map_results_{args.num_images}images.csv"
    print(f"\n[INFO] Saving results to {csv_filename}...")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'K', 'Num_Threads', 'Num_Images', 'Overall_mAP', 'mAP_50', 'mAP_75',
            'Total_Time_s', 'Total_Energy_J', 'Avg_Time_per_Image_s',
            'Avg_Power_W', 'Avg_Energy_per_Image_J', 'Num_Predictions'
        ])
        for r in all_results:
            writer.writerow([
                r['K'], r['num_threads'], r['num_images'],
                f"{r['overall_mAP']:.4f}", f"{r['mAP_50']:.4f}", f"{r['mAP_75']:.4f}",
                f"{r['total_time_s']:.3f}", f"{r['total_energy_J']:.2f}",
                f"{r['avg_time_s']:.3f}", f"{r['avg_power_W']:.2f}",
                f"{r['avg_energy_J']:.2f}", r['num_predictions']
            ])
    
    # Generate plots
    print(f"[INFO] Generating plots...")
    
    k_vals = [r['K'] for r in all_results]
    map_vals = [r['overall_mAP'] for r in all_results]
    power_vals = [r['avg_power_W'] for r in all_results]
    energy_vals = [r['avg_energy_J'] for r in all_results]
    time_vals = [r['avg_time_s'] for r in all_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Minimum Power Results ({args.num_images} images, {num_threads} thread)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: mAP vs K
    axes[0, 0].plot(k_vals, map_vals, 'o-', linewidth=2, markersize=6, color='green')
    axes[0, 0].set_xlabel('K (Number of Queries)')
    axes[0, 0].set_ylabel('mAP @ IoU=0.50:0.95')
    axes[0, 0].set_title('mAP vs K')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Power vs K
    axes[0, 1].plot(k_vals, power_vals, 'o-', linewidth=2, markersize=6, color='orange')
    axes[0, 1].set_xlabel('K (Number of Queries)')
    axes[0, 1].set_ylabel('Average Power (W)')
    axes[0, 1].set_title('Power vs K (Minimum CPU Utilization)')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Energy vs K
    axes[1, 0].plot(k_vals, energy_vals, 'o-', linewidth=2, markersize=6, color='steelblue')
    axes[1, 0].set_xlabel('K (Number of Queries)')
    axes[1, 0].set_ylabel('Average Energy per Image (J)')
    axes[1, 0].set_title('Energy vs K')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Time vs K
    axes[1, 1].plot(k_vals, time_vals, 'o-', linewidth=2, markersize=6, color='purple')
    axes[1, 1].set_xlabel('K (Number of Queries)')
    axes[1, 1].set_ylabel('Average Time per Image (s)')
    axes[1, 1].set_title('Inference Time vs K')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f"min_power_map_plots_{args.num_images}images.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"[INFO] Plots saved to {plot_filename}")
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - MINIMUM POWER CONFIGURATION")
    print(f"{'='*80}")
    print(f"{'K':>6} | {'mAP':>8} | {'Power(W)':>10} | {'Energy(J)':>11} | {'Time(s)':>9} | {'Predictions':>12}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['K']:>6} | {r['overall_mAP']:>8.4f} | {r['avg_power_W']:>10.2f} | "
              f"{r['avg_energy_J']:>11.2f} | {r['avg_time_s']:>9.3f} | {r['num_predictions']:>12}")
    print(f"{'='*80}")
    
    # Find optimal K (best mAP with minimum power)
    best_map = max(all_results, key=lambda x: x['overall_mAP'])
    min_power = min(all_results, key=lambda x: x['avg_power_W'])
    min_energy = min(all_results, key=lambda x: x['avg_energy_J'])
    
    print(f"\n[OPTIMAL CONFIGURATIONS]")
    print(f"  Best mAP:       K={best_map['K']:>4}, mAP={best_map['overall_mAP']:.4f}, Power={best_map['avg_power_W']:.2f}W")
    print(f"  Minimum Power:  K={min_power['K']:>4}, Power={min_power['avg_power_W']:.2f}W, mAP={min_power['overall_mAP']:.4f}")
    print(f"  Minimum Energy: K={min_energy['K']:>4}, Energy={min_energy['avg_energy_J']:.2f}J, mAP={min_energy['overall_mAP']:.4f}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RT-DETR Minimum Power mAP Calculation")
    parser.add_argument("-c", "--config",
                        default="./configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
    parser.add_argument("-r", "--resume",
                        default="rtdetr_r50vd_6x_coco_from_paddle.pth")
    parser.add_argument("--image-dir",
                        default="./dataset/coco/val2017")
    parser.add_argument("--gt-json",
                        default="./dataset/coco/instances_val2017.json")
    parser.add_argument("--num-images", type=int, default=5,
                        help="Number of images to process (default: 5)")
    parser.add_argument("--cpu-tdp", type=float, default=15.0,
                        help="CPU TDP in Watts per core (default: 15.0)")
    parser.add_argument("--conf-threshold", type=float, default=0.01,
                        help="Confidence threshold for predictions (default: 0.01)")

    args = parser.parse_args()
    main(args)
