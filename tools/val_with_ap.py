"""
Combined RT-DETR validation script that measures both:
1. Average Precision (mAP) on COCO images
2. Power consumption with different K values and target times

Generates plots:
- AP vs iteration (per image) for each K
- Power vs iteration (per image) for each K  
- Summary plot (mean AP and mean power/energy per K)
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
            "energy_J": power * wall_time,
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
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_combined_experiment(args, images_data, coco_gt, pm, k_values, target_times):
    """
    Run combined experiment measuring both AP and power for different K values.
    
    Returns:
        results: dict with structure {target_time: {k: {per_image_data, summary}}}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    all_results = {}
    
    for target_time in target_times:
        print(f"\n{'='*80}")
        print(f"TARGET TIME: {target_time}s")
        print(f"{'='*80}\n")
        
        target_results = {}
        
        for k in k_values:
            print(f"\n[K={k}] Loading model...")
            
            # Load model with K queries
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
            
            # Set thread count based on K and target_time
            base_scaling = k / 150.0
            time_scaling = 1.0 / target_time
            num_threads = max(1, min(pm.num_cores, int(base_scaling * time_scaling * pm.num_cores / 2)))
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            torch.set_num_threads(num_threads)
            
            print(f"[K={k}] Using {num_threads} threads for target time {target_time}s")
            
            # Warm-up
            print(f"[K={k}] Warming up...")
            sample_img = images_data[0]['tensor'].unsqueeze(0).to(device)
            sample_size = torch.tensor([[images_data[0]['orig_w'], images_data[0]['orig_h']]]).to(device)
            with torch.no_grad():
                for _ in range(3):
                    _ = model(sample_img, sample_size)
            
            # Process each image and collect metrics
            per_image_data = []
            all_predictions = []
            
            print(f"[K={k}] Processing {len(images_data)} images...")
            
            for img_idx, img_data in enumerate(images_data):
                img_tensor = img_data['tensor'].unsqueeze(0).to(device)
                orig_size = torch.tensor([[img_data['orig_w'], img_data['orig_h']]]).to(device)
                img_id = img_data['img_id']
                
                # Measure power during inference
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
                valid_mask = img_scores > args.score_threshold
                img_labels = img_labels[valid_mask]
                img_boxes = img_boxes[valid_mask]
                img_scores = img_scores[valid_mask]
                
                # Convert to COCO format and store
                image_predictions = []
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
                    image_predictions.append(pred)
                    all_predictions.append(pred)
                
                # Calculate AP for this single image
                if image_predictions:
                    try:
                        coco_dt = coco_gt.loadRes(image_predictions)
                        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                        coco_eval.params.imgIds = [img_id]
                        coco_eval.evaluate()
                        coco_eval.accumulate()
                        coco_eval.summarize()
                        img_ap = coco_eval.stats[0]  # AP @ IoU=0.50:0.95
                    except:
                        img_ap = 0.0
                else:
                    img_ap = 0.0
                
                per_image_data.append({
                    'image_idx': img_idx,
                    'image_id': img_id,
                    'ap': img_ap,
                    'power_W': stats['power_W'],
                    'energy_J': stats['energy_J'],
                    'time_s': stats['time_s'],
                    'num_predictions': len(image_predictions)
                })
                
                print(f"  Image {img_idx+1}/{len(images_data)}: AP={img_ap:.4f}, "
                      f"Power={stats['power_W']:.2f}W, Time={stats['time_s']:.4f}s")
            
            # Calculate overall mAP across all images
            if all_predictions:
                try:
                    coco_dt = coco_gt.loadRes(all_predictions)
                    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                    coco_eval.params.imgIds = [d['img_id'] for d in images_data]
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    overall_ap = coco_eval.stats[0]
                except:
                    overall_ap = 0.0
            else:
                overall_ap = 0.0
            
            # Calculate summary statistics
            mean_ap = np.mean([d['ap'] for d in per_image_data])
            mean_power = np.mean([d['power_W'] for d in per_image_data])
            mean_energy = np.mean([d['energy_J'] for d in per_image_data])
            mean_time = np.mean([d['time_s'] for d in per_image_data])
            
            target_results[k] = {
                'per_image': per_image_data,
                'summary': {
                    'k': k,
                    'target_time': target_time,
                    'num_threads': num_threads,
                    'overall_ap': overall_ap,
                    'mean_ap': mean_ap,
                    'mean_power': mean_power,
                    'mean_energy': mean_energy,
                    'mean_time': mean_time,
                    'total_predictions': sum(d['num_predictions'] for d in per_image_data)
                }
            }
            
            print(f"\n[K={k}] Summary: Overall AP={overall_ap:.4f}, Mean AP={mean_ap:.4f}, "
                  f"Mean Power={mean_power:.2f}W, Mean Time={mean_time:.4f}s\n")
        
        all_results[target_time] = target_results
    
    return all_results


# ============================================================
# PLOTTING
# ============================================================

def generate_plots(all_results, k_values, target_times, output_dir="."):
    """Generate all required plots."""
    
    # 1. AP vs iteration (per image) for each K - one plot per target time
    for target_time in target_times:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for k in k_values:
            per_image = all_results[target_time][k]['per_image']
            iterations = [d['image_idx'] + 1 for d in per_image]
            aps = [d['ap'] for d in per_image]
            ax.plot(iterations, aps, 'o-', label=f'K={k}', linewidth=2, markersize=6)
        
        ax.set_xlabel('Image Index', fontsize=12)
        ax.set_ylabel('Average Precision (AP)', fontsize=12)
        ax.set_title(f'AP vs Image (Target Time: {target_time}s)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ap_vs_image_t{int(target_time)}s.png", dpi=150)
        plt.close()
    
    # 2. Power vs iteration (per image) for each K - one plot per target time
    for target_time in target_times:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for k in k_values:
            per_image = all_results[target_time][k]['per_image']
            iterations = [d['image_idx'] + 1 for d in per_image]
            powers = [d['power_W'] for d in per_image]
            ax.plot(iterations, powers, 's-', label=f'K={k}', linewidth=2, markersize=6)
        
        ax.set_xlabel('Image Index', fontsize=12)
        ax.set_ylabel('Power (W)', fontsize=12)
        ax.set_title(f'Power vs Image (Target Time: {target_time}s)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/power_vs_image_t{int(target_time)}s.png", dpi=150)
        plt.close()
    
    # 3. Summary plots: Mean AP and Mean Power/Energy vs K
    fig, axes = plt.subplots(2, len(target_times), figsize=(6*len(target_times), 10))
    if len(target_times) == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, target_time in enumerate(target_times):
        # Mean AP vs K
        ax_ap = axes[0, idx]
        mean_aps = [all_results[target_time][k]['summary']['mean_ap'] for k in k_values]
        overall_aps = [all_results[target_time][k]['summary']['overall_ap'] for k in k_values]
        
        ax_ap.plot(k_values, mean_aps, 'bo-', label='Mean AP (per image)', linewidth=2, markersize=8)
        ax_ap.plot(k_values, overall_aps, 'go--', label='Overall AP', linewidth=2, markersize=8)
        ax_ap.set_xlabel('K (Number of Queries)', fontsize=11)
        ax_ap.set_ylabel('Average Precision', fontsize=11)
        ax_ap.set_title(f'AP vs K (Target: {target_time}s)', fontsize=12, fontweight='bold')
        ax_ap.legend()
        ax_ap.grid(True, alpha=0.3)
        
        # Mean Power vs K
        ax_power = axes[1, idx]
        mean_powers = [all_results[target_time][k]['summary']['mean_power'] for k in k_values]
        
        ax_power.plot(k_values, mean_powers, 'rs-', label='Mean Power (W)', linewidth=2, markersize=8)
        
        ax_power.set_xlabel('K (Number of Queries)', fontsize=11)
        ax_power.set_ylabel('Power (W)', fontsize=11)
        ax_power.set_title(f'Power vs K (Target: {target_time}s)', fontsize=12, fontweight='bold')
        ax_power.grid(True, alpha=0.3)
        ax_power.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_ap_power_vs_k.png", dpi=150)
    plt.close()
    
    print(f"[INFO] Plots saved to {output_dir}/")


# ============================================================
# MAIN
# ============================================================

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print(f"RT-DETR VALIDATION: AP + POWER ANALYSIS")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Number of images: {args.num_images}")
    print(f"K values: {args.k_values}")
    print(f"Target times: {args.target_times}")
    print(f"{'='*80}\n")
    
    pm = ProcessCPUMonitor(cpu_tdp=args.cpu_tdp)
    print(f"[Power Monitor] TDP/core={pm.cpu_tdp}W, Cores={pm.num_cores}")
    print(f"[Power Monitor] Max freq={'N/A' if not pm.freq_max else f'{pm.freq_max}MHz'}\n")
    
    # Load COCO ground truth
    print("Loading COCO dataset...")
    coco_gt = COCO(args.gt_json)
    image_ids = coco_gt.getImgIds()[:args.num_images]
    
    # Preload all images
    print(f"Loading {len(image_ids)} images...")
    images_data = []
    for img_id in image_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(args.image_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((640, 640))
        img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
        
        images_data.append({
            'img_id': img_id,
            'tensor': img_tensor,
            'orig_w': img_info['width'],
            'orig_h': img_info['height']
        })
    
    print(f"Images loaded.\n")
    
    # Parse K values and target times
    k_values = [int(k) for k in args.k_values.split(",")]
    target_times = [float(t) for t in args.target_times.split(",")]
    
    # Run combined experiment
    all_results = run_combined_experiment(args, images_data, coco_gt, pm, k_values, target_times)
    
    # Save results to CSV
    print("\nSaving results to CSV...")
    for target_time in target_times:
        csv_path = f"val_results_t{int(target_time)}s.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['K', 'Target_Time', 'Threads', 'Overall_AP', 'Mean_AP', 
                           'Mean_Power_W', 'Mean_Energy_J', 'Mean_Time_s', 'Total_Predictions'])
            for k in k_values:
                summary = all_results[target_time][k]['summary']
                writer.writerow([
                    summary['k'], summary['target_time'], summary['num_threads'],
                    f"{summary['overall_ap']:.4f}", f"{summary['mean_ap']:.4f}",
                    f"{summary['mean_power']:.2f}", f"{summary['mean_energy']:.2f}",
                    f"{summary['mean_time']:.4f}", summary['total_predictions']
                ])
        print(f"  Saved: {csv_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    generate_plots(all_results, k_values, target_times)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    for target_time in target_times:
        print(f"Target Time: {target_time}s")
        print(f"{'K':>6} | {'Overall AP':>10} | {'Mean AP':>10} | {'Power(W)':>10} | {'Energy(J)':>11} | {'Time(s)':>10}")
        print("-" * 75)
        for k in k_values:
            s = all_results[target_time][k]['summary']
            print(f"{k:>6} | {s['overall_ap']:>10.4f} | {s['mean_ap']:>10.4f} | "
                  f"{s['mean_power']:>10.2f} | {s['mean_energy']:>11.2f} | {s['mean_time']:>10.4f}")
        print()
    
    print(f"{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RT-DETR Validation: AP + Power Analysis")
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
    parser.add_argument("--score-threshold", type=float, default=0.3,
                       help="Confidence score threshold (default: 0.3)")
    parser.add_argument("--cpu-tdp", type=float, default=15.0,
                       help="CPU TDP in Watts for power estimation")
    parser.add_argument("--k-values", type=str,
                       default="50,100,200,300",
                       help="Comma-separated K values to test")
    parser.add_argument("--target-times", type=str,
                       default="1.0,2.0,3.0",
                       help="Comma-separated target times in seconds")
    
    args = parser.parse_args()
    main(args)
