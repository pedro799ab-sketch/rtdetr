"""
Power and mAP Measurement for RT-DETR with Dynamic K Values
Processes images in groups of 5 and measures both power and mAP at each K value
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
# mAP CALCULATION
# ============================================================

def calculate_map(coco_gt, predictions, image_ids, conf_threshold=0.1):
    """Calculate mAP using COCO evaluation."""
    if predictions is None or len(predictions) == 0:
        print(f"  [DEBUG] No predictions!")
        return {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
        }

    coco_results = []
    total_detections = 0
    
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
        
        total_detections += len(filtered_boxes)
        
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
                    'score': float(score),
                    "area": float(width * height),
                    "iscrowd": 0,
                })
    
    print(f"  [DEBUG] Total detections above threshold {conf_threshold}: {total_detections}")
    print(f"  [DEBUG] COCO results added: {len(coco_results)}")
    
    if len(coco_results) == 0:
        print(f"  [DEBUG] No valid COCO results after filtering!")
        return {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
        }
    
    # Check ground truth
    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_ids))
    print(f"  [DEBUG] Ground truth annotations: {len(gt_anns)}")
    if len(gt_anns) > 0:
        print(f"  [DEBUG] Sample GT box: {gt_anns[0]['bbox'][:4]}, category: {gt_anns[0]['category_id']}")
    if len(coco_results) > 0:
        print(f"  [DEBUG] Sample prediction: score={coco_results[0]['score']:.3f}, bbox={coco_results[0]['bbox'][:4]}, category={coco_results[0]['category_id']}")
    
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
        print(f"        Warning: mAP calculation failed: {e}")
        return {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
        }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device}")
    print(f"[INFO] Number of images: {args.num_images}")
    
    pm = ProcessCPUMonitor(cpu_tdp=args.cpu_tdp)
    print(f"[ProcessCPUMonitor] TDP/core={pm.cpu_tdp}W, Cores={pm.num_cores}, TDP_total={pm.cpu_tdp * pm.num_cores}W")
    
    # Set thread count
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
    torch.set_num_threads(args.num_threads)
    print(f"[INFO] Using {args.num_threads} threads")
    
    # Load COCO and images
    coco = COCO(args.gt_json)
    image_ids = coco.getImgIds()[:args.num_images]
    
    all_tensors = []
    image_ids_list = []
    for img_id in image_ids:
        info = coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(args.image_dir, info["file_name"])).convert("RGB")
        img = img.resize((640, 640))
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        all_tensors.append(t)
        image_ids_list.append(img_id)
    
    print(f"[INFO] Loaded {len(all_tensors)} images")
    
    # K values to test
    k_values = [5, 10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 500, 1000, 1500]
    print(f"[INFO] K values to test: {k_values}")
    print(f"[INFO] Processing images in groups of 5\n")
    
    all_results = []
    
    for k in k_values:
        print(f"{'='*80}")
        print(f"K={k}")
        print(f"{'='*80}")
        
        # Build model with K value
        cfg = YAMLConfig(args.config, resume=args.resume)
        
        # Set num_queries to K
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
        
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver._setup()
        solver.model.load_state_dict(state)
        
        # Create deploy model
        deploy_model = DeployModel(solver.model, solver.postprocessor).to(device).eval()
        
        # Warm-up
        sample_batch = all_tensors[0].unsqueeze(0).to(device)
        orig_size = torch.tensor([[640, 640]], device=device)
        with torch.no_grad():
            for _ in range(2):
                _ = deploy_model(sample_batch, orig_size)
        
        # Process in groups of 5 images
        group_size = 5
        num_groups = (len(all_tensors) + group_size - 1) // group_size
        
        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = min(start_idx + group_size, len(all_tensors))
            
            group_tensors = all_tensors[start_idx:end_idx]
            group_image_ids = image_ids_list[start_idx:end_idx]
            
            print(f"\n  Group {group_idx+1}/{num_groups} (Images {start_idx+1}-{end_idx}):")
            
            # Create batch
            batch = torch.stack(group_tensors).to(device)
            orig_sizes = torch.tensor([[640, 640]] * len(group_tensors), device=device)
            
            # Measure inference
            def run_inference():
                with torch.no_grad():
                    outputs = deploy_model(batch, orig_sizes)
                return outputs
            
            stats = pm.measure(run_inference)
            predictions = stats["result"]
            
            # Calculate mAP
            map_metrics = calculate_map(coco, predictions, group_image_ids)
            
            # Store results
            result = {
                "K": k,
                "group_idx": group_idx + 1,
                "start_img": start_idx + 1,
                "end_img": end_idx,
                "num_images": len(group_tensors),
                "time_s": stats['time_s'],
                "power_W": stats['power_W'],
                "energy_J": stats['power_W'] * stats['time_s'],
                "cpu_util": stats["cpu_utilization"],
                "freq_ratio": stats["freq_ratio"],
                "mAP": map_metrics["mAP"],
                "mAP_50": map_metrics["mAP_50"],
                "mAP_75": map_metrics["mAP_75"],
            }
            all_results.append(result)
            
            # Print results
            print(f"    Time: {stats['time_s']:.4f}s | Power: {stats['power_W']:.2f}W | Energy: {result['energy_J']:.2f}J")
            print(f"    mAP: {map_metrics['mAP']:.4f} | mAP@50: {map_metrics['mAP_50']:.4f} | CPU: {stats['cpu_utilization']*100:.1f}%")
        
        # Calculate K summary
        k_results = [r for r in all_results if r["K"] == k]
        avg_map = sum(r["mAP"] for r in k_results) / len(k_results)
        avg_power = sum(r["power_W"] for r in k_results) / len(k_results)
        avg_energy = sum(r["energy_J"] for r in k_results) / len(k_results)
        avg_time = sum(r["time_s"] for r in k_results) / len(k_results)
        
        print(f"\n  K={k} Summary:")
        print(f"    Avg mAP: {avg_map:.4f}")
        print(f"    Avg Power: {avg_power:.2f}W")
        print(f"    Avg Energy: {avg_energy:.2f}J")
        print(f"    Avg Time: {avg_time:.4f}s")
    
    # Save detailed results
    csv_path = "power_map_per5images.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "K", "Group", "Start_Image", "End_Image", "Num_Images",
            "Time_s", "Power_W", "Energy_J", "CPU_Util", "Freq_Ratio",
            "mAP", "mAP_50", "mAP_75"
        ])
        
        for r in all_results:
            writer.writerow([
                r["K"], r["group_idx"], r["start_img"], r["end_img"], r["num_images"],
                f"{r['time_s']:.4f}", f"{r['power_W']:.2f}", f"{r['energy_J']:.2f}",
                f"{r['cpu_util']:.4f}", f"{r['freq_ratio']:.4f}",
                f"{r['mAP']:.4f}", f"{r['mAP_50']:.4f}", f"{r['mAP_75']:.4f}"
            ])
    
    print(f"\n[INFO] Results saved to: {csv_path}")
    
    # Generate visualization
    print(f"[INFO] Generating visualization...")
    
    # Group by K
    k_values_unique = sorted(set(r["K"] for r in all_results))
    k_vals = []
    avg_maps = []
    avg_powers = []
    avg_energies = []
    avg_times = []
    
    for k in k_values_unique:
        k_results = [r for r in all_results if r["K"] == k]
        k_vals.append(k)
        avg_maps.append(sum(r["mAP"] for r in k_results) / len(k_results))
        avg_powers.append(sum(r["power_W"] for r in k_results) / len(k_results))
        avg_energies.append(sum(r["energy_J"] for r in k_results) / len(k_results))
        avg_times.append(sum(r["time_s"] for r in k_results) / len(k_results))
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Power and mAP Analysis per 5 Images at Different K Values', fontsize=14, fontweight='bold')
    
    # Plot 1: mAP vs K
    axes[0, 0].plot(k_vals, avg_maps, 'o-', color='green', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('K (Number of Queries)', fontsize=11)
    axes[0, 0].set_ylabel('Average mAP', fontsize=11)
    axes[0, 0].set_title('mAP vs K', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Plot 2: Power vs K
    axes[0, 1].plot(k_vals, avg_powers, 'o-', color='red', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('K (Number of Queries)', fontsize=11)
    axes[0, 1].set_ylabel('Average Power (W)', fontsize=11)
    axes[0, 1].set_title('Power vs K', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # Plot 3: Energy vs K
    axes[1, 0].plot(k_vals, avg_energies, 'o-', color='blue', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('K (Number of Queries)', fontsize=11)
    axes[1, 0].set_ylabel('Average Energy (J)', fontsize=11)
    axes[1, 0].set_title('Energy vs K', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Plot 4: Time vs K
    axes[1, 1].plot(k_vals, avg_times, 'o-', color='orange', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('K (Number of Queries)', fontsize=11)
    axes[1, 1].set_ylabel('Average Time (s)', fontsize=11)
    axes[1, 1].set_title('Time vs K', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    plot_path = "power_map_per5images.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Plot saved to: {plot_path}")
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RT-DETR Power and mAP Measurement per 5 Images")
    parser.add_argument("-c", "--config",
                        default="./configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
    parser.add_argument("-r", "--resume",
                        default="rtdetr_r50vd_6x_coco_from_paddle.pth")
    parser.add_argument("--image-dir",
                        default="./dataset/coco/subset_10/images")
    parser.add_argument("--gt-json",
                        default="./dataset/coco/subset_10/instances_train2017.json")
    parser.add_argument("--num-images", type=int, default=10,
                        help="Number of images to process")
    parser.add_argument("--cpu-tdp", type=float, default=15.0,
                        help="CPU TDP in Watts per core")
    parser.add_argument("--num-threads", type=int, default=1,
                        help="Number of threads to use")

    args = parser.parse_args()
    main(args)
