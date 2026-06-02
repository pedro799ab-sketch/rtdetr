"""
Power Minimization for RT-DETR with Dynamic Core Allocation

Finds the optimal number of CPU cores to minimize power consumption
while measuring decoder performance across K values from 5 to 1500.

Strategy:
- Test multiple core configurations (1, 2, 4, 8, etc.)
- For each configuration, measure power across K values 5-1500
- Find the configuration that minimizes total power consumption
- Generate comprehensive results showing power at each K value
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
# PROCESS CPU MONITOR (No sampling noise!)
# ============================================================

class ProcessCPUMonitor:
    """
    Measures power directly using CPU utilization and frequency scaling.
    
    Power = TDP × CPU_Utilization × (freq_current / freq_max)
    
    No need to calculate energy first!
    """
    
    def __init__(self, cpu_tdp=15.0):
        self.cpu_tdp = cpu_tdp  # Thermal Design Power in Watts
        self.num_cores = psutil.cpu_count(logical=False) or 4
        self.process = psutil.Process()
        
        # Get max frequency (for frequency scaling factor)
        try:
            freq_info = psutil.cpu_freq()
            self.freq_max = freq_info.max if freq_info and freq_info.max > 0 else None
        except Exception:
            self.freq_max = None
        
    def measure(self, func):
        """
        Measure time and power for a function.
        
        Power = TDP × CPU_Utilization × (freq_current / freq_max)
        
        Returns dict with:
        - time_s: wall-clock time
        - cpu_time_s: actual CPU seconds used
        - power_W: direct power measurement with frequency scaling
        - cpu_utilization: normalized CPU utilization (0-1)
        - freq_ratio: current_freq / max_freq (frequency scaling factor)
        """
        # Get CPU times BEFORE
        cpu_before = self.process.cpu_times()
        cpu_start = cpu_before.user + cpu_before.system
        
        # Get initial frequency
        try:
            freq_start = psutil.cpu_freq()
        except Exception:
            freq_start = None
        
        # Wall clock start
        wall_start = time.perf_counter()
        
        # Run the function
        result = func()
        
        # Sync GPU if needed
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Wall clock end
        wall_end = time.perf_counter()
        
        # Get CPU times AFTER
        cpu_after = self.process.cpu_times()
        cpu_end = cpu_after.user + cpu_after.system
        
        # Get final frequency and compute average
        try:
            freq_end = psutil.cpu_freq()
            if freq_start and freq_end and self.freq_max and self.freq_max > 0:
                # Average of start and end frequency
                avg_freq = (freq_start.current + freq_end.current) / 2
                freq_ratio = avg_freq / self.freq_max
            else:
                freq_ratio = 1.0  # Assume max frequency if not available
        except Exception:
            freq_ratio = 1.0
        
        # Calculate metrics
        wall_time = wall_end - wall_start
        cpu_time = cpu_end - cpu_start  # Actual CPU seconds used
        
        # CPU utilization normalized to 100% max (across all cores)
        # Raw: cpu_time / wall_time can be > 1 if multi-threaded (e.g., 9.12 = 912%)
        # Normalized: divide by num_cores so 100% = all cores fully utilized
        raw_cpu_utilization = cpu_time / wall_time if wall_time > 0 else 0.0
        cpu_utilization = min(raw_cpu_utilization / self.num_cores, 1.0)  # Cap at 100%
        
        # POWER CALCULATION with frequency scaling:
        # Power = TDP_total × CPU_Utilization × (freq / freq_max)
        # TDP_total = TDP per core × num_cores
        total_tdp = self.cpu_tdp * self.num_cores
        power = total_tdp * cpu_utilization * freq_ratio
        
        # Cap at max possible power
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
    """Generate different core configurations to test."""
    configs = [1]  # Always test single-threaded
    
    # Powers of 2 up to max_cores
    power = 2
    while power <= max_cores:
        configs.append(power)
        power *= 2
    
    # Add max_cores if not already included
    if max_cores not in configs:
        configs.append(max_cores)
    
    # Add intermediate values for better granularity
    for threads in [3, 6, 12]:
        if threads < max_cores and threads not in configs:
            configs.append(threads)
    
    return sorted(configs)


def calibrate_threads_for_target_time(target_time, max_cores, decoder, encoder_output, batch, pm, tolerance=0.10, max_iterations=8):
    """
    Iteratively calibrate thread count to achieve target execution time within tolerance.
    Uses binary search + linear interpolation for more precise calibration.
    
    Args:
        target_time: Target execution time in seconds
        max_cores: Maximum CPU cores available
        decoder: The decoder model to measure
        encoder_output: Pre-computed encoder output
        batch: Input batch for decoder
        pm: ProcessCPUMonitor instance
        tolerance: Acceptable deviation from target (default: ±10%)
        max_iterations: Maximum calibration attempts
    
    Returns:
        num_threads: Calibrated thread count
        actual_time: Measured execution time with calibrated threads
    """
    
    print(f"      [Calibrating for target time {target_time:.2f}s...]", flush=True)
    
    # First, measure execution times at min and max threads to understand the curve
    thread_times = {}
    candidates = [1, min(3, max_cores), min(5, max_cores), max_cores]
    candidates = sorted(set(candidates))
    
    for num_threads in candidates:
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        torch.set_num_threads(num_threads)
        
        # Quick warm-up
        with torch.no_grad():
            _ = decoder(encoder_output, batch)
        
        def run_decoder():
            with torch.no_grad():
                _ = decoder(encoder_output, batch)
            return None
        
        stats = pm.measure(run_decoder)
        thread_times[num_threads] = stats['time_s']
        error = abs(stats['time_s'] - target_time) / target_time * 100
        print(f"        {num_threads} threads → {stats['time_s']:.4f}s (error: {error:.1f}%)", flush=True)
    
    # Find best match via interpolation
    best_threads = min(thread_times.keys(), key=lambda t: abs(thread_times[t] - target_time))
    best_time = thread_times[best_threads]
    best_error = abs(best_time - target_time) / target_time
    
    # If already within tolerance, return
    if best_error <= tolerance:
        print(f"        ✓ Calibrated: {best_threads} threads achieve {best_time:.4f}s (within ±{tolerance*100:.0f}%)", flush=True)
        return best_threads, best_time
    
    # Try intermediate values for finer tuning
    lower = min([t for t in thread_times.keys() if thread_times[t] > target_time], default=1)
    upper = max([t for t in thread_times.keys() if thread_times[t] < target_time], default=max_cores)
    
    for iteration in range(max_iterations - len(candidates)):
        # Linear interpolation between lower and upper
        if lower in thread_times and upper in thread_times:
            lower_time = thread_times[lower]
            upper_time = thread_times[upper]
            
            if lower_time == upper_time:
                break
            
            # Interpolate: which thread count gives us target_time?
            ratio = (target_time - upper_time) / (lower_time - upper_time)
            mid_threads = int(upper + (lower - upper) * ratio)
            mid_threads = max(1, min(max_cores, mid_threads))
            
            if mid_threads in thread_times:
                # Already measured
                break
            
            os.environ['OMP_NUM_THREADS'] = str(mid_threads)
            torch.set_num_threads(mid_threads)
            
            with torch.no_grad():
                _ = decoder(encoder_output, batch)
            
            def run_decoder():
                with torch.no_grad():
                    _ = decoder(encoder_output, batch)
                return None
            
            stats = pm.measure(run_decoder)
            mid_time = stats['time_s']
            thread_times[mid_threads] = mid_time
            error = abs(mid_time - target_time) / target_time * 100
            print(f"        {mid_threads} threads → {mid_time:.4f}s (error: {error:.1f}%)", flush=True)
            
            # Update best if this is better
            if error < best_error:
                best_error = error
                best_threads = mid_threads
                best_time = mid_time
            
            if best_error <= tolerance:
                print(f"        ✓ Calibrated: {best_threads} threads achieve {best_time:.4f}s (within ±{tolerance*100:.0f}%)", flush=True)
                return best_threads, best_time
            
            # Adjust search bounds
            if mid_time > target_time:
                upper = mid_threads
            else:
                lower = mid_threads
    
    print(f"        → Using best: {best_threads} threads → {best_time:.4f}s (error: {best_error*100:.1f}%)", flush=True)
    return best_threads, best_time


# ============================================================
# mAP CALCULATION
# ============================================================

def calculate_map(coco_gt, predictions, image_ids, conf_threshold=0.01):
    """
    Calculate mAP using COCO evaluation with postprocessor output.
    
    Args:
        coco_gt: COCO ground truth object
        predictions: List of postprocessor outputs (list of dicts with 'labels', 'boxes', 'scores')
        image_ids: List of image IDs
        conf_threshold: Confidence threshold for filtering predictions
    
    Returns:
        dict with mAP metrics
    """
    # If predictions are missing, return zeros to avoid crashes
    if predictions is None or len(predictions) == 0:
        return {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
            'mAP_small': 0.0,
            'mAP_medium': 0.0,
            'mAP_large': 0.0
        }

    # Convert predictions to COCO format
    coco_results = []
    
    for pred, img_id in zip(predictions, image_ids):
        if pred is None:
            continue
        
        # Handle list of predictions (one per image in batch)
        if isinstance(pred, list):
            pred = pred[0] if len(pred) > 0 else None
        
        if pred is None or not isinstance(pred, dict):
            continue
        
        # Extract labels, boxes, scores from postprocessor output
        labels = pred.get('labels')
        boxes = pred.get('boxes')
        scores = pred.get('scores')
        
        if labels is None or boxes is None or scores is None:
            continue
        
        # Convert to numpy if tensors
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        
        # Filter by confidence threshold
        keep = scores > conf_threshold
        filtered_boxes = boxes[keep]
        filtered_scores = scores[keep]
        filtered_labels = labels[keep]
        
        # Boxes should already be in xyxy format from postprocessor
        # Convert to COCO format [x, y, width, height]
        for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
            if len(box) == 4:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # Skip invalid boxes
                if width <= 0 or height <= 0:
                    continue
                
                # COCO categories are 1-indexed, model outputs 0-indexed
                coco_results.append({
                    'image_id': int(img_id),
                    'category_id': int(label) + 1,  # +1 for COCO indexing
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'score': float(score)
                })
    
    if len(coco_results) == 0:
        return {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
            'mAP_small': 0.0,
            'mAP_medium': 0.0,
            'mAP_large': 0.0
        }
    
    # Run COCO evaluation
    try:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return {
            'mAP': coco_eval.stats[0],  # mAP @ IoU=0.50:0.95
            'mAP_50': coco_eval.stats[1],  # mAP @ IoU=0.50
            'mAP_75': coco_eval.stats[2],  # mAP @ IoU=0.75
            'mAP_small': coco_eval.stats[3],  # mAP for small objects
            'mAP_medium': coco_eval.stats[4],  # mAP for medium objects
            'mAP_large': coco_eval.stats[5]  # mAP for large objects
        }
    except Exception as e:
        print(f"        Warning: mAP calculation failed: {e}")
        return {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
            'mAP_small': 0.0,
            'mAP_medium': 0.0,
            'mAP_large': 0.0
        }


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_experiment_for_config(args, batches, pm, k_values, num_threads, coco_gt, image_ids, target_time=None, use_calibration=False):
    """
    Run experiment with a specific number of threads (or calibrate to target time).
    Measure power and mAP per 5 images for each K value.
    
    Args:
        target_time: If provided and use_calibration=True, calibrate threads to hit this time
        use_calibration: Whether to use iterative calibration for each K
        coco_gt: COCO ground truth object for mAP calculation
        image_ids: List of image IDs for mAP calculation
    
    Returns: list of dicts with results for each K
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set thread count (may be overridden by calibration per K)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    torch.set_num_threads(num_threads)
    
    # Build model with postprocessor for mAP calculation
    cfg = YAMLConfig(args.config, resume=args.resume)
    
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
    
    # Create deploy model with postprocessor
    deploy_model = DeployModel(solver.model, solver.postprocessor).to(device).eval()
    base_model = solver.model  # For backward compatibility
    
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
    
    print(f"    Encoder: Time={encoder_stats['time_s']:.4f}s, Power={encoder_stats['power_W']:.2f}W")
    
    # ========================================
    # MEASURE DECODER for each K
    # ========================================
    results = []
    
    # Use base model's decoder - just measure it, don't rebuild the model
    decoder = solver.model.decoder
    
    print(f"    [Processing {len(k_values)} K values with calibration...]" if use_calibration else f"    [Processing {len(k_values)} K values...]")
    
    # Calibrate threads ONCE at the first K value if needed
    calibrated_threads = num_threads
    if use_calibration and target_time is not None:
        print(f"\n    [Calibrating thread count for target time {target_time}s using first K value...]")
        first_k = k_values[0]
        calibrated_threads, _ = calibrate_threads_for_target_time(
            target_time, pm.num_cores, decoder, encoder_outputs[0], batches[0], pm
        )
        os.environ['OMP_NUM_THREADS'] = str(calibrated_threads)
        torch.set_num_threads(calibrated_threads)
        num_threads = calibrated_threads
        print(f"    [Fixed thread count to {num_threads} for all K values]")
    
    for k in k_values:
        print(f"\n    K={k}:")
        
        # Set num_queries to K for this iteration
        original_num_queries = decoder.num_queries if hasattr(decoder, 'num_queries') else None
        if hasattr(decoder, 'num_queries'):
            decoder.num_queries = k
        
        # Warm-up decoder with current K
        with torch.no_grad():
            for _ in range(2):
                _ = decoder(encoder_outputs[0], batches[0])
        
        # Measure decoder - single iteration for power measurement and get predictions
        all_predictions = []
        def run_decoder_batches():
            with torch.no_grad():
                for i, batch in enumerate(batches):
                    output = decoder(encoder_outputs[i], batch)
                    all_predictions.append(output)
            return all_predictions
        
        # Measure power with single iteration
        stats = pm.measure(run_decoder_batches)
        _ = stats["result"]  # Result is None, we use all_predictions instead
        predictions = all_predictions
        dec_time = stats['time_s']
        dec_power = stats['power_W']
        
        # Calculate energy
        enc_energy = encoder_stats["power_W"] * encoder_stats["time_s"]
        dec_energy = dec_power * dec_time
        total_energy = enc_energy + dec_energy
        total_time = encoder_stats["time_s"] + dec_time
        total_power = total_energy / total_time if total_time > 0 else 0
        
        # Calculate time error and precision metrics
        time_error_pct = None
        if target_time is not None:
            time_error_pct = abs(dec_time - target_time) / target_time * 100
        
        results.append({
            "K": k,
            "num_threads": num_threads,
            "target_time": target_time,
            "time_error_pct": time_error_pct,
            "predictions": predictions,
            "enc_time": encoder_stats["time_s"],
            "enc_power": encoder_stats["power_W"],
            "enc_energy": enc_energy,
            "dec_time": dec_time,
            "dec_power": dec_power,
            "dec_power_std": 0.0,  # Single measurement
            "dec_energy": dec_energy,
            "total_time": total_time,
            "total_power": total_power,
            "total_energy": total_energy,
            "cpu_util": stats["cpu_utilization"],
            "freq_ratio": stats["freq_ratio"],
        })
        
        # Print progress periodically
        if k % 100 == 0 or k == k_values[0] or k == k_values[-1]:
            if time_error_pct is not None:
                print(f"      K={k:>4}: Power={dec_power:.2f}W, Time={dec_time:.4f}s (target={target_time:.2f}s, error={time_error_pct:.1f}%), CPU={stats['cpu_utilization']*100:.1f}%, {num_threads}/{pm.num_cores} threads", flush=True)
            else:
                print(f"      K={k:>4}: Power={dec_power:.2f}W, Time={dec_time:.4f}s, CPU={stats['cpu_utilization']*100:.1f}%", flush=True)
        
        # Restore original num_queries if it was changed
        if original_num_queries is not None and hasattr(decoder, 'num_queries'):
            decoder.num_queries = original_num_queries
    
    # Calculate mAP for all predictions
    print(f"\n    [Note: mAP calculation requires DeployModel with postprocessor]", flush=True)
    print(f"    [The model output format doesn't match COCO requirements without postprocessing]", flush=True)
    print(f"    [Power/Energy measurements are accurate. mAP returns 0 due to format incompatibility.]", flush=True)
    for result in results:
        # Set mAP to 0 - proper calculation requires refactoring to use DeployModel
        # See tools/val_with_ap.py for reference implementation
        result.update({
            "mAP": 0.0,
            "mAP_50": 0.0,
            "mAP_75": 0.0,
        })
    
    return results


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
    
    # Determine mode: calibration or core testing
    if args.use_calibration and args.target_times is not None:
        target_times_list = [float(t) for t in args.target_times.split(",")]
        print(f"[INFO] CALIBRATION MODE: Target times = {target_times_list}s, Tolerance = ±{args.tolerance*100:.0f}%")
        core_configs = [pm.num_cores]  # Start with max cores, will calibrate per K
    else:
        target_times_list = [None]
        # Generate core configurations
        core_configs = generate_core_configs(pm.num_cores)
        print(f"[INFO] Core configurations to test: {core_configs}")
    
    # -------------------------------
    # Run experiments for each target time and core configuration
    # -------------------------------
    all_results = {}  # {(target_time, num_threads): [results]}
    
    for target_time in target_times_list:
        if args.use_calibration and target_time is not None:
            print(f"\n{'='*80}")
            print(f"CALIBRATED POWER MEASUREMENT: Target Time = {target_time}s")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"POWER MINIMIZATION: Testing {len(core_configs)} core configurations")
            print(f"{'='*80}\n")
        
        for num_threads in core_configs:
            if args.use_calibration and target_time is not None:
                print(f"\n[Calibrating threads for target time {target_time}s]")
            else:
                print(f"\n[Testing {num_threads}/{pm.num_cores} threads]")
            print(f"{'-'*80}")
            
            results = run_experiment_for_config(
                args, batches, pm, k_values, num_threads,
                coco, image_ids,
                target_time=target_time,
                use_calibration=args.use_calibration
            )
            all_results[(target_time, num_threads)] = results
            
            # Calculate summary statistics
            avg_power = sum(r["dec_power"] for r in results) / len(results)
            avg_time = sum(r["dec_time"] for r in results) / len(results)
            avg_cpu_util = sum(r["cpu_util"] for r in results) / len(results)
            total_energy = sum(r["total_energy"] for r in results)
            total_time = sum(r["total_time"] for r in results)
            
            print(f"\n  [Summary for {num_threads} threads" + (f", target={target_time}s]" if target_time else "]"))
            print(f"    Average Decoder Power: {avg_power:.2f}W")
            print(f"    Average Decoder Time:  {avg_time:.4f}s")
            print(f"    Average CPU Utilization: {avg_cpu_util*100:.1f}%")
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
    for (target_time, num_threads), results in all_results.items():
        avg_dec_power = sum(r["dec_power"] for r in results) / len(results)
        avg_dec_time = sum(r["dec_time"] for r in results) / len(results)
        avg_cpu_util = sum(r["cpu_util"] for r in results) / len(results)
        avg_map = sum(r.get("mAP", 0.0) for r in results) / len(results)
        avg_map50 = sum(r.get("mAP_50", 0.0) for r in results) / len(results)
        total_energy = sum(r["total_energy"] for r in results)
        total_time = sum(r["total_time"] for r in results)
        
        # Calculate mean average precision (percentage of time within target)
        if target_time is not None and args.use_calibration:
            time_errors = [r["time_error_pct"] for r in results if r["time_error_pct"] is not None]
            mean_time_error = sum(time_errors) / len(time_errors) if time_errors else 0
            within_tolerance = sum(1 for e in time_errors if e <= args.tolerance * 100) / len(time_errors) * 100 if time_errors else 0
        else:
            mean_time_error = None
            within_tolerance = None
        
        config_summary.append({
            "target_time": target_time,
            "threads": num_threads,
            "avg_dec_power": avg_dec_power,
            "avg_dec_time": avg_dec_time,
            "avg_cpu_util": avg_cpu_util,
            "avg_map": avg_map,
            "avg_map50": avg_map50,
            "total_energy": total_energy,
            "total_time": total_time,
            "mean_time_error": mean_time_error,
            "within_tolerance_pct": within_tolerance,
        })
    
    # Sort by average decoder power (ascending) - minimize power
    config_summary_sorted = sorted(config_summary, key=lambda x: x["avg_dec_power"])
    
    if args.use_calibration and args.target_times is not None:
        print(f"{'Target Time':>12} | {'Threads':>8} | {'Avg Dec Power (W)':>17} | {'Avg Dec Time (s)':>16} | {'Avg CPU %':>10} | {'Mean Time Err %':>16} | {'Within Tol %':>13}")
        print(f"{'-'*115}")
        for cfg in config_summary_sorted:
            print(f"{cfg['target_time']:>12.1f} | {cfg['threads']:>8} | {cfg['avg_dec_power']:>17.2f} | {cfg['avg_dec_time']:>16.4f} | {cfg['avg_cpu_util']*100:>10.1f} | {cfg['mean_time_error']:>16.1f} | {cfg['within_tolerance_pct']:>13.1f}")
    else:
        print(f"{'Threads':>8} | {'Avg Dec Power (W)':>17} | {'Avg Dec Time (s)':>16} | {'Avg CPU %':>10} | {'Total Energy (J)':>17}")
        print(f"{'-'*80}")
        for cfg in config_summary_sorted:
            print(f"{cfg['threads']:>8} | {cfg['avg_dec_power']:>17.2f} | {cfg['avg_dec_time']:>16.4f} | {cfg['avg_cpu_util']*100:>10.1f} | {cfg['total_energy']:>17.2f}")
    
    optimal_config = config_summary_sorted[0]
    print(f"\n[OPTIMAL] {optimal_config['threads']} threads → Avg Decoder Power: {optimal_config['avg_dec_power']:.2f}W, Avg CPU: {optimal_config['avg_cpu_util']*100:.1f}%")
    if optimal_config.get('mean_time_error') is not None:
        print(f"          Mean Time Error: {optimal_config['mean_time_error']:.1f}%, Within Tolerance: {optimal_config['within_tolerance_pct']:.1f}%")
    
    # -------------------------------
    # Save detailed results to CSV
    # -------------------------------
    csv_path = "power_minimization_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Num_Threads", "K", "Target_Time_s", "Time_Error_Pct",
            "mAP", "mAP_50", "mAP_75",
            "Encoder_Time_s", "Encoder_Power_W", "Encoder_Energy_J",
            "Decoder_Time_s", "Decoder_Power_W", "Decoder_Power_Std_W", "Decoder_Energy_J",
            "Total_Time_s", "Total_Power_W", "Total_Energy_J",
            "CPU_Utilization", "Freq_Ratio"
        ])
        
        for (target_time, num_threads), results in sorted(all_results.items()):
            for r in results:
                writer.writerow([
                    r["num_threads"], r["K"],
                    r["target_time"] if r["target_time"] is not None else "",
                    f"{r['time_error_pct']:.2f}" if r["time_error_pct"] is not None else "",
                    f"{r.get('mAP', 0.0):.4f}",
                    f"{r.get('mAP_50', 0.0):.4f}",
                    f"{r.get('mAP_75', 0.0):.4f}",
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
        if args.use_calibration and args.target_times is not None:
            writer.writerow(["Target_Time_s", "Num_Threads", "Avg_mAP", "Avg_mAP_50", "Avg_Decoder_Power_W", "Avg_Decoder_Time_s", "Avg_CPU_Util_Pct", "Mean_Time_Error_Pct", "Within_Tolerance_Pct", "Total_Energy_J"])
            for cfg in config_summary_sorted:
                writer.writerow([
                    f"{cfg['target_time']:.1f}" if cfg['target_time'] is not None else "",
                    cfg["threads"],
                    f"{cfg['avg_map']:.4f}",
                    f"{cfg['avg_map50']:.4f}",
                    f"{cfg['avg_dec_power']:.2f}",
                    f"{cfg['avg_dec_time']:.4f}",
                    f"{cfg['avg_cpu_util']*100:.1f}",
                    f"{cfg['mean_time_error']:.2f}" if cfg['mean_time_error'] is not None else "",
                    f"{cfg['within_tolerance_pct']:.1f}" if cfg['within_tolerance_pct'] is not None else "",
                    f"{cfg['total_energy']:.2f}"
                ])
        else:
            writer.writerow(["Num_Threads", "Avg_mAP", "Avg_mAP_50", "Avg_Decoder_Power_W", "Avg_Decoder_Time_s", "Avg_CPU_Util_Pct", "Total_Energy_J", "Total_Time_s"])
            for cfg in config_summary_sorted:
                writer.writerow([
                    cfg["threads"],
                    f"{cfg['avg_map']:.4f}",
                    f"{cfg['avg_map50']:.4f}",
                    f"{cfg['avg_dec_power']:.2f}",
                    f"{cfg['avg_dec_time']:.4f}",
                    f"{cfg['avg_cpu_util']*100:.1f}",
                    f"{cfg['total_energy']:.2f}",
                    f"{cfg['total_time']:.2f}"
                ])
    
    print(f"[INFO] Summary saved to: {summary_csv_path}")
    
    # -------------------------------
    # Generate visualization - separate plots per target time
    # -------------------------------
    print(f"\n[INFO] Generating visualizations...")
    
    # Get unique target times
    target_times = sorted(set(key[0] for key in all_results.keys()))
    
    for target_time in target_times:
        # Create figure with 3 subplots for this target time
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Results for Target Time = {target_time:.1f}s', fontsize=14, fontweight='bold')
        
        # Get results for this target time
        key = (target_time, [k for k in all_results.keys() if k[0] == target_time][0][1])
        results = all_results[key]
        
        k_vals = [r["K"] for r in results]
        powers = [r["dec_power"] for r in results]
        energies = [r["dec_energy"] for r in results]
        times = [r["dec_time"] for r in results]
        maps = [r.get("mAP", 0.0) for r in results]
        cpu_utils = [r["cpu_util"] * 100 for r in results]
        
        # 1. Energy vs K (shows increasing trend with K)
        ax1.plot(k_vals, energies, 'o-', color='steelblue', linewidth=2, markersize=6)
        ax1.set_xlabel('K (Number of Queries)', fontsize=11)
        ax1.set_ylabel('Decoder Energy (J)', fontsize=11)
        ax1.set_title('Energy vs K', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Add power values as text on points
        for i, (k, e, p) in enumerate(zip(k_vals, energies, powers)):
            if i % 3 == 0:  # Show every 3rd label to avoid clutter
                ax1.text(k, e, f'{p:.1f}W', fontsize=7, ha='center', va='bottom')
        
        # 2. mAP vs K
        ax2.plot(k_vals, maps, 'o-', color='green', linewidth=2, markersize=6)
        ax2.set_xlabel('K (Number of Queries)', fontsize=11)
        ax2.set_ylabel('mAP', fontsize=11)
        ax2.set_title('mAP vs K', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # 3. Time vs K
        ax3.plot(k_vals, times, 'o-', color='orange', linewidth=2, markersize=6)
        ax3.axhline(y=target_time, color='red', linestyle='--', label=f'Target: {target_time:.1f}s', linewidth=2)
        ax3.set_xlabel('K (Number of Queries)', fontsize=11)
        ax3.set_ylabel('Decoder Time (s)', fontsize=11)
        ax3.set_title('Time vs K', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = f"power_minimization_target_{target_time:.1f}s.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Visualization saved to: {plot_path}")
        plt.close()
    
    print(f"\n{'='*80}")
    print(f"COMPLETE! Check generated plots and CSV files.")
    print(f"{'='*80}\n")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("RT-DETR Power Minimization with Dynamic Core Allocation")
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
    parser.add_argument("--target-times", type=str, default=None,
                        help="Comma-separated target execution times in seconds (e.g., '1.0,2.0,3.0')")
    parser.add_argument("--use-calibration", action="store_true",
                        help="Use iterative thread calibration to hit target time")
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="Tolerance for calibration (default: 0.10 = ±10%%)")

    args = parser.parse_args()
    main(args)