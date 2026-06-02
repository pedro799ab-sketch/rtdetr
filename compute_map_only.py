"""
Compute mAP for 5 images using RT-DETR with proper postprocessing.

Simple script that:
1. Loads model with postprocessor
2. Runs inference on 5 images
3. Calculates mAP using COCO evaluation
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.core import YAMLConfig
from src.solver import TASKS


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


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Number of images: {args.num_images}")
    
    # Load COCO dataset
    print(f"[INFO] Loading COCO annotations from: {args.gt_json}")
    coco = COCO(args.gt_json)
    image_ids = coco.getImgIds()[:args.num_images]
    print(f"[INFO] Using {len(image_ids)} images")
    
    # Load model
    print(f"[INFO] Loading model config: {args.config}")
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint
    
    # Setup model
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    solver.model.load_state_dict(state)
    
    # Create deploy model with postprocessor
    deploy_model = DeployModel(solver.model, solver.postprocessor).to(device).eval()
    print(f"[INFO] Model loaded and ready for inference")
    
    # Load images and prepare batches
    print(f"\n[INFO] Loading {len(image_ids)} images...")
    all_tensors = []
    image_sizes = []
    
    for img_id in image_ids:
        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.image_dir, info["file_name"])
        img = Image.open(img_path).convert("RGB")
        
        # Store original size for postprocessor
        orig_size = torch.tensor([img.width, img.height]).unsqueeze(0)
        image_sizes.append(orig_size)
        
        # Resize to model input size
        img = img.resize((640, 640))
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        all_tensors.append(t)
        print(f"  [{len(all_tensors)}/{len(image_ids)}] Loaded {info['file_name']}")
    
    # Create batches
    batch_size = min(args.batch_size, len(all_tensors))
    batches = []
    batch_image_sizes = []
    batch_image_ids = []
    
    for i in range(0, len(all_tensors), batch_size):
        batch = torch.stack(all_tensors[i:i+batch_size]).to(device)
        batch_sizes = torch.cat(image_sizes[i:i+batch_size], dim=0).to(device)
        batch_ids = image_ids[i:i+batch_size]
        
        batches.append(batch)
        batch_image_sizes.append(batch_sizes)
        batch_image_ids.append(batch_ids)
    
    print(f"[INFO] Created {len(batches)} batches (batch_size={batch_size})\n")
    
    # Run inference
    print("[INFO] Running inference...")
    all_predictions_raw = []
    
    with torch.no_grad():
        for batch_idx, (batch, batch_sizes, batch_ids) in enumerate(zip(batches, batch_image_sizes, batch_image_ids)):
            print(f"  Batch {batch_idx + 1}/{len(batches)}...", end=" ", flush=True)
            
            # Forward pass with postprocessor - returns (labels, boxes, scores) tuple
            output = deploy_model(batch, batch_sizes)
            
            # output should be a tuple (labels, boxes, scores)
            if isinstance(output, tuple) and len(output) == 3:
                labels, boxes, scores = output
                all_predictions_raw.append({
                    'labels': labels,
                    'boxes': boxes,
                    'scores': scores,
                    'image_ids': batch_ids
                })
            else:
                print(f"Unexpected output format: {type(output)}")
            
            print("Done")
    
    # Convert predictions to COCO format
    print("\n[INFO] Converting to COCO format...")
    coco_results = []
    
    for batch_preds in all_predictions_raw:
        labels = batch_preds['labels']  # Shape: (batch_size, num_queries)
        boxes = batch_preds['boxes']      # Shape: (batch_size, num_queries, 4)
        scores = batch_preds['scores']    # Shape: (batch_size, num_queries)
        image_ids = batch_preds['image_ids']
        
        # Convert to numpy if tensors
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        
        # Process each image in the batch
        for batch_idx in range(len(image_ids)):
            img_id = image_ids[batch_idx]
            img_labels = labels[batch_idx]
            img_boxes = boxes[batch_idx]
            img_scores = scores[batch_idx]
            
            # Filter by confidence threshold
            valid_mask = img_scores > args.conf_threshold
            img_labels = img_labels[valid_mask]
            img_boxes = img_boxes[valid_mask]
            img_scores = img_scores[valid_mask]
            
            print(f"  Image {img_id}: {len(img_boxes)} detections (conf > {args.conf_threshold})")
            
            # Boxes are in xyxy format, convert to COCO format [x, y, width, height]
            for box, score, label in zip(img_boxes, img_scores, img_labels):
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
    
    print(f"\n[INFO] Total predictions in COCO format: {len(coco_results)}")
    
    if len(coco_results) == 0:
        print("[WARNING] No predictions generated!")
        return
    
    # Run COCO evaluation
    print("\n[INFO] Running COCO evaluation...")
    try:
        coco_dt = coco.loadRes(coco_results)
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.params.imgIds = image_ids  # Use the original image IDs list
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        print("\n" + "="*80)
        print("mAP RESULTS FOR 5 IMAGES")
        print("="*80)
        print(f"mAP @ IoU=0.50:0.95        : {coco_eval.stats[0]:.4f}")
        print(f"mAP @ IoU=0.50             : {coco_eval.stats[1]:.4f}")
        print(f"mAP @ IoU=0.75             : {coco_eval.stats[2]:.4f}")
        print(f"mAP (small objects)        : {coco_eval.stats[3]:.4f}")
        print(f"mAP (medium objects)       : {coco_eval.stats[4]:.4f}")
        print(f"mAP (large objects)        : {coco_eval.stats[5]:.4f}")
        print(f"mAR @ IoU=0.50:0.95, maxDets=1   : {coco_eval.stats[6]:.4f}")
        print(f"mAR @ IoU=0.50:0.95, maxDets=10  : {coco_eval.stats[7]:.4f}")
        print(f"mAR @ IoU=0.50:0.95, maxDets=100 : {coco_eval.stats[8]:.4f}")
        print(f"mAR (small objects)        : {coco_eval.stats[9]:.4f}")
        print(f"mAR (medium objects)       : {coco_eval.stats[10]:.4f}")
        print(f"mAR (large objects)        : {coco_eval.stats[11]:.4f}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"[ERROR] mAP calculation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute mAP for N images")
    parser.add_argument("-c", "--config",
                        default="./configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
    parser.add_argument("-r", "--resume",
                        default="rtdetr_r50vd_6x_coco_from_paddle.pth")
    parser.add_argument("--image-dir",
                        default="./dataset/coco/val2017")
    parser.add_argument("--gt-json",
                        default="./dataset/coco/instances_val2017.json")
    parser.add_argument("--num-images", type=int, default=5,
                        help="Number of images to compute mAP for (default: 5)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Batch size for inference (default: 5)")
    parser.add_argument("--conf-threshold", type=float, default=0.01,
                        help="Confidence threshold for filtering predictions (default: 0.01)")

    args = parser.parse_args()
    main(args)
