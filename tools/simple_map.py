"""
Simple script to calculate average precision (mAP) on 5 images from COCO dataset.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import transforms as T

from src.core import YAMLConfig
from src.solver import TASKS
from src.data.dataset import mscoco_label2category


def preprocess_image(img, target_size=640):
    """
    Preprocess image with padding to maintain aspect ratio.
    This matches the training preprocessing.
    """
    from PIL import Image
    import numpy as np
    
    # Get original size
    orig_w, orig_h = img.size
    
    # Calculate scaling factor to fit within target_size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Resize image maintaining aspect ratio
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Create padded image
    padded_img = Image.new('RGB', (target_size, target_size), (114, 114, 114))
    padded_img.paste(img_resized, (0, 0))
    
    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(padded_img)).permute(2, 0, 1).float() / 255.0
    
    return img_tensor, scale, new_w, new_h


def main():
    parser = argparse.ArgumentParser("Calculate mAP on COCO images")
    parser.add_argument("-c", "--config", 
                       default="configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
    parser.add_argument("-r", "--resume",
                       default="rtdetr_r50vd_6x_coco_from_paddle.pth")
    parser.add_argument("--num-images", type=int, default=5,
                       help="Number of images to evaluate (default: 5)")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                       help="Confidence score threshold (default: 0.3)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch size for processing (default: 10)")
    parser.add_argument("--gt-json",
                       default="dataset/coco/instances_val2017.json")
    args = parser.parse_args()
    
    device = torch.device('cpu')
    print(f"\n{'='*60}")
    print(f"CALCULATING mAP ON {args.num_images} COCO IMAGES")
    print(f"Score threshold: {args.score_threshold}")
    print(f"{'='*60}\n")
    
    # Load model in deploy mode
    print("Loading model...")
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.resume}...")
    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
        print("Loaded EMA weights")
    elif 'model' in checkpoint:
        state = checkpoint['model']
        print("Loaded model weights")
    else:
        state = checkpoint
        print("Loaded checkpoint weights")
    
    # Load state into config model
    cfg.model.load_state_dict(state)
    
    # Create deploy model with postprocessor
    class DeployModel(torch.nn.Module):
        def __init__(self, model, postprocessor):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = DeployModel(cfg.model, cfg.postprocessor).to(device).eval()
    
    # Load COCO ground truth
    print(f"Loading COCO ground truth from {args.gt_json}...")
    coco_gt = COCO(args.gt_json)
    
    # Get first N images
    image_ids = coco_gt.getImgIds()[:args.num_images]
    print(f"Selected {len(image_ids)} images\n")
    
    # Process images in batches to avoid memory issues
    print(f"Loading and processing images in batches of {args.batch_size}...")
    from PIL import Image
    import numpy as np
    
    predictions = []
    num_batches = (len(image_ids) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(image_ids))
        batch_image_ids = image_ids[start_idx:end_idx]
        
        print(f"\nBatch {batch_idx + 1}/{num_batches} (images {start_idx + 1}-{end_idx})...")
        
        # Load batch images
        all_images = []
        image_info_list = []
        
        for img_id in batch_image_ids:
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join('dataset/coco/val2017', img_info['file_name'])
            
            # Load and resize image (simple resize as per RT-DETR reference)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((640, 640))
            
            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            all_images.append(img_tensor)
            
            # Store info for bbox conversion
            image_info_list.append({
                'img_id': img_id,
                'orig_w': img_info['width'],
                'orig_h': img_info['height']
            })
        
        # Stack into batch
        batch = torch.stack(all_images).to(device)
        
        # Prepare original sizes tensor for postprocessor
        orig_target_sizes = torch.tensor([[info['orig_w'], info['orig_h']] 
                                          for info in image_info_list]).to(device)
        
        # Run inference with deploy model (returns labels, boxes, scores)
        with torch.no_grad():
            labels, boxes, scores = model(batch, orig_target_sizes)
            
            # Convert outputs to COCO format
            for i, img_id in enumerate(batch_image_ids):
                img_info_item = image_info_list[i]
                img_id_val = img_info_item['img_id']
                
                # Get predictions for this image
                img_labels = labels[i].cpu().numpy()  # [num_preds]
                img_boxes = boxes[i].cpu().numpy()    # [num_preds, 4] - x1,y1,x2,y2
                img_scores = scores[i].cpu().numpy()  # [num_preds]
                
                # Filter by score threshold
                valid_mask = img_scores > args.score_threshold
                img_labels = img_labels[valid_mask]
                img_boxes = img_boxes[valid_mask]
                img_scores = img_scores[valid_mask]
                
                # Convert to COCO format [x, y, w, h]
                for j in range(len(img_boxes)):
                    x1, y1, x2, y2 = img_boxes[j]
                    x = float(x1)
                    y = float(y1)
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    
                    # Map from 0-79 label to COCO category ID (1-90)
                    label_idx = int(img_labels[j])
                    coco_category_id = mscoco_label2category[label_idx]
                    
                    predictions.append({
                        "image_id": int(img_id_val),
                        "category_id": coco_category_id,
                        "bbox": [x, y, w, h],
                        "score": float(img_scores[j])
                    })
                
                if (i + 1) % 10 == 0 or i == len(batch_image_ids) - 1:
                    print(f"  Processed {start_idx + i + 1}/{len(image_ids)} images...")
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}\n")
    print(f"Total predictions (score > {args.score_threshold}): {len(predictions)}\n")
    
    if predictions:
        # Calculate mAP
        print("Computing mAP...\n")
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        # Evaluate only on the selected images
        coco_eval.params.imgIds = image_ids
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Images evaluated: {args.num_images}")
        print(f"Total predictions: {len(predictions)}")
        print(f"Average Precision (AP) @ IoU=0.50:0.95: {coco_eval.stats[0]:.4f}")
        print(f"Average Precision (AP) @ IoU=0.50:      {coco_eval.stats[1]:.4f}")
        print(f"Average Precision (AP) @ IoU=0.75:      {coco_eval.stats[2]:.4f}")
        print(f"{'='*60}\n")
        
        # Show ground truth info
        total_gt = 0
        for img_id in image_ids:
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            total_gt += len(ann_ids)
        print(f"NOTE: Total ground truth objects in {args.num_images} images: {total_gt}")
        print(f"NOTE: For reference, RT-DETR R50 achieves ~53% mAP on full COCO val2017 (5000 images)")
        print(f"NOTE: With only {args.num_images} images, results have high variance.\n")
    else:
        print(f"⚠ No predictions found with score > {args.score_threshold}\n")


if __name__ == "__main__":
    main()
