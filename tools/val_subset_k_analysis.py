"""Validate on a subset COCO JSON and compute AP at different K values and encoder AP.

Usage:
  python tools/val_subset_k_analysis.py \
    --ann-file dataset/coco/subset_10/instances.json \
    --images-dir dataset/coco/subset_10/images \
    --k-values 50 100 200 300 \
    --device cpu

This script:
1. Loads a model checkpoint (auto-selects EMA if available)
2. Runs inference on all images in the annotation file
3. Evaluates AP (COCO metrics) at different K (num_queries) by filtering top-K predictions
4. Reports encoder AP separately (constant across K)
"""
import os
import sys
import json
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.core import YAMLConfig


def load_model_and_cfg(config, resume, device):
    if config is None:
        config = "./configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"

    # auto-select resume if not provided
    if resume is None:
        import glob
        candidates = glob.glob('rtdetr*v2*_ema.pth') + glob.glob('*.pth')
        preferred = None
        for c in candidates:
            if 'ema' in os.path.basename(c):
                preferred = c
                break
        if preferred is None and candidates:
            preferred = candidates[0]
        resume = preferred

    cfg = YAMLConfig(config, resume=resume)

    # load checkpoint weights
    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint.get('model', None)

        if state is not None:
            cfg.model.load_state_dict(state)
        else:
            logging.warning('No model state found in checkpoint; using init weights')

    # Build model wrapper
    class ModelWrapper(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes, targets=None):
            outputs = self.model(images, targets)
            outputs_enc = {}
            enc_logits = outputs.get('enc_pred_logits', None) if isinstance(outputs, dict) else None
            enc_boxes = outputs.get('enc_pred_boxes', None) if isinstance(outputs, dict) else None
            if enc_logits is None or enc_boxes is None:
                logging.getLogger(__name__).warning("Encoder outputs missing; falling back to decoder outputs")
                enc_logits = outputs.get('pred_logits') if isinstance(outputs, dict) else None
                enc_boxes = outputs.get('pred_boxes') if isinstance(outputs, dict) else None

            outputs_enc['pred_logits'] = enc_logits
            outputs_enc['pred_boxes'] = enc_boxes
            outputs = self.postprocessor(outputs, orig_target_sizes)
            outputs_enc = self.postprocessor(outputs_enc, orig_target_sizes)
            return outputs, outputs_enc

    model = ModelWrapper(cfg)
    model.to(device)
    model.eval()
    return cfg, model


def run_inference_and_collect(ann_file, images_dir, model, device):
    """Run inference and collect all predictions (labels, boxes, scores) for each image."""
    with open(ann_file, 'r') as f:
        ann = json.load(f)

    images = ann.get('images', [])
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    # Store predictions: image_id -> (labels, boxes, scores) for decoder and encoder
    decoder_preds = {}
    encoder_preds = {}

    for im_meta in images:
        file_name = im_meta['file_name']
        img_id = im_meta['id']
        img_path = os.path.join(images_dir, file_name)
        if not os.path.exists(img_path):
            logging.warning(f'Image not found: {img_path}, skipping')
            continue

        im = Image.open(img_path).convert('RGB')
        w, h = im.size
        im_tensor = transform(im).unsqueeze(0).to(device)
        orig_target_sizes = torch.tensor([[w, h]], dtype=torch.int64, device=device)

        with torch.no_grad():
            out, out_enc = model(im_tensor, orig_target_sizes, targets=None)

        # Decoder outputs (labels, boxes, scores)
        labels = out[0][0].cpu()
        boxes = out[1][0].cpu()
        scores = out[2][0].cpu()
        decoder_preds[img_id] = (labels, boxes, scores)

        # Encoder outputs
        labels_enc = out_enc[0][0].cpu()
        boxes_enc = out_enc[1][0].cpu()
        scores_enc = out_enc[2][0].cpu()
        encoder_preds[img_id] = (labels_enc, boxes_enc, scores_enc)

    return decoder_preds, encoder_preds


def filter_topk(labels, boxes, scores, k):
    """Filter predictions to top-K by score."""
    if len(scores) > k:
        topk_idx = torch.topk(scores, k).indices
        labels = labels[topk_idx]
        boxes = boxes[topk_idx]
        scores = scores[topk_idx]
    return labels, boxes, scores


def predictions_to_coco_format(preds_dict, score_threshold=0.0):
    """Convert predictions dict to COCO result format (list of dicts)."""
    results = []
    for img_id, (labels, boxes, scores) in preds_dict.items():
        for i in range(len(scores)):
            score = float(scores[i])
            if score < score_threshold:
                continue
            label = int(labels[i])
            box = boxes[i].tolist()
            # COCO format: [xmin, ymin, width, height]
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            results.append({
                'image_id': img_id,
                'category_id': label + 1,  # COCO categories are 1-indexed
                'bbox': [xmin, ymin, width, height],
                'score': score
            })
    return results


def evaluate_coco(ann_file, results):
    """Evaluate COCO AP using pycocotools."""
    if len(results) == 0:
        print("No predictions to evaluate!")
        return {}

    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Return AP@0.5:0.95
    stats = {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5],
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-file', default='dataset/coco/subset_10/instances_train2017.json', help='COCO annotation file')
    parser.add_argument('--images-dir', default='dataset/coco/subset_10/images', help='Directory containing images')
    parser.add_argument('--config', '-c', type=str, help='Config YAML')
    parser.add_argument('--resume', '-r', type=str, help='Checkpoint to load')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu, cuda, mps)')
    parser.add_argument('--k-values', nargs='+', type=int, default=[50, 100, 200, 300], help='K values to test')
    parser.add_argument('--score-threshold', type=float, default=0.0, help='Score threshold for evaluation')
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    print(f"\n=== Validation on {args.ann_file} ===")
    print(f"Device: {device}")
    print(f"K values: {args.k_values}")
    print()

    cfg, model = load_model_and_cfg(args.config, args.resume, device)
    decoder_preds, encoder_preds = run_inference_and_collect(args.ann_file, args.images_dir, model, device)

    print(f"Collected predictions for {len(decoder_preds)} images\n")

    # Evaluate decoder at different K values
    print("=" * 60)
    print("DECODER AP @ DIFFERENT K VALUES")
    print("=" * 60)
    for k in args.k_values:
        filtered_preds = {}
        for img_id, (labels, boxes, scores) in decoder_preds.items():
            filtered_preds[img_id] = filter_topk(labels, boxes, scores, k)
        
        results = predictions_to_coco_format(filtered_preds, score_threshold=args.score_threshold)
        print(f"\n--- K = {k} (top-{k} predictions) ---")
        stats = evaluate_coco(args.ann_file, results)
        print(f"  AP (0.5:0.95): {stats.get('AP', 0.0):.4f}")
        print(f"  AP50:          {stats.get('AP50', 0.0):.4f}")
        print(f"  AP75:          {stats.get('AP75', 0.0):.4f}")

    # Evaluate encoder (constant, no K filtering)
    print("\n" + "=" * 60)
    print("ENCODER AP (CONSTANT)")
    print("=" * 60)
    encoder_results = predictions_to_coco_format(encoder_preds, score_threshold=args.score_threshold)
    print(f"\n--- Encoder predictions (all queries) ---")
    encoder_stats = evaluate_coco(args.ann_file, encoder_results)
    print(f"  AP (0.5:0.95): {encoder_stats.get('AP', 0.0):.4f}")
    print(f"  AP50:          {encoder_stats.get('AP50', 0.0):.4f}")
    print(f"  AP75:          {encoder_stats.get('AP75', 0.0):.4f}")
    print()
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
