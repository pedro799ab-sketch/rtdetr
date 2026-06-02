"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time
import os 
import sys 
import logging
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn 
from PIL import Image, ImageDraw, ImageFont

import torchvision.transforms as T

from src.core import YAMLConfig


class Model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        
    def forward(self, images, orig_target_sizes,targets=None):
        outputs = self.model(images,targets)
        outputs_enc = {}
        # Some model variants may not produce encoder-topk outputs ('enc_pred_logits').
        # Fall back to decoder outputs if encoder outputs are missing.
        logger = logging.getLogger(__name__)
        enc_logits = outputs.get('enc_pred_logits', None) if isinstance(outputs, dict) else None
        enc_boxes = outputs.get('enc_pred_boxes', None) if isinstance(outputs, dict) else None
        if enc_logits is None or enc_boxes is None:
            logger.warning("Encoder 'enc_pred_*' outputs missing; falling back to decoder predictions for encoder visualization.")
            enc_logits = outputs.get('pred_logits') if isinstance(outputs, dict) else None
            enc_boxes = outputs.get('pred_boxes') if isinstance(outputs, dict) else None

        outputs_enc['pred_logits'] = enc_logits
        outputs_enc['pred_boxes']  = enc_boxes
        outputs = self.postprocessor(outputs, orig_target_sizes)
        outputs_enc = self.postprocessor(outputs_enc, orig_target_sizes)
        return outputs, outputs_enc
    
    #     outputs_mod, outputs_enc = self.model(images,targets)
    #     outputs     = self.postprocessor(outputs_mod, orig_target_sizes)
    #     outputs_enc = self.postprocessor(outputs_enc, orig_target_sizes)
    #     return outputs, outputs_enc


# def detect(im, model,orig_target_sizes,jpg_size):
#     # img = transform(im).unsqueeze(0)
#     # assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

#    #  print("\n3. Running inference...")
#    #  start_time = time.time()
#    #  outputs = model(im, orig_target_sizes)
#    #  torch.cuda.synchronize()
#    #  end_time = time.time()
    
#    #  print(f"\n4. Inference complete in { (end_time - start_time) * 1000:.2f} ms.")

#     # # keep only predictions with 0.7+ confidence
#     # probas = outputs[2].softmax(-1)[0, :]
#     # # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
#     # keep = probas.max(-1).values > 0.7

#     # # convert boxes from [0; 1] to image scales
#     # bboxes_scaled = rescale_bboxes(outputs[1][0, keep], jpg_size)
#     # # bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
#     # return probas[keep], bboxes_scaled
#     return outputs

# --- Visualization Utility Function ---
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def visualize_detections(image_pil, boxes, scores, labels, class_names=COCO_CLASSES, threshold=0.5,color = 'red'):
    """
    Draws bounding boxes on a PIL image. This function is a general-purpose utility.

    Args:
        image_pil (PIL.Image.Image): The image to draw on.
        boxes (torch.Tensor): A tensor of bounding boxes (shape: [N, 4]).
        scores (torch.Tensor): A tensor of confidence scores (shape: [N]).
        labels (torch.Tensor): A tensor of class labels (shape: [N]).
        class_names (list): A list of strings corresponding to class labels.
        threshold (float): The confidence threshold for displaying detections.

    Returns:
        PIL.Image.Image: The image with detections drawn on it.
    """
    img_draw = image_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Ensure tensors are on CPU and converted to NumPy for processing
    # boxes = boxes.cpu().numpy()
    # scores = scores.cpu().numpy()
    # labels = labels.cpu().numpy()

    boxes  =  boxes.detach().numpy()
    scores = scores.detach().numpy()
    if labels is not None:
        labels = labels.detach().numpy()
    
    count = 0
    for i in range(len(scores)):
        if labels is not None:
            score = scores[i]
            if score < threshold:
                continue
            
            count += 1
            box = boxes[i]
            label_idx = int(labels[i])
            
            xmin, ymin, xmax, ymax = box
            class_name = class_names[label_idx] if label_idx < len(class_names) else f'CLS-{label_idx}'
            # color = 'red' # Keep it simple or use a color map
            
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=3)
            
            text = f"{class_name}: {score:.2f}"
            
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            text_bbox = draw.textbbox((xmin, ymin), text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((xmin, ymin), text, fill="white", font=font)

        else:
            score = scores[i]
            if score < threshold:
                continue
            
            count += 1
            box = boxes[i]
            
            xmin, ymin, xmax, ymax = box
            # color = 'red' # Keep it simple or use a color map
            
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=3)
            
            text = f"{score:.2f}"
            
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            text_bbox = draw.textbbox((xmin, ymin), text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((xmin, ymin), text, fill="white", font=font)

        
    print(f"   - Found {count} objects above threshold {threshold}.")
    return img_draw



def main(args, ):
    """main
    """

    if args.config is None:
        args.config = "./configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"

    # If user didn't provide a resume checkpoint, try to find a sensible default
    if args.resume is None:
        # Prefer EMA checkpoint for rtdetrv2 if present in repo
        import glob
        candidates = glob.glob('rtdetr*v2*_ema.pth') + glob.glob('*.pth')
        preferred = None
        for c in candidates:
            if 'ema' in os.path.basename(c):
                preferred = c
                break
        if preferred is None and candidates:
            preferred = candidates[0]
        if preferred is not None:
            print(f"auto-resume: using checkpoint {preferred}")
            args.resume = preferred

    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')


    rtdetr = Model(cfg)
    logger = logging.getLogger(__name__)
    device = torch.device(args.device)
    rtdetr.to(device)

    # standard PyTorch mean-std input image normalization
    # transform = T.Compose([
    # T.Resize(800),
    # T.ToTensor(),
    # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])


    img_path = "./dataset/000000469174.jpg"
    im = Image.open(img_path).convert("RGB")
    w, h = im.size

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    im_tensor = transform(im).unsqueeze(0).to(device)
    # original size of the image
    orig_target_sizes = torch.tensor([[w, h]], dtype=torch.int64, device=device)

    logger.info(f"   - Original image size: {w}x{h}")
    logger.info(f"   - Input tensor shape: {im_tensor.shape}")


    rtdetr.eval()

    # outputs = detect(im_tensor, rtdetr, orig_target_sizes, im.size)


    # LUNCH THE MODEL / NETWORK

    logger.info("\n3. Running inference...")
    start_time = time.time()
    try:
        outputs, outputs_enc = rtdetr(im_tensor, orig_target_sizes, targets=None)
    except Exception:
        logger.exception('Error during inference')
        if args.debug:
            try:
                import pdb
                pdb.post_mortem()
            except Exception:
                logger.debug('pdb not available or cannot drop into post-mortem')
        raise

    # device-aware synchronize
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == 'mps' and getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

    end_time = time.time()
    logger.info(f"\n4. Inference complete in { (end_time - start_time) * 1000:.2f} ms.")



    print("\n5. Post-processing and saving output image...")
    # output_labels = outputs['labels'][0]
    # output_boxes  = outputs['boxes'][0]
    # output_scores = outputs['scores'][0]

    # outputs is (labels, boxes, scores) in deploy mode: [B, K] etc.
    output_labels = outputs[0][0]
    output_boxes  = outputs[1][0]
    output_scores = outputs[2][0]

    # Count detections above threshold and warn if none found
    vis_threshold = args.vis_threshold
    det_count = int((output_scores >= vis_threshold).sum().item())
    logger = logging.getLogger(__name__)
    if det_count == 0:
        logger.warning(f'No detections found above threshold {vis_threshold}. Try passing a trained checkpoint via --resume or lower --vis-threshold.')
        # For debug, log top-10 scores
        topk = torch.topk(output_scores.flatten(), min(10, output_scores.numel())).values
        logger.debug(f'Top scores (top-10): {topk.tolist()}')

    # Use the new, separate visualization function
    result_image = visualize_detections(
        im, 
        output_boxes, 
        output_scores, 
        output_labels, 
        threshold=0.25
    )

    outputs_enc_labels = outputs_enc[0][0]
    outputs_enc_boxes  = outputs_enc[1][0]
    outputs_enc_scores = outputs_enc[2][0]
    result_image_enc = visualize_detections(
        im, 
        outputs_enc_boxes, 
        outputs_enc_scores, 
        outputs_enc_labels, 
        threshold=0.25,
        color='blue'
    )

    # result_image_2 = visualize_detections(
    #     im, 
    #     output_boxes, 
    #     output_scores, 
    #     output_labels, 
    #     threshold=0.4
    # )

    # Save outputs using a safe config lookup (some configs may not define RTDETRTransformer)
    num_queries = 'N'
    try:
        num_queries = cfg.yaml_cfg.get('RTDETRTransformer', {}).get('num_queries', num_queries)
    except Exception:
        # cfg.yaml_cfg might not be a dict or may be missing; fall back to 'N'
        pass

    output_file = f"{img_path[0:-4]}_d{num_queries}v1.jpg"
    output_file_enc = f"{img_path[0:-4]}_d{num_queries}v1-encoder_feats.jpg"
    result_image.save(output_file)
    result_image_enc.save(output_file_enc)

    logger = logging.getLogger(__name__)
    logger.info(f"   - Output image with detections saved to: {os.path.abspath(output_file)}")
    logger.info(f"   - Encoder output image saved to: {os.path.abspath(output_file_enc)}")
    logger.info("\n--- finished successfully ---")







if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--output_file', '-o', type=str, default='model.onnx')
    parser.add_argument('--input_size', '-s', type=int, default=640)
    parser.add_argument('--device', '-d', type=str, default=None,
                        help='Device to run on (e.g. "cuda:0", "mps", or "cpu").')
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode with DEBUG logs and post-mortem pdb on exception')
    parser.add_argument('--vis-threshold', type=float, default=0.25,
                        help='Visualization confidence threshold (default: 0.25)')
    

    args = parser.parse_args()

    # Auto-select device: prefer CUDA, then MPS (Apple Silicon), then CPU
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda:0'
        elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    if args.debug:
        logger.debug('Debug mode enabled')

    try:
        main(args)
    except Exception:
        logger.exception('Unhandled exception in main')
        if args.debug:
            # Try to drop to post-mortem pdb if running interactively
            try:
                import pdb
                pdb.post_mortem()
            except Exception:
                logger.debug('Could not run pdb.post_mortem()')
        raise
