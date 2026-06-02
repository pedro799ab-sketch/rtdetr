#!/bin/bash
# Run power and mAP analysis for 1000 images, one K value at a time

echo "Starting analysis for 1000 images..."

K_VALUES=(5 10 15 20 25 30 40 50 100 200 300 500 1000 1500)
OUTPUT_FILE="power_map_1000images_results.csv"

# Create CSV header
echo "K,Threads,CPU_Util_%,mAP,mAP_50,mAP_75,Avg_Time_s,Avg_Power_W,Avg_Energy_J,Total_Time_s,Total_Energy_J,Num_Predictions" > $OUTPUT_FILE

for K in "${K_VALUES[@]}"; do
    echo "================================"
    echo "Processing K=$K..."
    echo "================================"
    
    # Run for single K value
    python -c "
import os
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import psutil
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time

from src.core import YAMLConfig
from src.data.dataset import mscoco_label2category

class ProcessCPUMonitor:
    def __init__(self, cpu_tdp=15.0):
        self.cpu_tdp = cpu_tdp
        self.num_cores = psutil.cpu_count(logical=False) or 4
        self.process = psutil.Process()
        try:
            freq_info = psutil.cpu_freq()
            self.freq_max = freq_info.max if freq_info and freq_info.max > 0 else None
        except:
            self.freq_max = None
    
    def measure(self, func):
        cpu_before = self.process.cpu_times()
        cpu_start = cpu_before.user + cpu_before.system
        try:
            freq_start = psutil.cpu_freq()
        except:
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
        except:
            freq_ratio = 1.0
        wall_time = wall_end - wall_start
        cpu_time = cpu_end - cpu_start
        raw_cpu_utilization = cpu_time / wall_time if wall_time > 0 else 0.0
        cpu_utilization = min(raw_cpu_utilization / self.num_cores, 1.0)
        total_tdp = self.cpu_tdp * self.num_cores
        power = total_tdp * cpu_utilization * freq_ratio
        power = min(power, total_tdp)
        return {'result': result, 'time_s': wall_time, 'power_W': power, 'cpu_utilization': cpu_utilization}

class DeployModel(torch.nn.Module):
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model.deploy() if hasattr(model, 'deploy') else model
        self.postprocessor = postprocessor.deploy() if hasattr(postprocessor, 'deploy') else postprocessor
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes=orig_target_sizes)

def get_optimal_threads(k):
    if k <= 20: return 1
    elif k <= 50: return 2
    elif k <= 100: return 3
    elif k <= 200: return 4
    elif k <= 300: return 6
    elif k <= 500: return 8
    elif k <= 1000: return 9
    else: return 10

# Main
k = $K
device = 'cpu'
pm = ProcessCPUMonitor(cpu_tdp=15.0)
optimal_threads = get_optimal_threads(k)
os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
torch.set_num_threads(optimal_threads)

print(f'Loading model K={k}, threads={optimal_threads}...')
cfg = YAMLConfig('./configs/rtdetr/rtdetr_r50vd_6x_coco.yml', resume='rtdetr_r50vd_6x_coco_from_paddle.pth')
if 'RTDETRTransformer' in cfg.yaml_cfg:
    cfg.yaml_cfg['RTDETRTransformer']['num_queries'] = k
if 'RTDETRPostProcessor' in cfg.yaml_cfg:
    cfg.yaml_cfg['RTDETRPostProcessor']['num_top_queries'] = k
if 'RTDETRTransformerv2' in cfg.yaml_cfg:
    cfg.yaml_cfg['RTDETRTransformerv2']['num_queries'] = k

checkpoint = torch.load('rtdetr_r50vd_6x_coco_from_paddle.pth', map_location='cpu')
if 'ema' in checkpoint:
    state = checkpoint['ema']['module']
elif 'model' in checkpoint:
    state = checkpoint['model']
else:
    state = checkpoint
cfg.model.load_state_dict(state)
model = DeployModel(cfg.model, cfg.postprocessor).to(device).eval()

print('Loading COCO dataset...')
coco = COCO('./dataset/coco/instances_val2017.json')
image_ids = coco.getImgIds()[:1000]
print(f'Processing {len(image_ids)} images...')

all_predictions = []
total_time = 0
total_power = 0
cpu_utils = []

for idx, img_id in enumerate(image_ids):
    if idx % 100 == 0:
        print(f'  Progress: {idx}/{len(image_ids)} images...')
    
    info = coco.loadImgs(img_id)[0]
    img_path = os.path.join('./dataset/coco/val2017', info['file_name'])
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.width, img.height
    img = img.resize((640, 640))
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    orig_size = torch.tensor([[orig_w, orig_h]]).to(device)
    
    def run_inference():
        with torch.no_grad():
            return model(img_tensor, orig_size)
    
    stats = pm.measure(run_inference)
    labels, boxes, scores = stats['result']
    
    img_labels = labels[0].cpu().numpy()
    img_boxes = boxes[0].cpu().numpy()
    img_scores = scores[0].cpu().numpy()
    valid_mask = img_scores > 0.01
    img_labels = img_labels[valid_mask]
    img_boxes = img_boxes[valid_mask]
    img_scores = img_scores[valid_mask]
    
    for j in range(len(img_boxes)):
        x1, y1, x2, y2 = img_boxes[j]
        label_idx = int(img_labels[j])
        coco_category_id = mscoco_label2category[label_idx]
        pred = {
            'image_id': int(img_id),
            'category_id': coco_category_id,
            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            'score': float(img_scores[j])
        }
        all_predictions.append(pred)
    
    total_time += stats['time_s']
    total_power += stats['power_W']
    cpu_utils.append(stats['cpu_utilization'])

print('Calculating mAP...')
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
    except:
        overall_map = map_50 = map_75 = 0.0
else:
    overall_map = map_50 = map_75 = 0.0

avg_time = total_time / len(image_ids)
avg_power = total_power / len(image_ids)
avg_energy = avg_power * avg_time
avg_cpu_util = np.mean(cpu_utils)
total_energy = total_power * total_time / len(image_ids)

print(f'{k},{optimal_threads},{avg_cpu_util*100:.1f},{overall_map:.4f},{map_50:.4f},{map_75:.4f},{avg_time:.4f},{avg_power:.2f},{avg_energy:.2f},{total_time:.2f},{total_energy:.2f},{len(all_predictions)}')
" >> $OUTPUT_FILE 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✓ K=$K completed successfully"
    else
        echo "✗ K=$K failed"
    fi
done

echo ""
echo "================================"
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "================================"
