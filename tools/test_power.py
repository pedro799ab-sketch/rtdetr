#!/usr/bin/env python
"""Quick test of power measurement without full pipeline"""

import os
import sys
import time
import torch
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.solver import TASKS

print("[1] Device check...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"    Device: {device}")

print("[2] Loading config...")
cfg = YAMLConfig("./configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
print(f"    Config loaded")

print("[3] Building model...")
solver = TASKS[cfg.yaml_cfg["task"]](cfg)
solver._setup()
model = solver.model.to(device).eval()
print(f"    Model built: {type(model)}")

print("[4] Loading checkpoint...")
state_dict = torch.load("rtdetr_r50vd_6x_coco_from_paddle.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)
print(f"    Checkpoint loaded")

print("[5] Create dummy input...")
batch_size = 5
dummy_img = torch.randn(batch_size, 3, 640, 640).to(device)
print(f"    Input shape: {dummy_img.shape}")

print("[6] Testing encoder...")
start = time.perf_counter()
with torch.no_grad():
    encoder_out = model.backbone(dummy_img)
elapsed = time.perf_counter() - start
print(f"    Encoder time: {elapsed:.4f}s")

print("[7] Testing decoder...")
K_values = [5, 10, 15, 20, 25, 30]
for K in K_values:
    # Create dummy encoder output with shape matching model expectations
    try:
        start = time.perf_counter()
        with torch.no_grad():
            output = model.decoder(encoder_out, {"K": K})
        elapsed = time.perf_counter() - start
        print(f"    K={K:>3}: {elapsed:.4f}s")
    except Exception as e:
        print(f"    K={K:>3}: ERROR - {e}")
        break

print("\n[DONE] Test completed successfully!")
