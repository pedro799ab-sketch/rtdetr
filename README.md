# ⚡ RT-DETR Power, Energy & Accuracy Analysis

> Research-grade experimental framework for studying the **power / energy / latency / mAP trade-offs** of the [RT-DETR](https://github.com/lyuwenyu/RT-DETR) real-time object detector on CPU.

This repository contains my custom experimental pipeline built on top of RT-DETR / RT-DETRv2 to answer one core question:

> **How does the number of decoder queries `K` and the available compute budget affect *power*, *energy*, *latency* and *detection accuracy (mAP)* on a real CPU?**

The work covers calibrated time-budget execution, per-K power sweeps, CPU-thread sweeps, energy bookkeeping, COCO mAP evaluation on subsets, and reproducible plotting.

---

## 🧠 What's interesting here

- 📊 **End-to-end measurement pipeline** — runs RT-DETR inference on COCO val while logging wall time, CPU time, utilization, estimated power and energy.
- 🎚️ **Decoder-query (`K`) sweeps** — measures how detection quality (mAP / AP50 / AP75) and cost scale with the number of object queries.
- ⏱️ **Time-budget calibration** — automatically adapts the number of threads to hit target inference times (1s, 2s, 3s, 4s, 5s) and re-measures power / mAP at each operating point.
- 🔋 **Power minimization sweeps** — searches the `(K, num_threads)` space to find Pareto-optimal *low-power-but-still-accurate* configurations.
- 🐳 **Containerized** — ships with a `Dockerfile` + `docker-compose.yml` for reproducible runs.
- 📈 **Plot generators** — every analysis has a matching script that produces publication-style figures.

> 🔬 Built and validated on **Apple Silicon (MPS)** and standard x86 CPUs.

---

## 🖼️ Example results

| Analysis | Description |
|---|---|
| `power_K.csv`, `energy_K.csv` | Power & energy as a function of `K` |
| `power_K_time{1..5}s.csv` | Power vs `K` under fixed time budgets |
| `val_results_t{1..5}s.csv` | mAP / AP50 / AP75 under fixed time budgets |
| `min_power_map_results_5images.csv` | Minimum-power configuration per `K` |
| `optimal_cpu_results_5images.csv` | Optimal CPU-thread count per `K` |
| `power_map_1000images_summary.csv` | Large-scale (1000 images) `K` sweep |

See [`ANALYSIS_SUMMARY.md`](./ANALYSIS_SUMMARY.md), [`POWER_ANALYSIS_SUMMARY.md`](./POWER_ANALYSIS_SUMMARY.md), [`POWER_MEASUREMENT_REFERENCE.md`](./POWER_MEASUREMENT_REFERENCE.md) and [`VISUALIZATION_GUIDE.md`](./VISUALIZATION_GUIDE.md) for a full write-up of the methodology and findings.

---

## 🗂️ Repository layout

```
.
├── configs/                  # RT-DETR / RT-DETRv2 model configs (YAML)
├── src/                      # RT-DETR source (model, data, solver, zoo) – upstream base
├── tools/                    # 🔧 Custom analysis & evaluation scripts
│   ├── Val.py                #   COCO eval with K sweep + power logging
│   ├── Val_per5images.py     #   Per-image / per-5-images granular eval
│   ├── val_with_ap.py        #   mAP/AP50/AP75 evaluation pipeline
│   ├── val_subset_k_analysis.py
│   ├── minimize_power.py     #   (K, threads) Pareto search for min power
│   ├── power_vs_K_time.py    #   Power vs K under fixed time budgets
│   ├── power_vs_time_analysis.py
│   ├── generate_analysis_plots.py
│   ├── simple_map.py
│   ├── run_profile.py        #   CPU/perf profiler wrapper
│   ├── test_power.py
│   ├── train.py
│   ├── export_onnx.py / export_trt.py / onnx2trt.sh
│   └── AP.py / convert_csv.py
├── min_power_map_sweep.py    # Top-level sweep entry points
├── optimal_cpu_per_k.py
├── power_map_per5images.py
├── compute_map_only.py
├── embed_photos.py
├── generate_power_map_plots.py
├── plot_k_vs_power_map.py
├── quick_plot_3images.py
├── main.py                   # Quick inference demo
├── Dockerfile, docker-compose.yml
├── requirements.txt
└── *.csv / *.md              # Experimental results + write-ups
```

---

## 🚀 Quick start

### 1. Clone and install

```bash
git clone https://github.com/pedro799ab-sketch/rtdetr.git
cd rtdetr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch / torchvision compatibility:

| torch | torchvision |
|---|---|
| 2.4 | 0.19 |
| 2.2 | 0.17 |
| 2.1 | 0.16 |
| 2.0 | 0.15 |

### 2. Download pretrained weights

The `.pth` checkpoints are **not** stored in this repo (too large). Download from the upstream RT-DETR / RT-DETRv2 releases and place them at the repo root:

- `rtdetr_r50vd_6x_coco_from_paddle.pth`
- `rtdetrv2_r50vd_6x_coco_ema.pth`

See [`README_RT-DETR_upstream.md`](./README_RT-DETR_upstream.md) for the full model zoo and direct download links.

### 3. Prepare the COCO dataset

Place COCO 2017 under `dataset/coco/` (or symlink it). The expected layout is the standard COCO `images/` + `annotations/` directory tree referenced from `configs/dataset/coco_detection.yml`.

### 4. Run a quick inference

```bash
python main.py --device mps     # Apple Silicon
python main.py --device cuda    # NVIDIA GPU
python main.py --device cpu
```

### 5. Reproduce the power / mAP analyses

```bash
# Power & energy as a function of K (number of decoder queries)
python tools/Val.py --num-images 50 --batch-size 10 \
    --k-values "5,10,20,50,100,200,300"

# Limit parallelism for a different power profile
OMP_NUM_THREADS=2 python tools/Val.py --num-images 50 --batch-size 10 \
    --k-values "5,50,100,200,300"

# Power vs K under fixed time budgets (1s..5s)
python tools/power_vs_K_time.py

# Find minimum-power configuration that preserves mAP
python tools/minimize_power.py

# Large-scale 1000-image sweep
bash run_1000images_analysis.sh

# Regenerate all plots from CSVs
python tools/generate_analysis_plots.py
python generate_power_map_plots.py
python plot_k_vs_power_map.py
```

### 6. Or just run it in Docker

```bash
docker compose up --build
```

---

## 🔬 Methodology (TL;DR)

Power is estimated from CPU utilization, frequency, and TDP:

$$
P = \text{TDP} \cdot U_{\text{CPU}} \cdot \frac{f}{f_{\max}}
\qquad
E = P \cdot t
$$

For each `K` (and optionally each thread count / time budget) we record:
`wall_time`, `cpu_time`, `cpu_util`, `power_W`, `energy_J`, `mAP`, `AP50`, `AP75`, `AR`. This lets us draw the **(cost, accuracy)** Pareto front directly from real measurements rather than from FLOPs estimates.

Full details: [`POWER_MEASUREMENT_REFERENCE.md`](./POWER_MEASUREMENT_REFERENCE.md).

---

## 📜 License & credits

- My contributions (analysis pipeline, scripts under `tools/`, sweep drivers at the repo root, all `*_ANALYSIS*.md` / `*_REFERENCE*.md` / `VISUALIZATION_GUIDE.md` and result CSVs) are released under the **MIT License** — see [`LICENSE`](./LICENSE).
- The underlying **RT-DETR / RT-DETRv2** model code in `src/`, `configs/` and `references/` is the work of Lyu Wenyu et al. and remains under its original Apache 2.0 License. Upstream repo: <https://github.com/lyuwenyu/RT-DETR>.

If you use this work, please also cite the original RT-DETR paper.

---

## 👤 Author

**Ahmed Badr** — building efficient computer-vision systems and energy-aware ML.

If you're hiring for ML / Computer Vision / Efficient AI roles, feel free to reach out. 🙂
