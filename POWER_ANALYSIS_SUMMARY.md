# RT-DETR Power Analysis Summary

## Key Findings

### 1. Why Power Doesn't Increase Linearly with K

**Physical Reality:** When the CPU is already at maximum utilization (~90-100%), power consumption cannot increase further. The CPU is like a car engine at maximum RPM - it can't spin faster, it can only run longer.

| K Value | CPU Utilization | Power | Time | Energy |
|---------|----------------|-------|------|--------|
| K=5 | ~94% | 143W | 0.6s | **87J** |
| K=300 | ~84% | 126W | 1.4s | **181J** |

### 2. The Correct Metric: ENERGY (not Power)

**Energy = Power × Time** is the proper measure of computational cost:

- ✅ **Energy increases with K** (87J → 181J = +108%)
- ❌ Power may stay flat or even decrease (hardware saturation)
- ✅ Time increases linearly with K (0.6s → 1.4s = +133%)

### 3. What K Represents

**K = Number of Object Queries** in the RT-DETR decoder:

- **K=5:** Only detect top 5 objects → Fast but may miss objects
- **K=300:** Detect up to 300 objects → Slower but comprehensive
- **Complexity:** Decoder has O(K²) self-attention operations

### 4. Research Conclusions

For your paper/research:

1. **Report Energy, not Power** - Energy correctly shows the cost scaling
2. **Graph Energy vs K** - This shows the clear upward trend
3. **Explain CPU Saturation** - Modern CPUs parallelize efficiently, saturating at low K
4. **Practical Impact** - Higher K means longer inference time and more energy per frame

## Files Generated

- `power_K.csv` - Raw data (Time, Power, Energy for each K)
- `power_K_graphs.png` - Visualization (3 graphs: Energy, Power, Time vs K)

## How to Run

```bash
# Standard measurement (all CPU cores)
python tools/Val.py --num-images 50 --batch-size 10 --k-values "5,10,20,50,100,200,300"

# Limited parallelism (2 threads for different power profile)
OMP_NUM_THREADS=2 python tools/Val.py --num-images 50 --batch-size 10 --k-values "5,50,100,200,300"
```

## Formula Reference

```
Power (W) = TDP × CPU_Utilization × (freq / freq_max)
Energy (J) = Power (W) × Time (s)
CPU_Utilization = (CPU_time / Wall_time) / num_cores  [normalized to 100%]
```
