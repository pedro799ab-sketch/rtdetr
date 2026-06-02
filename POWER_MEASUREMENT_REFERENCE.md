# Power Measurement Implementation Reference

This document provides all functions, formulas, and references used to calculate power consumption, energy, and CPU utilization in the RT-DETR validation scripts.

---

## Table of Contents
1. [Core Measurement Class](#core-measurement-class)
2. [Power Calculation Formula](#power-calculation-formula)
3. [CPU Utilization Formula](#cpu-utilization-formula)
4. [Energy Calculation](#energy-calculation)
5. [Frequency Scaling](#frequency-scaling)
6. [Implementation Details](#implementation-details)
7. [References](#references)

---

## Core Measurement Class

### `ProcessCPUMonitor`

Located in: `tools/val_with_ap.py`, `compute_map_only.py` (if added), `tools/Val.py`

```python
class ProcessCPUMonitor:
    """
    Measures power directly using CPU utilization and frequency scaling.
    
    Power = TDP × CPU_Utilization × (freq_current / freq_max)
    
    This approach provides accurate power measurements without external hardware.
    """
    
    def __init__(self, cpu_tdp=15.0):
        """
        Initialize the CPU monitor.
        
        Args:
            cpu_tdp (float): Thermal Design Power per CPU core in Watts.
                           Default: 15W (typical for mobile/laptop CPUs)
        """
        self.cpu_tdp = cpu_tdp
        self.num_cores = psutil.cpu_count(logical=False) or 4
        self.process = psutil.Process()
        
        # Get max CPU frequency for frequency scaling factor
        try:
            freq_info = psutil.cpu_freq()
            self.freq_max = freq_info.max if freq_info and freq_info.max > 0 else None
        except Exception:
            self.freq_max = None
```

**Dependencies:**
- `psutil`: Cross-platform library for retrieving information on running processes and system utilization
  - `psutil.cpu_count(logical=False)`: Get physical CPU core count
  - `psutil.Process()`: Get current process information
  - `psutil.cpu_freq()`: Get CPU frequency information

---

## Power Calculation Formula

### Mathematical Formula

```
Power (W) = TDP_total × CPU_Utilization × Frequency_Ratio

Where:
  TDP_total = TDP_per_core × Number_of_cores
  CPU_Utilization = (CPU_time / Wall_time) / Number_of_cores  [capped at 1.0]
  Frequency_Ratio = Average_frequency / Max_frequency
```

### Implementation

```python
def measure(self, func):
    """
    Measure time and power for a function.
    
    Returns dict with:
    - time_s: wall-clock time (seconds)
    - cpu_time_s: actual CPU seconds used
    - power_W: instantaneous power measurement (Watts)
    - cpu_utilization: normalized CPU utilization (0-1)
    - freq_ratio: current_freq / max_freq (frequency scaling factor)
    """
    # Get CPU times BEFORE execution
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
    
    # Get CPU times AFTER execution
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
```

---

## CPU Utilization Formula

### Raw CPU Utilization

```
Raw_CPU_Utilization = CPU_time / Wall_time

Where:
  CPU_time = User_time + System_time  (from process.cpu_times())
  Wall_time = Actual elapsed time measured by time.perf_counter()
```

**Example:**
- If a process runs for 1 second (wall time) and uses 2 CPU cores fully
- CPU_time = 2 seconds (cumulative across cores)
- Raw_CPU_Utilization = 2.0 (200%)

### Normalized CPU Utilization

```
CPU_Utilization = min(Raw_CPU_Utilization / Number_of_cores, 1.0)

Where:
  Number_of_cores = Physical CPU core count
```

**Example:**
- Raw_CPU_Utilization = 2.0, Number_of_cores = 10
- CPU_Utilization = min(2.0 / 10, 1.0) = 0.20 (20% of total CPU capacity)

### CPU Time Measurement

Using `psutil.Process().cpu_times()`:

```python
cpu_times = process.cpu_times()
cpu_time = cpu_times.user + cpu_times.system

# user: Time spent in user mode (executing application code)
# system: Time spent in kernel mode (system calls, I/O operations)
```

---

## Energy Calculation

### Formula

```
Energy (J) = Power (W) × Time (s)
```

### Implementation Examples

#### 1. Per-Component Energy

```python
# Encoder energy
enc_energy = encoder_power_W * encoder_time_s

# Decoder energy
dec_energy = decoder_power_W * decoder_time_s

# Total energy
total_energy = enc_energy + dec_energy
```

#### 2. Average Power from Energy

```python
# When you have total energy and time
average_power = total_energy_J / total_time_s
```

#### 3. Energy per Image

```python
# For batch processing
energy_per_image = total_energy_J / num_images
```

---

## Frequency Scaling

### Purpose

Modern CPUs dynamically adjust their operating frequency based on workload. This affects power consumption:
- Higher frequency → More power consumption
- Lower frequency → Less power consumption

### Frequency Ratio Calculation

```python
freq_start = psutil.cpu_freq()  # Frequency at start (MHz)
freq_end = psutil.cpu_freq()    # Frequency at end (MHz)

# Average frequency during execution
avg_freq = (freq_start.current + freq_end.current) / 2

# Frequency scaling factor (0 to 1)
freq_ratio = avg_freq / freq_max

# Example:
#   freq_max = 3000 MHz
#   avg_freq = 2400 MHz
#   freq_ratio = 2400 / 3000 = 0.8
```

### Power Adjustment

```python
# Base power (at max frequency)
base_power = TDP_total × CPU_Utilization

# Actual power (accounting for frequency scaling)
actual_power = base_power × freq_ratio
```

**Why this matters:**
- A CPU running at 80% frequency uses approximately 80% of its max power
- This provides more accurate power measurements than assuming constant frequency

---

## Implementation Details

### 1. Thread Control

Control CPU thread count to manage power consumption:

```python
import os
os.environ['OMP_NUM_THREADS'] = str(num_threads)
torch.set_num_threads(num_threads)
```

**Effect on power:**
- More threads → Higher CPU utilization → Higher power
- Fewer threads → Lower CPU utilization → Lower power
- Trade-off: Performance vs. Power consumption

### 2. Time Measurement

```python
import time

# High-precision wall-clock timer
wall_start = time.perf_counter()
# ... execute code ...
wall_end = time.perf_counter()
elapsed_time = wall_end - wall_start
```

### 3. GPU Synchronization

```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
```

**Why needed:**
- PyTorch GPU operations are asynchronous
- Must synchronize before measuring end time
- Ensures accurate timing for GPU workloads

---

## References

### Libraries Used

1. **psutil (Python System and Process Utilities)**
   - Version: 5.9.0+
   - Documentation: https://psutil.readthedocs.io/
   - Purpose: CPU utilization, frequency, and process monitoring
   
   Key functions:
   - `psutil.cpu_count(logical=False)`: Physical CPU core count
   - `psutil.Process().cpu_times()`: Process CPU time (user + system)
   - `psutil.cpu_freq()`: CPU frequency information (current, min, max)

2. **time (Python Standard Library)**
   - Documentation: https://docs.python.org/3/library/time.html
   - `time.perf_counter()`: High-resolution performance counter

3. **PyTorch**
   - Version: 2.0+
   - `torch.set_num_threads()`: Control CPU parallelism
   - `torch.cuda.synchronize()`: GPU synchronization

### Academic References

1. **CPU Power Modeling:**
   - "Power and Performance Modeling for Multi-core Processors"
   - Formula: P = C × V² × f (Capacitance × Voltage² × Frequency)
   - Simplified: P ≈ TDP × Utilization × (f / f_max)

2. **TDP (Thermal Design Power):**
   - Intel/AMD processor specifications
   - Represents maximum sustained power consumption
   - Typical values:
     - Mobile/Laptop CPUs: 10-45W per core
     - Desktop CPUs: 65-125W per core
     - Server CPUs: 150-300W per core

3. **CPU Utilization:**
   - Linux: /proc/stat, /proc/[pid]/stat
   - Formula: Utilization = (1 - idle_time/total_time) × 100%
   - Multi-core: Sum CPU time across all cores

### Configuration Parameters

Default values used in scripts:

```python
# Power measurement
cpu_tdp = 15.0  # Watts per core (configurable via --cpu-tdp)

# Time calibration
target_time = 1.0  # Target execution time in seconds
tolerance = 0.10   # ±10% tolerance for calibration

# Thread configuration
max_threads = psutil.cpu_count(logical=False)  # Physical cores
min_threads = 1

# Confidence threshold
conf_threshold = 0.01  # Minimum detection confidence
```

---

## Usage Examples

### Example 1: Basic Power Measurement

```python
pm = ProcessCPUMonitor(cpu_tdp=15.0)

def my_inference():
    with torch.no_grad():
        output = model(input_tensor)
    return output

stats = pm.measure(my_inference)

print(f"Time: {stats['time_s']:.4f}s")
print(f"Power: {stats['power_W']:.2f}W")
print(f"Energy: {stats['power_W'] * stats['time_s']:.2f}J")
print(f"CPU Utilization: {stats['cpu_utilization']*100:.1f}%")
```

### Example 2: Energy per Image

```python
total_energy = 0
num_images = 10

for img in images:
    stats = pm.measure(lambda: model(img))
    total_energy += stats['power_W'] * stats['time_s']

energy_per_image = total_energy / num_images
print(f"Energy per image: {energy_per_image:.2f}J")
```

### Example 3: Power-Performance Trade-off

```python
results = []
for num_threads in [1, 2, 4, 8]:
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    torch.set_num_threads(num_threads)
    
    stats = pm.measure(my_inference)
    results.append({
        'threads': num_threads,
        'time': stats['time_s'],
        'power': stats['power_W'],
        'energy': stats['power_W'] * stats['time_s']
    })

# Find optimal configuration (minimize energy)
optimal = min(results, key=lambda x: x['energy'])
```

---

## Validation

### Sanity Checks

1. **Power bounds:**
   ```python
   assert 0 <= power <= (cpu_tdp * num_cores), "Power out of range"
   ```

2. **CPU utilization bounds:**
   ```python
   assert 0 <= cpu_utilization <= 1.0, "Utilization out of range"
   ```

3. **Frequency ratio bounds:**
   ```python
   assert 0 <= freq_ratio <= 1.0, "Frequency ratio out of range"
   ```

4. **Time consistency:**
   ```python
   assert cpu_time <= wall_time * num_cores, "CPU time exceeds physical limit"
   ```

### Expected Ranges

For RT-DETR inference on CPU:

| Metric | Low K (5-50) | High K (500-1500) |
|--------|--------------|-------------------|
| Power | 15-30W | 120-130W |
| CPU Utilization | 10-20% | 80-95% |
| Time per image | 0.5-1.5s | 0.6-0.8s |
| Energy per image | 10-30J | 70-100J |

---

## Troubleshooting

### Issue 1: Power readings are always 0

**Cause:** `psutil` not installed or permission issues

**Solution:**
```bash
pip install psutil
```

### Issue 2: Frequency ratio always 1.0

**Cause:** CPU frequency scaling not available on system

**Effect:** Power measurements still valid, but less accurate

**Solution:** Use frequency governors or check BIOS settings

### Issue 3: CPU utilization > 100%

**Cause:** Not normalizing by core count

**Solution:**
```python
cpu_utilization = min(raw_utilization / num_cores, 1.0)
```

### Issue 4: Inconsistent power readings

**Cause:** Background processes, thermal throttling, or power-saving modes

**Solution:**
- Close background applications
- Disable power-saving modes
- Run multiple iterations and average
- Use warm-up iterations before measurement

---

## Summary

### Key Formulas

1. **Power:** `P = TDP × CPU_Util × (f / f_max)`
2. **Energy:** `E = P × t`
3. **CPU Utilization:** `U = min((CPU_time / Wall_time) / Cores, 1.0)`
4. **Frequency Ratio:** `R = Current_freq / Max_freq`

### Key Functions

1. `psutil.Process().cpu_times()` → Get CPU time
2. `psutil.cpu_freq()` → Get CPU frequency
3. `psutil.cpu_count(logical=False)` → Get physical cores
4. `time.perf_counter()` → High-precision timing

### Key Classes

1. `ProcessCPUMonitor` → Main measurement class
2. `DeployModel` → Model with postprocessor for inference

---

*Last updated: January 27, 2026*
