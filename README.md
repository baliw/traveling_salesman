# ga_tsp — SIMD & GPU Accelerated Genetic Algorithm for TSP

A high-performance genetic algorithm solver for the Traveling Salesman Problem
with multi-threading support, platform-specific SIMD optimizations, and optional Metal GPU compute.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  CPU Path (default)                                     │
│  ┌───────────┐   ┌───────────┐   ┌──────────────────┐  │
│  │  Breed    │──▶│ Evaluate  │──▶│  Sort / Select   │  │
│  │(multi-CPU)│   │ (SIMD)    │   │  (scalar)        │  │
│  └───────────┘   └───────────┘   └──────────────────┘  │
│       ▲               ▲                                  │
│  Multi-threaded  x86: AVX2+FMA (4 legs/cycle)           │
│  breeding pool   ARM64: NEON+FMA (4 legs/cycle, dual)   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  GPU Path (--gpu, macOS Apple Silicon only)             │
│  ┌───────────┐   ┌───────────┐   ┌──────────────────┐  │
│  │  Breed    │──▶│ Evaluate  │──▶│  Sort / Select   │  │
│  │(multi-CPU)│   │ (Metal)   │   │  (CPU)           │  │
│  └───────────┘   └───────────┘   └──────────────────┘  │
│       ▲               ▲               ▲                  │
│  Writes directly  Zero-copy GPU    Reads flat f32       │
│  into Metal buf   dispatch only    distance array       │
│  (dual ping-pong) (no upload!)     (no download!)       │
└─────────────────────────────────────────────────────────┘
```

## Usage

```bash
cargo build --release
./target/release/ga_tsp [OPTIONS]

Options:
  -c, --cities <N>       Number of cities       [default: 500]
  -p, --pop <N>          Population size         [default: 30000]
  -g, --generations <N>  Number of generations   [default: 1000]
  -t, --threads <N>      CPU threads for breeding [default: 1]
      --gpu              Use Metal GPU compute (macOS Apple Silicon only)
  -h, --help             Show this help
```

### Examples

```bash
# Default: 500 cities, 30K population, single-threaded breeding + SIMD evaluation
./target/release/ga_tsp

# Multi-threaded breeding on 4-core machine
./target/release/ga_tsp -t 4

# Quick test with 4 CPU threads
./target/release/ga_tsp -c 50 -p 1000 -g 100 -t 4

# Large run: GPU evaluation + 8 CPU threads for breeding
./target/release/ga_tsp -c 1000 -p 100000 -g 2000 --gpu -t 8

# Large CPU-only run with multi-threaded breeding
./target/release/ga_tsp -c 2000 -p 50000 -g 500 -t 4
```

## Platform Support

| Platform | CPU SIMD | GPU |
|---|---|---|
| macOS Apple Silicon | NEON+FMA (2×f64) | Metal (--gpu) |
| macOS Intel | AVX2+FMA (4×f64) | Not supported |
| Linux x86_64 | AVX2+FMA (4×f64) | --gpu gracefully falls back |
| Linux ARM64 | NEON+FMA (2×f64) | --gpu gracefully falls back |

## Multi-Threading Architecture

The `-t` flag enables parallel breeding across CPU threads:

- **Persistent worker pool**: Threads created once and reused for all generations
- **Barrier synchronization**: Workers coordinate via start/end barriers each generation
- **Zero-copy population sharing**: Main thread shares population reference to workers
- **Per-worker RNG**: Each thread uses independent PRNG seeded per-generation
- **Arena warmup**: Two warmup cycles prime per-thread malloc arenas (avoids 800ms first-gen penalty)
- **Scalability**: Breeding parallelizes linearly; SIMD evaluation remains single-threaded (unless --gpu)

**Performance**: On 8-core CPU with 30K population, `-t 8` provides ~6-7× speedup in breeding phase.

## SIMD Optimizations

### x86_64 (AVX2+FMA)
- **Tour distance**: `_mm256_i64gather_pd` loads 4 non-contiguous city coordinates,
  FMA computes 4 squared distances, vectorized sqrt — 4 tour legs per cycle
- **Population stats**: AVX2 min/max/sum reduction over distance array

### AArch64 (NEON+FMA) — optimized for M2 Ultra P-cores
- **Tour distance**: Manual 2-lane coordinate gather via `vcombine_f64`/`vld1_f64`,
  `vfmaq_f64` for fused d², `vsqrtq_f64` for vectorized sqrt
- **Dual accumulator**: Two independent sum registers hide the 12-cycle sqrt
  latency on M2's out-of-order pipeline (vs 4-cycle FMA)
- **4 legs per iteration**: Two NEON operations process 4 tour legs total

### Scalar fallback
- Automatic on CPUs without AVX2 (x86) — NEON is always present on AArch64

## Metal GPU Compute — M2 Ultra optimized

The `--gpu` flag offloads tour distance evaluation to the Metal GPU:

- **Zero-copy flat-buffer pipeline**: Population tours live in Metal shared-memory
  buffers (`StorageModeShared`). CPU breeds directly into buffers, GPU evaluates
  in-place — no upload/download copies, no `usize↔u32` conversions
- **Dual ping-pong tour buffers**: CPU reads parents from buffer A while writing
  children to buffer B, then GPU evaluates B and swaps
- **Threadgroup shared memory**: City coordinates (4KB for 500 cities) loaded
  into fast threadgroup memory (~1 cycle vs ~100+ for device memory)
- **float4 vectorization**: 4 tour legs per iteration using GPU's native float4 ALUs
- **FMA**: `fma(dy, dy, dx*dx)` fused into single GPU instruction
- **Dispatch**: 256 threads/threadgroup (8 SIMD groups of 32), up to 2048 cities

### When GPU helps most
- **Best with `-t N`**: GPU evaluates while CPU breeds in parallel — combine with
  multi-threaded breeding for peak throughput
- **Large populations** (>50K): more parallel work saturates GPU cores
- **All sizes benefit**: Zero-copy eliminates the per-generation data copy overhead
  (60MB at 30K pop, 200MB at 100K pop)

## Performance

Benchmarked on Apple M2 Ultra, 500 cities, default parameters.

### 30K population, 200 generations

```bash
# CPU single-threaded (NEON+FMA)
./target/release/ga_tsp -g 200
# ~15s

# CPU 16-threaded
./target/release/ga_tsp -t 16 -g 200
# 3.02s

# GPU single-threaded breeding
./target/release/ga_tsp --gpu -g 200
# 9.56s

# GPU + 16-threaded breeding
./target/release/ga_tsp --gpu -t 16 -g 200
# 1.18s  (2.6× faster than CPU 16T)
```

| Mode | Time | vs CPU 16T |
|---|---|---|
| CPU 1T | ~15s | 0.2× |
| CPU 16T | 3.02s | 1.0× |
| GPU 1T | 9.56s | 0.3× |
| **GPU + 16T** | **1.18s** | **2.6×** |

### 100K population, 100 generations

```bash
# CPU 16-threaded
./target/release/ga_tsp -t 16 -p 100000 -g 100
# 4.52s

# GPU + 16-threaded breeding
./target/release/ga_tsp --gpu -t 16 -p 100000 -g 100
# 2.00s  (2.3× faster than CPU 16T)
```

| Mode | Time | vs CPU 16T |
|---|---|---|
| CPU 1T | 30.67s | 0.15× |
| CPU 16T | 4.52s | 1.0× |
| GPU 1T | 20.56s | 0.22× |
| **GPU + 16T** | **2.00s** | **2.3×** |

The GPU's zero-copy flat-buffer pipeline eliminates the ~60MB per-generation data copy
(200MB at 100K pop), making `--gpu -t 16` the fastest mode at all population sizes.

## GA Parameters

| Parameter | Value | Notes |
|---|---|---|
| Tournament size | 7 | Balances selection pressure vs diversity |
| Crossover rate | 95% | Order Crossover (OX1) |
| Mutation rate | 35% | Swap, reverse-segment, or insert |
| Double mutation | 15% | Extra mutation for diversity |
| Elite count | pop/600 | Auto-scaled, minimum 2 |
| Fitness reference | 85% of NN | Nearest-neighbor heuristic baseline |
