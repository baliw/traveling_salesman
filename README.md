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
│       ▲               ▲                                  │
│  Parallel CPU    Metal compute shader: float4 vec'd,    │
│  breeding        threadgroup shared mem, 256 threads/tg │
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

- **Unified memory**: `StorageModeShared` buffers — zero-copy between CPU/GPU
- **Threadgroup shared memory**: City coordinates (4KB for 500 cities) loaded
  into fast threadgroup memory (~1 cycle vs ~100+ for device memory)
- **float4 vectorization**: 4 tour legs per iteration using GPU's native float4 ALUs
- **FMA**: `fma(dy, dy, dx*dx)` fused into single GPU instruction
- **Dispatch**: 256 threads/threadgroup (8 SIMD groups of 32), up to 2048 cities

### When GPU helps most
- **Large populations** (>50K): more parallel work saturates 76 GPU cores
- **Large city counts** (>500): higher compute-to-transfer ratio
- **30K pop / 500 cities**: modest 2-4× speedup (CPU SIMD is already fast)
- **100K pop / 1000 cities**: significant GPU advantage

## GA Parameters

| Parameter | Value | Notes |
|---|---|---|
| Tournament size | 7 | Balances selection pressure vs diversity |
| Crossover rate | 95% | Order Crossover (OX1) |
| Mutation rate | 35% | Swap, reverse-segment, or insert |
| Double mutation | 15% | Extra mutation for diversity |
| Elite count | pop/600 | Auto-scaled, minimum 2 |
| Fitness reference | 85% of NN | Nearest-neighbor heuristic baseline |
