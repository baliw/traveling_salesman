#![allow(clippy::needless_range_loop, unused_imports)]

use rand::prelude::*;
use std::io::{self, Write};
use std::sync::{Arc, Barrier, Mutex, Condvar};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ═══════════════════════════════════════════════════════════════════════════════
//  Signal handling — restore cursor on Ctrl-C
// ═══════════════════════════════════════════════════════════════════════════════
extern "C" fn handle_sigint(_sig: i32) {
    // POSIX write(2) is async-signal-safe — write to BOTH stdout and stderr
    // to guarantee the terminal sees it regardless of redirection.
    #[cfg(unix)]
    unsafe {
        extern "C" { fn write(fd: i32, buf: *const u8, count: usize) -> isize; }
        write(1, SHOW_CURSOR.as_ptr(), SHOW_CURSOR.len());
        write(1, b"\n" as *const u8, 1);
    }
    #[cfg(not(unix))]
    {
        let _ = io::stdout().write_all(SHOW_CURSOR.as_bytes());
    }
    std::process::exit(130); // 128 + SIGINT(2)
}

#[cfg(unix)]
extern "C" {
    fn signal(signum: i32, handler: extern "C" fn(i32)) -> usize;
}

fn install_signal_handler() {
    #[cfg(unix)]
    unsafe {
        signal(2 /* SIGINT */, handle_sigint);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Defaults (overridable via CLI)
// ═══════════════════════════════════════════════════════════════════════════════
const DEFAULT_NUM_CITIES: usize = 500;
const DEFAULT_POP_SIZE: usize = 30_000;
const DEFAULT_NUM_GENERATIONS: usize = 1000;

const TOURNAMENT_SIZE: usize = 7;
const CROSSOVER_RATE: f64 = 0.95;
const MUTATION_RATE: f64 = 0.35;
const DOUBLE_MUTATION_RATE: f64 = 0.15;
const MAP_SIZE: f64 = 1000.0;

// ═══════════════════════════════════════════════════════════════════════════════
//  CLI Configuration
// ═══════════════════════════════════════════════════════════════════════════════
struct Config {
    num_cities: usize,
    pop_size: usize,
    num_generations: usize,
    elite_count: usize,
    use_gpu: bool,
    num_threads: usize,
}

impl Config {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut num_cities = DEFAULT_NUM_CITIES;
        let mut pop_size = DEFAULT_POP_SIZE;
        let mut num_generations = DEFAULT_NUM_GENERATIONS;
        let mut use_gpu = false;
        let mut num_threads: usize = 1;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--cities" | "-c" => {
                    i += 1;
                    num_cities = args.get(i).and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| { eprintln!("Error: --cities requires a number"); std::process::exit(1); });
                }
                "--pop" | "-p" => {
                    i += 1;
                    pop_size = args.get(i).and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| { eprintln!("Error: --pop requires a number"); std::process::exit(1); });
                }
                "--generations" | "-g" => {
                    i += 1;
                    num_generations = args.get(i).and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| { eprintln!("Error: --generations requires a number"); std::process::exit(1); });
                }
                "--threads" | "-t" => {
                    i += 1;
                    num_threads = args.get(i).and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| { eprintln!("Error: --threads requires a number"); std::process::exit(1); });
                }
                "--gpu" => {
                    use_gpu = true;
                }
                "--help" | "-h" => {
                    eprintln!("Usage: ga_tsp [OPTIONS]");
                    eprintln!();
                    eprintln!("Options:");
                    eprintln!("  -c, --cities <N>       Number of cities       [default: {}]", DEFAULT_NUM_CITIES);
                    eprintln!("  -p, --pop <N>          Population size         [default: {}]", DEFAULT_POP_SIZE);
                    eprintln!("  -g, --generations <N>  Number of generations   [default: {}]", DEFAULT_NUM_GENERATIONS);
                    eprintln!("  -t, --threads <N>      CPU threads for breeding [default: 1]");
                    eprintln!("      --gpu              Use Metal GPU compute (macOS Apple Silicon only)");
                    eprintln!("  -h, --help             Show this help");
                    std::process::exit(0);
                }
                other => { eprintln!("Unknown option: {other}. Use --help."); std::process::exit(1); }
            }
            i += 1;
        }
        if num_cities < 4 { eprintln!("Error: need at least 4 cities"); std::process::exit(1); }
        if pop_size < 10 { eprintln!("Error: population must be at least 10"); std::process::exit(1); }
        if num_threads < 1 { num_threads = 1; }
        let elite_count = (pop_size / 600).max(2);
        Config { num_cities, pop_size, num_generations, elite_count, use_gpu, num_threads }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  ANSI Escape Codes
// ═══════════════════════════════════════════════════════════════════════════════
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const WHITE: &str = "\x1b[97m";
const MAGENTA: &str = "\x1b[35m";
const ORANGE: &str = "\x1b[38;5;208m";
const GRAY: &str = "\x1b[38;5;245m";
const HIDE_CURSOR: &str = "\x1b[?25l";
/// Show cursor + restore default cursor style (blink).
/// \x1b[?25h = DECTCEM show, \x1b[0 q = reset cursor shape to terminal default.
const SHOW_CURSOR: &str = "\x1b[?25h";

// ═══════════════════════════════════════════════════════════════════════════════
//  Data Structures
// ═══════════════════════════════════════════════════════════════════════════════

/// Cities in Structure-of-Arrays layout — x[] and y[] are contiguous for SIMD.
/// At 500 cities: 8KB total, permanently L1-resident on any modern CPU.
struct Cities {
    x: Vec<f64>,
    y: Vec<f64>,
    n: usize,
}

struct Individual {
    tour: Vec<usize>,
    distance: f64,
}

impl Clone for Individual {
    fn clone(&self) -> Self {
        Individual { tour: self.tour.clone(), distance: self.distance }
    }
}

struct GenStats {
    generation: usize,
    best_pct_nn: f64,   // best distance as % of NN baseline (lower = better)
    worst_pct_nn: f64,
    avg_pct_nn: f64,
    best_distance: f64,
    elapsed_ms: u128,
}

struct PopStats {
    best_d: f64,
    worst_d: f64,
    sum_d: f64,
    best_i: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Cities — construction, scalar distance, nearest-neighbor heuristic
// ═══════════════════════════════════════════════════════════════════════════════
impl Cities {
    fn new(n: usize, rng: &mut impl Rng) -> Self {
        let x: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * MAP_SIZE).collect();
        let y: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * MAP_SIZE).collect();
        Cities { x, y, n }
    }

    #[inline(always)]
    fn dist(&self, a: usize, b: usize) -> f64 {
        let dx = self.x[a] - self.x[b];
        let dy = self.y[a] - self.y[b];
        (dx * dx + dy * dy).sqrt()
    }

    /// Tour distance — dispatches to the best available SIMD path.
    #[inline(always)]
    fn tour_distance(&self, tour: &[usize]) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.tour_distance_avx2(tour) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on AArch64 — always available.
            return unsafe { self.tour_distance_neon(tour) };
        }
        #[allow(unreachable_code)]
        self.tour_distance_scalar(tour)
    }

    fn tour_distance_scalar(&self, tour: &[usize]) -> f64 {
        let n = tour.len();
        let mut total = 0.0;
        for i in 0..n - 1 {
            total += self.dist(tour[i], tour[i + 1]);
        }
        total + self.dist(tour[n - 1], tour[0])
    }

    fn nearest_neighbor_distance(&self) -> f64 {
        let starts = self.n.min(50);
        let mut best = f64::MAX;
        for start in 0..starts {
            let mut visited = vec![false; self.n];
            let mut cur = start;
            visited[cur] = true;
            let mut total = 0.0;
            for _ in 1..self.n {
                let mut nn = 0;
                let mut nn_d = f64::MAX;
                for j in 0..self.n {
                    if !visited[j] {
                        let d = self.dist(cur, j);
                        if d < nn_d { nn = j; nn_d = d; }
                    }
                }
                visited[nn] = true;
                total += nn_d;
                cur = nn;
            }
            total += self.dist(cur, start);
            if total < best { best = total; }
        }
        best
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  x86_64 AVX2+FMA — 4 tour legs per cycle
//
//  Uses gather to load non-contiguous city coordinates into 256-bit registers,
//  FMA for d² = dx·dx + dy·dy, and vectorized sqrt.
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(target_arch = "x86_64")]
impl Cities {
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tour_distance_avx2(&self, tour: &[usize]) -> f64 {
        let n = tour.len();
        let xp = self.x.as_ptr();
        let yp = self.y.as_ptr();
        let mut sum = _mm256_setzero_pd();
        let open_legs = n - 1;
        let mut i = 0;

        while i + 4 <= open_legs {
            let idx_a = _mm256_set_epi64x(
                tour[i + 3] as i64, tour[i + 2] as i64,
                tour[i + 1] as i64, tour[i] as i64,
            );
            let idx_b = _mm256_set_epi64x(
                tour[i + 4] as i64, tour[i + 3] as i64,
                tour[i + 2] as i64, tour[i + 1] as i64,
            );
            let xa = _mm256_i64gather_pd::<8>(xp, idx_a);
            let ya = _mm256_i64gather_pd::<8>(yp, idx_a);
            let xb = _mm256_i64gather_pd::<8>(xp, idx_b);
            let yb = _mm256_i64gather_pd::<8>(yp, idx_b);
            let dx = _mm256_sub_pd(xa, xb);
            let dy = _mm256_sub_pd(ya, yb);
            let d2 = _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx));
            sum = _mm256_add_pd(sum, _mm256_sqrt_pd(d2));
            i += 4;
        }

        // Horizontal sum
        let hi128 = _mm256_extractf128_pd::<1>(sum);
        let lo128 = _mm256_castpd256_pd128(sum);
        let pair = _mm_add_pd(hi128, lo128);
        let pair = _mm_hadd_pd(pair, pair);
        let mut total = _mm_cvtsd_f64(pair);

        while i < open_legs {
            total += self.dist(tour[i], tour[i + 1]);
            i += 1;
        }
        total + self.dist(tour[n - 1], tour[0])
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  AArch64 NEON+FMA — 2 tour legs per cycle
//
//  Optimized for Apple M2 Ultra P-cores:
//    • 4-wide f64 FP pipeline (executes 2×float64x2 per cycle)
//    • Fused multiply-accumulate via vfmaq_f64
//    • No gather — coordinates loaded with scalar index + vld1q_lane_f64
//    • Two independent accumulators to hide sqrt latency (~12 cycles on M2)
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(target_arch = "aarch64")]
impl Cities {
    #[target_feature(enable = "neon")]
    unsafe fn tour_distance_neon(&self, tour: &[usize]) -> f64 {
        let n = tour.len();
        let xp = self.x.as_ptr();
        let yp = self.y.as_ptr();
        let open_legs = n - 1;
        let mut i = 0;

        // Two independent accumulators — hides sqrt latency on M2's
        // out-of-order pipeline (sqrt is 12 cycles, FMA is 4 cycles).
        let mut sum0 = vdupq_n_f64(0.0);
        let mut sum1 = vdupq_n_f64(0.0);

        // Process 4 legs per iteration (2 NEON ops × 2 f64 lanes)
        while i + 4 <= open_legs {
            // Leg pair 0: tour[i]→tour[i+1], tour[i+1]→tour[i+2]
            let a0 = tour[i];
            let b0 = tour[i + 1];
            let a1 = tour[i + 1];
            let b1 = tour[i + 2];

            // Manual 2-lane coordinate gather (no hardware gather on ARM)
            let xa_0 = vcombine_f64(
                vld1_f64(xp.add(a0)),
                vld1_f64(xp.add(a1)),
            );
            let xb_0 = vcombine_f64(
                vld1_f64(xp.add(b0)),
                vld1_f64(xp.add(b1)),
            );
            let ya_0 = vcombine_f64(
                vld1_f64(yp.add(a0)),
                vld1_f64(yp.add(a1)),
            );
            let yb_0 = vcombine_f64(
                vld1_f64(yp.add(b0)),
                vld1_f64(yp.add(b1)),
            );

            let dx0 = vsubq_f64(xa_0, xb_0);
            let dy0 = vsubq_f64(ya_0, yb_0);
            // d² = dx*dx + dy*dy  via FMA
            let d2_0 = vfmaq_f64(vmulq_f64(dx0, dx0), dy0, dy0);
            sum0 = vaddq_f64(sum0, vsqrtq_f64(d2_0));

            // Leg pair 1: tour[i+2]→tour[i+3], tour[i+3]→tour[i+4]
            let a2 = tour[i + 2];
            let b2 = tour[i + 3];
            let a3 = tour[i + 3];
            let b3 = tour[i + 4];

            let xa_1 = vcombine_f64(
                vld1_f64(xp.add(a2)),
                vld1_f64(xp.add(a3)),
            );
            let xb_1 = vcombine_f64(
                vld1_f64(xp.add(b2)),
                vld1_f64(xp.add(b3)),
            );
            let ya_1 = vcombine_f64(
                vld1_f64(yp.add(a2)),
                vld1_f64(yp.add(a3)),
            );
            let yb_1 = vcombine_f64(
                vld1_f64(yp.add(b2)),
                vld1_f64(yp.add(b3)),
            );

            let dx1 = vsubq_f64(xa_1, xb_1);
            let dy1 = vsubq_f64(ya_1, yb_1);
            let d2_1 = vfmaq_f64(vmulq_f64(dx1, dx1), dy1, dy1);
            sum1 = vaddq_f64(sum1, vsqrtq_f64(d2_1));

            i += 4;
        }

        // Process remaining legs 2 at a time
        while i + 2 <= open_legs {
            let a0 = tour[i];
            let b0 = tour[i + 1];
            let a1 = tour[i + 1];
            let b1 = tour[i + 2];

            let xa = vcombine_f64(vld1_f64(xp.add(a0)), vld1_f64(xp.add(a1)));
            let xb = vcombine_f64(vld1_f64(xp.add(b0)), vld1_f64(xp.add(b1)));
            let ya = vcombine_f64(vld1_f64(yp.add(a0)), vld1_f64(yp.add(a1)));
            let yb = vcombine_f64(vld1_f64(yp.add(b0)), vld1_f64(yp.add(b1)));

            let dx = vsubq_f64(xa, xb);
            let dy = vsubq_f64(ya, yb);
            let d2 = vfmaq_f64(vmulq_f64(dx, dx), dy, dy);
            sum0 = vaddq_f64(sum0, vsqrtq_f64(d2));
            i += 2;
        }

        // Merge accumulators and extract scalar
        let combined = vaddq_f64(sum0, sum1);
        let mut total = vgetq_lane_f64::<0>(combined) + vgetq_lane_f64::<1>(combined);

        // Scalar remainder
        while i < open_legs {
            total += self.dist(tour[i], tour[i + 1]);
            i += 1;
        }
        total + self.dist(tour[n - 1], tour[0])
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Population Statistics
// ═══════════════════════════════════════════════════════════════════════════════
fn compute_stats(pop: &[Individual]) -> PopStats {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { compute_stats_avx2(pop) };
        }
    }
    // NEON stats and scalar share the same structure — the compiler
    // auto-vectorizes the simple loop on AArch64 extremely well, and
    // manual NEON doesn't help for this trivial reduction.
    compute_stats_scalar(pop)
}

fn compute_stats_scalar(pop: &[Individual]) -> PopStats {
    let mut best_d = f64::MAX;
    let mut worst_d = 0.0f64;
    let mut sum_d = 0.0;
    let mut best_i = 0;
    for (i, ind) in pop.iter().enumerate() {
        let d = ind.distance;
        sum_d += d;
        if d < best_d { best_d = d; best_i = i; }
        if d > worst_d { worst_d = d; }
    }
    PopStats { best_d, worst_d, sum_d, best_i }
}

/// Compute population stats from a flat f32 distance array (GPU mode).
fn compute_stats_flat(distances: &[f32]) -> PopStats {
    let mut best_d = f32::MAX;
    let mut worst_d: f32 = 0.0;
    let mut sum_d: f64 = 0.0;
    let mut best_i = 0;
    for (i, &d) in distances.iter().enumerate() {
        sum_d += d as f64;
        if d < best_d { best_d = d; best_i = i; }
        if d > worst_d { worst_d = d; }
    }
    PopStats { best_d: best_d as f64, worst_d: worst_d as f64, sum_d, best_i }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_stats_avx2(pop: &[Individual]) -> PopStats {
    let distances: Vec<f64> = pop.iter().map(|ind| ind.distance).collect();
    let n = distances.len();
    let ptr = distances.as_ptr();

    let mut vsum = _mm256_setzero_pd();
    let mut vmin = _mm256_set1_pd(f64::MAX);
    let mut vmax = _mm256_set1_pd(0.0);

    let mut i = 0;
    while i + 4 <= n {
        let v = _mm256_loadu_pd(ptr.add(i));
        vsum = _mm256_add_pd(vsum, v);
        vmin = _mm256_min_pd(vmin, v);
        vmax = _mm256_max_pd(vmax, v);
        i += 4;
    }

    let hi = _mm256_extractf128_pd::<1>(vsum);
    let lo = _mm256_castpd256_pd128(vsum);
    let s2 = _mm_hadd_pd(_mm_add_pd(hi, lo), _mm_add_pd(hi, lo));
    let mut sum_d = _mm_cvtsd_f64(s2);

    let hi = _mm256_extractf128_pd::<1>(vmin);
    let lo = _mm256_castpd256_pd128(vmin);
    let m2 = _mm_min_pd(hi, lo);
    let mut best_d = _mm_cvtsd_f64(_mm_min_pd(m2, _mm_unpackhi_pd(m2, m2)));

    let hi = _mm256_extractf128_pd::<1>(vmax);
    let lo = _mm256_castpd256_pd128(vmax);
    let m2 = _mm_max_pd(hi, lo);
    let mut worst_d = _mm_cvtsd_f64(_mm_max_pd(m2, _mm_unpackhi_pd(m2, m2)));

    while i < n {
        let d = distances[i];
        sum_d += d;
        if d < best_d { best_d = d; }
        if d > worst_d { worst_d = d; }
        i += 1;
    }

    let mut best_i = 0;
    for (j, d) in distances.iter().enumerate() {
        if *d <= best_d { best_i = j; break; }
    }
    PopStats { best_d, worst_d, sum_d, best_i }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  GA Operators
// ═══════════════════════════════════════════════════════════════════════════════
#[inline(always)]
fn tournament_select<'a>(pop: &'a [Individual], rng: &mut SmallRng) -> &'a Individual {
    let len = pop.len();
    let mut best: Option<&Individual> = None;
    for _ in 0..TOURNAMENT_SIZE {
        let ind = &pop[rng.gen_range(0..len)];
        if best.map_or(true, |b| ind.distance < b.distance) { best = Some(ind); }
    }
    best.unwrap()
}

fn order_crossover(p1: &[usize], p2: &[usize], rng: &mut SmallRng) -> Vec<usize> {
    let n = p1.len();
    let (mut a, mut b) = (rng.gen_range(0..n), rng.gen_range(0..n));
    if a > b { std::mem::swap(&mut a, &mut b); }

    let mut child = vec![0usize; n];
    let mut used = vec![false; n];
    for i in a..=b { child[i] = p1[i]; used[p1[i]] = true; }

    let mut pos = (b + 1) % n;
    let mut p2i = (b + 1) % n;
    for _ in 0..(n - (b - a + 1)) {
        while used[p2[p2i]] { p2i = (p2i + 1) % n; }
        child[pos] = p2[p2i];
        pos = (pos + 1) % n;
        p2i = (p2i + 1) % n;
    }
    child
}

fn mutate(tour: &mut Vec<usize>, rng: &mut SmallRng) {
    let n = tour.len();
    match rng.gen_range(0u8..3) {
        0 => { let i = rng.gen_range(0..n); let j = rng.gen_range(0..n); tour.swap(i, j); }
        1 => {
            let mut i = rng.gen_range(0..n);
            let mut j = rng.gen_range(0..n);
            if i > j { std::mem::swap(&mut i, &mut j); }
            tour[i..=j].reverse();
        }
        _ => {
            let i = rng.gen_range(0..n);
            let city = tour.remove(i);
            let j = rng.gen_range(0..tour.len());
            tour.insert(j, city);
        }
    }
}

#[inline(never)]
fn breed_one(pop: &[Individual], cities: &Cities, rng: &mut SmallRng) -> Individual {
    let p1 = tournament_select(pop, rng);
    let p2 = tournament_select(pop, rng);

    let mut child_tour = if rng.gen::<f64>() < CROSSOVER_RATE {
        order_crossover(&p1.tour, &p2.tour, rng)
    } else {
        p1.tour.clone()
    };

    if rng.gen::<f64>() < MUTATION_RATE { mutate(&mut child_tour, rng); }
    if rng.gen::<f64>() < DOUBLE_MUTATION_RATE { mutate(&mut child_tour, rng); }

    let d = cities.tour_distance(&child_tour);
    Individual { tour: child_tour, distance: d }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Flat-Buffer GA Operators (GPU zero-copy mode)
//
//  These operate on flat u32/f32 arrays stored directly in Metal buffers,
//  eliminating the Vec<Individual> indirection and 60MB+ per-gen copies.
// ═══════════════════════════════════════════════════════════════════════════════

/// Tournament selection from flat f32 distance array. Returns winner index.
#[inline(always)]
fn tournament_select_flat(distances: &[f32], rng: &mut SmallRng) -> usize {
    let len = distances.len();
    let mut best_idx = rng.gen_range(0..len);
    let mut best_d = distances[best_idx];
    for _ in 1..TOURNAMENT_SIZE {
        let idx = rng.gen_range(0..len);
        let d = distances[idx];
        if d < best_d { best_d = d; best_idx = idx; }
    }
    best_idx
}

/// Order crossover on flat u32 tour slices.
/// `p1`, `p2` are parent tour slices of length `n`.
/// `child` is the destination slice (must be length `n`).
/// `used` is a reusable scratch buffer (must be length >= `n`).
fn order_crossover_flat(p1: &[u32], p2: &[u32], child: &mut [u32], rng: &mut SmallRng, used: &mut [bool]) {
    let n = p1.len();
    let (mut a, mut b) = (rng.gen_range(0..n), rng.gen_range(0..n));
    if a > b { std::mem::swap(&mut a, &mut b); }

    used[..n].fill(false);
    for i in a..=b { child[i] = p1[i]; used[p1[i] as usize] = true; }

    let mut pos = (b + 1) % n;
    let mut p2i = (b + 1) % n;
    for _ in 0..(n - (b - a + 1)) {
        while used[p2[p2i] as usize] { p2i = (p2i + 1) % n; }
        child[pos] = p2[p2i];
        pos = (pos + 1) % n;
        p2i = (p2i + 1) % n;
    }
}

/// Mutate a flat u32 tour slice in-place.
fn mutate_flat(tour: &mut [u32], rng: &mut SmallRng) {
    let n = tour.len();
    match rng.gen_range(0u8..3) {
        0 => { let i = rng.gen_range(0..n); let j = rng.gen_range(0..n); tour.swap(i, j); }
        1 => {
            let mut i = rng.gen_range(0..n);
            let mut j = rng.gen_range(0..n);
            if i > j { std::mem::swap(&mut i, &mut j); }
            tour[i..=j].reverse();
        }
        _ => {
            let i = rng.gen_range(0..n);
            let city = tour[i];
            // Shift left to close gap (memmove-safe)
            if i < n - 1 { tour.copy_within(i + 1..n, i); }
            let j = rng.gen_range(0..n - 1);
            // Shift right to make room (memmove-safe)
            if j < n - 1 { tour.copy_within(j..n - 1, j + 1); }
            tour[j] = city;
        }
    }
}

/// Breed one offspring directly into a flat destination slice.
/// Reads parents from `src_tours` (indexed by tournament on `distances`),
/// writes child tour into `dest` (a slice of exactly `nc` u32s).
#[inline(never)]
fn breed_one_into_dest(
    src_tours: &[u32],
    distances: &[f32],
    dest: &mut [u32],
    nc: usize,
    rng: &mut SmallRng,
    used_buf: &mut [bool],
) {
    let p1_idx = tournament_select_flat(distances, rng);
    let p2_idx = tournament_select_flat(distances, rng);

    let p1 = &src_tours[p1_idx * nc..(p1_idx + 1) * nc];
    let p2 = &src_tours[p2_idx * nc..(p2_idx + 1) * nc];

    if rng.gen::<f64>() < CROSSOVER_RATE {
        order_crossover_flat(p1, p2, dest, rng, used_buf);
    } else {
        dest.copy_from_slice(p1);
    }

    if rng.gen::<f64>() < MUTATION_RATE { mutate_flat(dest, rng); }
    if rng.gen::<f64>() < DOUBLE_MUTATION_RATE { mutate_flat(dest, rng); }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Metal GPU Compute — M2 Ultra optimized
//
//  Architecture:
//    CPU breeds tours (branchy, RNG-heavy) → bulk transfer to GPU →
//    GPU evaluates all tour distances in one dispatch → distances back to CPU
//
//  M2 Ultra GPU: 76 cores × 32-wide SIMD = 2,432 threads in flight.
//  With 30K pop and 256 threads/threadgroup → 118 threadgroups → ~1.6 waves.
//  Unified memory (800 GB/s) means zero-copy buffer sharing.
//
//  Kernel optimizations:
//    • City coords loaded into threadgroup shared memory (4KB for 500 cities)
//    • float4 vectorized distance accumulation (4 legs per iteration)
//    • FMA: fma(dx,dx, dy*dy) fused into single instruction
//    • f32 on GPU (2× throughput vs f64, sufficient for TSP scoring)
// ═══════════════════════════════════════════════════════════════════════════════

/// Metal Shading Language kernel — compiled at runtime on macOS.
#[cfg(target_os = "macos")]
const METAL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ─── M2 Ultra-tuned tour distance kernel ───────────────────────────────
//
// Each thread evaluates one individual's complete tour distance.
// City coordinates are loaded into threadgroup shared memory for
// fast repeated access (threadgroup memory: ~1 cycle vs device: ~100+).
//
// Supports up to 2048 cities via shared memory (2048 × 2 × 4B = 16KB
// out of 32KB available per threadgroup on M2).

kernel void evaluate_tours(
    device const uint*  tours       [[buffer(0)]],   // [pop_size × num_cities]
    device const float* city_x      [[buffer(1)]],   // [num_cities]
    device const float* city_y      [[buffer(2)]],   // [num_cities]
    device float*       distances   [[buffer(3)]],   // [pop_size] output
    constant uint&      num_cities  [[buffer(4)]],
    constant uint&      pop_size    [[buffer(5)]],
    uint tid     [[thread_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // ── Load city coords into threadgroup shared memory ──
    // 2048 cities max: 2 × 2048 × 4B = 16KB < 32KB threadgroup limit
    threadgroup float sx[2048];
    threadgroup float sy[2048];

    for (uint i = lid; i < num_cities; i += tg_size) {
        sx[i] = city_x[i];
        sy[i] = city_y[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Guard: don't process past population
    if (tid >= pop_size) return;

    uint base = tid * num_cities;

    // ── Vectorized distance accumulation (4 legs per iteration) ──
    // float4 operations use all 4 ALU lanes of an M2 GPU execution unit.
    float4 acc = float4(0.0);
    uint prev = tours[base];
    uint i = 1;

    for (; i + 3 < num_cities; i += 4) {
        uint c0 = tours[base + i];
        uint c1 = tours[base + i + 1];
        uint c2 = tours[base + i + 2];
        uint c3 = tours[base + i + 3];

        float4 dx = float4(
            sx[prev] - sx[c0],
            sx[c0]   - sx[c1],
            sx[c1]   - sx[c2],
            sx[c2]   - sx[c3]
        );
        float4 dy = float4(
            sy[prev] - sy[c0],
            sy[c0]   - sy[c1],
            sy[c1]   - sy[c2],
            sy[c2]   - sy[c3]
        );

        // FMA: fma(dy, dy, dx*dx) = dy² + dx²
        acc += sqrt(fma(dy, dy, dx * dx));
        prev = c3;
    }

    float total = acc.x + acc.y + acc.z + acc.w;

    // Scalar remainder
    for (; i < num_cities; i++) {
        uint cur = tours[base + i];
        float dx = sx[prev] - sx[cur];
        float dy = sy[prev] - sy[cur];
        total += sqrt(fma(dy, dy, dx * dx));
        prev = cur;
    }

    // Closing leg: last → first
    uint first = tours[base];
    float dx = sx[prev] - sx[first];
    float dy = sy[prev] - sy[first];
    total += sqrt(fma(dy, dy, dx * dx));

    distances[tid] = total;
}
"#;

/// GPU evaluator — manages Metal device, pipeline, and dual tour buffers.
///
/// Zero-copy architecture: CPU breeds directly into Metal shared-memory buffers,
/// GPU evaluates in-place. Dual tour buffers (ping-pong) allow safe concurrent
/// reads from the "source" (last evaluated) and writes to the "destination"
/// (next generation) without data races.
#[cfg(target_os = "macos")]
mod gpu {
    use metal::*;
    use std::cell::Cell;

    pub struct GpuEvaluator {
        pipeline: ComputePipelineState,
        command_queue: CommandQueue,
        tours_bufs: [Buffer; 2],    // ping-pong tour buffers
        city_x_buf: Buffer,
        city_y_buf: Buffer,
        distances_buf: Buffer,
        num_cities_buf: Buffer,
        pop_size_buf: Buffer,
        pop_size: usize,
        num_cities: usize,
        gpu_name: String,
        _gpu_cores: u64,
        current: Cell<usize>,      // index of last-evaluated tours buffer (0 or 1)
    }

    impl GpuEvaluator {
        pub fn new(cities_x: &[f64], cities_y: &[f64], pop_size: usize) -> Result<Self, String> {
            let device = Device::system_default()
                .ok_or_else(|| "No Metal GPU device found".to_string())?;

            let gpu_name = device.name().to_string();
            let gpu_cores = device.max_threads_per_threadgroup().width as u64;

            let options = CompileOptions::new();
            let library = device.new_library_with_source(super::METAL_SHADER, &options)
                .map_err(|e| format!("Metal shader compile error: {e}"))?;
            let kernel = library.get_function("evaluate_tours", None)
                .map_err(|e| format!("Metal function error: {e}"))?;
            let pipeline = device.new_compute_pipeline_state_with_function(&kernel)
                .map_err(|e| format!("Metal pipeline error: {e}"))?;
            let command_queue = device.new_command_queue();

            let num_cities = cities_x.len();

            // ── Allocate shared-memory buffers (zero-copy on Apple Silicon) ──
            let tour_bytes = (pop_size * num_cities * std::mem::size_of::<u32>()) as u64;
            let city_bytes = (num_cities * std::mem::size_of::<f32>()) as u64;
            let dist_bytes = (pop_size * std::mem::size_of::<f32>()) as u64;
            let uint_bytes = std::mem::size_of::<u32>() as u64;

            let res_opts = MTLResourceOptions::StorageModeShared;

            // Two tour buffers for ping-pong: CPU writes next gen to one
            // while GPU reads current gen from the other.
            let tours_buf_a = device.new_buffer(tour_bytes, res_opts);
            let tours_buf_b = device.new_buffer(tour_bytes, res_opts);
            let city_x_buf = device.new_buffer(city_bytes, res_opts);
            let city_y_buf = device.new_buffer(city_bytes, res_opts);
            let distances_buf = device.new_buffer(dist_bytes, res_opts);
            let num_cities_buf = device.new_buffer(uint_bytes, res_opts);
            let pop_size_buf = device.new_buffer(uint_bytes, res_opts);

            // Upload city coordinates (f64 → f32 conversion, one-time)
            {
                let ptr = city_x_buf.contents() as *mut f32;
                for i in 0..num_cities {
                    unsafe { *ptr.add(i) = cities_x[i] as f32; }
                }
            }
            {
                let ptr = city_y_buf.contents() as *mut f32;
                for i in 0..num_cities {
                    unsafe { *ptr.add(i) = cities_y[i] as f32; }
                }
            }

            // Upload constants
            unsafe {
                *(num_cities_buf.contents() as *mut u32) = num_cities as u32;
                *(pop_size_buf.contents() as *mut u32) = pop_size as u32;
            }

            Ok(GpuEvaluator {
                pipeline, command_queue,
                tours_bufs: [tours_buf_a, tours_buf_b],
                city_x_buf, city_y_buf, distances_buf,
                num_cities_buf, pop_size_buf,
                pop_size, num_cities, gpu_name, _gpu_cores: gpu_cores,
                current: Cell::new(0),
            })
        }

        /// Mutable slice of the "current" tour buffer — for initial population setup.
        /// SAFETY: caller must not alias with other references to the same buffer.
        pub unsafe fn current_tours_mut(&self) -> &mut [u32] {
            let ptr = self.tours_bufs[self.current.get()].contents() as *mut u32;
            std::slice::from_raw_parts_mut(ptr, self.pop_size * self.num_cities)
        }

        /// Immutable slice of the "current" (last-evaluated) tour buffer.
        /// SAFETY: caller must ensure GPU is not writing to this buffer.
        pub unsafe fn src_tours(&self) -> &[u32] {
            let ptr = self.tours_bufs[self.current.get()].contents() as *const u32;
            std::slice::from_raw_parts(ptr, self.pop_size * self.num_cities)
        }

        /// Mutable slice of the "next" tour buffer (destination for breeding).
        /// SAFETY: caller must not alias with other references to the same buffer.
        pub unsafe fn dst_tours_mut(&self) -> &mut [u32] {
            let idx = 1 - self.current.get();
            let ptr = self.tours_bufs[idx].contents() as *mut u32;
            std::slice::from_raw_parts_mut(ptr, self.pop_size * self.num_cities)
        }

        /// Distances slice (valid after evaluate_current or evaluate_dst_and_swap).
        /// SAFETY: caller must ensure GPU has completed evaluation.
        pub unsafe fn distances(&self) -> &[f32] {
            let ptr = self.distances_buf.contents() as *const f32;
            std::slice::from_raw_parts(ptr, self.pop_size)
        }

        /// Evaluate the current tour buffer (used for initial population).
        pub fn evaluate_current(&self) {
            self.dispatch_evaluation(&self.tours_bufs[self.current.get()]);
        }

        /// Evaluate the destination buffer, then swap it to become current.
        pub fn evaluate_dst_and_swap(&self) {
            let dst_idx = 1 - self.current.get();
            self.dispatch_evaluation(&self.tours_bufs[dst_idx]);
            self.current.set(dst_idx);
        }

        /// Dispatch GPU compute on a specific tour buffer and wait for completion.
        fn dispatch_evaluation(&self, tours_buf: &Buffer) {
            let cmd_buf = self.command_queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(0, Some(tours_buf), 0);
            encoder.set_buffer(1, Some(&self.city_x_buf), 0);
            encoder.set_buffer(2, Some(&self.city_y_buf), 0);
            encoder.set_buffer(3, Some(&self.distances_buf), 0);
            encoder.set_buffer(4, Some(&self.num_cities_buf), 0);
            encoder.set_buffer(5, Some(&self.pop_size_buf), 0);

            let tg_size = MTLSize { width: 256, height: 1, depth: 1 };
            let grid_size = MTLSize { width: self.pop_size as u64, height: 1, depth: 1 };
            encoder.dispatch_threads(grid_size, tg_size);
            encoder.end_encoding();

            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }

        pub fn label(&self) -> String {
            format!("Metal GPU ({})", self.gpu_name)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  GPU Stub — non-macOS platforms
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(not(target_os = "macos"))]
mod gpu {
    pub struct GpuEvaluator;
    impl GpuEvaluator {
        pub fn new(_cx: &[f64], _cy: &[f64], _pop: usize) -> Result<Self, String> {
            Err("Metal GPU requires macOS with Apple Silicon".to_string())
        }
        pub unsafe fn current_tours_mut(&self) -> &mut [u32] { &mut [] }
        pub unsafe fn src_tours(&self) -> &[u32] { &[] }
        pub unsafe fn dst_tours_mut(&self) -> &mut [u32] { &mut [] }
        pub unsafe fn distances(&self) -> &[f32] { &[] }
        pub fn evaluate_current(&self) {}
        pub fn evaluate_dst_and_swap(&self) {}
        pub fn label(&self) -> String { String::new() }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Terminal Size Detection
// ═══════════════════════════════════════════════════════════════════════════════
fn get_terminal_size() -> (usize, usize) {
    // Returns (cols, rows)
    #[cfg(unix)]
    {
        #[repr(C)]
        struct Winsize { ws_row: u16, ws_col: u16, _xpixel: u16, _ypixel: u16 }
        extern "C" { fn ioctl(fd: i32, request: u64, ...) -> i32; }
        #[cfg(target_os = "linux")]
        const TIOCGWINSZ: u64 = 0x5413;
        #[cfg(target_os = "macos")]
        const TIOCGWINSZ: u64 = 0x40087468;
        let mut ws = std::mem::MaybeUninit::<Winsize>::uninit();
        if unsafe { ioctl(1, TIOCGWINSZ, ws.as_mut_ptr()) } == 0 {
            let ws = unsafe { ws.assume_init() };
            return (ws.ws_col as usize, ws.ws_row as usize);
        }
    }
    (80, 40)
}

/// Layout dimensions computed from terminal size.
struct Layout {
    cols: usize,        // terminal width
    inner: usize,       // usable width inside borders (cols - 2)
    data_rows: usize,   // number of generation rows in the table
    map_w: usize,       // map character width
    map_h: usize,       // map character height (not counting borders)
}

impl Layout {
    fn from_terminal(cols: usize, rows: usize) -> Self {
        let cols = cols.max(60);
        let rows = rows.max(30);
        let inner = cols.saturating_sub(2);

        // Fixed overhead lines:
        //   3 title box + 1 info + 1 sep + 1 header + 1 sep +
        //   1 sep + 1 status + 1 sep + 1 map_label + 2 map_borders + 1 finish_line
        let overhead = 14;
        let available = rows.saturating_sub(overhead);

        // Give map ~35% of available, data rows ~65%
        let map_h = (available * 35 / 100).clamp(6, 40);
        let data_rows = available.saturating_sub(map_h).max(4);

        let map_w = cols.saturating_sub(6); // 2 indent + 2 border + 2 margin

        Layout { cols, inner, data_rows, map_w, map_h }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Display
// ═══════════════════════════════════════════════════════════════════════════════

/// Color for pct-of-NN (lower is better).
fn pct_color(p: f64) -> &'static str {
    if p <= 85.0 { GREEN } else if p <= 100.0 { YELLOW } else if p <= 120.0 { ORANGE } else { RED }
}

fn render_map(cities: &Cities, tour: &[usize], w: usize, h: usize) -> Vec<String> {
    let w = w.max(10);
    let h = h.max(4);
    let mut grid = vec![vec![0u8; w]; h];

    for i in 0..tour.len() {
        let a = tour[i];
        let b = tour[(i + 1) % tour.len()];
        let x1 = ((cities.x[a] / MAP_SIZE) * (w as f64 - 1.0)) as i32;
        let y1 = ((cities.y[a] / MAP_SIZE) * (h as f64 - 1.0)) as i32;
        let x2 = ((cities.x[b] / MAP_SIZE) * (w as f64 - 1.0)) as i32;
        let y2 = ((cities.y[b] / MAP_SIZE) * (h as f64 - 1.0)) as i32;
        let steps = (x2 - x1).unsigned_abs().max((y2 - y1).unsigned_abs()) as usize;
        if steps == 0 { continue; }
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let x = (x1 as f64 + t * (x2 - x1) as f64).round() as usize;
            let y = (y1 as f64 + t * (y2 - y1) as f64).round() as usize;
            if x < w && y < h && grid[y][x] == 0 { grid[y][x] = 1; }
        }
    }

    for idx in 0..cities.n {
        let x = ((cities.x[idx] / MAP_SIZE) * (w as f64 - 1.0)).round() as usize;
        let y = ((cities.y[idx] / MAP_SIZE) * (h as f64 - 1.0)).round() as usize;
        if x < w && y < h {
            grid[y][x] = if idx == tour[0] { 3 } else { 2 };
        }
    }

    let border: String = "\u{2500}".repeat(w);
    let mut lines = Vec::with_capacity(h + 2);
    lines.push(format!("  {DIM}\u{250c}{border}\u{2510}{RESET}"));
    for row in &grid {
        let mut line = format!("  {DIM}\u{2502}{RESET}");
        for &cell in row {
            match cell {
                3 => line.push_str(&format!("{BOLD}{GREEN}\u{25c6}{RESET}")),
                2 => line.push_str(&format!("{CYAN}\u{2022}{RESET}")),
                1 => line.push_str(&format!("{GRAY}\u{00b7}{RESET}")),
                _ => line.push(' '),
            }
        }
        line.push_str(&format!("{DIM}\u{2502}{RESET}"));
        lines.push(line);
    }
    lines.push(format!("  {DIM}\u{2514}{border}\u{2518}{RESET}"));
    lines
}

fn render(
    out: &mut impl Write,
    cfg: &Config,
    history: &[GenStats],
    cities: &Cities,
    best_tour: &[usize],
    g_best_pct: f64,
    g_best_dist: f64,
    nn_dist: f64,
    t0: &Instant,
    done: bool,
    engine_label: &str,
    layout: &Layout,
) {
    let w = layout.cols;
    let iw = layout.inner;

    write!(out, "\x1b[H").unwrap();

    // ── Title box ──
    let title = format!(
        "GENETIC ALGORITHM \u{2500}\u{2500} TSP ({} cities \u{00d7} {} pop \u{00d7} {engine_label})",
        cfg.num_cities, cfg.pop_size
    );
    writeln!(out, "{BOLD}{CYAN}\u{2554}{}\u{2557}{RESET}\x1b[K", "\u{2550}".repeat(iw)).unwrap();
    writeln!(out, "{BOLD}{CYAN}\u{2551}{RESET}  {BOLD}{WHITE}{title:<tw$}{CYAN}\u{2551}{RESET}\x1b[K", tw = iw - 2).unwrap();
    writeln!(out, "{BOLD}{CYAN}\u{255a}{}\u{255d}{RESET}\x1b[K", "\u{2550}".repeat(iw)).unwrap();

    // ── Info line with NN baseline ──
    let elapsed = t0.elapsed().as_secs_f64();
    let gen_now = history.len().saturating_sub(1);
    let gps = if elapsed > 0.0 && gen_now > 0 { gen_now as f64 / elapsed } else { 0.0 };
    writeln!(
        out,
        "  {DIM}Gens:{RESET}{YELLOW}{gen_now}/{}{RESET} \
         {DIM}NN:{RESET}{CYAN}{nn_dist:.0}{RESET} \
         {DIM}Tourney:{RESET}{YELLOW}{TOURNAMENT_SIZE}{RESET} \
         {DIM}Xover:{RESET}{YELLOW}{:.0}%{RESET} \
         {DIM}Mutate:{RESET}{YELLOW}{:.0}%{RESET} \
         {DIM}Elite:{RESET}{YELLOW}{}{RESET} \
         {DIM}Speed:{RESET}{MAGENTA}{gps:.1}/s{RESET} \
         {DIM}Time:{RESET}{CYAN}{elapsed:.1}s{RESET}\x1b[K",
        cfg.num_generations, CROSSOVER_RATE * 100.0, MUTATION_RATE * 100.0, cfg.elite_count,
    ).unwrap();

    let sep = format!("{DIM}{}{RESET}", "\u{2500}".repeat(w));
    writeln!(out, "{sep}\x1b[K").unwrap();

    // ── Column header ──
    // Columns: GEN  BEST%  AVG%  WORST%  DISTANCE  ms
    writeln!(
        out,
        "  {BOLD}{WHITE} GEN    BEST%   AVG%  WORST%     DISTANCE     ms{RESET}\x1b[K",
    ).unwrap();
    writeln!(out, "{sep}\x1b[K").unwrap();

    // ── Data rows ──
    let display_rows = layout.data_rows;
    let start_i = history.len().saturating_sub(display_rows);
    for stats in &history[start_i..] {
        writeln!(
            out,
            "  {DIM}{:>4}{RESET}  {BOLD}{}{:>5.1}%{RESET}  {}{:>5.1}%{RESET}  {}{:>5.1}%{RESET}  {CYAN}{:>10.1}{RESET}  {DIM}{:>5}{RESET}\x1b[K",
            stats.generation,
            pct_color(stats.best_pct_nn), stats.best_pct_nn,
            pct_color(stats.avg_pct_nn), stats.avg_pct_nn,
            pct_color(stats.worst_pct_nn), stats.worst_pct_nn,
            stats.best_distance,
            stats.elapsed_ms,
        ).unwrap();
    }
    // Fill remaining rows
    let shown = history.len().min(display_rows);
    for _ in shown..display_rows {
        writeln!(out, "\x1b[K").unwrap();
    }

    // ── Status line ──
    writeln!(out, "{sep}\x1b[K").unwrap();
    let status = if done {
        format!("{BOLD}{GREEN}\u{2713} COMPLETE{RESET}")
    } else {
        format!("{BOLD}{YELLOW}\u{21bb} EVOLVING{RESET}")
    };
    let gc = pct_color(g_best_pct);
    let eta_str = if !done && gen_now > 0 {
        let remaining = (cfg.num_generations - gen_now) as f64 * (elapsed / gen_now as f64);
        let m = (remaining / 60.0).floor() as u64;
        let s = (remaining % 60.0).round() as u64;
        format!("  {DIM}ETA:{RESET}{CYAN}{m}m{s:02}s{RESET}")
    } else if done {
        format!("  {DIM}Total:{RESET}{CYAN}{elapsed:.1}s{RESET}")
    } else { String::new() };
    writeln!(
        out,
        "  {status}  {BOLD}{WHITE}Best:{RESET} {BOLD}{gc}{g_best_pct:>5.1}%{RESET} {DIM}of NN{RESET}\
         {BOLD}{WHITE}  Dist:{RESET} {BOLD}{CYAN}{g_best_dist:>9.1}{RESET}{eta_str}\x1b[K"
    ).unwrap();

    // ── Map ──
    writeln!(out, "{sep}\x1b[K").unwrap();
    writeln!(
        out,
        "  {BOLD}{WHITE}Best Tour{RESET}  {DIM}({GREEN}\u{25c6}{RESET}{DIM}=start  \
         {CYAN}\u{2022}{RESET}{DIM}=city  {GRAY}\u{00b7}{RESET}{DIM}=route){RESET}\x1b[K"
    ).unwrap();
    for line in render_map(cities, best_tour, layout.map_w, layout.map_h) {
        writeln!(out, "{line}\x1b[K").unwrap();
    }
    out.flush().unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Detect SIMD features at startup
// ═══════════════════════════════════════════════════════════════════════════════
fn detect_simd_label() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return "AVX2+FMA";
        } else if is_x86_feature_detected!("avx2") {
            return "AVX2";
        }
        return "x86 scalar";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "NEON+FMA";
    }
    #[allow(unreachable_code)]
    "scalar"
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════════
fn main() {
    let cfg = Config::from_args();

    let simd_label = detect_simd_label();

    let mut rng = SmallRng::from_entropy();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let cities = Cities::new(cfg.num_cities, &mut rng);
    let nn_dist = cities.nearest_neighbor_distance();

    // ── GPU initialization (after cities exist) ──
    let gpu_eval = if cfg.use_gpu {
        match gpu::GpuEvaluator::new(&cities.x, &cities.y, cfg.pop_size) {
            Ok(g) => {
                eprintln!("  GPU initialized: {}", g.label());
                Some(g)
            }
            Err(e) => {
                eprintln!("  GPU init failed: {e}");
                eprintln!("  Falling back to CPU ({simd_label})");
                None
            }
        }
    } else {
        None
    };

    let engine_label: String = if let Some(ref g) = gpu_eval {
        if cfg.num_threads > 1 {
            format!("{} + CPU {}T", g.label(), cfg.num_threads)
        } else {
            g.label()
        }
    } else if cfg.num_threads > 1 {
        format!("{simd_label}, {}-thread", cfg.num_threads)
    } else {
        format!("{simd_label}, single-thread")
    };

    if cfg.num_threads > 1 {
        let hw_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if cfg.num_threads > hw_threads {
            eprintln!(
                "  Warning: {} threads requested but only {} hardware threads available",
                cfg.num_threads, hw_threads
            );
        }
    }

    write!(out, "{HIDE_CURSOR}\x1b[2J").unwrap();
    out.flush().unwrap();
    install_signal_handler();
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        eprint!("{SHOW_CURSOR}");
        default_hook(info);
    }));

    let t0 = Instant::now();
    let mut history: Vec<GenStats> = Vec::with_capacity(cfg.num_generations + 1);
    let mut g_best_dist = f64::MAX;
    let mut g_best_tour: Vec<usize> = Vec::new();
    let mut g_best_pct = 999.0f64;

    // ── Dynamic terminal layout ──
    let (tc, tr) = get_terminal_size();
    let mut layout = Layout::from_terminal(tc, tr);
    let mut render_count: usize = 0;

    if let Some(ref gpu) = gpu_eval {
        // ═════════════════════════════════════════════════════════════════════
        //  GPU mode — zero-copy flat buffer pipeline
        //
        //  Population lives entirely in Metal shared-memory buffers:
        //    tours:     [pop_size × num_cities] u32 (dual ping-pong buffers)
        //    distances: [pop_size] f32
        //
        //  CPU breeds directly into buffers → GPU evaluates in-place.
        //  No Vec<Individual>, no 60MB copy, no usize↔u32 conversions.
        // ═════════════════════════════════════════════════════════════════════
        let nc = cfg.num_cities;
        let ps = cfg.pop_size;

        // Initialize population directly in GPU buffer (Fisher-Yates shuffle)
        {
            let tours = unsafe { gpu.current_tours_mut() };
            for i in 0..ps {
                let base = i * nc;
                for j in 0..nc { tours[base + j] = j as u32; }
                for j in (1..nc).rev() {
                    let k = rng.gen_range(0..=j);
                    tours.swap(base + j, base + k);
                }
            }
        }
        gpu.evaluate_current();

        // Reusable index buffer for elite selection
        let mut indices: Vec<usize> = (0..ps).collect();

        for gen in 0..=cfg.num_generations {
            let gen_t = Instant::now();

            if render_count % 10 == 0 {
                let (tc, tr) = get_terminal_size();
                layout = Layout::from_terminal(tc, tr);
            }
            render_count += 1;

            if gen > 0 {
                let distances = unsafe { gpu.distances() };

                // Find elite indices via partial sort on distances
                for (i, idx) in indices.iter_mut().enumerate() { *idx = i; }
                indices.select_nth_unstable_by(cfg.elite_count, |&a, &b|
                    distances[a].partial_cmp(&distances[b]).unwrap()
                );

                {
                    let src = unsafe { gpu.src_tours() };
                    let dst = unsafe { gpu.dst_tours_mut() };

                    // Copy elites from src to front of dst
                    for i in 0..cfg.elite_count {
                        let src_idx = indices[i];
                        dst[i * nc..(i + 1) * nc]
                            .copy_from_slice(&src[src_idx * nc..(src_idx + 1) * nc]);
                    }

                    // Breed offspring
                    if cfg.num_threads <= 1 {
                        let mut used_buf = vec![false; nc];
                        for i in cfg.elite_count..ps {
                            let dest = &mut dst[i * nc..(i + 1) * nc];
                            breed_one_into_dest(src, distances, dest, nc, &mut rng, &mut used_buf);
                        }
                    } else {
                        // Multi-threaded breeding directly into flat GPU buffer.
                        // Each thread writes to non-overlapping regions of dst.
                        let offspring_count = ps - cfg.elite_count;
                        let nt = cfg.num_threads;
                        // Cast to usize for Send across thread::scope (standard pattern).
                        // SAFETY: pointers remain valid for the scope's lifetime — they
                        // come from Metal shared-memory buffers that outlive the scope.
                        let src_addr = src.as_ptr() as usize;
                        let dst_addr = dst.as_mut_ptr() as usize;
                        let dist_addr = distances.as_ptr() as usize;
                        let total_tours = ps * nc;

                        std::thread::scope(|s| {
                            let mut offset = cfg.elite_count;
                            for t in 0..nt {
                                let chunk = offspring_count / nt
                                    + if t < offspring_count % nt { 1 } else { 0 };
                                let my_start = offset;
                                let my_end = offset + chunk;
                                offset += chunk;
                                let seed = rng.gen::<u64>();

                                s.spawn(move || {
                                    // SAFETY: src and dist are read-only (shared across threads).
                                    // Each thread writes to a disjoint region of dst via raw ptr.
                                    let src_sl = unsafe {
                                        std::slice::from_raw_parts(
                                            src_addr as *const u32, total_tours,
                                        )
                                    };
                                    let dist_sl = unsafe {
                                        std::slice::from_raw_parts(
                                            dist_addr as *const f32, ps,
                                        )
                                    };
                                    let mut t_rng = SmallRng::seed_from_u64(seed);
                                    let mut used_buf = vec![false; nc];

                                    for i in my_start..my_end {
                                        let dest = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                (dst_addr as *mut u32).add(i * nc), nc,
                                            )
                                        };
                                        breed_one_into_dest(
                                            src_sl, dist_sl, dest, nc,
                                            &mut t_rng, &mut used_buf,
                                        );
                                    }
                                });
                            }
                        });
                    }
                }

                gpu.evaluate_dst_and_swap();
            }

            // Stats and render from flat GPU buffers
            let distances = unsafe { gpu.distances() };
            let tours = unsafe { gpu.src_tours() };
            collect_stats_and_render_gpu(
                distances, tours, nc, &cities, &cfg, &mut history,
                &mut g_best_dist, &mut g_best_tour, &mut g_best_pct,
                gen, gen_t, &t0, nn_dist, &engine_label, &layout, &mut out,
            );
        }
    } else {
        // ═════════════════════════════════════════════════════════════════════
        //  CPU mode — Vec<Individual> population
        // ═════════════════════════════════════════════════════════════════════
        let mut pop: Vec<Individual> = (0..cfg.pop_size)
            .map(|_| {
                let mut tour: Vec<usize> = (0..cfg.num_cities).collect();
                tour.shuffle(&mut rng);
                let distance = cities.tour_distance(&tour);
                Individual { tour, distance }
            })
            .collect();

        if cfg.num_threads <= 1 {
            // ─────────────────────────────────────────────────────────────────
            //  Single-threaded generation loop
            // ─────────────────────────────────────────────────────────────────
            for gen in 0..=cfg.num_generations {
                let gen_t = Instant::now();

                if render_count % 10 == 0 {
                    let (tc, tr) = get_terminal_size();
                    layout = Layout::from_terminal(tc, tr);
                }
                render_count += 1;

                if gen > 0 {
                    pop.select_nth_unstable_by(cfg.elite_count, |a, b|
                        a.distance.partial_cmp(&b.distance).unwrap()
                    );
                    let elites: Vec<Individual> = pop[..cfg.elite_count].to_vec();
                    let offspring_count = cfg.pop_size - elites.len();

                    let mut next = Vec::with_capacity(cfg.pop_size);
                    next.extend(elites);
                    for _ in 0..offspring_count {
                        next.push(breed_one(&pop, &cities, &mut rng));
                    }
                    pop = next;
                }

                collect_stats_and_render(
                    &pop, &cities, &cfg, &mut history, &mut g_best_dist,
                    &mut g_best_tour, &mut g_best_pct, gen, gen_t, &t0, nn_dist,
                    &engine_label, &layout, &mut out,
                );
            }
        } else {
            // ─────────────────────────────────────────────────────────────────
            //  Multi-threaded generation loop — persistent worker pool
            // ─────────────────────────────────────────────────────────────────
            let nt = cfg.num_threads;

            struct SharedPopSlice {
                ptr: std::cell::UnsafeCell<(*const Individual, usize)>,
            }
            unsafe impl Sync for SharedPopSlice {}

            let shared_pop = SharedPopSlice {
                ptr: std::cell::UnsafeCell::new((std::ptr::null(), 0)),
            };
            let start_barrier = Barrier::new(nt + 1);
            let end_barrier = Barrier::new(nt + 1);
            let shutdown = AtomicBool::new(false);

            let result_slots: Vec<Mutex<Vec<Individual>>> =
                (0..nt).map(|_| Mutex::new(Vec::new())).collect();
            let chunk_sizes: Vec<AtomicU64> = (0..nt).map(|_| AtomicU64::new(0)).collect();
            let seeds: Vec<AtomicU64> = (0..nt).map(|_| AtomicU64::new(0)).collect();

            std::thread::scope(|s| {
                for t in 0..nt {
                    let shared_pop = &shared_pop;
                    let start_barrier = &start_barrier;
                    let end_barrier = &end_barrier;
                    let shutdown = &shutdown;
                    let result_slot = &result_slots[t];
                    let chunk_size_atom = &chunk_sizes[t];
                    let seed_atom = &seeds[t];
                    let cities_ref = &cities;

                    std::thread::Builder::new()
                        .stack_size(2 * 1024 * 1024)
                        .name(format!("breed-{t}"))
                        .spawn_scoped(s, move || {
                            let mut t_rng;
                            loop {
                                start_barrier.wait();
                                if shutdown.load(Ordering::Relaxed) { break; }

                                let chunk_size = chunk_size_atom.load(Ordering::Relaxed) as usize;
                                let seed = seed_atom.load(Ordering::Relaxed);
                                t_rng = SmallRng::seed_from_u64(seed);

                                let (ptr, len) = unsafe { *shared_pop.ptr.get() };
                                let pop_slice = unsafe { std::slice::from_raw_parts(ptr, len) };

                                let mut batch = Vec::with_capacity(chunk_size);
                                for _ in 0..chunk_size {
                                    batch.push(breed_one(pop_slice, cities_ref, &mut t_rng));
                                }

                                *result_slot.lock().unwrap() = batch;
                                end_barrier.wait();
                            }
                        })
                        .expect("failed to spawn worker thread");
                }

                // Warmup: force per-thread malloc arena initialization
                for _warmup in 0..2 {
                    let warmup_count = cfg.pop_size - cfg.elite_count;
                    for t in 0..nt {
                        let cs = warmup_count / nt
                            + if t < warmup_count % nt { 1 } else { 0 };
                        chunk_sizes[t].store(cs as u64, Ordering::Relaxed);
                        seeds[t].store(rng.gen::<u64>(), Ordering::Relaxed);
                    }
                    unsafe { *shared_pop.ptr.get() = (pop.as_ptr(), pop.len()); }
                    start_barrier.wait();
                    end_barrier.wait();
                    for slot in &result_slots {
                        slot.lock().unwrap().clear();
                    }
                }

                for gen in 0..=cfg.num_generations {
                    let gen_t = Instant::now();

                    if render_count % 10 == 0 {
                        let (tc, tr) = get_terminal_size();
                        layout = Layout::from_terminal(tc, tr);
                    }
                    render_count += 1;

                    if gen > 0 {
                        pop.select_nth_unstable_by(cfg.elite_count, |a, b|
                            a.distance.partial_cmp(&b.distance).unwrap()
                        );
                        let elites: Vec<Individual> = pop[..cfg.elite_count].to_vec();
                        let offspring_count = cfg.pop_size - elites.len();

                        for t in 0..nt {
                            let cs = offspring_count / nt
                                + if t < offspring_count % nt { 1 } else { 0 };
                            chunk_sizes[t].store(cs as u64, Ordering::Relaxed);
                            seeds[t].store(rng.gen::<u64>(), Ordering::Relaxed);
                        }

                        unsafe {
                            *shared_pop.ptr.get() = (pop.as_ptr(), pop.len());
                        }

                        start_barrier.wait();
                        end_barrier.wait();

                        let mut next = Vec::with_capacity(cfg.pop_size);
                        next.extend(elites);
                        for slot in &result_slots {
                            let mut batch = slot.lock().unwrap();
                            next.append(&mut batch);
                        }
                        pop = next;
                    }

                    collect_stats_and_render(
                        &pop, &cities, &cfg, &mut history, &mut g_best_dist,
                        &mut g_best_tour, &mut g_best_pct, gen, gen_t, &t0, nn_dist,
                        &engine_label, &layout, &mut out,
                    );
                }

                shutdown.store(true, Ordering::Relaxed);
                start_barrier.wait();
            }); // scope: all workers joined here
        }
    }

    let improvement = if nn_dist > 0.0 { (1.0 - g_best_dist / nn_dist) * 100.0 } else { 0.0 };
    writeln!(
        out,
        "\n  {BOLD}{WHITE}Finished in {CYAN}{:.2}s{RESET}  {DIM}({engine_label}){RESET}",
        t0.elapsed().as_secs_f64()
    ).unwrap();
    writeln!(
        out,
        "  {DIM}NN baseline: {nn_dist:.1}  |  GA best: {g_best_dist:.1}  |  Improvement: {improvement:+.1}%{RESET}"
    ).unwrap();
    writeln!(out, "{SHOW_CURSOR}").unwrap();
    out.flush().unwrap();
}

/// Extract stats computation and rendering into a helper to avoid
/// duplicating this block between single-threaded and multi-threaded paths.
#[inline(never)]
fn collect_stats_and_render(
    pop: &[Individual],
    cities: &Cities,
    cfg: &Config,
    history: &mut Vec<GenStats>,
    g_best_dist: &mut f64,
    g_best_tour: &mut Vec<usize>,
    g_best_pct: &mut f64,
    gen: usize,
    gen_t: Instant,
    t0: &Instant,
    nn_dist: f64,
    engine_label: &str,
    layout: &Layout,
    out: &mut io::StdoutLock,
) {
    let stats = compute_stats(pop);
    let avg_d = stats.sum_d / pop.len() as f64;

    // Percentage of NN baseline (lower = better)
    let best_pct = (stats.best_d / nn_dist) * 100.0;
    let worst_pct = (stats.worst_d / nn_dist) * 100.0;
    let avg_pct = (avg_d / nn_dist) * 100.0;

    if stats.best_d < *g_best_dist {
        *g_best_dist = stats.best_d;
        *g_best_tour = pop[stats.best_i].tour.clone();
        *g_best_pct = best_pct;
    }

    history.push(GenStats {
        generation: gen,
        best_pct_nn: best_pct,
        worst_pct_nn: worst_pct,
        avg_pct_nn: avg_pct,
        best_distance: stats.best_d,
        elapsed_ms: gen_t.elapsed().as_millis(),
    });

    render(
        out, cfg, history, cities, g_best_tour,
        *g_best_pct, *g_best_dist, nn_dist, t0, gen == cfg.num_generations,
        engine_label, layout,
    );
}

/// GPU mode variant — reads stats from flat f32 distance array
/// and tours from flat u32 tour buffer.
#[inline(never)]
fn collect_stats_and_render_gpu(
    distances: &[f32],
    tours: &[u32],
    nc: usize,
    cities: &Cities,
    cfg: &Config,
    history: &mut Vec<GenStats>,
    g_best_dist: &mut f64,
    g_best_tour: &mut Vec<usize>,
    g_best_pct: &mut f64,
    gen: usize,
    gen_t: Instant,
    t0: &Instant,
    nn_dist: f64,
    engine_label: &str,
    layout: &Layout,
    out: &mut io::StdoutLock,
) {
    let stats = compute_stats_flat(distances);
    let avg_d = stats.sum_d / distances.len() as f64;

    let best_pct = (stats.best_d / nn_dist) * 100.0;
    let worst_pct = (stats.worst_d / nn_dist) * 100.0;
    let avg_pct = (avg_d / nn_dist) * 100.0;

    if stats.best_d < *g_best_dist {
        *g_best_dist = stats.best_d;
        let base = stats.best_i * nc;
        *g_best_tour = tours[base..base + nc].iter().map(|&c| c as usize).collect();
        *g_best_pct = best_pct;
    }

    history.push(GenStats {
        generation: gen,
        best_pct_nn: best_pct,
        worst_pct_nn: worst_pct,
        avg_pct_nn: avg_pct,
        best_distance: stats.best_d,
        elapsed_ms: gen_t.elapsed().as_millis(),
    });

    render(
        out, cfg, history, cities, g_best_tour,
        *g_best_pct, *g_best_dist, nn_dist, t0, gen == cfg.num_generations,
        engine_label, layout,
    );
}
