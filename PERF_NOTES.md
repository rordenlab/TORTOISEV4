# PERF_NOTES — measured facts, harness commands, and traps

Companion to `plan_optimize.md`. This file is knowledge, not a plan: it is what someone
starting the optimisation work with no prior context needs, and it stays true regardless of
which optimisation is attempted.

Everything numeric here was measured on **this host** (RTX 4070 Ti SUPER, CUDA 13.0, driver
595.84, 32-thread CPU) unless stated. Re-measure before trusting any of it elsewhere.

---

## 1. Where the time actually goes

**MEASURED 2026-08-21** with the `[PROFILE]` stage timers (`src/main/tortoise_profile.h`),
binary `30d04142`, three `fast` runs on an idle machine. Named stages account for **99.2 %**
of wall time.

| stage | P1 | P2 | P3 | median | % of median TOTAL |
|---|---:|---:|---:|---:|---:|
| **DIFFPREP** | 382 | 375 | 382 | **382** | 54 % |
| — of which `DIFFPREP.Register` | 317 | 309 | 316 | **316** | **45 %** |
| **EPI (DRBUDDI)** | 257 | 277 | 297 | **277** | 39 % |
| — of which the DRBUDDI stage loop | 208 | 229 | 246 | **229** | 32 % |
| Denoising | 21 | 22 | 22 | 22 | 3 % |
| RigidToStructural | 17 | 17 | 16 | 17 | 2 % |
| FinalData | 6 | 6 | 6 | 6 | 1 % |
| Gibbs | 5 | 5 | 5 | 5 | 1 % |
| Import | 0.4 | 0.4 | 0.4 | 0.4 | — |
| **TOTAL** | 695 | 708 | 734 | **708** | |

**The bottleneck is DIFFPREP volume registration, not DRBUDDI.** `DIFFPREP.Register` is 45 % of
wall clock — larger than the entire DRBUDDI stage loop (32 %, which does reproduce the 33 % figure
previously recorded here). It runs three times per `fast` run: twice on the 138-volume `ap` dataset
(two epochs, ~154 s each) and once on the 10-volume `pa` dataset (~8 s).

Note the stability difference, which matters when grading a change: `Register` spans 309–317 s
across the three runs (2.6 %), while `EPI` spans 257–297 s (15 %). A DRBUDDI-side win needs more
repeats to be visible than a registration-side win of the same size.

Volume routing, printed by the same instrumentation as
`[PROFILE] DIFFPREP.RegisterSplit gpu_vols N cpu_vols M threads T`:

| dataset | volumes | GPU | ITK CPU | OMP threads |
|---|---:|---:|---:|---:|
| `ap` (up) | 138 | 48 | 90 | 31 |
| `pa` (down) | 10 | 10 | 0 | 1 |

`Nt = 31` here, so `max_t_per_pass = 15 + 31 - 1 = 45` and `npass = 4`; the fourth pass holds only
3 volumes, which is under the GPU quota and therefore goes entirely to the GPU. That is where
48/90 comes from rather than the 45/93 that §4's arithmetic predicts for `Nt = 32`.

**Wall clock by dataset** (CUDA / WebGPU, current binaries):

| dataset | CUDA | WebGPU | peak GPU CUDA | peak GPU WebGPU |
|---|---:|---:|---:|---:|
| fast (100×100×58, 138+10 vols) | 12:09 | 12:51 | 4378 MiB | 4084 MiB |
| medium (140×140×81, 102+102) | 22:27 | 20:08 | 5992 MiB | 5685 MiB |
| slow (140×140×92, 297+3) | 44:22 | 42:11 | 2672 MiB | 2660 MiB |

WebGPU is at or slightly better than parity, and **below** CUDA on peak GPU in every case —
it has no pitched-allocation row padding and no `cudaArray` duplication.

---

## 2. There is no GPU utilisation data. At all.

`benchmark/<ds>/<backend>/gpu_samples.csv` is `timestamp,pid,used_memory` — **memory only**,
1 Hz, from `nvidia-smi --query-compute-apps`. The column that looks like utilisation is the
PID.

### 2.1 MEASURED 2026-08-21 — utilisation at last, and it kills the pooling hypothesis

One `fast` run (binary `877652a1`, post-M1) sampled at 200 ms with
`nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.sm`, aligned to the
`[PROFILE]` stage boundaries by anchoring the sampler's last sample to the process end:

| phase | s | mean util | <10 % | >90 % |
|---|---:|---:|---:|---:|
| Denoise + Gibbs | 28 | 0.0 % | 100 % | 0 % |
| DIFFPREP | 304 | 34.8 % | 41 % | 0 % |
| EPI/DRBUDDI early | ~118 | 51.0 % | 36 % | 16 % |
| **EPI/DRBUDDI late (stages 25+)** | ~146 | **93.3 %** | 5 % | **95 %** |
| Rigid + FinalData | 28 | 0.0 % | 100 % | 0 % |
| whole run | 623 | 48.4 % | 37 % | 25 % |

**The expensive DRBUDDI stages are kernel-bound, not host-bound.** 95 % of samples above 90 %
means removing *every* `cudaMalloc`, `cudaFree` and `cudaDeviceSynchronize` in that region could
recover at most the missing 7 % — ~10 s, **1.6 % of the run**. §3's "~22 000 allocate/free pairs"
is a real count but not a real cost where it matters, and the WebGPU port's experience does
**not** transfer: a Dawn queue round-trip is far more expensive than a CUDA sync.

**That bound was too optimistic about the ceiling and is REFINED, not overturned, by §2.2.**
Utilisation alone says "at most 7 % recoverable in the late stages"; it cannot say what the idle
is *made of*, and averaged over all of DRBUDDI the idle is 25 %, not 7 %. See §2.2 — an `nsys`
trace shows allocation is ~80 % of that idle, which makes candidate B worth roughly 5.7 %
end-to-end. Do not cite this section alone to dismiss pooling.

Caveat on what `utilization.gpu` means: percent of sampled intervals in which **any** kernel was
resident. It is not occupancy, and back-to-back tiny kernels with sub-sample gaps read as busy.
It bounds host-stall time from above, which is exactly what was needed here.

The two 0 % phases (56 s, 9 % of the run) are pure CPU — denoising, Gibbs, rigid-to-structural
and final data generation never touch the GPU at all.

### 2.2 MEASURED 2026-08-21 — `nsys`: allocation is ~80 % of DRBUDDI's idle GPU

`nsys profile -t cuda --delay=430 --duration=45` on the post-M1 binary, a 45 s window inside the
DRBUDDI stage loop. (`nsys` SIGTERMs the process when the window closes — `rc=143` is expected,
and the run produces no output, so its wall time is not a timing sample.)

GPU kernel time in the window: **36.58 s of 45 s = 81 % busy**, leaving **8.4 s idle**. Host-side
CUDA API time in the same window, single-threaded and summing to the full 45 s:

| API | calls | total | note |
|---|---:|---:|---|
| `cudaDeviceSynchronize` | 335 403 | 37.17 s | waiting for kernels — *this is the productive part* |
| `cudaFree` | 108 133 | **3.61 s** | |
| `cudaMalloc3D` | 37 527 | **2.22 s** | pitched image buffers |
| `cudaMemcpyToSymbol` | 153 199 | 0.61 s | |
| `cudaLaunchKernel` | 298 365 | 0.56 s | |
| `cudaMemcpy` | 70 626 | 0.38 s | reduction results, D2H |
| `cudaMemset3D` | 29 628 | 0.09 s | |
| `cudaMalloc` | 70 626 | 0.08 s | flat reduction scratch |

**Allocation (`cudaFree` + `cudaMalloc3D` + `cudaMemset3D` + `cudaMalloc`) = 6.0 s, which is
~80 % of the 8.4 s of idle GPU.** Non-allocation overhead is only ~1.5 s.

**Do not read that as 5.7 % end-to-end — MEASURED, it is not.** The 6.0 s is dominated by the
*pitched* traffic (`cudaMalloc3D` 2.22 s plus most of `cudaFree`). Reusing only the **flat**
reduction scratch removes ~0.08 s of `cudaMalloc` and roughly two thirds of `cudaFree`'s
call count, i.e. ~2.4 s per 45 s ⇒ ~14 s over DRBUDDI ⇒ **~2 % end-to-end, which is inside the
run-to-run spread.** A `fast` run with the flat-scratch change measured TOTAL 640 s against an
M1 median of 632 s — indistinguishable. The change is retained because it is free and correct,
not because it is a measurable win. The pitched buffers are where DRBUDDI's remaining allocator
cost lives, and they are blocked by §2.3.

Note `cudaFree` is called 108 133 times against 70 626 `cudaMalloc` + 37 527 `cudaMalloc3D` =
108 153 allocations: the counts match, so nothing is leaking and nothing is being pooled today.

**Two distinct targets, with very different risk:**

1. **Flat reduction scratch** — 70 626 `cudaMalloc`/`cudaFree` pairs of *identical size*
   (`sizeof(float)*gSize`). A single reused buffer removes them all. Bit-exact by construction:
   same size, and every element is written by the first reduction kernel before the second reads
   it, so there is no stale-read path.
2. **Pitched image buffers** — 37 527 `cudaMalloc3D`. Higher value but this is where §5.5 bites:
   reductions iterate over `pitch/sizeof(float)*sy*sz`, i.e. **over the row padding**, and
   `Allocate()` memsets only `extent.width` per row. Today that padding happens to read as zero.
   A pool that hands back a previously-used block would hand back *stale* padding, and the
   pitch-spanning reductions (`SumImage`, `AddToUpdateField`, `PreprocessImage`,
   `InvertField`) would silently change their results. Pooling these is only safe if the full
   `pitch*sy*sz` region is zeroed on reuse — which also removes the latent §5.5 hazard, since
   CUDA never guaranteed that padding was zero in the first place.

### 2.4 MEASURED 2026-08-21 — DIFFPREP registration is 88 % allocator, 12 % compute

The same `nsys` treatment with the window moved into DIFFPREP registration
(`--delay=210 --duration=45`, post-M1 binary). Only the GPU thread issues CUDA calls during
registration — the other 30 OMP threads are in ITK — so these host times are single-threaded and
directly comparable to the 45 s of wall clock:

| | time | calls |
|---|---:|---:|
| **GPU kernel time** | **5.27 s** | — |
| `cudaFree` | 14.89 s | 1 430 576 |
| `cudaDeviceSynchronize` | 8.01 s | 1 335 905 |
| `cudaMalloc` | 6.44 s | 1 334 480 |
| `cudaLaunchKernel` | 4.05 s | 1 335 193 |
| `cudaMemcpy` | 3.47 s | 667 241 |
| `cudaMalloc3D` + memsets | 3.47 s | ~573 000 |

**The GPU is computing for 11.7 % of the window.** `cudaMalloc` + `cudaFree` alone are 21.3 s —
**47 % of registration wall clock** — across **1.33 million allocations in 45 seconds**
(~30 000/s).

The kernels identify the caller unambiguously: `QuadraticTransformImage_kernel` plus the joint
histogram/entropy chain, 95 320 instances each — the **mutual-information metric**
(`compute_entropy.cu`) driving the registration optimiser. 1 334 480 / 95 320 = **exactly 14
`cudaMalloc` per metric evaluation**, matching the 14 flat malloc sites in that file. Per
evaluation: **55 µs of kernel against 423 µs of API overhead.**

**This is the largest single opportunity in the pipeline, and it compounds with M1.** Registration
is 38 % of the post-M1 run. Removing the per-evaluation allocation should cut GPU per-volume cost
several-fold, which in turn *raises the optimal `GPU_CPU_ratio`* (§4.1): at ~0.3 s/volume the
balance point moves to ~108 GPU volumes with a single CPU round, i.e. a registration critical path
near 60 s against today's 120 s. That is worth roughly **19 % end-to-end on its own**, on top of
M1's 10.7 %.

Note this is the same defect class as §2.2 but an order of magnitude larger, and it was invisible
to the utilisation sampler alone: DIFFPREP's 34.8 % mean (§2.1) mixes the idle GPU thread with the
CPU-only phases, and only the trace separates "GPU has no work" from "GPU has work but the host is
in the allocator".

### 2.3 §5.5 IS NOT DORMANT — it fired, 2026-08-21. Read this before touching allocation.

§5.5 records that CUDA's reductions read uninitialised row padding, and concludes the divergence
is "dormant, and that is measured, not assumed", on the evidence that every golden vector replays
bit-exactly. **That evidence was misread.** Bit-exact replay does not show the padding is zero; it
shows the padding is *the same* on every run, because the process makes the *same sequence of
allocations* every time.

Demonstrated: removing 19 `cudaMalloc`/`cudaFree` pairs per reduction — a change that touches no
arithmetic whatsoever — moved the allocator's subsequent addresses, so pitched buffers landed on
different residue, and `gpu_replay` went from 132/132 to **126/132**. All six failures were
`InvertField`, whose reductions are written as
`ScalarFindMax<<<...>>>(scale_image.ptr, scale_image.pitch/sizeof(float)*sy*sz, ...)` — reading
the padding explicitly. Magnitude `rel = 0.08`, i.e. **8 %**, not a rounding difference.

`medium` failed the same way; `slow` did not.

**Consequence: "padding reads as zero" is a property of one exact allocation sequence, not of the
allocator.** Any change to allocation traffic — pooling, `cudaMallocAsync`, reordering, removing a
temporary — can change numerical results while leaving every kernel untouched. This is the single
biggest hazard for the remaining optimisation work, and it is invisible to code review.

**The fix, applied:** zero the *whole* pitched allocation rather than `extent.width` per row:

| where | change |
|---|---|
| `cuda_image.cxx` `CUDAIMAGE::Allocate()` | `cudaMemset3D(p,0,extent)` → `cudaMemset(p.ptr,0,p.pitch*sy*sz)` |
| `cuda_image_utilities.cu` `InvertField` | same, for its two raw `cudaMalloc3D` buffers (`scale_image`, `composed_field`) |

After both, all three record sets replay bit-exactly again (132 / 118 / 106) **with** the
allocation change in place. That the numbers return to the originals is the proof that the
reference's padding really was zero for its own sequence — the fix reproduces it deterministically
instead of by luck.

**Still outstanding:** there are **55 raw `cudaMalloc3D` sites** outside `cuda_image.cxx` (30 in
`compute_metric.cu` alone), 48 of them paired with a `cudaMemset3D(...,0,extent)` that leaves
padding untouched. Only `InvertField`'s two are currently read over pitch, so the rest are latent
in exactly the way §5.5 wrongly believed all of them were. **Any future pooling of pitched buffers
must zero full pitch at every one of those sites first.** Note that a handful deliberately memset
to byte **1**, not 0 (§5.5, metric images) — those must not be converted.

**Stage wall timers now exist** (§1) — `src/main/tortoise_profile.h`, always on, one
`[PROFILE] <name> <seconds>` line per scope on stderr. They are *wall* timers, so they still
do not separate GPU time from CPU work inside a stage; they were named `TORTOISE_GPU_PROFILE`
in the port plan and are not gated on an environment variable, because the stage prints they
replaced were already unconditional. Do not cite the per-stage table in `benchmark/README.md`
as evidence of GPU time.

To get real utilisation:
```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,clocks.sm \
           --format=csv,noheader,nounits -lms 200 > util.csv
```

---

## 3. Allocation and synchronisation, counted

```
205   cudaMalloc / cudaFree call sites in src/cuda_src/
        compute_metric.cu          78
        cuda_image_utilities.cu    37
        compute_entropy.cu         28
        cuda_image_utilities.cxx   18

117   cudaDeviceSynchronize() calls in src/cuda_src/*.cu
        cuda_image_utilities.cu    45
        compute_metric.cu          41
        compute_entropy.cu         18
```

Six allocate/free pairs per metric evaluation. `fast` runs 4 metrics × 926 DRBUDDI
iterations ⇒ **~22 000 allocate/free pairs in DRBUDDI alone**, before image allocations.

**The WebGPU port already proved the payoff of fixing this class of problem.** It went from
**12.6× slower than CUDA to parity** through five changes, in rough order of effect:

1. removing the per-dispatch host wait (largest single win),
2. device-side buffer clear instead of uploading a host vector of zeros,
3. device-to-device copy instead of a host round-trip,
4. device-side min/max instead of downloading every voxel to reduce on the host,
5. caching small readback staging buffers.

**CUDA still has the pattern (1) fixed in the port.** It is plausible that CUDA is now behind
its own port on synchronisation discipline. That is the first hypothesis worth testing.

---

## 4. Why GPU utilisation looks low during DIFFPREP — by design

`src/main/DIFFPREP.cxx:589`:
```c
const int GPU_CPU_ratio = 15;   // ratio of how much GPU is faster than a CPU per volume
```
and `:599`: `max_t_per_pass = NGPUs*GPU_CPU_ratio + Nt - NGPUs`.

On `fast` this routes most volumes to the ITK CPU path. The split is pure index arithmetic and
fully deterministic. **Measured** (`NGPUs=1`, `Nt=31`): **48 volumes to the GPU and 90 to the
CPU**, not the 45/93 that `Nt=32` would give — see §1. `Nt` is what OpenMP actually hands back,
so do not assume it equals the core count.

So the GPU is deliberately given a minority of registration work while 31 CPU threads take
the rest. **"Use the GPU more" is not automatically faster** — the CPU path is 31-way
parallel, the GPU path is serial. `GPU_CPU_ratio` is a hand-tuned guess; if GPU operations
get materially faster it becomes wrong, and re-deriving it from a measured sweep may be the
largest end-to-end win available.

Second benefit: the ITK CPU path is the source of the pipeline's non-reproducibility (below),
so moving work to the GPU makes results *more* deterministic.

### 4.1 MEASURED 2026-08-21 — `GPU_CPU_ratio = 15` is wrong by more than 2×

Per-thread timers inside the registration loop
(`[PROFILE] DIFFPREP.RegisterThread <thr> nvols <n> <seconds>`, thread 0 is the GPU),
binary `a2b6d94d`, `fast`:

| path | volumes/thread | wall | per volume |
|---|---:|---:|---:|
| GPU (thread 0) | 48 | 46.8 s / 63.6 s | **≈1.3 s** |
| ITK CPU (threads 1–30) | 3 | 99.1 – 157.1 s, mean 136.8 | **≈45.6 s** |

**The measured ratio is ≈35×, not 15.** The registration call ends when the *slowest* thread
does, so the critical path is the CPU threads at ~137–157 s while the GPU has been idle since
~60 s. That idle GPU is ~13 % of total pipeline wall clock.

Consequences for optimisation planning:

- **Speeding up the CUDA registration kernels buys nothing on its own.** The GPU path is 9 % of
  wall clock and is *off* the critical path; halving it saves zero seconds until the split is
  rebalanced. Sync/allocator work (§3) pays off in DRBUDDI, which is GPU-bound, not here.
- **Rebalancing is where the time is.** Modelling critical path as
  `max(G*1.3, ceil((138-G)/30)*45.6)` over the ratios the formula can actually produce:

  | `GPU_CPU_ratio` | GPU vols | CPU vols | CPU rounds | predicted critical path |
  |---:|---:|---:|---:|---:|
  | 15 (current) | 48 | 90 | 3 | **137 s** |
  | 20 | 60 | 90 | 3 | 137 s |
  | **30 or 35** | **78** | **60** | **2** | **≈104 s** |
  | 45 | 90 | 48 | 2 | 120 s |
  | ∞ (all GPU) | 138 | 0 | 0 | 184 s |

  Optimum is ≈30–35, worth ~33 s on each of the two `ap` calls ⇒ **~9 % end-to-end**. Note 20
  changes nothing: `npass` drops to 3 but the CPU still gets 90 volumes.
- The two compound. Making the GPU path faster moves the optimum ratio *up*: at 0.65 s/volume
  the balance point is ≈96 GPU volumes and ≈64 s, worth ~21 % end-to-end.
- **This is not arithmetic-neutral.** It moves volumes between the CUDA and ITK implementations,
  which is a scheduling-policy change, not a bit-exact one (plan_optimize.md §2C). It moves
  results in the *same* direction as the existing run-to-run nondeterminism, and toward *more*
  determinism, since the GPU path is the reproducible one.

**Prediction for `medium` and `slow`, recorded before either was run** (routing is pure index
arithmetic, so the split is exact; the times scale the `fast` per-volume costs by matrix size).
`ceil(C/30)` rounds is what actually sets the CPU cost, which is why a bigger GPU share does not
always help:

| dataset | ratio | GPU vols | CPU vols | CPU rounds | predicted critical path |
|---|---:|---:|---:|---:|---:|
| `medium` (102 vols, 2.7× `fast` voxels) | 15 | 42 | 60 | 2 | 246 s |
| | **35** | 70 | 32 | 2 | **246 s — neutral** |
| `slow` (297 vols, 3.1× `fast` voxels) | 15 | 105 | 192 | 7 | 987 s |
| | **35** | 175 | 122 | 5 | **≈705 s — ~280 s saved** |

So `medium` should be unchanged (the CPU still needs 2 rounds either way; only the GPU share
grows, and it stays under the CPU cost) and `slow` should gain ~10 %. If `medium` comes back
*slower*, the per-volume GPU cost scales worse with matrix size than assumed and the ratio needs
to be dataset-aware rather than a constant.

**OUTCOME (2026-08-21): the prediction was right about the mechanism and wrong about the size,
because it was overtaken by M3 and M5.** It modelled `GPU_CPU_ratio = 35` at the *then*-current
GPU speed, under which `medium` really would have stayed at 2 CPU rounds and gained nothing.
What actually shipped made the GPU path 1.8x faster (M3) and replaced the constant with a runtime
split (M5), and the queue got `medium` down to **one** CPU round:

| dataset | predicted (ratio 35) | measured (M5 dynamic) |
|---|---|---|
| `medium` | 70 GPU / 32 CPU, 2 rounds, "neutral" | **72 GPU / 30 CPU, 1 round, 22:27 -> 16:39 = −25.8 %** |

Measured on `medium`: `t_gpu` 1.32–1.43 s against `t_cpu` 60–67 s, a real ratio of ~45x (versus
~85x on `fast` — the gap does narrow with matrix size, which is exactly why a single constant was
never going to fit all three datasets). Peak GPU 5992 -> 6152 MiB, +2.7 %.

---

## 5. The pipeline is not run-to-run reproducible, and this shapes all measurement

Two runs of the same CUDA binary on the same input differ in **~35 % of voxels** (`fast`;
~55 % on `medium`, ~78 % on `slow`). This is not a defect in the GPU code — it is the ITK CPU
registration path, whose work-unit count depends on how many threads are available.

**Measured CUDA-vs-CUDA floors** (same binary, same input):

| dataset | pairs | `max\|diff\|/p99` | Pearson r |
|---|---:|---|---|
| fast | 6 (4 runs) | 3.046 – 6.319 | 0.999275 – 0.999833 |
| medium | 3 (3 runs) | 1.720 – 2.737 | 0.999371 – 0.999910 |
| slow | 3 (3 runs) | 3.346 – 5.402 | 0.998723 – 0.999517 |

**Wall clock is noisy too:** CUDA's `fast` spans **11:49–13:10** over four runs of one binary
(sd 36 s, ~11 %).

Two consequences that have cost this project real time:

- **Report n ≥ 3 runs and the spread.** Four successive speedup figures (1.14×, 1.16×, 1.18×)
  and three Pearson r values were published here and every one had to be retracted, each
  being a single-run comparison of an 11 %-spread quantity.
- **Measure on an idle machine.** Compressing a file during a floor measurement changed the
  result, via the ITK thread-count mechanism above.

Setting `OMP_NUM_THREADS=1` makes the pipeline deterministic (every volume goes to the GPU
path) but single-threads everything. There is also a `TORTOISE_DETERMINISTIC_GPU` build
switch. Both are validation tools, not performance configurations.

---

## 6. The accuracy apparatus you must not break

Correctness is established **per kernel**, not end-to-end.

- **381 golden-vector records**: 132 `fast` + 118 `medium` + 106 `slow` + 25 synthetic.
  Each holds `{op, params, inputs, outputs}` captured at a CUDA host wrapper; the outputs
  *are* the CUDA reference.
- **`gpu_replay --all <dir>` must be BIT-EXACT.** This is the anchor for everything. If it
  stops being bit-exact, the change was not arithmetic-neutral.
- **`webgpu_replay`** grades the port against those records and against 25 analytic
  references (needed because CUDA's texture sampler is itself lossy).
- **`revalidate.sh`** runs all 15 gates in one command.

**Which optimisations are safe:**

| safe (bit-exact) | not safe (changes results) |
|---|---|
| memory pooling, allocation reuse | changing summation order |
| removing `cudaDeviceSynchronize` | fusing reductions that sum |
| streams / overlap | anything altering FP contraction |
| device-side copy instead of host round-trip | changing kernel block sizes *if* it changes reduction order |
| fusing **min/max** (order-independent) | |

Floating-point addition is not associative. A reduction-order change breaks bit-exact replay
and destroys the anchor. If one is ever justified it needs explicit sign-off and full
recapture of all three datasets (~80 minutes of GPU time).

**Row padding is read by several reductions, and that is NOT dormant — see §2.3.** Removing
allocation traffic (not arithmetic) moved `InvertField` by 8 % and broke bit-exact replay. Zero
the full `pitch*sy*sz` region, not `extent.width` per row, before any allocation change.

**Never relax a tolerance to make an optimisation pass.** Every tolerance here was derived
from a measurement — the elementwise gate is 1e-5 because the same CUDA source built with
`-fmad=false` disagrees with itself by 2.67e-6.

---

## 7. Harness commands

```bash
cd /home/chris/src/TORTOISEV4
source benchmark/scripts/gpu_env.sh          # ALWAYS - pins to the discrete NVIDIA GPU

# build (see CLAUDE.md 2.4 for the full COMMON flags)
$CMAKE --build build_cuda   -j30
$CMAKE --build build_webgpu -j30

# the anchor - MUST be bit-exact
bin/gpu_replay    --all benchmark/fast/CAPTURE/golden_vectors     # 132
bin/gpu_replay    --all benchmark/slow/CAPTURE/golden_vectors     # 106

# the port
bin/webgpu_replay --all benchmark/fast/CAPTURE/golden_vectors     # 132
bin/webgpu_replay --all benchmark/medium/CAPTURE/golden_vectors   # 118
bin/webgpu_replay --all benchmark/slow/CAPTURE/golden_vectors     # 106
bin/webgpu_replay --all benchmark/synthetic_vectors               #  25

# everything, 15 gates
benchmark/scripts/revalidate.sh --compare

# one timed run + memory sampling
benchmark/scripts/run_cuda_reference.sh fast CUDA
benchmark/scripts/finalize_run.sh       fast CUDA

# per-op microbenchmark (READ THE CAVEATS IN CLAUDE.md 7.1 BEFORE QUOTING IT)
bin/gpu_replay --bench    benchmark/fast/CAPTURE/golden_vectors
bin/webgpu_replay --bench benchmark/fast/CAPTURE/golden_vectors
```

**`benchmark/` is gitignored in its entirety** (~148 GB) and copied between machines. A fresh
clone has none of it, and every `benchmark/...` path cited in the docs will resolve to
nothing.

---

## 8. Traps that have already cost time

**Silent-success failures.** Three separate scripts reported success while doing nothing:
`run_cuda_reference.sh` documented capture hooks it never enabled; `revalidate.sh` printed
`ALL GATES PASSED` while blind to 106 records; `drbuddi_isolated.sh snapshot` printed
"fixture frozen" after 109 permission-denied errors. **Check exit status, and verify the
operation had its intended effect.** All three are fixed; the pattern is not.

**Stale binaries.** Every run records `exe_sha256` in its `provenance.json`, and the gates
refuse to grade output produced by a different binary. Rebuild → re-run the affected
evidence. A comment-only source edit usually produces an identical object file (verify by
comparing the `.o`), but **any change to a `.wgsl` file changes the embedded shader string
and therefore the binary**.

**Don't edit a running bash script.** Bash reads by byte offset; editing mid-run executes
garbage. Confirm nothing is executing it — resolve `/proc/<pid>/exe`, and note that
`pgrep -f <script>` also matches the shell running your check.

**Pass paths positionally, not via environment variables.** An env var can be forgotten at
any call site and fails at the *end* of a long run. This wasted a completed 12-minute
pipeline run.

**`--bench` numbers are a poor proxy for pipeline cost.** It times each op once with fresh
host transfers; the pipeline runs many dispatches on resident data. One op
(`PreprocessImage`, 7.19 M voxels) is 75 % of all compared output floats in the suite and is
a one-shot setup pass in the real pipeline. Trust `ITERATION_TIME` and stage timers.

---

## 9. Sizing

Peak GPU is driven by **metric count × matrix size**, and the coefficient spans ~18 % across
measured configurations. Plan with the maximum, **1 887 MiB per metric-megavoxel**, not the
mean. Worst case — 4 metrics at `slow`'s matrix — is **≈13.3 GiB against a 16 GiB card**.

This matters for pooling: the current per-operation allocate/free pattern is *why* peak GPU
sits below CUDA's. A pool trades memory for latency. Sample `nvidia-smi` after every pooling
change.

---

## 10. Release record

Per `plan_optimize.md` §5. One entry per accepted milestone.

### M1 — `GPU_CPU_ratio` 15 → 35 (accepted 2026-08-21)

```
date        2026-08-21
binary      baseline 30d04142  ->  M1 877652a1
change      src/main/DIFFPREP.cxx: const int GPU_CPU_ratio = 15 -> 35
dataset     fast, CUDA, idle machine, n=3 per configuration
wall        baseline 11:35 / 11:48 / 12:14   median 11:48  (708 s)
            M1       10:20 / 10:32 / 10:45   median 10:32  (632 s)   -10.7 %
Register    baseline 317 / 309 / 316         median 316 s
            M1       236 / 242 / 241         median 241 s           -23.7 %
EPI/DRBUDDI baseline median 277 s -> M1 median 269 s   (unchanged, as predicted)
peak GPU    4380 MiB -> 4378 MiB             (unchanged)
replay      gpu_replay --all fast: 132/132 bit-exact, before and after
decision    ACCEPTED. Gate was >=10 % stage median improvement + bit-exact replay.
```

The two configurations' TOTAL ranges are **disjoint** (695–734 vs 620–644), so this is not a
draw from the 11 % run-to-run spread that has produced retracted figures here before.

**What it is not:** this is a scheduling-policy change, not a bit-exact one. It moves 30 volumes
per registration call from the ITK CPU path to the CUDA path, so end-to-end output moves — in the
same direction and within the same band as the existing run-to-run nondeterminism, and toward
*more* determinism. Kernel golden vectors are untouched, which is why `gpu_replay` is still
bit-exact. Approved explicitly by the repo owner before implementation.

**Still open after M1:** end-to-end is 10.7 % against a 15 % target. The remaining budget is the
DRBUDDI stage loop (median 269 s, 43 % of the new total) — see §4.1 for why speeding up the CUDA
registration kernels specifically would still buy nothing.

### M2/M3 — allocation reuse (accepted 2026-08-21)

```
date        2026-08-21
binary      877652a1 (M1)  ->  c2cf05a7
changes     src/cuda_src/reduction_scratch.h                 NEW: per-thread, per-call-site scratch
            compute_metric.cu, cuda_image_utilities.cu       18 reduction dev_out sites reuse it
            compute_entropy.cu                               14 MI-metric sites reuse it (slots 0..13)
            cuda_image.cxx, cuda_image_utilities.cu          zero FULL pitch, not extent.width (2.3)
dataset     fast, CUDA, idle machine, n=3
wall        M1  10:20 / 10:32 / 10:45   median 10:32  (632 s)
            M3  10:00 /  9:58 /  9:22   median  9:58  (598 s)   -5.4 % vs M1
Register    241 s -> 233 s
EPI/DRBUDDI 269 s -> 242 s     (DRBUDDI shares ComputeJointEntropy, so it gained too)
GPU thread  78 volumes in 75/98 s -> 41/54 s   = 1.26 -> 0.687 s per volume, 1.8x
replay      132/118/106 bit-exact on all three datasets
decision    ACCEPTED.
```

**Cumulative against the pre-optimisation baseline: 708 s -> 598 s = 15.5 %**, which meets the
plan's 15 % target. Ranges are disjoint (695–734 vs 562–600).

Two things worth separating, because they were nearly conflated:

- The **DRBUDDI** flat-scratch reuse is worth ~2 % and is inside the noise band (§2.2). It is kept
  because it is free and correct, not because it was measured as a win.
- The **DIFFPREP MI-metric** reuse is the real change: it is what took the GPU registration path
  from 1.26 to 0.687 s per volume, exactly as §2.4's trace predicted.

**And it immediately invalidated `GPU_CPU_ratio = 35`.** With the GPU path 1.8x faster the CPU
threads became the critical path again (GPU 54 s vs CPU 115 s), so most of the win was left on
the table — the end-to-end gain was only 5.4 % against a stage gain of 1.8x. That is the third
time the constant needed retuning, and the reason M5 replaced it with a measured runtime split
rather than a fourth hand-picked number.

### M5 — dynamic GPU/CPU load balancing (accepted 2026-08-21)

```
date        2026-08-21
binary      c2cf05a7 (M3)  ->  1dda0221
change      src/main/DIFFPREP.cxx: GPU_CPU_ratio, max_t_per_pass, npass and the
            gpu_ids_per_thread/my_threads construction all DELETED; volumes are pulled
            from a shared atomic counter, GPU-ness bound to the thread index.
dataset     fast, CUDA, idle machine, n=3
wall        M3  10:00 / 9:58 / 9:22   median  9:58  (598 s)
            M5   8:03 / 8:02 / 8:26   median  8:03  (482 s)   -19.4 % vs M3
Register    233 s -> 128 s
split       108 GPU / 30 CPU on all three runs, DISCOVERED not configured
            measured t_gpu 0.507-0.569 s, t_cpu 45.6 s  =>  real ratio ~85x
pa dataset  10 volumes, 10 GPU / 0 CPU, 3.71 s  (bootstrap behaves)
peak GPU    4380 MiB -> 4398 MiB  (+0.4 %, gate was <=5 %)
replay      132/118/106 bit-exact on all three datasets
decision    ACCEPTED.
```

**The queue reproduced the hand-computed optimum without being told it.** 108/30 is exactly the
split `GPU_CPU_ratio = 54` would have produced, arrived at from measured per-volume times. The
constant is gone rather than retuned a third time.

**The feared nondeterminism did not materialise, and the reason is structural.** The split was
108/30 in every run. Because the retire rule compares `remaining * t_gpu` against `t_cpu` and the
two differ by ~85x, the crossover is sharp: it takes a large perturbation to move the decision by
even one volume. It also routes *more* volumes (108 vs 78) to the GPU path, which is the
reproducible one, so end-to-end output should be **more** stable than before, not less.

**Why the retire rule is not optional.** A plain work queue is *worse* than the tuned static split
here. At 0.51 s vs 45.6 s per volume, a CPU thread that picks up a volume near the end adds its
whole 45 s as a tail: modelling front/back pulling gives ~114 s against the static split's 74 s.
The rule — take another volume only if the GPU could not clear everything remaining in less time
than one CPU volume costs — is what makes the queue competitive, and it is also what makes it
adaptive: with a slow GPU the CPU threads keep helping to the very end.

---

## 11. Cumulative result

| configuration | binary | wall median (n=3) | TOTAL | Register | vs baseline |
|---|---|---:|---:|---:|---:|
| baseline (`GPU_CPU_ratio=15`) | `30d04142` | 11:48 | 708 s | 316 s | — |
| M1 ratio 35 | `877652a1` | 10:32 | 632 s | 241 s | −10.7 % |
| M2/M3 allocation reuse | `c2cf05a7` | 9:58 | 598 s | 233 s | −15.5 % |
| **M5 dynamic split** | `1dda0221` | **8:03** | **482 s** | **128 s** | **−31.9 %** |

**1.47x end-to-end on `fast`**, against a 15 % target. Registration alone went 316 s -> 128 s
(2.5x). `gpu_replay` stayed bit-exact on all 381 records at every step, and peak GPU moved +0.4 %.

The three wins in order of size: fixing the CPU/GPU split (twice, then removing it), and removing
1.33 M allocations per 45 s from the MI metric. **No kernel arithmetic was changed anywhere.**

### 11.1 `medium` and `slow` confirmation (plan M4 gate)

One run each on binary `1dda0221`, against the baselines recorded in §1:

| dataset | baseline | M5 | change | peak GPU | Register |
|---|---:|---:|---:|---|---:|
| `fast` (n=3) | 11:48 | **8:03** | **−31.9 %** | 4380 → 4398 MiB (+0.4 %) | 316 → 128 s |
| `medium` | 22:27 | **16:39** | **−25.8 %** | 5992 → 6152 MiB (+2.7 %) | — |
| `slow` | 44:22 | **33:11** | **−25.2 %** | 2672 → **2946 MiB (+10.3 %)** | — |

No dataset regressed. The split the queue chose, and the ratio it measured, differ per dataset —
which is the empirical answer to whether one constant could ever have served all three:

| dataset | volumes | split chosen | `t_gpu` | `t_cpu` | implied ratio |
|---|---:|---|---:|---:|---:|
| `fast` | 138 | 108 / 30 | 0.51 s | 45.6 s | **~85x** |
| `medium` | 102 | 72 / 30 | 1.32–1.43 s | 60–67 s | **~45x** |
| `slow` | 297 | 209 / 88, then 183 / 114 | 1.53–1.81 s | 89–94 s | **~50–58x** |

Note `slow`'s two registration calls chose *different* splits (209/88 then 183/114) as the
estimates refined — the adaptivity is doing real work, not just reproducing a fixed answer.

**`slow` breached the plan's 5 % peak-memory gate (+10.3 %), and that was fixed rather than
waived.** Cause: of the 14 cached MI-metric buffers, two are sized `Nblocks * Nbins^2` and
therefore scale with the image, so retaining them between evaluations forced them to coexist with
allocations that previously had that space free. The other twelve are `Nbins`-sized and
image-independent. The two large ones were returned to `cudaMalloc`/`cudaFree` (binary
`9b0d50fb`), which still leaves 4 allocation calls per evaluation against the original 29 — an
86 % reduction rather than 100 %.

**Do not "restore" those two to the cache for symmetry.** The memory gate is why they are
different, and §9's worst case (4 metrics at `slow`'s matrix, ~13.3 GiB against a 16 GiB card)
is why the gate matters.


### 11.2 Peak GPU on `slow` — UNEXPLAINED, and one hypothesis is ruled out

`slow`'s peak GPU rose 2672 -> 2946 MiB (+10.3 %) somewhere across M1–M5, breaching the plan's
5 % gate. `fast` (+0.4 %) and `medium` (+2.7 %) are inside it, so it scales with something.

**Ruled out: retained MI-metric scratch persisting from DIFFPREP into DRBUDDI.** The obvious
story was that the cached `Nblocks*Nbins^2` histograms stayed resident through DRBUDDI, the most
memory-hungry phase. `CudaScratchRelease()` was added and wired into the end of the registration
loop so each thread drops its blocks. **Peak was still 2946 MiB** — identical. Wall time was also
identical (33:08 vs 33:11), so the release is free; it is retained as hygiene, but it is not the
cause.

Two candidates remain, neither tested:
- DRBUDDI's **own** `ComputeJointEntropy` scratch, which the registration-scoped release does not
  cover and which is sized for DRBUDDI's images.
- Allocation churn from routing ~2x as many volumes through the GPU path (209 vs 105 on `slow`),
  raising the driver's watermark without any single allocation being larger.

**Judgement, for the record:** 2946 MiB against a 16 GiB card, in exchange for 25 % less wall
clock, is a trade worth making — but §9's worst case (4 metrics at `slow`'s matrix, ~13.3 GiB)
would become ~14.6 GiB, cutting the margin from 2.7 GiB to 1.4 GiB. **That is the owner's call,
not a detail to wave through**, and it is the one gate in this work that is not met.


## 12. DRBUDDI per-iteration time is NOT a usable metric at n=3

**Measured 2026-08-22, and it invalidated four rounds of my own attribution.** Two consecutive
`fast` runs of the *same* binary (`b135c702`, `CUDA_I1` and `CUDA_I2`):

| run | iterations | sum of `ITERATION_TIME` | **s/iter** | EPI |
|---|---:|---:|---:|---:|
| I1 | 1150 | 266.7 s | **0.2319** | 314 s |
| I2 | 1097 | 189.8 s | **0.1730** | 237 s |

**A 34 % swing, same code, same input, back to back.** Iteration *counts* are stable (±3 %), so
this is per-iteration cost, not convergence.

What this cost: across four builds the median s/iter read 0.168 -> 0.207 -> 0.211 -> 0.232, which
looks like a clean monotonic regression and is not one. Three separate hypotheses were formed and
"confirmed" from n=1 measurements — that full-pitch memsets cost 40 s, then that padding-only
`cudaMemset2D` was the fix, then that the 8 added memsets were the culprit — and each was refuted
by the next run. The final step *increased* s/iter after work was **removed**, which is what
finally exposed the noise.

A simple arithmetic check would have pre-empted all of it: `nsys` measured `cudaMemset3D` at
**0.44 s per 45 s window**, so no change to memset behaviour can possibly cost 40 s. When a
measurement and an order-of-magnitude estimate disagree by 100x, the measurement is the thing to
doubt first.

**Rules that follow, for anyone optimising the DRBUDDI side:**

- **Never accept an EPI or s/iter delta below ~35 % at n=3.** The noise band swallows it.
- `DIFFPREP.Register` (spread ±1 s over nine runs) and the CPU-only stages (`Denoising`,
  ±0.3 s) *are* reliable. Prefer them as the signal.
- Interleave A/B runs rather than comparing against numbers measured hours earlier — the machine
  had been under continuous load for 7 h, and while `nvidia-smi` showed no thermal or power
  throttling, drift of unknown origin cannot be excluded across that span.
- Check the arithmetic before believing a regression.


## 13. Final configuration and what is actually established

Binary `840a3a88`. Replay: **132 / 118 / 106 fast/medium/slow + 25 synthetic, all bit-exact.**

**Changes kept, in order of measured value:**

| change | where | evidence |
|---|---|---|
| Dynamic GPU/CPU work queue (deletes `GPU_CPU_ratio`) | `DIFFPREP.cxx` | `Register` 233 -> 128 s |
| `GPU_CPU_ratio` 15 -> 35 (superseded by the above) | `DIFFPREP.cxx` | `Register` 316 -> 241 s |
| MI-metric allocation reuse (14 sites, 12 cached) | `compute_entropy.cu` | GPU path 1.26 -> 0.69 s/volume |
| `-march=x86-64-v3` (AVX2 + FMA) | `CMakeLists.txt` | Denoising 22.1 -> 15.6 s |
| Reduction scratch reuse (18 sites) | `compute_metric.cu`, `cuda_image_utilities.cu` | below noise; kept as free + correct |
| Full-pitch zeroing of every pitched allocation | 57 sites across `src/cuda_src/` | correctness (§2.3); cost below noise |
| Phase-scoped scratch release | `reduction_scratch.h`, `DIFFPREP.cxx` | memory hygiene; did NOT fix §11.2 |
| Stage wall timers | `tortoise_profile.h` + 3 files | the instrument everything else rests on |

**Established, with tight spreads:**

| | baseline | final | basis |
|---|---:|---:|---|
| `DIFFPREP.Register` | 316 s | **128–135 s** | 21 runs, ±1 s within group |
| Denoising | 22.1 s | **15.6 s** | 6 runs, ±0.3 s |
| `medium` wall | 22:27 | **16:39** | n=1 |
| `slow` wall | 44:22 | **33:11** | n=1 |
| GPU registration | 1.26 s/vol | **0.51 s/vol** | 9 runs |

**`fast` end-to-end: 708 s -> roughly 480–525 s.** Quote it as a **range**, not a median.
Everything from M5 onward is indistinguishable on TOTAL (§12), because DRBUDDI now dominates the
run and its per-iteration time swings 34 % between identical runs. A single point estimate here
would be exactly the over-claim this file has had to retract three times before.

**Not achieved / open:**

- **Peak GPU on `slow` is +10.3 %** (2946 vs 2672 MiB), the one plan gate not met. Cause unknown;
  the leading hypothesis was disproved (§11.2). Owner's call — see the margin analysis there.
- `revalidate.sh --compare` has not been run end to end. Every suite inside it has been run
  individually and passes, but the one-command gate and its end-to-end smoke comparison have not.
- LTO was never tried. Offered, deferred in favour of the arch flag.
- Pitched-buffer pooling in DRBUDDI (~3-4 % modelled) is now *unblocked* by the full-pitch
  zeroing, but §12 means it cannot be validated on `fast` at n=3.


## 14. Kernel launch geometry — the swapped arguments are a real defect, not cosmetics

**Measured 2026-08-22.** CLAUDE.md §0.1 records the `kernel<<<blockSize, gridSize>>>` naming
throughout `src/cuda_src/` as an upstream oddity where "behaviour is correct; only the naming is
wrong". The first half is true. The second is not: the arguments really are swapped, so the
**grid-shaped values are passed as block dimensions**.

With `BLOCKSIZE 32`, `blockSize=(32,32,32)` and `gridSize=ceil(dim/32)`:

| volume | effective blockDim | threads/block | in x |
|---|---|---:|---:|
| `slow` 140x140x92 | (5,5,3) | 75 | 5 |
| `fast` 100x100x58 | (4,4,2) | 32 | 4 |
| downsampled DRBUDDI stage ~25x25x15 | **(1,1,1)** | **1** | **1** |

A 32-lane warp therefore spanned 4 consecutive voxels across 8 pitch-separated rows: eight
scattered transactions where one coalesced access would do. **This independently explains §2.1's
51 % GPU utilisation in DRBUDDI's early (downsampled) stages** — there the launches were running
*one thread per block*. The `while(gridSize.x*y*z > 1024)` clamp beside each launch exists only
to keep the "grid" inside the threads-per-block limit, which is the tell.

**Fix:** `ElementwiseLaunch()` / `ElementwiseLaunchM()` give `block=(32,8,1)` (utilities) and
`(32,4,1)` (metrics, register-heavier), with the launch arguments in the correct order. Applied to
**46 elementwise launches** across `cuda_image_utilities.cu` and `compute_metric.cu`.

**Bit-exact by construction, and verified:** every converted kernel computes one output element
per thread from its own `(i,j,k)` with no reduction, so block shape cannot affect the result.
Reductions (`ScalarFindSum`/`ScalarFindMax`, `gSize/bSize`) were deliberately left alone — block
size there sets summation order (§6). All 381 records replay bit-exact after every build.

**Measured per kernel** (`nsys`, same 45 s DRBUDDI window, std-dev ~0.1 %):

| kernel | before | after | change |
|---|---:|---:|---:|
| `NegateImage_kernel` | 422.7 us | 257.3 us | **-39.1 %** |
| `computeFiniteDiffStructs` | 31.34 ms | 22.04 ms | **-29.7 %** |
| `UpdateInvertField_kernel` | 566.1 us | 447.5 us | **-20.9 %** |
| `ComputeFieldLocalNormImage` | 198.0 us | 192.9 us | -2.6 % |
| `ComposeFields_kernel` | 427.2 us | 422.2 us | -1.2 % |

`ComposeFields` barely moves because it is trilinear-interpolation-bound - scattered gathers
dominate, so write coalescing cannot help it. `NegateImage` gains most because it is pure
streaming. That the effect tracks each kernel's access pattern is the evidence the mechanism is
the one claimed.

**End-to-end (n=3), and unusually for DRBUDDI work, the ranges DO NOT overlap:**

| | pre-geometry | post-geometry |
|---|---|---|
| s/iter | 0.173 - 0.232 (med 0.206) | **0.137 - 0.170 (med 0.167)** |
| TOTAL | 484 - 562 (med 521) | **447 - 485 (med 474)** |

-19 % on s/iter, matching the ~21 % of kernel time nsys predicted. Two independent instruments
agreeing is what §12 said end-to-end alone could never establish. The variance also **shrank**
(34 % spread -> 2 %), consistent with the kernels becoming cleanly bandwidth-bound instead of
dependent on scheduling luck.

**Propagated to WebGPU.** The port had inherited the equivalent shape: `const uint32_t wg = 4`, a
4x4x4 workgroup with 4 invocations in x, in 12 places. NVIDIA subgroups are 32 wide under Vulkan
too, so it costs the same way. Changed to `(32,4,1)` in the 8 elementwise backends;
`reductions.wgsl` (the only shader using `var<workgroup>`) and `compute_entropy.cxx` were left
alone. All four WebGPU suites still pass: **132 / 118 / 106 / 25**.

**Not done:** six single-launch files (`warp_image`, `resample_image`,
`quadratic_transform_image`, `rigid_transform_image`, `gaussian_smooth_image`, `compute_mi_cuda`)
still use the swapped form. They are internally consistent, so they are correct and merely
suboptimal. `QuadraticTransformImage_kernel` is 30 % of DIFFPREP's *kernel* time - but DIFFPREP
registration is allocator-bound, not kernel-bound, so the end-to-end value is small.

**Trap for whoever finishes them:** one setup block in `cuda_image_utilities.cu` had a slightly
different `while` body and escaped the regex while its launch was swapped, leaving
`blockDim=(32,32,32)` = 32768 threads - an illegal launch. Convert the setup and the launch
together, and check `grep -c "dim3 blockSize(BLOCKSIZE"` reaches 0.


## 15. The dynamic split, demonstrated across two backends

The strongest evidence for M5 is not the `fast` speedup - it is that the **same code chose
different splits for CUDA and WebGPU on the same machine and the same data**, because the two
backends have genuinely different per-volume costs:

| | CUDA (`72219555`) | WebGPU (`d22fcefc`) |
|---|---:|---:|
| measured `t_gpu` | 0.51 s/volume | **1.36 s/volume** |
| measured `t_cpu` | 45.6 s/volume | 44.5 s/volume |
| implied ratio | ~85x | **~33x** |
| split chosen | **108 GPU / 30 CPU** | **77 GPU / 61 CPU** |
| `fast` wall | 7:54 | 10:39 (baseline 12:51, **-17 %**) |
| peak GPU | 4038-4396 MiB | 4084 MiB (unchanged) |

The WebGPU registration path is 2.7x slower per volume than CUDA's, so the queue gave it 31 fewer
GPU volumes. **A constant tuned on CUDA would have been wrong here by construction**: at
`GPU_CPU_ratio = 54` WebGPU would take 108 GPU volumes at 1.36 s = 147 s of critical path against
the ~105 s the queue actually achieves. That is on one machine with one CPU - the error across
different *hardware* would be larger, which is the case §4.1 could only argue in the abstract.

WebGPU gains less end-to-end (-17 % vs CUDA's -33 %) for the same reason: a slower GPU path leaves
it more CPU-bound, so less work can move off the ITK threads.

---

## 16. Final state

| dataset / backend | baseline | final | change |
|---|---:|---:|---:|
| `fast` CUDA (n=3) | 11:48 | **7:54** (447-485 s, med 474) | **-33 %** |
| `fast` WebGPU (n=1) | 12:51 | **10:39** | -17 % |
| `medium` CUDA (n=1) | 22:27 | 16:39 | -25.8 % |
| `slow` CUDA (n=1) | 44:22 | 33:11 | -25.2 % |
| `DIFFPREP.Register` | 316 s | **128 s** | **-59 %** |
| Denoising | 22.1 s | 15.6 s | -30 % |
| DRBUDDI s/iter | 0.223 | **0.167** | -25 % |

`medium`/`slow` figures predate the kernel geometry work (§14) and should improve further; they
have not been re-run.

**Replay is bit-exact at every step: 132 fast + 118 medium + 106 slow + 25 synthetic, on both
CUDA and WebGPU. No kernel arithmetic was changed anywhere in this work.**


## 17. ITK pins its own libraries to a 2008 ISA — and silently overrides yours

**Found 2026-08-22.** `ITKSetStandardCompilerFlags.cmake:319` appends `-march=corei7` (Nehalem,
2008: SSE4.2, no AVX) to `CMAKE_CXX_FLAGS` for every ITK build. Because **the last `-march` on the
command line wins in GCC**, passing `-DCMAKE_CXX_FLAGS="-march=x86-64-v3"` is silently discarded:

    CXX_FLAGS = -march=x86-64-v3  -mtune=generic -march=corei7 ... -O3 -DNDEBUG
                ^ ours, position 3              ^ ITK's, position 6 - THIS ONE WINS

The first rebuild attempt produced libraries **byte-identical** to the stock build. Nothing failed;
the flag was in `CMakeCache.txt` exactly as requested. Only comparing library sizes (`diff` on
`ls -l`, which returned nothing at all) exposed it - and the near-miss conclusion would have been
"rebuilding ITK buys nothing", which is wrong.

**The fix is to land the flag *after* ITK's**, via the per-config variable, which CMake appends
last (`CXX_FLAGS = ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_<CONFIG>}`):

    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=x86-64-v3"
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -march=x86-64-v3"

Result - vector instruction counts (`vfmadd|vfmsub|ymm`) in the static libraries:

| library | stock | `-march=x86-64-v3` |
|---|---:|---:|
| **`libitkvnl-6.0`** (vnl_matrix, vnl_svd) | **0** | **87 363** |
| `libITKCommon-6.0` | 70 | 1 055 |
| `libITKOptimizers-6.0` | 0 | 483 |
| `libITKStatistics-6.0` | 0 | 93 |

**This does not affect TORTOISE's own objects** - our build sets its flags independently and
carries no `corei7` (verified). It affects only code that lives in ITK's prebuilt `.a` files.
Most ITK code TORTOISE executes is header templates instantiated in *our* translation units, so
it already gets our flags; `vnl` is the significant exception.

**Built as an OPT-IN tree, at the owner's request:** `$LIB/InsightToolkit-6.0b02_build_v3`,
same source and options as the stock build. `$LIB/InsightToolkit-6.0b02_build` is untouched.
Select with `-DITK_DIR=$LIB/InsightToolkit-6.0b02_build_v3`; revert by pointing back. ~300 MB,
and the whole build takes about 2 minutes at `-j10`.

**MEASURED 2026-08-22: it buys nothing. Do not adopt it.**

Interleaved A/B (itkv3, stock, itkv3, stock), each arm a pipeline run truncated at the first
`RegisterSplit` line - ~4 min per arm rather than a full 8 min run, since `Denoising` and
`t_gpu`/`t_cpu` both print by then:

| arm | Denoising | `t_cpu` | `t_gpu` |
|---|---:|---:|---:|
| itkv3 #1 | 15.255 | 43.464 | 0.504 |
| stock #1 | 15.536 | 43.606 | 0.504 |
| itkv3 #2 | 15.576 | 43.840 | 0.509 |
| stock #2 | 15.368 | 43.377 | 0.514 |
| **itkv3 mean** | **15.416** | **43.652** | |
| **stock mean** | **15.452** | **43.492** | |

Denoising differs by **0.2 %**; `t_cpu` is **0.4 % slower** with the optimised library. Both are
inside the run-to-run spread and the sign flips between rounds. `t_gpu` is unchanged, as it must
be - ITK is not in the CUDA path - which is the internal control confirming the comparison is
sound. The binary really does link the vectorised code: 188 702 AVX2/FMA instructions against the
stock build's 39 820.

**Conclusion: DIFFPREP's CPU registration time is not in vnl or any ITK prebuilt library.** The
-6 % `t_cpu` from `-march` (§14) was the ceiling for that path; the remainder is header templates
already compiled with our flags, or code that does not vectorise.

**Methodological note - interleave, do not compare across time.** `itkv3 #1` (15.255) against the
morning's stock baseline (15.56) reads as a 2 % win, and would have been adopted on that basis.
The same binary's second run gave 15.576. This is the §12 failure mode in miniature; the
interleaved control is what prevented it.

The `_build_v3` tree is left in place (~300 MB, harmless, opt-in via `ITK_DIR`) since it cost
2 minutes to build and documents the §17 flag-ordering trap. Nothing selects it.


## 18. NegateImage fusion — and a case where the port was ahead of the reference

**Measured 2026-08-22, binary `ad99418e`.** `InvertField`'s loop ran a standalone
`NegateImage_kernel` over the whole field purely to flip signs - a full read-and-rewrite pass,
~22.7k launches per 45 s window, computing nothing. Folded into `ComposeFields_kernel` behind a
`negate` flag (defaulted off in the header, so the other caller is untouched).

**Equivalence, established before measuring and confirmed by replay:** unary negation is exact in
IEEE and the same operator was used; the only consumer in between,
`ComputeFieldLocalNormImage`, sums SQUARES (`v*v/spc/spc`) and is therefore sign-symmetric; and
`UpdateInvertField_kernel` already received the negated field. All 381 records replay bit-exact.

| kernel | geometry only | + fusion | |
|---|---:|---:|---|
| **`NegateImage_kernel`** | 257.3 us | **eliminated** | |
| `UpdateInvertField_kernel` | 447.5 us | **428.0 us** | **-4.4 %** |
| `ComposeFields_kernel` | 422.2 us | 425.6 us | +0.8 % (does the negate now) |
| `ComputeFieldLocalNormImage` | 192.9 us | 192.2 us | - |
| `computeFiniteDiffStructs` (control) | 22.04 ms | 22.10 ms | +0.3 % |

Per iteration the group went **1319.9 us -> 1045.8 us, -20.8 %** - more than the ~14 % predicted
from removing the pass alone. The surplus is `UpdateInvertField` speeding up 4.4 % **without being
modified**: dropping the intervening full-field rewrite leaves `composed_field` resident in cache
when it is read. Removing a memory pass is worth more than the pass itself costs.

**The WebGPU port did not need this change - it never had the defect.** Its shader entry is
`local_norm_and_negate`: it reads the field once, writes the norm and the negated values in the
same pass. The port, written later, had already collapsed a redundancy the CUDA reference carries.
CUDA now matches it with a different split (compose+negate vs norm+negate); both are equivalent
and both remove the standalone pass.

**Three kernels is the floor for this loop.** `ComputeFieldLocalNormImage`'s output is *reduced*
(`ScalarFindMax`/`ScalarFindSum`) before `UpdateInvertField` can use `m_MaxErrorNorm`, so no
further fusion is possible without changing the algorithm.


## 19. `__restrict__` is a null; the remaining GPU target is the correlation-window kernel

**`__restrict__` (measured, then REVERTED).** Added to the 15 base-pointer declarations in the
four hot kernels, after verifying every call site passes distinct buffers. Result, same 45 s
window: `ComposeFields` 425.6->426.0, `UpdateInvertField` 428.0->427.7,
`ComputeFieldLocalNormImage` 192.2->192.1, `computeFiniteDiffStructs` 22.10->22.12 ms. **Every
kernel within 0.1 %.** These kernels load, compute, then store once at the end, so aliasing never
constrained the scheduler. Reverted: no measurable gain, and it leaves a footgun - a future caller
passing `output == main_field` would get silently wrong results with no diagnostic.

**`computeFiniteDiffStructs` is now the largest single kernel: 22.5 % of GPU time, ~22 ms/call.**
It is a 3D correlation window:

    #define WIN_RAD 5   WIN_RAD_Z 3    -> 11x11x7  =   847 voxels/output
    #define WIN_RAD_JAC 9  ..._Z 4     -> 19x19x9  = 3 249 voxels/output

**Which regime is it in? Unresolved, and `ncu` cannot answer it here.** Nsight Compute fails with
`ERR_NVGPUCTRPERM`; GPU performance counters need `NVreg_RestrictProfilingToAdminUsers=0`, which
needs root, and there is no sudo on this machine (CLAUDE.md 2.3). Analytically:

| assumption | traffic/call | implied |
|---|---:|---|
| no intra-warp reuse (3 249 x 2 images x 4 B x 0.58 Mvox) | ~15 GB | ~100 % of peak bandwidth |
| with intra-warp coalescing (~12x, windows overlap 18/19 columns) | ~1.2 GB | **~56 GB/s = ~8 % of peak** |
| compute (~5 flop x 3 249 x 0.58 Mvox) | 9.4 GFLOP | **~425 GFLOP/s = ~1 % of peak** |

The second is the realistic one, and it saturates neither bandwidth nor compute - so the kernel is
most likely **latency-bound** on the serial nested loop with dependent accumulator chains.

**Therefore prefer REGISTER BLOCKING over shared-memory tiling.** Tiling attacks bandwidth, which
is probably not the constraint. Having each thread compute 2-4 outputs along x gives independent
accumulator chains (ILP) *and* amortises the window loads, so it wins under either hypothesis.
**Bit-exact**: same operands, same summation order, only more in flight. Shared-memory tiling is
also bit-exact and remains the fallback if blocking proves bandwidth-limited after all.

**Off-limits without sign-off:** running sums / summed-area tables would take the window from
O(w^3) to O(1) per voxel - by far the largest theoretical win - but they change summation order
and would break bit-exact replay (§6).


## 20. Register blocking made it WORSE — the window kernel wants occupancy, not ILP

**Measured 2026-08-22.** §19 predicted register blocking would help `computeFiniteDiffStructs`
because the kernel saturates neither bandwidth nor compute. It does the opposite:

| `FD_GROUP` | ms/call | |
|---:|---:|---|
| 1 | **22.12** | |
| 4 | **35.14** | **+59 % - much worse** |

Each thread carries its own `suma2, suma, sumac, sumc2, sumc, N` accumulators, so 4 outputs per
thread quadruples that live state. The register pressure cuts warps resident per SM - and if the
kernel is latency-bound, **occupancy is precisely the mechanism hiding that latency**. Blocking
traded away the thing that was working.

Block starvation is ruled out: at `FD_GROUP=4` on `fast` the grid is still ~1450 blocks
(1 x 25 x 58) across 66 SMs.

**Reverted to `FD_GROUP=1`.** The `FiniteDiffLaunch` helper and the `FD_GROUP` define are kept -
at 1 they reproduce the previous geometry exactly, and they leave the knob in place with the
measurement recorded beside it. **A dedicated factor was necessary rather than raising the global
`PER_GROUP`**: 5 of the 12 kernels launched with that geometry (`AddToUpdateField2`,
`ComputeMetric_CC`, `ComputeMetric_CCJacSSingle`, `ComputeMetric_MSJacSingle`, `NegateImage2`)
have no `PER_GROUP*ii` loop and would have silently covered only 1/N of the x range.

**Direction this points:** the kernel wants *more* occupancy, not more work per thread. Untested
levers, cheapest first - shrink live state (recompute rather than hold), try `blk=(32,8,1)` or
`(64,2,1)` at `FD_GROUP=1`, or `__launch_bounds__` to cap registers. **`ncu` would answer this in
one run** (achieved occupancy, registers/thread, stall reasons) but is blocked by
`ERR_NVGPUCTRPERM` - it needs `NVreg_RestrictProfilingToAdminUsers=0`, i.e. root, and this machine
has no sudo. Without counters, each hypothesis costs a build plus an nsys run to test.


## 21. Full validation on the committed code (branch `optimize`, e879b9c)

Run as a single clean sequence 2026-08-22: build all three configs once, regenerate the reference
evidence with those exact binaries, then gate. No source edits in between - the discipline that
was violated twice earlier in this work.

Binaries: CUDA `3e1dff9b`, WebGPU `d22fcefc`.

| gate | result |
|---|---|
| build cpu / cuda / webgpu | **PASS** clean |
| CUDA self-replay `fast` (bitwise) | **PASS 132/132** |
| CUDA self-replay `slow` (bitwise) | **PASS 106/106** |
| WebGPU replay fast / medium / slow / synthetic | **PASS 132 / 118 / 106 / 25** |
| webgpu_probe (adapter + allocation) | **PASS** |
| negative tests | **PASS** |
| end-to-end `fast` smoke | **PASS** |
| record manifest | FAIL - stale index (pre-existing) |
| DIFFPREP pa registration | FAIL - stale baseline |
| isolated DRBUDDI Step2 | FAIL - stale reference class |

**Zero correctness failures. All three FAILs are baseline/bookkeeping staleness**, and all three
require *replacing a stored baseline*, which `plan_optimize.md` §3 forbids as part of performance
work ("Do not recapture goldens, loosen a tolerance, replace a baseline... Those actions are a
separate approved change"). They are therefore left failing, deliberately:

- `record manifest` - `benchmark/MANIFEST.json` (08-21 02:49) predates a recapture of the record
  trees (07:14-08:33 the same morning). **Predates this work entirely.**
- `DIFFPREP pa registration` - `diffprep_pa_baseline.json` (08-20 00:48) was recorded against CUDA
  binary `a20b5dcd`. The summary line reads "outside NOFMA-relative thresholds", which is
  misleading; the detail is "baseline recorded against a different CUDA binary - re-baseline".
- `isolated DRBUDDI Step2` - `DRB_C1` was produced by `a20b5dcd`. Re-baselining also needs
  `build_cuda_nofma` rebuilt by hand; `revalidate.sh` does not build it.

**Performance on the validated binaries:**

| | baseline | validated | change |
|---|---:|---:|---:|
| `fast` CUDA wall (n=2) | 11:48 | **6:56 - 7:03** | **-41 %** |
| `fast` CUDA TOTAL | 708 s | 415 / 423 s | |
| `DIFFPREP.Register` | 316 s | 127 / 128 s | -60 % |
| DRBUDDI s/iter | 0.223 | 0.115 / 0.124 | -46 % |
| `fast` WebGPU wall (n=1) | 12:51 | **10:37** | -17 % |

The `fast` CUDA result improved again over the §16 figures because the `NegateImage` fusion (§18)
landed after those were taken: post-fusion 415-423 s does not overlap the geometry-only 447-485 s.

**WebGPU's remaining gap is its own registration path**: `t_gpu` 1.256 s/volume against CUDA's
0.51 s. That is why the queue gives it 78 GPU volumes rather than 108, and why it gains 17 %
where CUDA gains 41 %. It is the port's next target and is independent of this work.
