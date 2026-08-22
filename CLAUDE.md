# CLAUDE.md — TORTOISE V4 / WebGPU port

Working notes for this repository. Written for a session with no prior context.

---

## 0. Read this before touching anything

### 0.0 The governing principle: **MIMIC THE REFERENCE, DO NOT IMPROVE IT**

This is a *faithful port*, not a better implementation. The WebGPU backend must reproduce the CUDA
backend's behaviour **including its quirks and including its absence of defensive checks**. Adding
a guard the reference does not have is a divergence, and a divergence is a bug — it changes which
inputs produce which outputs and silently invalidates the golden-vector comparison.

Verified against the source, so it does not need re-deriving:

- **The CUDA kernels in `src/cuda_src/*.cu` contain ZERO NaN/Inf checks.** `grep -n "isnan\|isinf"
  src/cuda_src/*.cu` returns nothing. The port must not add any.
- Their **only** guards are *degeneracy thresholds*, and each has a WGSL counterpart that must
  match exactly — that is the complete list:

| Guard | Where |
|---|---|
| `LIMCC (1E-10)`, `LIMCCSK (1E-5)`, `LIMCCJAC (1E-5)` | `src/cuda_src/compute_metric.cu:10-12`, mirrored in `src/main/compute_metrics_*.h` |
| `det = -1 + 1E-5` when the Jacobian goes non-positive | `compute_metric.cu:230, 501, 594, 703` |
| `if(detf <= 0) detf = 1E-5;` (and `detF2` likewise) | `compute_metric.cu:1495-1498, 1086, 1174, 1264` |
| `if(nrm != 0)` | `cuda_image_utilities.cu:467` |
| `if(magnitude > 1E-20)` | `cuda_image_utilities.cu:413` (`ScaleUpdateField`) — **was missing from this table**; the port reproduces it at `image_utilities.cxx:129-130`, including CUDA's NaN behaviour. A reviewer applying the table literally would have deleted it |
| `if(magnitude != 0)` | `cuda_image_utilities.cu:316` |

- **Aborting on a WebGPU failure is faithful mimicry, not an addition.** `cuda_utils.h`'s
  `gpuErrchk` calls `gpuAssert(..., bool abort = true)`, which prints and `exit(code)`s on any
  device error. So `wgpuctx::Dispatch` aborting on a failed validation, and `Download` aborting on
  a failed `MapAsync`, match the reference's own error discipline.

**`FillBuffer` reproduces a reference bug on purpose (added 2026-08-20).** `CUDAIMAGE::FillBuffer`
(`cuda_image.h:51-55`) calls `cudaMemset3D(PitchedFloatData, val, extent)`, whose second parameter is
an **`int` byte value** — so the float argument is truncated to `int` and its low byte written to
every *byte*, not to every float:

| call | reference result per float |
|---|---|
| `FillBuffer(0)` | `0.0f` |
| `FillBuffer(1)` | `0x01010101` = **2.3694e-38**, not `1.0` |
| `FillBuffer(-1)` | `0xFFFFFFFF` = **NaN**, not `-1.0` |

`src/webgpu_src/gpu_image.cxx` filled each *float* with `val` — the obvious reading, what ITK's
`FillBuffer` does, and a §0.0 violation. It now reproduces the byte fill. **Do not "fix" it back.**

Reachability was **checked, not assumed**: this overload has *no call site in the GPU pipeline at
all*. Every apparent one binds to ITK's `FillBuffer` instead — passing an `itk::Vector`
(`run_drbuddi_stage.cxx:193-220`, inside the `#else` CPU branch closing at `:222`) or acting on an
`ImageType3D` (`create_mask.cxx:107`, `DIFFPREP.cxx:1321`, `FINALDATA.cxx:985`,
`compute_metrics_*.h`). The divergence is therefore **latent**, and no golden vector covers the
function — a future `FillBuffer(1)` wired to the GPU path would have diverged silently.

**Two over-defensive checks were added during the port and then deliberately REVERTED.** Do not
re-introduce them:

1. A `components_per_voxel` check in `GPUIMAGE::CudaImageToITKImage` (`src/webgpu_src/gpu_image.cxx`).
   `CUDAIMAGE::CudaImageToITKImage` reads `NumVoxels()` floats regardless of the component count;
   the WebGPU version now does the same, with a comment saying so.
2. Zero-filling the destination on a failed readback in `src/webgpu_src/webgpu_context.cxx`. Handing
   back zeros makes a GPU failure look like a legitimate result; it now increments the error counter,
   prints the reason and `exit(1)`s — which is what `gpuErrchk` does.

The one place where *defence is correct* is the **test harness**, which is not part of the ported
computation — see §4.5.

### 0.0b RESOLVED — the port's one deliberate divergence no longer exists

**Status as of 2026-08-21: there are NO deliberate divergences between the WebGPU port and the
CUDA reference.** This section used to argue at length that the port was knowingly *more correct*
than CUDA on one operation. That argument is obsolete — the CUDA bug it described was fixed.

**What it was.** `FieldFindMaxLocalNorm` (`cuda_image_utilities.cu`) reduced the maximum
spacing-normalised norm over a 3-component displacement field by indexing the buffer **flat**
(`gArr[3*i]`, `gArr[3*i+1]`, `gArr[3*i+2]`) over `pitch/sizeof(float)/3*sz.y*sz.z` elements. The
buffer is pitched (`cudaMalloc3D`, 512-byte-granular rows), so triples align with voxels only when
`(pitch/4) % 3 == 0`. When they drift, components of *different* voxels are combined with their axes
mis-assigned, and the sweep additionally reads row padding that `CUDAIMAGE::Allocate` never clears.
The port instead computed the true per-voxel maximum, matching the CPU/ITK implementation in
`drbuddi_image_utilities.cxx`.

**What changed.** The CUDA kernel was fixed — commit `0a0e44c`, branch
`fix/fieldfindmaxlocalnorm-pitched-indexing`, one file, kernel plus its two call sites. It now takes
the volume dimensions and row pitch and reduces over `sz.x*sz.y*sz.z` real voxels.

**Measured effect of the fix**, on a record built to force the drift (39x17x9, anisotropic spacing,
`pitch/4 = 128`), against an analytically computed reference:

| | relative error |
|---|---:|
| before | **0.204** — 20 % wrong |
| after | **7.97e-08** — float rounding |

Verified isolated: of 25 analytic reference records, **24 replay byte-identically** before and
after; only the record built to expose the bug moves.

**And on real captured data.** All artefacts were recaptured against the fixed binary. The three
`slow` `ScaleUpdateField` records that previously diverged by `rel = 0.001651` now report:

| record | before | after |
|---|---|---|
| `ScaleUpdateField.0 / .2 / .4` | `XDIVERGE rel 0.001651` | **`PASS rel 0` — bit-exact** |

`rel = 0` is byte-identical output between the WGSL kernel and the fixed CUDA kernel.

**Consequences — do not restore any of this:**
- The `expected_divergence` mechanism in `gpu_record.h` / `webgpu_replay_main.cxx` is retained but
  **has no users**. It is a general capability (a record declares a known, quantified divergence and
  the tool requires it to be present *and* the right size). Leave it; do not add users casually.
- `FieldFindMaxLocalNorm/` and `cuda_bug_demo/` document the bug for upstream. They describe a
  **fixed** defect.
- Any pre-fix golden vector is invalid. The magnitude normalises the update field on every DRBUDDI
  iteration, so the whole captured trajectory shifted — not merely the `ScaleUpdateField` records.

### 0.1 Standing rules

| Rule | Why |
|---|---|
| **Never add a check the reference does not have.** | §0.0. This includes NaN/Inf guards, shape assertions and "safe" fallbacks in the ported kernels and wrappers. |
| **Never modify `README.md`.** | It is the upstream project README, not ours. |
| **Never relax a tolerance to make a test go green.** | Every tolerance in this project was *derived from a measurement*. §5 records those measurements. Re-deriving or loosening them destroys the evidence chain. |
| **Compute must run on the NVIDIA RTX 4070 Ti SUPER only.** | The host also exposes an AMD iGPU and llvmpipe. Both would run the kernels *correctly* and report meaningless performance. See §3. |
| **`gpu_replay` (CUDA self-replay) must stay bit-exact.** | It is the anchor for everything else. If it ever stops being bit-exact, something changed in the CUDA path and the whole comparison basis is void. |
| **Do not "fix" upstream oddities encountered while porting.** | Non-goals: §1.1. **EXCEPT the swapped `kernel<<<blockSize, gridSize>>>` launches, which this rule used to call cosmetic — that was wrong.** The arguments really are swapped, so grid-shaped values become *block* dimensions: `blockDim=(4,4,2)` on `fast`, and **(1,1,1) — one thread per block** — on downsampled DRBUDDI stages. Results were correct, performance was not. Fixed for 46 elementwise launches (`PERF_NOTES.md` §14); reductions keep their geometry because block size sets summation order. Do not revert. |
| **`audit_response.md` and `benchmark/milestones/` are gitignored** (`audit_*.md`, `plan_*.md`). | Deliberate, by the repo owner. They are internal working documents. Do not assume a fresh clone has them, and do not make this file depend on them. |

> ## Working on PERFORMANCE, not the port?
> Read **`PERF_NOTES.md`** (committed) and **`plan_optimize.md`** (gitignored) instead of
> starting here. **Both bullets this banner used to carry are now obsolete** and are kept only
> so nobody re-derives them: stage wall timers now exist (`src/main/tortoise_profile.h`, always
> on, `[PROFILE] <name> <seconds>` on stderr) and utilisation *has* been measured
> (`PERF_NOTES.md` §2.1). The bottleneck was **not** DRBUDDI: it was DIFFPREP registration at
> 45 % of wall clock (§1).
>
> `fast` is currently **708 s → ~474 s (−33 %)**, all of it bit-exact. Read §12 before trusting
> any DRBUDDI-side measurement — its per-iteration time swings 34 % between identical runs, and
> three separate attributions were made and retracted because of it.
>
> This file still governs: the fidelity rules (§0), the validation architecture (§4) and the
> derived tolerances (§5) apply unchanged to optimisation work. `gpu_replay` must stay
> bit-exact.

> ## Open opportunities, and one outstanding obligation
>
> **OPPORTUNITY — running sums for the correlation-window kernels (needs sign-off).**
> `computeFiniteDiffStructs` is the largest single GPU kernel (~22 % of kernel time, ~22 ms per
> call). It evaluates a **19×19×9 = 3 249-voxel** window per output voxel
> (`WIN_RAD_JAC`/`WIN_RAD_JAC_Z` in `compute_metric.cu`; the CC path is 11×11×7 = 847). Adjacent
> outputs share ~95 % of their window, so a **running sum / summed-area formulation reduces the
> per-voxel cost from O(w³) to O(1)** — by far the largest theoretical win left in the codebase,
> plausibly an order of magnitude on that kernel.
>
> **It is not bit-exact, and that is the whole problem.** Incremental sums accumulate in a
> different order, and floating-point addition is not associative, so `gpu_replay` would stop
> being byte-identical — the anchor the entire validation architecture rests on (§4, §5). Doing
> it requires: explicit owner sign-off, a full recapture of all 381 golden vectors across three
> datasets (~80 min GPU), re-verification of the WebGPU port against the new records, and a fresh
> derivation of every tolerance in §5. **Do not start it as ordinary performance work.**
>
> Cheaper, bit-exact levers on the same kernel are listed in `PERF_NOTES.md` §20 — note that
> register blocking was tried and made it **59 % slower**, so the kernel wants occupancy, not
> ILP. `ncu` would settle the direction in one run but is blocked here: `ERR_NVGPUCTRPERM` needs
> `NVreg_RestrictProfilingToAdminUsers=0`, i.e. root, and there is no sudo on this machine.
>
> **OBLIGATION — a full `revalidate.sh --compare` has NOT been run against the current binary.**
> Every gate inside it has been run individually and passes (`gpu_replay` 132/118/106 bit-exact,
> `webgpu_replay` 132/118/106/25, adapter probe, negative tests), but the one-command suite and
> its end-to-end smoke comparison have not completed on the final code. It must be run as a
> **single clean sequence** — regenerate the reference evidence, *then* revalidate, with **no
> source edits in between**. Editing source mid-sequence invalidates the evidence and re-trips
> the staleness gates; that mistake was made twice during this work. Budget ~70 min.
>
> One gate is known-failing and is **not** caused by this work: `record manifest` reports drift
> because `benchmark/MANIFEST.json` (dated 08-21 02:49) predates a recapture of the record trees
> (07:14–08:33 the same morning). Regenerating it is baseline bookkeeping and is the owner's call.

Authoritative documents, in order of precedence:
**this file** → `benchmark/README.md` (datasets, baselines, methodology) →
`benchmark/END_TO_END_VARIABILITY.md` (why the end-to-end statistic is not a gate).

This file is self-contained: the goal contract is §1.1, the milestone gates are §7, and every
cross-reference in the source points at a section here.

> **THE ENTIRE `benchmark/` TREE IS GITIGNORED** (owner's decision, 2026-08-21) — datasets,
> golden-vector records, harness scripts, `README.md`, `MANIFEST.json` and
> `END_TO_END_VARIABILITY.md` alike. It is ~114 GB and the harness is useless without the data it
> drives, so the whole tree is copied between machines rather than committed. **Every
> `benchmark/...` path cited in this file therefore resolves to nothing in a fresh clone.** Those
> citations are kept because they are exact and useful to anyone who *has* the tree; they are not
> broken links to something that should have been committed. Ask the owner for the tree.
>
> What a clone does get: the port itself (`src/webgpu_src/`, `src/gpu_src/`, `src/cuda_src/`,
> `src/tools/GPUReplay/`), the build system, `cuda_bug_demo/`, and this file.

> `audit_response.md` (external-review correspondence) and `benchmark/milestones/M1.md`–`M6.md`
> (per-milestone close-out evidence) are internal working documents and are **gitignored**, so
> citations of them here resolve to nothing in a clone. Nothing load-bearing for review depends on
> them: this file, `benchmark/README.md`, `benchmark/END_TO_END_VARIABILITY.md`,
> `benchmark/scripts/` and the source are the complete set.

---

## 1. What this is

**TORTOISE V4** (`https://tortoise.nibib.nih.gov`) is the NIH diffusion-MRI preprocessing suite:
import, denoising, Gibbs correction, motion/eddy correction (**DIFFPREP**), susceptibility
correction via blip-up/blip-down (**DRBUDDI**), tensor/MAPMRI fitting, output resampling. C++ /
ITK / Boost / Eigen, with CUDA kernels for the GPU-accelerated path.

**The current work** is a *faithful* WebGPU/WGSL port of the CUDA kernels reachable from the
default `TORTOISEProcess` DIFFPREP + DRBUDDI path, so that non-NVIDIA platforms (macOS/Metal in
particular) can be targeted later. Dawn is the WebGPU implementation, chosen because it is
C++-native and already has a Metal backend.

### 1.1 Goal contract — the original brief, and what was delivered

> Add a `USEWEBGPU` build of `TORTOISEProcess` and `DRBUDDI` that ports the CUDA kernels reachable
> from the default DIFFPREP + DRBUDDI pipeline to WGSL, preserves the existing CUDA public wrapper
> interface, uses only a verified discrete NVIDIA Vulkan adapter, and passes the kernel, stage and
> end-to-end CUDA-comparison gates on `fast`, `medium` and `slow`. Keep `USECUDA` and CPU builds
> working; leave DRTAMAS and non-default TVVF out of scope.

**In scope:** the 20 host wrappers measured as reachable at runtime (§4).

**Non-goals** (do not drift into these): algorithmic changes, refactors, performance work beyond
parity, DRTAMAS, TVVF (`--DRBUDDI_transformation_type SyN` is the default), and any CUDA path not
reachable from the default pipeline. The only fp64 kernels in the tree are DRTAMAS-only, so **the
port needs no fp64 emulation** — WebGPU has none.

**Delivery rules that still bind:** do not relax a tolerance to turn a failure green — find and
document the cause first (§0.1). Record commands, revisions, adapter identity and comparisons under
`benchmark/`. A failure caused by missing input metadata is an external blocker, not permission to
guess.

**Definition of done, and where it stands:**

| Requirement | Status |
|---|---|
| `USECUDA=1` and CPU builds remain valid | **met** — all 7 `.cu` files byte-identical to `9a65714` |
| WebGPU build produces both executables | **met** |
| every reachable wrapper has WGSL coverage | **met** — 20/20 |
| `fast` passes every applicable gate | **met** |
| `medium` passes every applicable gate | **met** |
| `slow` passes every applicable gate | **run 2026-08-20 on both backends** and passes the smoke comparison (r = 0.999949, d = 4.88 — the *best* r of the three). Its CUDA-vs-CUDA floor is being measured; until it exists this is a gross-breakage bound, not a correctness gate |
| benchmark evidence checked in | **partial** — harness and summaries committed; record trees and `milestones/` gitignored by the owner |

**All three datasets now run on both backends**, and `compare_outputs.py` exits zero for each —
which is what the original M6 gate asked for. Be precise about what that is worth: the end-to-end
comparison was demoted to a **smoke test** (§7), so "exits zero" means no gross breakage, not
correctness. The load-bearing evidence remains the per-kernel golden vectors, which now cover all
three matrix sizes.

End-to-end, current measurements on this host:

| dataset | CUDA wall | WebGPU wall | CUDA peak GPU | WebGPU peak GPU | ratio |
|---|---:|---:|---:|---:|---:|
| `fast` | 12:04 | 13:22 | 4 378 MiB | **4 084 MiB** | 0.93× |
| `medium` | 22:01 | 19:56 | 5 940 MiB | **5 685 MiB** | 0.96× |
| `slow` | 43:03 | 42:11 | 2 790 MiB | **2 660 MiB** | 0.95× |

**Wall-clock ratios and smoke `r` are deliberately NOT quoted here.** Both move run to run —
CUDA's own `fast` time spans 11:49–13:10 across 4 runs of one binary (~11 %), and two runs of the
*same* WebGPU binary produced smoke r = 0.999102 and 0.999208. Every previously-embedded figure in
this file (1.14×, 1.16×, 1.18×, and three different `r` values) went stale within a day, and each
one had to be retracted. **Read the current values from the artefacts** — `benchmark/<ds>/<backend>/
provenance.json` and `benchmark/LAST_REVALIDATE.json` — not from prose. The durable statements are:
`fast` is at parity within CUDA's own spread, `medium` and `slow` are at or slightly better than
parity, and all three clear the smoke bounds.

Host RSS is within 1–5 % on all three. **Peak GPU is below CUDA on every dataset** (0.93–0.96×),
as predicted: the port drops pitched-allocation row padding and the `cudaArray` duplication, so a
figure *above* CUDA would indicate a buffer-cache leak.

**Peak GPU is the trustworthy column.** It is identical across all four CUDA `fast` runs, so unlike
wall clock it is not a single-sample figure. Host RSS is within 1–5 % on all three datasets.

Check `exe_sha256` in each `provenance.json` before quoting any row: these were produced over
several days across several builds, and a row from a superseded binary is not evidence about the
current one.

---

## 2. Layout and build

### 2.1 Source layout

| Path | Contents |
|---|---|
| `src/main/` | pipeline: `DIFFPREP.cxx`, `DRBUDDI.cxx`, `DRBUDDI_Diffeo.cxx`, `run_drbuddi_stage.cxx`, `TORTOISE.cxx`, `defines.h` |
| `src/cuda_src/` | the CUDA backend: 7 `.cu` kernel files + thin `.cxx` host wrappers + `cuda_image.{h,cxx}` (`CUDAIMAGE`). Also `gpu_capture.{h,cxx}` — **new**, the golden-vector capture hook |
| `src/webgpu_src/` | **new** WebGPU backend. `webgpu_context.{h,cxx}` (instance/adapter/device singleton, pipeline cache, adapter policy), `gpu_image.{h,cxx}` (`GPUIMAGE`, with `using CUDAIMAGE = GPUIMAGE;` so `main/` compiles unchanged), one `.cxx` per CUDA wrapper, `shaders/*.wgsl` |
| `src/gpu_src/` | **new** backend-selection shims. One header per interface, each `#ifdef USEWEBGPU` → `webgpu_src/…` `#else` → `cuda_src/…`. `target_include_directories(<tgt> BEFORE PRIVATE ${WEBGPU_INCS})` makes it shadow `cuda_src` — **per WebGPU target**, not directory-wide (see §2.2) |
| `src/tools/GPUReplay/` | **new** `gpu_replay_main.cxx` (CUDA self-replay), `webgpu_replay_main.cxx` (WebGPU vs CUDA / vs exact), `gpu_record.h` (record schema v2), `negative_tests.sh` |
| `benchmark/` | datasets, harness (`scripts/`), evidence (`README.md`, `milestones/`). Tracked: `scripts/`, `README.md`, `MANIFEST.json`, `LAST_REVALIDATE.json`, `END_TO_END_VARIABILITY.md`. **Untracked: the multi-GB data AND `milestones/`** (owner's call, 2026-08-20 — the M1–M6 close-outs are internal working evidence) |
| `TORTOISEV4/CMakeLists.txt` | the single build file (~836 lines). `TORTOISEV4/cmake/embed_wgsl.cmake` wraps `.wgsl` into raw-string headers |

The insertion point for the whole port is one typedef: `DRBUDDI_Diffeo.h` has
`using CurrentImageType = CUDAIMAGE`, and the WebGPU backend supplies
`using CUDAIMAGE = GPUIMAGE;` so the pipeline's *algorithms* need no rewrite.

**`src/main/` is NOT unchanged**, despite an earlier claim here to that effect: `git diff --stat
src/main/` shows 13 files, +86/-77 — `DIFFPREP.cxx`, `DRBUDDI_Diffeo.{cxx,h}`, `OmpThreadBase.h`,
`TORTOISE_global.cxx`, `defines.h`, `drbuddi_structs.h`,
`itkDIFFPREPGradientDescentOptimizerv4.{cxx,h}`, `register_dwi_to_slice.h`,
`run_drbuddi_stage.{cxx,h}`, `run_drbuddi_stage_TVVF.cxx`. The changes are backend-selection
plumbing, not algorithmic. `CUDAIMAGE` appears ~45 times across 10 files in `src/main/` alone — an
earlier "~8 references" figure here was wrong.

### 2.2 Build options

`TORTOISEV4/CMakeLists.txt` defines three relevant options, all defaulting to `0`:

| Option | Effect |
|---|---|
| `USECUDA` | adds `-DUSECUDA` **and** `-DUSEGPU`; builds `TORTOISEProcess_cuda`, `DRBUDDI_cuda`, `DRTAMAS_cuda`, `gpu_replay` |
| `USEWEBGPU` | adds `-DUSEWEBGPU` **and** `-DUSEGPU`; builds `TORTOISEProcess_webgpu`, `DRBUDDI_webgpu`, `webgpu_probe`, `webgpu_replay`. Requires `${DAWN_DIR}/build/src/dawn/native/libwebgpu_dawn.a` (fatal error if absent) |
| neither | CPU build: `TORTOISEProcess`, `DRBUDDI`, plus all the standalone tools |

`USEGPU` is **not** a user-settable option — it is added on the command line by both GPU branches
(and defined in `src/main/defines.h` as a fallback). It means "some GPU backend exists".

**Correction to an earlier note here:** `cudaSetDevice`/`cudaGetDeviceCount` are no longer in
`DIFFPREP.cxx` at all. The port replaced that block with backend-neutral `GPUDeviceIds()` /
`GPUSetDevice()` (`DIFFPREP.cxx:594,718,760`), whose CUDA bodies live in `src/gpu_src/gpu_device.h`,
and the surrounding guards moved from `USECUDA` to `USEGPU`. Do not follow the old note when
editing guards. It is a **command-line macro** rather than something relying on `defines.h` being
included first: several files use `#ifdef USEGPU` without a direct include, and if the macro were
invisible the GPU branch would silently compile out and fall back to CPU — a failure that still
"works".

**Scoping (audit fix — do not revert to `add_definitions`):** in the `USEWEBGPU` branch, `USEWEBGPU`
/ `USEGPU` (`set(WEBGPU_DEFS …)`) and the shim include path (`set(WEBGPU_INCS …/gpu_src …/cuda_src)`)
are applied **per target** via `target_compile_definitions`/`target_include_directories` on
`webgpu_probe`, `TORTOISEProcess_webgpu`, `DRBUDDI_webgpu` and `webgpu_replay` only. Directory-wide
`add_definitions(-DUSEGPU)` was leaking into the plain CPU targets defined later in the same file,
compiling GPU code paths into binaries that link neither the WebGPU sources nor Dawn. The `USECUDA`
branch still uses `add_definitions`, which is safe there because that branch does not also build the
CPU targets.

Other notes on the build file:
- WebGPU targets are `CXX_STANDARD 20` (Dawn's C++ headers require it); the rest of the tree is C++14.
- The WebGPU build **cannot** be fully static: Dawn `dlopen()`s the Vulkan loader, and a `-static`
  binary segfaults on the first `dlopen`. The CPU build's `-static` linker flag is suppressed when
  `USEWEBGPU` is set. CUDA targets use `-static-libgcc -static-libstdc++` only.
- `metric_common.wgsl` is a prelude **textually prepended** into the metric shaders, not a
  standalone module; the WGSL-embedding glob explicitly removes it from `WGSL_SOURCES`, **and it is
  listed in every embed command's `DEPENDS`** (audit fix). Without that, editing `metric_common.wgsl`
  left every generated `*.wgsl.h` stale and the build reported success while running old shader code.

### 2.3 Dependencies — all local, **there is no sudo on this machine**

**The paths in this section are THIS machine's** (Linux, the CUDA development host). They are
written out concretely because a worked example beats a placeholder, but every one of them is
machine-specific — on another machine set `LIB` and `SRC` to wherever you put things. Nothing in the
build system hard-codes them any more: `DAWN_DIR` must be passed explicitly (§2.2), and the harness
scripts derive the repo root from their own location.

Everything lives under **`/home/chris/src/tortoise_libraries`** (the "LIB prefix"):

| Dependency | Location |
|---|---|
| cmake 3.29.6 | `$LIB/cmake/bin/cmake` — use this, not any system cmake |
| ITK 6.0b02 (Release) | `$LIB/InsightToolkit-6.0b02_build` |
| Boost 1.86 (static) | `$LIB/boost186/{include,lib}` |
| Eigen 3.4 | `$LIB/local/share/eigen3/cmake` |
| FFTW 3.3.10 | `$LIB/local/{include,lib}` |
| Dawn (WebGPU) | `$LIB/dawn`, revision pinned in `$LIB/dawn.pin` (currently `7e23399c7cb679d4009749d11e4e0e10025bb0b7`) |
| CUDA 13.0 | `/usr/local/cuda` (hard-coded in `CMakeLists.txt`) |

`CMakeLists.txt` hard-codes `BOOST_ROOT /usr/local/boost186` and an ITK path from the original
author's machine — **both are wrong here** and must be overridden on the command line. FFTW and
Eigen headers/libs are *not* in any CMake search path either, so they are handed to the
compiler/linker via `CPATH` / `LIBRARY_PATH`.

Build helper scripts live in the LIB prefix, not in the repo:
- `$LIB/build_tortoise.sh <cpu|cuda>` — configures and builds those two configs.
  **It has no `webgpu` mode**; use the command below (a worthwhile small addition).
- `$LIB/build_dawn.sh` — clones/pins and builds Dawn, **Vulkan backend only**
  (`-DDAWN_ENABLE_DESKTOP_GL=OFF -DDAWN_ENABLE_OPENGLES=OFF -DDAWN_ENABLE_NULL=OFF`), monolithic
  static. Compiling the GL backends out means there is no build-time path to the iGPU at all.

### 2.4 Exact build commands

```bash
LIB=/home/chris/src/tortoise_libraries
SRC=/home/chris/src/TORTOISEV4
CMAKE=$LIB/cmake/bin/cmake
export CPATH=$LIB/local/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$LIB/local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}

COMMON="-DUSE_VTK=0 -DCMAKE_BUILD_TYPE=Release \
  -DITK_DIR=$LIB/InsightToolkit-6.0b02_build \
  -DEigen3_DIR=$LIB/local/share/eigen3/cmake \
  -DBoost_INCLUDE_DIR=$LIB/boost186/include \
  -DBoost_IOSTREAMS_LIBRARY_RELEASE=$LIB/boost186/lib/libboost_iostreams.a \
  -DBoost_FILESYSTEM_LIBRARY_RELEASE=$LIB/boost186/lib/libboost_filesystem.a \
  -DBoost_SYSTEM_LIBRARY_RELEASE=$LIB/boost186/lib/libboost_system.a"

# CPU      -> bin/TORTOISEProcess, bin/DRBUDDI
$CMAKE -S $SRC/TORTOISEV4 -B $SRC/build_cpu    -DUSECUDA=0 $COMMON && $CMAKE --build $SRC/build_cpu -j30

# CUDA     -> bin/TORTOISEProcess_cuda, bin/DRBUDDI_cuda, bin/gpu_replay
$CMAKE -S $SRC/TORTOISEV4 -B $SRC/build_cuda   -DUSECUDA=1 -DCMAKE_CUDA_ARCHITECTURES=89 $COMMON \
  && $CMAKE --build $SRC/build_cuda -j30

# WebGPU   -> bin/TORTOISEProcess_webgpu, bin/DRBUDDI_webgpu, bin/webgpu_replay, bin/webgpu_probe
$CMAKE -S $SRC/TORTOISEV4 -B $SRC/build_webgpu -DUSECUDA=0 -DUSEWEBGPU=1 \
  -DDAWN_DIR=$LIB/dawn $COMMON && $CMAKE --build $SRC/build_webgpu -j30
```

All configurations write executables to **`$SRC/bin/`** (shared), so the three builds overwrite
each other's *shared-named* targets only where names collide — they do not; each config's targets
have distinct names. Existing build trees: `build_cpu/`, `build_cuda/`, `build_webgpu/`, and
`build_cuda_nofma/` (CUDA with `CMAKE_CUDA_FLAGS=-fmad=false`, used for the §5.1 experiment;
its replay binary is `bin/gpu_replay_nofma`).

`sm_89` = RTX 4070 Ti SUPER (Ada). CUDA 13.0, driver 595.84.

---

## 3. GPU selection policy — NVIDIA discrete only

This host enumerates **three** Vulkan devices when unpinned (measured by
`benchmark/scripts/vkprobe.c`):

| # | vendorID | type | device |
|---|---|---|---|
| 0 | `0x1002` | integrated | AMD Ryzen 9 7950X (RADV RAPHAEL_MENDOCINO) — `0f:00.0`, display only |
| 1 | `0x10DE` | **discrete** | **NVIDIA GeForce RTX 4070 Ti SUPER** — `01:00.0`, the only legitimate compute device |
| 2 | `0x10005` | cpu | llvmpipe (LLVM 20.1.2) — software rasteriser |

llvmpipe is the subtler hazard: it produces *correct* results at glacial speed and would still
"pass" a correctness gate while invalidating every benchmark number. That is why the policy is a
**conjunction**, not a vendor test.

Two independent, deliberately redundant mechanisms:

**(a) Environment — `benchmark/scripts/gpu_env.sh`.** Source it before every run.
```
CUDA_VISIBLE_DEVICES=0                    # read by CUDA
VK_ICD_FILENAMES / VK_DRIVER_FILES = /usr/share/vulkan/icd.d/nvidia_icd.json
TORTOISE_WEBGPU_VENDOR_ID=0x10DE          # READ by SelectAdapter (overrides the vendor check)
__NV_PRIME_RENDER_OFFLOAD=1 / __GLX_VENDOR_LIBRARY_NAME=nvidia
```
With it sourced, `vkprobe` sees exactly one device — the Vulkan ICD variables do the real work here.

**`WGPU_BACKEND=vulkan` and `TORTOISE_WEBGPU_ADAPTER_TYPE=discrete` are also exported but are NOT
read by anything in `src/`** (`grep` returns nothing). `WGPU_BACKEND` is a wgpu-native variable, not
a Dawn one — Dawn takes its backend from the hardcoded `opts.backendType`. They are retained in case
a future backend honours them, and are annotated as inert in `gpu_env.sh`. Do not count them as a
layer of protection: the discrete/Vulkan requirements are enforced unconditionally in code, which is
strictly stronger since it cannot be unset.

**(b) In-process check — `src/webgpu_src/webgpu_context.cxx`, `SelectAdapter()`.** Enumerates
adapters and requires `vendorID == 0x10DE` **AND** `adapterType == DiscreteGPU` **AND**
`backendType == Vulkan`. On no match it **aborts with the full adapter list printed** rather than
falling back. `TORTOISE_WEBGPU_VENDOR_ID` overrides the vendor; `TORTOISE_WEBGPU_ALLOW_ANY_ADAPTER=1`
relaxes the policy **only** for testing the rejection path on machines without an NVIDIA card —
never for real runs.

The selected adapter's name/vendor/type/backend/driver is logged at startup and recorded in each
run's `provenance.json`.

**`TORTOISE_WEBGPU_ALLOW_ANY_ADAPTER` prefers a conforming adapter (fixed 2026-08-20).** The
override used to latch adapter[0] on the first loop iteration and never look further, so on this
host — AMD iGPU at [0], NVIDIA at [1] — setting it selected the **integrated GPU even though the
conforming card was present**, which is precisely what §3 exists to prevent. `SelectAdapter` now
tracks a conforming `best` and a separate `fallback`, and uses the fallback only when nothing
conforms. Verified: unpinned with the override set, it selects the RTX 4070 Ti SUPER; with
`TORTOISE_WEBGPU_VENDOR_ID=0xDEAD` it still refuses and prints the adapter list.

### 3.1 Running on another platform (macOS/Metal) — the policy is configurable

`opts.backendType` used to be **hardcoded** to Vulkan, which made the port's own stated goal
unreachable without editing source. Both the backend and the adapter-type requirement are now
environment-selectable. **The defaults are unchanged**, so this host still demands an NVIDIA
discrete Vulkan adapter and every benchmark number stays comparable:

| variable | default | accepts |
|---|---|---|
| `TORTOISE_WEBGPU_BACKEND` | `vulkan` | `vulkan` \| `metal` \| `d3d12` \| `opengl` \| `auto` |
| `TORTOISE_WEBGPU_ADAPTER_TYPE` | `discrete` | `discrete` \| `integrated` \| `any` |
| `TORTOISE_WEBGPU_VENDOR_ID` | `0x10DE` | any vendor ID |

`TORTOISE_WEBGPU_ADAPTER_TYPE` was exported by `gpu_env.sh` from the start and **had no reader** —
it is now honoured. An unrecognised value for either is a hard error, not a silent fallback.

**On Apple Silicon**, all three defaults fail: the vendor is Apple, the GPU reports as
`IntegratedGPU` (requiring `discrete` rejects the only GPU present), and the backend is Metal.
Start from:

```bash
export TORTOISE_WEBGPU_BACKEND=metal
export TORTOISE_WEBGPU_ADAPTER_TYPE=integrated
export TORTOISE_WEBGPU_VENDOR_ID=0x106B     # Apple; or TORTOISE_WEBGPU_ALLOW_ANY_ADAPTER=1
bin/webgpu_probe                            # confirm the adapter before anything else
```

**This has never been executed on Metal.** The change was verified only in the directions testable
here: the default path still selects the RTX 4070 Ti SUPER, bad values are rejected, and
`TORTOISE_WEBGPU_BACKEND=metal` fails cleanly with "no metal adapters found" on Linux. Treat the
macOS recipe as untested.

**Validation on macOS needs no CUDA.** `bin/webgpu_replay` links no CUDA libraries, and each golden
vector carries its own `out` tensors (CUDA's recorded output), so `webgpu_replay --all <records>`
grades Metal against the same reference and the same tolerances this backend passes. That is the
entire point of the record design — see §4.1. What macOS *cannot* do is **capture new records**;
those require `TORTOISEProcess_cuda`.

---

## 4. Validation architecture

The pipeline **cannot reproduce itself** end-to-end (§5.4), so correctness is established at the
*kernel* level with deterministic golden vectors. This is the load-bearing design decision.

### 4.1 The four layers

| Layer | What it proves | Determinism |
|---|---|---|
| **Golden-vector capture** (`TORTOISE_GPU_CAPTURE`) | records `{op, params, inputs, outputs}` at each of the 20 CUDA host wrappers | — |
| **`gpu_replay`** (CUDA) | CUDA re-running a captured record reproduces its output **bit-for-bit** | exact |
| **`webgpu_replay`** | WGSL vs CUDA golden vectors, and vs **exact analytic references** | tolerance-gated |
| **`compare_outputs.py`** | full-pipeline smoke test, **against the measured floor** (§5.4), never an absolute | non-deterministic |

Capture hooks live at the *host wrappers* (`WarpImage_cuda`, `ComputeMetric_MSJac_cuda`, …), not
the kernels — 3 lines each, inert unless `TORTOISE_GPU_CAPTURE` is set. Record schema is **v2**
(`gpucap::SCHEMA_VERSION` in `src/cuda_src/gpu_capture.h` must equal `RECORD_SCHEMA_VERSION` in
`src/tools/GPUReplay/gpu_record.h`). Each record is a `record.json` sidecar plus raw fp32 blobs
with FNV-1a digests.

**Reachability is determined by runtime capture, not grep.** Static counting suggested 23 wrappers;
an instrumented `fast` run captured exactly **20**. The three absentees (`ComputeEntropy`,
`NegateField`, `ComputeImageGradientImg`) are dead — their apparent callers are the *CPU/ITK*
namesakes in `src/main/drbuddi_image_utilities.cxx`, or a commented-out call site. Name-based
searching cannot distinguish the ITK and CUDA implementations; do not re-open this by grepping.

The 20 ported ops (all registered in `webgpu_replay_main.cxx`'s `BuildRegistry()`):
`AddImages`, `AddToUpdateField`, `ComposeFields`, `ComputeJointEntropy`, `ComputeMetric_CC`,
`ComputeMetric_CCJacS`, `ComputeMetric_CCSK`, `ComputeMetric_MSJac`, `ContrainDefFields` *(sic —
upstream spelling)*, `GaussianSmoothImage`, `InvertField`, `MultiplyImage`, `MultiplyImages`,
`PreprocessImage`, `QuadraticTransformImageC`, `ResampleImage`, `RestrictPhase`, `ScaleUpdateField`,
`SumImage`, `WarpImage`.

### 4.2 Datasets

| Dataset | Matrix | Voxel | Up | Down | Structural | DRBUDDI metrics |
|---|---|---|---:|---:|---|---:|
| `fast` | 100×100×58 | 2.2 mm | 138 | 10 | T2w 176×256×256 (+ T1w 192×256×256, capture only) | **4** (6 with both structurals) |
| `medium` | 140×140×81 | 1.71 mm | 102 | 102 | none | 2 |
| `slow` | 140×140×92 | 1.5 mm | 297 | 3 | none | 1 |

`fast` is the iteration and **coverage** dataset — it carries a structural image so it runs all 4
DRBUDDI metrics, making its operation set a superset of the other two. `medium`/`slow` are the
non-overfitting and memory gates.

`slow` shipped without JSON sidecars and with gradient-file basenames that do not match the NIfTI
basenames. **This is resolved** — the harness prepares it automatically (pairs gradient files,
writes minimal sidecars with `PhaseEncodingDirection`, drops `part-phase` volumes), applying the
fix only to the copies under `<backend>/`, never to `In/`. Phase encoding per the dataset owner:
`dir-AP` → `j`, `dir-PA` → `j-` (the *opposite* sign convention to `fast`/`medium`; harmless,
because DRBUDDI solves for the midpoint and the axis is what matters). Only `PhaseEncodingDirection`
is ever read by TORTOISE — `TotalReadoutTime`/`EffectiveEchoSpacing` appear nowhere in the source.

`benchmark/<ds>/In/` is **never written to**. TORTOISE resolves symlinks when choosing its output
path, so the harness *copies* inputs into `<ds>/<backend>/` rather than linking them.

### 4.3 Exact commands

```bash
cd /home/chris/src/TORTOISEV4
source benchmark/scripts/gpu_env.sh          # ALWAYS, before any GPU run

# --- capture golden vectors (CUDA build) ---
TORTOISE_GPU_CAPTURE=benchmark/fast/CAPTURE/golden_vectors TORTOISE_GPU_CAPTURE_N=6 \
  bin/TORTOISEProcess_cuda --up_data ap.nii --down_data pa.nii --structural T2w.nii
#   N=6 is what the committed artefacts were captured with (22 shape classes x 6 = 132
#   records). The BUILT-IN default is still 2 - passing it explicitly is required, or you
#   get 44 records and revalidate.sh fails at 44/132. Quotas are keyed per SHAPE CLASS
#   (e.g. GaussianSmoothImage.v1.* scalar vs .v3.* 3-component field) — see §5.5.

# --- CUDA self-replay: MUST be bit-exact ---
bin/gpu_replay --all benchmark/fast/CAPTURE/golden_vectors   # 132 records, must be bit-exact
src/tools/GPUReplay/negative_tests.sh benchmark/fast/CAPTURE/golden_vectors/WarpImage.0

# --- WebGPU replay ---
bin/webgpu_replay --list benchmark/fast/CAPTURE/golden_vectors     # coverage: ported/unported
bin/webgpu_replay --all  benchmark/fast/CAPTURE/golden_vectors     # vs CUDA,  fast   (132 records)
bin/webgpu_replay --all  benchmark/medium/CAPTURE/golden_vectors   # vs CUDA,  medium (118 records)
bin/webgpu_replay --all  benchmark/synthetic_vectors               # vs EXACT  (25 records)

# --- regenerate the synthetic exact-reference records ---
benchmark/scripts/make_synthetic_records.py benchmark/synthetic_vectors

# --- adapter / allocation smoke test ---
bin/webgpu_probe

# --- full pipeline + profile ---
benchmark/scripts/run_cuda_reference.sh fast CUDA     # also accepts WebGPU | CPU
benchmark/scripts/finalize_run.sh       fast CUDA     # writes provenance.json + stage_cache
benchmark/scripts/compare_outputs.py fast CUDA WebGPU --floor      # or --floor-d=<D> --floor-r=<R>
```

### 4.4 Synthetic exact-reference records — why they exist

`make_synthetic_records.py` builds records whose expected output is computed **analytically**. A
linear field interpolates *exactly* under trilinear interpolation, so a linear ramp gives a
reference that a sampling bug cannot satisfy. This matters because **CUDA is the less accurate
implementation** for texture-sampled ops (§5.2) — validating only against CUDA would mean measuring
the port against a known-lossy reference.

The 25 synthetic records also cover edge cases the captured vectors never reach: border overhang
(`WarpImage.synth_border`, `ResampleImage.synth_overhang`), oblique direction matrices
(`*.synth_oblique`), a degenerate single-slice volume (`WarpImage.synth_thin`), and the
3-component field path with the small-variance blend (`GaussianSmoothImage.synth_field*`).

Stage isolation: `benchmark/<ds>/CUDA/stage_cache/` is a `chmod a-w` copy of the DIFFPREP
`*_temp_proc/` intermediates, so DRBUDDI can in principle be replayed standalone without
re-running DIFFPREP and with no chance of mutating the reference. **The standalone executable
was fixed on 2026-08-20** and the isolated replay now runs — see §6.1.

### 4.5 NaN policy in the harness — **divergence only**

In `src/tools/GPUReplay/gpu_record.h` (`Compare()` / `Outcome::check()`):

- **Matching NaNs between the two backends count as AGREEMENT.** If CUDA produces NaN at a voxel and
  WebGPU produces NaN at the same voxel, the port reproduced the reference — that is a pass. ±Inf
  compares equal by `a == b`.
- **A non-finite value in one backend only is a FAILURE** (`d.nnonfinite` → `ok = false`, reported as
  "N non-finite mismatch(es) (NaN/Inf in one backend only)"), and the tolerance sweep is negated
  (`if(!(diff <= limit)) d.nexceed++`) so NaN counts as an exceedance rather than sneaking through.

**Why this exists, and what it is not.** `std::max(x, NaN)` returns `x`, so before this change a NaN
difference never raised `maxabs` and the comparison silently **PASSED**. This is a fix to the
*comparison*, not a judgement that NaN is undesirable. It is fully consistent with §0.0: the
reference kernels contain no NaN checks and neither do the ported ones — the harness merely refuses
to call a NaN-vs-number difference a match.

Two other correctness fixes in the same file and in the context, from the audit:

- `Compare()` **throws** on an output size mismatch instead of comparing the common prefix. A wrapper
  returning the wrong dimensions used to report PASS — the one thing this harness must never do.
- `wgpuctx::Upload()` no longer rounds the length up when padding to 4 bytes (that read past the
  caller's buffer); it stages through a padded host copy instead.
- The context's `g_bytes` / `g_errors` counters are `std::atomic<uint64_t>` — they are read and
  written from the pipeline's OpenMP threads.
- `gpu_capture.cxx` writes the **real directory leaf** into `manifest.jsonl`. Variant records live in
  `op.v<N>.<seq>` (see §5.5), so emitting `op.<seq>` made them unlocatable from the manifest.

---

## 5. Measured numerical findings — DO NOT re-derive or undo

These four are the hardest-won results in the project. Each replaced a plausible-but-wrong
assumption. They are implemented in `src/tools/GPUReplay/webgpu_replay_main.cxx`.

### 5.1 The elementwise tolerance is **1e-5**, and it is a floor, not a fudge

`ResampleImage` — chosen first *precisely because* it uses no hardware filtering, so it isolates
the buffer/index/arithmetic layer — reproduced CUDA to `1.45e-6`, just over this project's original
`1e-6`. Rather than relax the number, CUDA was **rebuilt with `-fmad=false`** (`build_cuda_nofma/`)
and replayed against its own FMA-built golden vectors:

| Comparison | max relative difference |
|---|---:|
| WebGPU vs CUDA (default, FMA on) | **1.45e-6** |
| CUDA `-fmad=false` vs CUDA `-fmad=true` | **2.67e-6** |

The same source, same compiler, one contraction flag apart, **disagrees with itself more than the
WGSL port does**. nvcc contracts `a*b+c` into FMA by default and WGSL exposes no contraction
control (only an explicit `fma()` builtin), so bit-exactness across backends is unattainable in
principle. A 1e-6 gate sits *below the reference implementation's own noise floor*.

**Tolerance: 1e-5**, ≈4× the measured compiler-induced spread. A real porting bug (wrong index,
wrong boundary rule, wrong interpolation order) produces errors orders of magnitude larger.

**There is a SECOND gate, and this section used to omit it.** `tol` above is max-normalised —
`max|a-b| / max|b|` — a single global allowance set by the largest voxel in the volume. On a metric
update field whose max is 1.9e7, `tol = 1e-5` permits an absolute error of 194 at *every* element,
and a port returning zero for the ~34 % of nonzero elements below that would have PASSED. So
`webgpu_replay_main.cxx` also applies a **per-element** gate:

    |a-b| <= elem_tol * (|b| + rms(ref))          elem_tol: 1e-4 elementwise/exact,
                                                            1e-3 guarded, 2e-2 texture

Be accurate about how much this buys. `rms` is **not** an outlier-robust denominator — it is
outlier-dominated — and the measured effect is a **median band shrink of 1.08×**, with 50 of 150
tensors actually *looser* per-element. It materially tightens the small elementwise ops and barely
touches the metric kernels. **The metric kernels' real evidence is the expression-by-expression
source review, not this gate.** Do not quote the withdrawn "3.6× margin".

### 5.2 CUDA's texture sampler is **lossy**; the WGSL port is the accurate one

First A/B suggested CUDA did not quantise filter weights. It was an insensitive test:

| Test | Result | What it shows |
|---|---:|---|
| `WarpImage` real data, exact fp32 vs CUDA | 1.45e-6 | misleading — source/target share a grid, sub-voxel displacements, weights near 0/1 |
| `QuadraticTransformImageC` real data, exact fp32 vs CUDA | 2.85e-3 | resamples *between* grids (33³ @6.6mm → 20×25×17 @7.33mm), so weights are arbitrary |
| **synthetic linear ramp, CUDA vs exact trilinear** | **3.32e-5** | **CUDA's sampler is lossy** |
| synthetic linear ramp, WebGPU vs exact trilinear | passes at 1e-5 | the port computes true trilinear |

Emulating the documented `floor(w*256)/256` did **not** reproduce CUDA's numbers (it was worse than
exact), so the hardware's rounding model is not established — and does not need to be. The goal is
portability, not bug-compatibility. `TORTOISE_WEBGPU_CUDA_TEXFILTER=1` is retained purely as a
diagnostic for re-testing on other hardware (relevant to the Metal port).

**Consequence — texture-sampling ops (`WarpImage`, `QuadraticTransformImageC`) get a two-sided gate:**

| Gate | Tolerance | Role |
|---|---:|---|
| vs **exact trilinear** (synthetic, `reference == "exact"`) | **1e-5** | **primary correctness gate** |
| vs **CUDA** golden vectors | **5e-3** (observed max 3.6e-3) | bounds CUDA's own sampler loss; catches gross errors only |

Related, and equally important: **WebGPU cannot reproduce the CUDA sampler with a sampler.**
`GPUAddressMode` has no `clamp-to-border` and no border colour, while the CUDA textures use
`cudaAddressModeBorder`. Manual trilinear over a storage buffer is therefore **mandatory**, not a
preference — and is also the portable choice for Metal. Consequently no sampler is used anywhere,
"textures" are plain storage buffers, and `CreateTexture()` is a **no-op** in the WebGPU backend
(removing the `cudaArray` duplication CUDA carries).

**Four sampling kernels, three different out-of-domain rules.** Assuming one shared convention
would have produced quietly wrong edges in three of the four:

| Kernel | Out-of-domain rule |
|---|---|
| `ResampleImage` | whole output voxel zeroed if *any* axis is outside `[0,N-1]` |
| `QuadraticTransformImageC` | guarded *before* sampling; out-of-domain voxels left at the wrapper's memset zero |
| `WarpImage` | **unguarded**; per-neighbour border zero → a one-voxel halo gets nonzero partial values |
| `ComposeFields` | falls back to the **main field's displacement** — neither zeroed nor blended |

Coordinate convention: all sample at `tex3D(tex, iw+0.5, jw+0.5, kw+0.5)` with unnormalised
coords, so CUDA's internal `u = x - 0.5` recovers exactly `iw`. **WGSL uses `iw` directly — no
half-texel juggling.**

### 5.3 Guarded kernels use an **outlier-fraction** rule, not max-error

`ComposeFields` exceeded tolerance on **52 of 95 823** elements. CUDA-vs-CUDA under the same FMA
toggle also exceeds it, on 1 element. Kernels with a domain-guard branch flip that branch when a
coordinate lands within 1 ulp of the boundary, and the two branches return genuinely different
things — the error is *discontinuous*, so max-error is the wrong statistic.

**Rule: guarded ops pass if exceedances are confined to ≤ 0.1 % (`max_outlier_frac = 1e-3`) of
elements.** A real bug moves far more than that.

Guarded ops (`IsGuarded()` in `webgpu_replay_main.cxx`): `ComposeFields`, `ResampleImage`,
`QuadraticTransformImageC`, and **`InvertField`** — the last is a fixed-point iteration that calls
`ComposeFields` ~40 times and so inherits the discontinuity.

### 5.4 The pipeline is **not run-to-run reproducible** — absolute end-to-end gates are unachievable

Two runs of the *same CUDA binary* on the *same input*, capture disabled:

| Comparison | differing voxels | max\|diff\|/p99 | Pearson r |
|---|---:|---:|---:|
| CUDA run A vs CUDA run B | **35.08 %** | 3.68 | 0.999518 |
| CUDA reference vs instrumented run | 35.06 % | 4.93 | 0.999374 |

The two figures agree, which **exonerates the capture hooks** — and in any case every golden vector
replays bit-exact, so the kernels are deterministic on identical inputs.

Localised precisely by diffing intermediates:

| Stage output | A vs B |
|---|---|
| `ap_proc.nii` (import + denoise + Gibbs) | **identical** |
| `ap_proc_moteddy_transformations.txt` (DIFFPREP registration) | **differs** — 1914 of 3312 parameters |
| everything downstream | differs, as a consequence |

Differences are mostly noise (median 3e-5) with a long tail (max 0.245) — the signature of an
iterative optimiser where a tiny early perturbation sends a few volumes down a different
convergence path.

**MECHANISM ESTABLISHED 2026-08-19 — the earlier "thread scheduling decides which volume goes
where" hypothesis was WRONG.** The GPU/CPU assignment is *fully deterministic*:
`gpu_ids_per_thread` (`DIFFPREP.cxx:600-634`) is pure index arithmetic over `Nvols`, `NGPUs` and
`Nt = omp_get_max_threads()`, and the loop is `schedule(static,1)`, so a given volume always takes
the same path on the same machine. **The nondeterminism is inside the ITK CPU registration path
itself.**

The code makes a sharp, testable prediction, and a single `fast` run contains both cases:

| | volumes | assignment (`GPU_CPU_ratio = 15`, `NGPUs = 1`, `Nt = 32`) | reproducible? |
|---|---:|---|---|
| `pa` (down) | 10 | 10 ≤ 15 → **all GPU** | **yes — 0/240 params differ** |
| `ap` (up) | 138 | 45 GPU, **93 ITK CPU** | **no — 1918/3312 params differ** |

Confirmed by `benchmark/scripts/demo_diffprep_nondeterminism.sh`, a self-contained,
shareable reproducer. This also independently corroborates that the CUDA kernels are deterministic
(consistent with every golden vector replaying bit-exactly).

**Making it deterministic, and the cost.** `OMP_NUM_THREADS=1` forces every volume onto the GPU
path with **no code change** (`Nt = 1` makes `max_t_per_pass = GPU_CPU_ratio*NGPUs`, so no pass
exceeds the GPU quota) — but it single-threads the *entire* pipeline, and a `fast` CUDA run uses
~17× parallelism. The targeted alternative is one constant at `DIFFPREP.cxx:589`: raise
`GPU_CPU_ratio` above `Nvols` so the CPU branch is never taken, costing ~3× on registration only
(138 volumes serially on one GPU vs 3 passes of 15-GPU + 31-CPU in parallel). **Neither is proposed
— `DIFFPREP.cxx` is upstream and outside the port's remit.** Recorded so the cost is known.

(Metric sampling is ruled out: 100 %, strategy `NONE`, no reseeding.)

**FLOOR RE-DERIVED 2026-08-18 — the original numbers were wrong.** The floor above (r ≥ 0.999518,
`max|diff|/p99` ≤ 3.68) came from **n = 2** and took the better pair. Measured properly over
**6 pairwise comparisons from 4 independent CUDA runs** (`benchmark/scripts/measure_floor.py`,
`benchmark/fast/CUDA{,_C,_D,_E}`):

| statistic | min | max | mean | sd |
|---|---:|---:|---:|---:|
| `max\|diff\|/p99` | 4.860 | 5.846 | 5.394 | 0.496 |
| Pearson r | 0.999428 | 0.999715 | 0.999569 | 0.000115 |

**CUDA fails the old gate against itself in all 6 pairs** (every d exceeds 3.68; two r values fall
below 0.999518).

**SUPERSEDED AGAIN 2026-08-20 — and this table is the stale one.** The 08-18 derivation cited
`benchmark/fast/CUDA{,_C,_D,_E}`, but `benchmark/fast/CUDA` was **re-run on 08-19**, so those
constants stopped being reproducible from the directories named as their source; an audit found them
unreproducible. The floor was re-derived from 6 pairs over 4 runs **all sharing one binary**
(`e692f7d4`), and `measure_floor.py` now refuses a mixed set. Current values, pinned in
`benchmark/scripts/floor_fast.json`:

| statistic | min | max | mean | sd | **gate (mean ∓ 3 sd)** |
|---|---:|---:|---:|---:|---:|
| `max\|diff\|/p99` | 3.483 | 5.846 | 5.161 | 0.926 | **≤ 7.939** |
| Pearson r | 0.999485 | 0.999715 | 0.999579 | 0.000105 | **≥ 0.999264** |

An earlier version of this section quoted **r ≥ 0.999224, d ≤ 6.88** as what `compare_outputs.py`
uses. It does not, and has not since 08-20 — the constants are `FLOOR_R = 0.999264`,
`FLOOR_D = 7.939`. Read the script, not this paragraph, if the two ever disagree again.

**And none of it is a gate any more.** The end-to-end comparison was demoted to `--smoke` on
08-19 (§7); the floor is retained for reference only.

This is a **correction of an inadequate measurement, not a relaxation**: it was derived from CUDA
runs only, with no WebGPU output consulted, and it made the gate *stricter* in the sense that it
now means something. The standing rule against relaxing tolerances is intact.

**Three consequences that must not be forgotten:**

1. Per-voxel max-diff is the **wrong** end-to-end metric. `max|diff|/p99 = 3.68` between two runs
   of the *same code* proves it measures edge placement, not correctness.
2. The end-to-end gate is a comparison **against the floor**, not an absolute:
   `r_webgpu ≥ r_floor` with a diff distribution inside the run-to-run spread. Use bare
   `compare_outputs.py --floor` (or `--floor-r=`/`--floor-d=`). **Not** `--floor=<value>` — that
   hits a back-compat branch which sets only the diff gate and silently leaves Pearson at the
   nominal 0.99999 this very section calls unachievable. The plan's nominal `r ≥ 0.99999` / `max|diff|/p99 ≤ 1e-3`
   **cannot be met by CUDA against itself** and must never be applied as written.
3. The trustworthy gate is the **stage-isolated / kernel golden vectors**. The full run is a smoke
   test only.

Evidence: the floor was re-derived from `benchmark/fast/CUDA{,_C,_D,_E}` (6 pairs). Note
`benchmark/fast/DET_A` is **not** part of it — it is an aborted `DETERMINISTIC_GPU` +
`--disable_itk_threads` run (0-byte `time.txt`, no final output) and `DET_B` does not exist. An
earlier version of this line cited them as the CUDA-vs-CUDA pair; that was wrong.

### 5.5 Two smaller traps worth remembering

- **Capture quotas are per shape class.** "First N calls per op" biased capture toward the first
  pipeline phase: DIFFPREP smooths *scalars* and runs first, so `GaussianSmoothImage` never got a
  slot for DRBUDDI's *3-component fields*, leaving the field path (including `AdjustFieldBoundary`)
  with **zero coverage**. Quotas are now keyed `GaussianSmoothImage.v1.*` / `.v3.*` (and likewise
  `ResampleImage.v1/.v3`). Do not revert this.
- **Row padding — NOT uniformly "small", corrected 2026-08-20.** CUDA's reductions size themselves
  as `pitch/sizeof(float)*sy*sz`, which includes row padding; `Allocate()`
  (`cuda_image.cxx:114-118`) memsets `extent.width` bytes per row, **not `pitch`**, so that padding
  is *never initialised* — and several wrappers do not memset at all (`AddImages`,
  `MultiplyImage(s)`, `DivideImages`, `NegateField`). WebGPU's flat buffers have no padding and the
  port reduces exactly `NumFloats()`/`NumVoxels()` — the *intended* extent, which is what the
  CPU/ITK path computes. Structurally the same situation as §0.0b.

  An earlier version of this note called the divergence "small". That is true only of the
  **summations** (`SumImage`, `AddToUpdateField`'s `ScalarFindSumSq`), where padding contributes
  additively and is dwarfed by the data. It is **false for `PreprocessImage`'s
  `ScalarFindMax`/`ScalarFindMin`** (`cuda_image_utilities.cu:1162,1176`) and `InvertField`'s
  `ScalarFindMax` (`:1069`): **max/min do not average**, so a single garbage float in the padding
  sets the result outright and rescales the whole image. The error is unbounded in principle.

  **In practice it is dormant, and that is measured, not assumed:** all 132 `fast` and 118 `medium`
  golden vectors replay **bit-exactly** under `gpu_replay`, and two CUDA runs of the isolated DRBUDDI
  Step2 from a frozen fixture are bit-identical. Both would be impossible if the padding varied
  between runs, so on this allocator it reads as zero. Two consequences follow that a reader should
  not have to re-derive:
    - Zero padding still *participates*: `ScalarFindMax` seeds at `-1`, so CUDA's `PreprocessImage`
      max is `max(true_max, 0, -1)`. On an image whose values are all negative CUDA returns `0`
      where the port returns the true (negative) maximum. No such image arises in the pipeline —
      `PreprocessImage` runs on intensity images — so this is recorded, not guarded against.
    - It is a *candidate* contributor to CUDA-vs-CUDA end-to-end non-reproducibility (§5.4), which
      is currently attributed wholly to the ITK CPU registration path. The bit-exact replays argue
      against it, but it has not been separately excluded on a machine with a dirtier allocator.

- **Metric images are memset to BYTE 1 on the reference side, not 0** (found 2026-08-20, previously
  unrecorded). `compute_metric.cu:836, 1326, 1719, 2133, 2440` call
  `cudaMemset3D(metric_image, 1, extent)` — the same `int`-byte-value parameter as §0.0's
  `FillBuffer`, so every float starts at `0x01010101` = **2.3694e-38**, not `0.0`. The port
  zero-fills via `Allocate()`. MSJac and CCJacS write the metric only on the interior, so CUDA's
  untouched boundary voxels (plus its row padding) each contribute ~2.4e-38 to the reduced metric
  where the port contributes 0 — of order **1e-32 total against metric values of order 1**. Far
  below any tolerance and invisible to every golden vector; recorded so nobody re-derives it or
  mistakes it for a defect.

Other implementation notes: Gaussian smoothing deliberately reads taps straight from the buffer
instead of staging lines in workgroup memory (arithmetically identical, and avoids tying workgroup
size to an image dimension that can exceed the 256-invocation portable limit); boundary taps are
**skipped, not renormalised**, exactly reproducing the reference's truncated-kernel edge behaviour.

---

## 6. Pre-existing upstream bugs — NOT caused by this work

### 6.1 The standalone `DRBUDDI` executables — FIXED, and verified 2026-08-20

**Was:** `src/main/DRBUDDI.cxx` under `-DDRBUDDIALONE` did `this->stream = &((*stream));`, reading
the uninitialised member `stream`, dereferencing it and taking its address, against
`src/main/DRBUDDIBase.h:82`. The first log write faulted, so every standalone target segfaulted on
any input.

**Now:** line 50 reads `this->stream = &std::cout;`. Applied by the repo owner.

**Verified with real arguments, which matters** — an earlier "verification" ran the binaries with no
arguments and concluded the fix held. That test was worthless: `DRBUDDI_main.cxx:103` returns
`EXIT_FAILURE` before the constructor at `:118`, so it could not reach the fixed line at all.
A real invocation now runs:

```bash
bin/DRBUDDI_cuda --up_data ap.nii --up_json ap.json --down_data pa.nii
#   -> "Starting DRBUDDI Processing..." (written THROUGH this->stream, so the fix took)
#   -> 385 log lines, DRBUDDI diffeo iterations converging, no crash
```

`DRBUDDI_webgpu` reaches the same point. This **unblocks the isolated-DRBUDDI replay** that M5
records as blocked.

**But standalone DRBUDDI needs inputs `TORTOISEProcess` normally generates, and fails badly without
them.** It expects a `<basename>.bmtxt` next to each NIfTI; the datasets under `benchmark/` carry
only `.bval`/`.bvec`, because TORTOISE writes the b-matrix during import. With it missing:

```
vnl_matrix<T>::read_ascii: Called with bad stream
Segmentation fault in v3p_netlib_dsvdc_
  <- vnl_svd<double>::vnl_svd
  <- DRBUDDIBase::CreateCorrectionImage
  <- DRBUDDI::Step0_CreateImages
```

`CreateCorrectionImage` does not check that `read_ascii` succeeded before handing the matrix to
`vnl_svd`. **This is a second, distinct upstream bug**, unrelated to the stream fix and still
present. It is not caused by the port and is not the port's to fix — but anyone acting on "standalone
DRBUDDI works now" will hit it, so it is recorded here. Generate a `.bmtxt` from the `.bval`/`.bvec`
(6 columns: `bxx, 2bxy, 2bxz, byy, 2byz, bzz`) and it proceeds normally.

### 6.2 Open review item (from `audit_response.md`, deliberately not actioned)

`DRTAMAS_cuda` had to gain `gpu_capture.cxx` in its source list. This is a *consequence* of
instrumenting shared translation units (`cuda_image_utilities.cxx`, `compute_metric.cxx`,
`resample_image.cxx`, `gaussian_smooth_image.cxx`), which DRTAMAS already compiles and which now
reference `gpucap::Rec` — removing it produces undefined symbols. The clean elimination is a
compile-time switch (`TORTOISE_GPU_CAPTURE_ENABLED`, on only for pipeline targets), a ~23-site
mechanical change requiring the M1 golden vectors to be re-run. **Deliberately not done**; raised
as an explicit choice. The hooks are inert when `TORTOISE_GPU_CAPTURE` is unset.

### 6.3 Audit fixes already applied — do not undo

Full rationale in `audit_response.md`; the diff is in the working tree. Each of these is fixed:

| Fix | Where | What it was |
|---|---|---|
| `metric_common.wgsl` added to the embed `DEPENDS` | `TORTOISEV4/CMakeLists.txt` (wgsl embed block) | it is textually prepended into the metric shaders, so editing it left the generated headers stale and the build "succeeded" running old shader code |
| `Compare()` throws on output size mismatch | `src/tools/GPUReplay/gpu_record.h` (`Compare`) | comparing the common prefix let a wrong-shaped result report PASS |
| NaN divergence detection | `gpu_record.h` — see **§4.5** | `std::max(x, NaN)` returns `x`, so NaN differences were invisible and PASSED |
| `Upload()` no longer reads past the caller's buffer | `webgpu_context.cxx` (`Upload`) | rounding the byte count up to a multiple of 4 over-read the source; stages through a padded copy now |
| Counters made atomic | `webgpu_context.cxx` — `std::atomic<uint64_t> g_bytes, g_errors` | raced under OpenMP |
| Capture manifest records the real variant directory | `src/cuda_src/gpu_capture.cxx:230-234` | `op.v<N>.<seq>` records were written as `op.<seq>` and became unlocatable |
| WebGPU defs/includes scoped **per target** | `CMakeLists.txt` (`WEBGPU_DEFS`/`WEBGPU_INCS`) and each `target_*` call | directory-wide `-DUSEGPU` was being applied to the plain CPU targets |
| `wgpuctx::Dispatch` aborts on failed validation | `webgpu_context.cxx` (`Dispatch`/`DispatchAt`) | a rejected dispatch leaves its output untouched, indistinguishable from a legitimately zero update field. Aborting matches `gpuErrchk` (§0.0) |
| Failed readback aborts instead of zero-filling | `webgpu_context.cxx` (`Download`) | previously dereferenced an unmapped buffer; the intermediate "zero-fill" fix was itself reverted as over-defensive (§0.0) |
| float-vs-double faithfulness | `src/webgpu_src/quadratic_transform_image.cxx:78-90` | the reference copies params into a `float params_arr[]` and tests **that float copy** against `1E-10` for the `do_cubic`/`phase` decision. The port compared the doubles, so a value that rounds across the threshold in float picked a different branch |
| `DRBUDDI_webgpu` target added | `CMakeLists.txt` (`DRBUDDI_webgpu` block) | the goal contract names both executables. (It no longer crashes at startup — §6.1 is fixed and verified) |
| `BytesAllocated()` → `BytesAllocatedCumulative()` | `webgpu_context.{h,cxx}` | it never decreases and is *not* live usage; the peak-GPU figures come from the 1 Hz `nvidia-smi` sampler, never this counter |
| `.gitignore` `benchmark/` → `benchmark/*` | `.gitignore` | git does not descend into an excluded *directory*, so the negations for `scripts/`, `README.md` etc. were inert. Now the harness is committable and the datasets stay excluded. (`milestones/` was subsequently moved back OUT of the tracked set — see §2.1) |

---

## 7. Status

Milestone gates. No milestone could start before the previous one's gate passed:

| # | Gate that had to pass |
|---|---|
| M0 | a clean rerun of `run_cuda_reference.sh <ds> CUDA` reproduces the recorded layout and profile |
| M1 | CUDA replay of every captured record is **byte-identical** to its captured output; unknown/truncated/incompatible records fail clearly |
| M2 | CPU, CUDA and WebGPU all configure and build; startup rejects a non-NVIDIA or non-discrete adapter; allocation smoke test covers the `medium` volume |
| M3 | every P1 captured vector passes the §5 tolerances on `fast` and `medium` |
| M4 | all reachable P2 vectors pass; no unaccounted reachable wrapper remains |
| M5 | all P3 vectors pass; isolated DRBUDDI output meets the tolerances |
| M6 | `compare_outputs.py` exits zero for `fast`, `medium` **and `slow`**; CPU and CUDA builds still pass; benchmark evidence present |

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| M0 | CUDA baseline: build/revision/adapter/invocation/timings/memory for all 3 datasets; immutable stage cache; `slow` data blocker | **done** | `benchmark/README.md`, `benchmark/<ds>/CUDA/provenance.json` |
| M1 | Capture hooks, schema v2, `gpu_replay` + registry, one record per reachable wrapper | **done, gate passed** | `benchmark/milestones/M1.md` — 20/20 ops, CUDA self-replay bit-exact, 6/6 negative tests |
| M2 | `USEWEBGPU` plumbing, pinned Dawn, `USEGPU` guards, `GPUIMAGE`, WGSL embedding, adapter policy | **done, gate passed** | `benchmark/milestones/M2.md` |
| M3 | P1 kernels: warp, resample, quadratic transform, Gaussian smooth | **done, gate passed** | `benchmark/milestones/M3.md` — counts as recorded at M3 close-out; the suite has grown since (see its HISTORICAL COUNTS banner) |
| M4 | P2: all reachable `cuda_image_utilities` + `compute_entropy` wrappers | **done, gate passed** | `benchmark/milestones/M4.md` |
| M5 | P3: MSJac, CC, CCSK, CCJacS + `InvertField`/`ComposeFields` | **done, gate passed** — all 20 reachable ops ported. (The isolated-DRBUDDI replay this row once recorded as blocked is **unblocked**: §6.1 is fixed and the Step2 gate now runs and passes) | `benchmark/milestones/M5.md` |
| M6 | Full WebGPU pipeline on `fast`/`medium`/`slow`, comparison JSON/CSV, `benchmark/README.md` updated | **PARTIAL** — `fast` passes (**at parity within CUDA's own run-to-run spread** — 12 m 54 s against a CUDA range of 11 m 49 s – 13 m 10 s; peak GPU 0.93×) and `medium` passes (**0.90×** — faster than CUDA). `slow` **never run**, and the `medium`/`slow` CUDA-vs-CUDA floors were never measured. The contract names all three datasets, so this is incomplete | `benchmark/milestones/M6.md` |

**All gates currently pass on `fast`.** Re-validated after the round-3
audit fixes via `benchmark/scripts/revalidate.sh --compare` (all three builds clean):

| Suite | Result |
|---|---|
| CUDA self-replay (`gpu_replay --all`, `fast`) | **132/132 bit-exact** |
| `webgpu_replay` vs CUDA, `fast` | **132/132** |
| `webgpu_replay` vs CUDA, `medium` | **118/118** |
| `webgpu_replay` vs exact analytic, `benchmark/synthetic_vectors` | **25/25** |
| **isolated DRBUDDI Step2** (`drbuddi_isolated.sh`) | Reference side is **exact**: two CUDA runs from a frozen fixture are bit-identical (815 = 815 iterations), so a difference is attributable to the backend. Port comparison is a *regression* gate vs the NOFMA control — magnitude 1.08–1.17× on every statistic. **Do NOT cite the differing-voxel fraction**: it is saturated (98.5–99 % of non-zero support for both the port and a mere compiler flag) and cannot discriminate — `M6.md` retracts that reasoning. |
| end-to-end `fast` — **SMOKE TEST, not a gate** | Demoted 2026-08-19. Four runs of identical code span r = 0.999067–0.999314; the old gate (0.999224) sat *inside* that spread, so it passed 1 of 4 on the draw. NOFMA (CUDA + one compiler flag) sits at 0.999208 and fails it too. Now `--smoke`: **r ≥ 0.9986**, spread ≤ 20, catching gross breakage only (0.99 was tried first and
missed 100 % of single-volume zeroings; 0.9986 still misses 70 of 148 — see `END_TO_END_VARIABILITY.md`). Full record: `benchmark/END_TO_END_VARIABILITY.md` |

**The end-to-end statistic is no longer a gate at all** (2026-08-19). The WebGPU run
also routes 93 of 138 volumes through the non-reproducible ITK CPU path, so the same build measured
r = 0.999067 and r = 0.999314 on consecutive runs. Use the **isolated DRBUDDI comparison** above,
which is bit-exact on the reference side, and the per-kernel golden vectors. `revalidate.sh` fails
the end-to-end gate outright when the binary is newer than the output.

**If a gate fails — do not relax it.** That rule stands, but the paragraph that used to sit here
(arguing about a 35.0978 % differing fraction and a 5.028 spread against an n = 2 floor) is
**entirely superseded** and has been removed: the floor was re-measured twice, the end-to-end
comparison was demoted to a smoke test, and the row-padding hypothesis was ruled out (entropy reduces
over the histogram, never over pitched image memory). Current end-to-end figures on the current
binary: **r = 0.999102, max|diff|/p99 = 4.940, differing 35.0781 %** — the last of which is
indistinguishable from CUDA-vs-CUDA's 35.08 %. Full analysis and all five retractions:
`benchmark/milestones/M6.md`.

Artefact counts (on disk, and what `revalidate.sh` pins): **132** golden-vector records on
`fast` covering all 20 ops; **118** on `medium` covering **18** ops (no structural image
there, so `ComputeMetric_CCJacS` and `ComputeMetric_CCSK` never run); **106** on `slow`
covering **16** ops (one DRBUDDI metric, so no `ComputeMetric_CC`/`AddToUpdateField` either) —
the only large-matrix coverage, captured 2026-08-21; **25** synthetic exact-reference records.
**381** distinct records in total. Counting
`GaussianSmoothImage.v1/.v3` and `ResampleImage.v1/.v3` as distinct makes `medium` look
like 20 ops - it is not.

### 7.1 RESOLVED (mostly) — the WebGPU slowdown was host round-trips

**HISTORICAL — NOT REPRODUCIBLE TODAY.** These figures were measured when the `fast` capture held
**44** records; `--bench` iterates the whole capture directory, which now holds **132**, so none of
the absolute times, the 1.54× ratio, or `PreprocessImage`'s share reproduces as written. The
*conclusions* (host round-trips dominated; the five listed fixes) stand and are corroborated
end-to-end. Re-measure before quoting any number here.

**READ THE CAVEATS BEFORE USING THESE NUMBERS.** Measured per-op on the then-44 golden vectors
(`webgpu_replay --bench` / `gpu_replay --bench`):

> **Caveat 1 — the two benches are NOT symmetric, and the bias runs against WebGPU.**
> `gpu_replay_main.cxx`'s `CHECK_OUT` calls `Compare(...)` with no `tol` (defaults 0);
> `webgpu_replay_main.cxx`'s passes `out.tol`, and `Compare`'s exceedance sweep is gated on
> `if(tol > 0)`. So **WebGPU runs two passes over every output buffer and CUDA runs one.** An
> earlier version of this section claimed the ratio was "conservative" because both paid an
> identical term - that was exactly backwards.
>
> **Caveat 2 — the suite is dominated by one op that is not a pipeline hotspot.**
> `PreprocessImage`'s two records are 237x237x128 = 7.19 M voxels each; they are **75.4 % of all
> compared output floats** in the suite (next largest op: 5.3 %). Its wrapper issues only 2
> reductions + 1 dispatch; the measured time is mostly the harness's own 27.4 MiB upload, 27.4 MiB
> download and comparison. **Excluding it, the other 19 ops are ~1.15x CUDA** - and that is still
> measured while paying caveat 1.
>
> In the real pipeline `PreprocessImage` is called only from
> `DRBUDDI_Diffeo::SetImagesForMetrics` (`DRBUDDI_Diffeo.cxx:886-968`), a one-shot setup pass on
> images already resident on the device. Optimising it optimises a benchmark artefact.
>
> **Caveat 3 - and this is the big one: the per-op bench is a poor proxy for pipeline cost.**
> It times each op ONCE with fresh host transfers. The pipeline runs many dispatches per DRBUDDI
> iteration on resident data, so it is dominated by accumulated per-dispatch overhead that a
> single-op measurement barely samples. Trust `ITERATION_TIME`, not this table, for pipeline
> performance.

| | CUDA | WebGPU before | WebGPU after |
|---|---:|---:|---:|
| Total over 20 ops | 63.9 ms | 186.9 ms | **98.7 ms** |
| ratio vs CUDA | 1.0× | 2.9× | **1.54×** |

Earlier the DRBUDDI *per-iteration* gap was 33×. Five changes, each gated on all four replay suites:

| Change | Where | Effect |
|---|---|---|
| Removed the per-dispatch host wait | `webgpu_context.cxx` `DispatchAt` | the largest single win. It mirrored `cudaDeviceSynchronize()` — the reference's *shape* but not its cost. Never load-bearing: queue ops are ordered and Dawn inserts read-after-write barriers; only `Download` must reach the host, and it already waits. `TORTOISE_WEBGPU_SYNC_EACH_DISPATCH=1` restores it for bisecting |
| Device-side `ClearBuffer` | `wgpuctx::Zero` | was a host vector of zeros + full upload on **every** image allocation |
| Device-to-device `CopyBuffer` | `gpu_image.cxx` `DuplicateFromCUDAImage` | was a blocking map + two full host transits, several times per DRBUDDI iteration |
| Device-side min/max | `image_utilities.cxx` `PreprocessImage` | was a **full blocking readback of every voxel** to reduce on the host. Needed a new `ReduceOp::Min`; min/max are order-independent and involve no rounding, so results are bit-identical |
| Cached small readback staging buffers (≤64 KiB) | `webgpu_context.cxx` `Download` | a reduction returns one float; creating and destroying a staging buffer for 4 bytes dominated. `InvertField` 11.5→4.0 ms, `ComputeJointEntropy` 4.0→0.8 ms |

Deliberately bounded: the staging cache is capped at 64 KiB per size and is **not** a general buffer
pool, because the per-op allocate/free pattern is what keeps peak GPU at 0.93× CUDA (§7.4).

**Still open.** `PreprocessImage` is 63.9 ms — **64.7 %** of WebGPU's total and 1.9× CUDA's 33.7 ms
(itself the slowest CUDA op). It runs two reductions plus a kernel; fusing min and max into one pass
would halve its reduction work. Dispatch batching across ops was **not** done — it is the riskiest
change (it alters when buffers become visible to later dispatches) and the remaining gap no longer
justifies it.

**End-to-end confirmed since.** **Wall clock: quote the SPREAD, not a single-run ratio.** CUDA's own `fast` runtime varies
**11:49.20 – 13:09.95** across 4 runs of one binary (mean 12:25.8, sd 36.2 s — a ~11 % spread).
The WebGPU run at **12:54.57** therefore sits **inside CUDA's own range** (+0.79 sd), giving
**1.04× the CUDA mean** — and 0.98× against CUDA's slowest run. Successive point estimates of
1.14× / 1.16× / 1.18× all appeared in these documents at various times; each compared single runs of
a quantity with 11 % spread, which is the same n = 2 error §5.4 records for the accuracy floor.
State it as: **`fast` is at parity with CUDA within run-to-run noise** (n = 4 CUDA, n = 1 WebGPU —
the port's own spread is unmeasured, so this is not a claim that it is faster). `medium`
`medium` 19 m 56 s (**0.90× — faster than CUDA**). See `benchmark/milestones/M6.md`.

### 7.1b Historical — the original diagnosis

**Measured, on `fast`, same host, same `gpu_env.sh` pinning.** Per-iteration times are printed by
DRBUDDI itself (`ITERATION_TIME` in `run.log`), so this is a direct like-for-like comparison:

| Stage 25/28, `fast` | CUDA (`benchmark/fast/CUDA/run.log`) | WebGPU (`benchmark/fast/WebGPU/run.log`) |
|---|---:|---:|
| iteration 1 | 0.72 s | 16.53 s |
| iterations 2-4 | 0.73 / 0.80 / 2.06 s | 89.08 / 89.54 / 89.28 s |

Across the whole CUDA run, iterations range **0.01 s – 2.13 s**. WebGPU is sitting at **~89 s** per
iteration in the same stages — roughly **40-120×** slower. Whole-run reference: CUDA `fast` finished
in **11:37** wall (`benchmark/fast/CUDA/time.txt`); the WebGPU run was still inside DRBUDDI stage
25/28 at 35 minutes.

**This is a performance problem only. Every replay suite passes, so the arithmetic is right.**

Suspected causes, all identified in the source and all *deliberate* mimicry decisions taken during
the port:

| Cause | Where | Detail |
|---|---|---|
| Per-dispatch submit-and-wait | `webgpu_context.cxx:425-431` (`DispatchAt`) | every single dispatch does `Submit` + `OnSubmittedWorkDone` + spin on `ProcessEvents()`. This mirrors the CUDA wrappers' `cudaDeviceSynchronize()` after every kernel — but a CUDA sync is far cheaper than a WebGPU queue round-trip |
| Per-call buffer allocation | `src/webgpu_src/reductions.cxx:29-50` | each `Reduce()` creates **4** buffers (2 storage, 2 uniform), issues **2** synchronised dispatches, then a `Download` (which itself creates a staging buffer, maps, spins, unmaps, destroys). DRBUDDI calls `Reduce` several times per metric per iteration — see the 12 call sites in `compute_metric.cxx`, `image_utilities.cxx`, `invert_field.cxx` |
| Host round-trip for a device-to-device copy | `gpu_image.cxx:35-37` (`DuplicateFromCUDAImage`) | downloads the whole image to a `std::vector<float>` and re-uploads it |
| Host round-trip to zero a buffer | `webgpu_context.cxx:267-270` (`wgpuctx::Zero`) | allocates a host vector of zeros and `WriteBuffer`s it |

Remedies, in rough order of expected payoff:

1. **Batch dispatches into a single compute pass** and sync only where a result is actually read
   back to the host.
2. **Pool/reuse buffers** instead of allocating per call (especially the reduction partials, the
   uniform blocks, and the download staging buffer).
3. **Device-side zero-fill** (a trivial fill shader or `ClearBuffer`) instead of uploading zeros.
4. **Device-to-device copy** (`CopyBufferToBuffer`) in `DuplicateFromCUDAImage`.

**All four are performance-only changes and each one needs a full golden-vector re-run afterwards**
(`gpu_replay --all` plus the three `webgpu_replay` suites) — batching in particular changes when
buffers are visible to subsequent dispatches, which is exactly the kind of change that can alter
results without altering any arithmetic.

Note the tension with §7.4: the sizing note says to preserve the per-op allocate/free pattern
because it keeps the footprint low. Pooling trades memory for latency; measure the peak against the
`nvidia-smi` sampler when you do it.

### 7.2 As of 2026-08-18 — **HISTORICAL SNAPSHOT, superseded**

> Kept because it records how the performance problem looked before it was fixed. Every
> performance figure below is stale: `fast` is now **14 m 14 s = 1.18× CUDA**, not 12.6×.
> The `pgrep -f` note is also wrong in the other direction — see the correction inline.

- The **end-to-end WebGPU run on `fast` completed**: exit 0, 2 h 26 m 24 s wall (12.6× CUDA's
  11 m 38 s), peak host RSS 3.67 GiB (1.04× CUDA), **peak GPU 4 092 MiB (0.93× CUDA)**.
  `ap_TORTOISE_final.nii` written under `benchmark/fast/WebGPU/`. The sub-CUDA GPU footprint
  confirms §7.4's prediction and rules out a buffer-cache leak.
- `/usr/bin/time` recorded **236 % CPU**, 15 553 s user / **5 263 s system** — so the slowdown is
  not the purely single-threaded stall an early spot-check suggested. The large system-time share is
  consistent with (but does not prove) the host-round-trip hypothesis in §7.1.
- **`benchmark/scripts/revalidate.sh`** is new: rebuilds all three configs and runs every gate in one
  command, refusing to start while a WebGPU job still holds the GPU. Use it after any change.
  (Note: match processes with `pgrep -f`, never `pgrep -x` — Linux truncates `comm` to 15 chars, so
  `TORTOISEProcess_webgpu` never matches by name. **But `pgrep -f <script>` also matches the
  shell running the check itself**, which reports a finished job as live; `revalidate.sh` resolves
  `/proc/<pid>/exe` instead of pattern-matching, and any new check should do the same.)
- `bin/DRBUDDI_webgpu` is **now built**. *(Superseded: the §6.1 startup segfault this bullet expected
  to persist was fixed on 2026-08-20 and both binaries were retested with real arguments — see §6.1.
  The no-argument test cited here could not reach the faulting line at all.)*
- **`TORTOISE_GPU_PROFILE=1` was specified in the original plan but NOT implemented.** There are no
  per-stage GPU timers anywhere in `src/` (`grep -rn TORTOISE_GPU_PROFILE src/` returns nothing).
  Consequently **no speedup claim may be derived from the per-stage table** in
  `benchmark/README.md`: without scoped timers, GPU time cannot be separated from the CPU-only work
  inside DIFFPREP. The only directly-measured GPU time available is DRBUDDI's own `ITERATION_TIME`
  lines (§7.1), which is why §7.1 uses those and nothing else.
- Working tree is **dirty**: 25+ modified tracked files plus untracked `.gitignore`, `CLAUDE.md`,
  `TORTOISEV4/cmake/`, `benchmark/`, `src/gpu_src/`, `src/webgpu_src/`, `src/tools/GPUReplay/`,
  `src/cuda_src/gpu_capture.{h,cxx}`. Base commit `9a65714`. **As of 2026-08-20 none of the port work was committed**; if you are
  reading this from inside a commit, that sentence has been overtaken — check `git log`.

### 7.3 Next steps

1. ~~**Resolve the M6 end-to-end failure.**~~ — **moot.** There is no end-to-end failure: the
   comparison was demoted to a smoke test (§7) and passes. Retained for the reasoning:
   (a) measure the `fast` CUDA-vs-CUDA floor properly — several more pairs, so the gate rests on a
   distribution rather than n = 2 set at its optimistic end; (b) ~~test the §5.5 row-padding hypothesis~~ — **done and ruled out**; entropy reduces over the
   histogram, never pitched memory. The divergence was localised to DIFFPREP registration and then
   shown to be the same class of perturbation a `-fmad=false` compiler flag produces.
2. ~~**Address §7.1.**~~ — **done.** The 12.6× slowdown that once blocked `medium`/`slow` is gone
   (`fast` at parity within CUDA's own spread, `medium` 0.90×). `slow` remains un-run, but on runtime grounds it no longer is.
   Re-run `revalidate.sh --compare` after any change.
3. ~~Re-run the `fast` end-to-end pipeline on the current binary~~ — **done 2026-08-20**, together
   with the `pa` and isolated-Step2 evidence, after the fidelity fixes changed the binary. All 13
   gates pass with zero skips (`benchmark/LAST_REVALIDATE.json`). `medium`/`slow` still pending.
4. Measure the CUDA-vs-CUDA floor for `medium` and `slow` — never measured, and the end-to-end gate
   is meaningless without it. Running those datasets before measuring their floors would produce
   numbers that cannot be graded.
5. Implement the `TORTOISE_GPU_PROFILE=1` scoped timers, or drop the per-stage speedup table from
   `benchmark/README.md`. Until one or the other happens, that table cannot support a speedup claim.
6. Decide (owner's call) on: ~~the §6.1 one-line `DRBUDDI` stream fix~~ (**applied by the owner**), the §6.2
   `TORTOISE_GPU_CAPTURE_ENABLED` switch, and whether `plan_*.md` / `audit_*.md` should be
   un-gitignored.

### 7.4 Sizing target for the port

Peak GPU memory is driven by **metric count × matrix size**, not dataset size — but the coefficient
is NOT as stable as this note once claimed. "Consistent to within 2 %" was true of `fast` and
`medium` alone; across all four measured configurations it spans **18 %**:

| configuration | metrics | Mvox | metric-Mvox | peak GPU | MiB per metric-Mvox |
|---|---:|---:|---:|---:|---:|
| `fast`, 1 structural | 4 | 0.580 | 2.32 | 4 378 | 1 887 |
| `medium` | 2 | 1.588 | 3.18 | 5 940 | 1 871 |
| `fast`, **2 structurals** | **6** | 0.580 | 3.48 | 5 698 | **1 637** |
| `slow` | 1 | 1.803 | 1.80 | 2 790 | 1 547 |

The coefficient *falls* as metric count rises, so some buffers are evidently shared across metrics
and the scaling is sublinear in metric count. Fitting an affine model does not help — least squares
returns a slightly negative intercept, which is unphysical — so treat it as proportional with ~20 %
scatter and **plan with the maximum coefficient (1 887), not the mean**.

Worst case on that basis — 4 metrics at `slow`'s matrix — is **≈13.3 GiB against a 16 GiB card**.
Still feasible, but the margin is thinner than a mean-based estimate suggests.

The 2-structural configuration is reproducible with `STRUCTURALS=all` (opt-in;
`run_cuda_reference.sh`). `--structural` is repeatable and
`DRBUDDI_Diffeo::SetDefaultStages` does `for(int s=0;s<Nstr;s++)` for CCSK and CCJacS, so each
additional structural adds two metrics — the only way to exercise those two metrics beyond a single
instance. CUDA peaks: `fast` 4 378 MiB,
`medium` 5 940 MiB, `slow` 2 790 MiB. The worst case is a large matrix carrying a structural *and*
enough down volumes for tensor fitting — a 4-metric run at `slow`'s matrix, which by the maximum
coefficient above is **≈13.3 GiB** against the card's 16 GiB. (An earlier ≈13.7 GiB here used the
rounded 1.9 GiB coefficient; 13.3 follows this section's own rule of planning with 1 887.) WebGPU drops pitched allocations (no row padding) and the `cudaArray`
duplication, **so its footprint should be equal or slightly lower; a regression indicates a leak in
the buffer cache.** Preserve the CUDA wrappers' per-op allocate/free pattern rather than replacing
it with long-lived pools.
