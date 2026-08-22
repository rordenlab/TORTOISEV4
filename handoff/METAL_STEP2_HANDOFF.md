# Isolated DRBUDDI Step2 — handoff for the macOS/Metal port

Regenerated 2026-08-22 on the Linux CUDA host, branch `optimize`.
CUDA `3e1dff9b`, WebGPU `d22fcefc`, deterministic CUDA `083e682f`.

**This document replaces an earlier version that was substantially wrong.** That version
claimed the frozen fixture was contaminated and that the isolated gate was unusable.
Both claims came from reading stale `DRB_*` directories (an older naming convention the
script no longer writes) instead of the directories the runs actually produced. Corrected
below; the retractions are listed at the end so nothing is silently changed.

---

## The isolated Step2 gate works. Use it.

**Measured: two CUDA runs of `drbuddi_isolated.sh run <tag> CUDA` from the same fixture
are BIT-IDENTICAL** — all five artefacts, `r = 1.00000000`, 0.00 % of elements differing.
No special build required. `--DRBUDDI_step 2` skips Step0/Step1, and everything Step2
depends on is then fixed, so any difference is attributable to the GPU backend.

### The clean cross-backend delta (Linux, same machine, same fixture)

This is the number your Metal delta should be compared against:

| artefact | max\|d\| | max\|d\|/p99 | rel_rms | Pearson r | differ% |
|---|---:|---:|---:|---:|---:|
| `deformation_FINV.nii.gz` | 4.947 | 0.858 | **0.0483** | **0.99887** | 96.7 |
| `deformation_MINV.nii.gz` | 4.317 | 0.709 | 0.0488 | 0.99884 | 96.7 |
| `blip_up_b0_corrected.nii` | 8864 | 0.702 | 0.0168 | 0.99982 | 86.4 |
| `blip_down_b0_corrected.nii` | 5312 | 0.474 | 0.0169 | 0.99982 | 85.9 |
| `b0_corrected_final.nii` | 3698 | 0.329 | 0.00775 | 0.99996 | 85.4 |

Since CUDA-vs-CUDA is exactly zero, that whole table is backend difference — a correct
second backend landing in this range is a pass. Note the deformation fields differ by
~5 % RMS despite bit-exact kernels: Step2 is a fixed-point iteration and amplifies
last-ulp differences. High `differ%` with low `rel_rms` is the expected signature.

---

## What IS nondeterministic: DRBUDDI Step1, end-to-end only

Two **end-to-end** runs of a `-DDETERMINISTIC_GPU=1` build, both routing all 138 volumes
to the GPU (`gpu_vols 138 cpu_vols 0`), still differ — 35 % of voxels, DRBUDDI iteration
counts 901 vs 731. Comparing every intermediate localises it exactly:

| | file |
|---|---|
| IDENTICAL | `ap_proc.nii` (import/denoise/Gibbs), `ap_proc_moteddy_transformations.txt` (DIFFPREP registration), `blip_up/down_b0.nii`, `*_FA.nii`, `*_quad.nii` (DRBUDDI **Step0**), `b0_str_registration_target.nii` (**Step1's input**) |
| DIFFERS | **`b0_to_str_rigidtrans.hdf5`**, **`bdown_to_bup_rigidtrans.hdf5`** (**Step1's output**), then everything downstream |

**Step1's input is bit-identical and its output is not.** DRBUDDI Step1's ITK rigid
registration is multithreaded and non-reproducible. `DETERMINISTIC_GPU` does not touch
it — that flag only pins DIFFPREP's volume routing (which it does correctly: the
transformations file is bit-identical).

This explains the macOS observation of `r = 0.999424` between two runs with **identical**
76/62 splits: it was never Metal, never routing, and never host arithmetic — the
CPU-only stages match across platforms at `r = 1.000000000000`.

**Consequence: end-to-end comparison cannot be made exact by any flag.** Use the
isolated Step2 gate, which sidesteps Step1 entirely.

---

## How to run it

```bash
# add a Metal case next to the existing ones in drbuddi_isolated.sh:
#     Metal)  EXE=$ROOT/bin/TORTOISEProcess_metal ;;

cp -a DRB_FIXTURE  <repo>/benchmark/fast/DRB_FIXTURE
chmod -R a-w       <repo>/benchmark/fast/DRB_FIXTURE

benchmark/scripts/drbuddi_isolated.sh run M1 Metal
python3 delta.py benchmark/fast/M1/ap_temp_proc DET_REFERENCE Metal CUDA
```

**Output path is `benchmark/fast/<TAG>`** — no `DRB_` prefix. The script writes
`benchmark/$DS/$TAG` and `compare` reads the same place; it is self-consistent. Empty
`DRB_*` skeletons in an older tree are from a superseded convention and are exactly what
misled the earlier version of this document.

Run it **twice** and `cmp` the two before trusting any cross-backend number. If Metal
does not reproduce itself, that is the finding, and no comparison against CUDA means
anything until it does.

---

## Payload

| path | size | what |
|---|---:|---|
| `DRB_FIXTURE/` | 2.1 GB | frozen Step2 input |
| `DET_REFERENCE/` | 383 MB | CUDA Step2 output, verified reproducible |
| `WEBGPU_REFERENCE/` | 384 MB | WebGPU Step2 output — the worked example above |
| `delta.py`, `compare_outputs.py`, `SHA256SUMS` | small | tooling + transfer verification |

`ap_TORTOISE_final.nii` (608 MB) is excluded: it is the pipeline's end-to-end output,
snapshotted into the fixture incidentally, and Step2 never reads it. Verified by running
Step2 from the trimmed fixture.

Not transferable via git: 10 files exceed GitHub's 100 MB per-file limit and 2.9 GB would
permanently inflate a 13 MB repository. Use `rsync`.

---

## The primary gate needs none of this

Per CLAUDE.md §3.1 the per-kernel golden vectors are self-contained — `webgpu_replay`
links no CUDA and every record carries CUDA's recorded output. Build with
`benchmark/scripts/make_reference_bundle.sh`. **If you are stalled, start there**: a
failure names the specific kernel, where Step2 only tells you the stage moved.

Suggested order: `webgpu_probe` (confirm adapter — the Metal defaults in §3.1 are
untested) → `synthetic` (25, analytic references) → `fast` (132, all 20 ops) →
`medium` (118) → `slow` (106) → Step2 last.

---

## Retractions from the previous version

1. "The fixture is frozen on disk but not in effect" — **false**. Two fresh CUDA runs are
   bit-identical. The `blip_up_b0.nii` difference was against a stale directory.
2. "Neither mechanism alone is sufficient" — **false**. `--DRBUDDI_step 2` alone suffices;
   `DETERMINISTIC_GPU` is not needed for the isolated gate.
3. "CUDA differs from itself by more than WebGPU differs from CUDA" — **false**, same
   cause. CUDA-vs-CUDA is zero.
4. Cross-backend `rel_rms 0.107 / r 0.99426` — **wrong figures**, computed from two stale
   runs built by different binaries. Correct: 0.0483 / 0.99887.
5. "A slower backend gets more nondeterminism" — the *mechanism* is real (the dynamic
   queue sizes the split from measured throughput, and CUDA 108/30 vs WebGPU 78/60 is
   measured), but it does not affect the isolated gate, which bypasses DIFFPREP routing.
   It applies to end-to-end comparison only.
6. `benchmark/fast/DRB_M1/...` in the run instructions — a path the script never produces.
