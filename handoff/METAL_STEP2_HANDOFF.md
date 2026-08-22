# Isolated DRBUDDI Step2 — handoff for the macOS/Metal port

Generated 2026-08-22 on the Linux CUDA host, branch `optimize` @ `ce37d17`.
Binaries: CUDA `3e1dff9b`, WebGPU `d22fcefc`.

---

## READ THIS FIRST — the reference is not exact any more

`CLAUDE.md` §7 and `drbuddi_isolated.sh`'s own header both claim that two CUDA runs
from the frozen fixture are **bit-identical**, so any Step2 difference is
attributable to the backend. **That is no longer true**, and it was discovered while
building this payload.

Measured, two CUDA runs, identical fixture, identical binary:

| comparison | `deformation_FINV` rel_rms | Pearson r | DRBUDDI iterations |
|---|---:|---:|---:|
| **CUDA vs CUDA** | **0.129** | 0.99170 | **901 vs 731** |
| CUDA vs WebGPU | 0.107 | 0.99426 | — |

**CUDA differs from itself by more than WebGPU differs from CUDA.**

### Why

`drbuddi_isolated.sh run` copies the fixture into a working directory and then
launches the *full* pipeline with `--DRBUDDI_step 2` (TORTOISEProcess has no flag to
skip DIFFPREP). DIFFPREP therefore still executes and **overwrites files inside the
copied `ap_temp_proc/`** — including `blip_up_b0.nii`, which is a Step2 *input*.
Verified: that file differs between the two runs.

This was previously harmless because DIFFPREP's GPU/CPU volume routing was static
index arithmetic and therefore deterministic. The `optimize` branch replaced it with a
**dynamic work queue** whose split is measured at runtime, so routing is now
timing-dependent, DIFFPREP output varies run to run, and that variation propagates into
Step2's inputs.

So: the fixture is frozen on disk, but not in effect.

### It is worse than that: a slower backend gets MORE nondeterminism

The dynamic queue sizes the CPU/GPU split from **measured** per-volume throughput. A
slower GPU therefore takes fewer volumes and pushes more onto the ITK CPU path - which
is the non-reproducible one. Measured on this host, same machine, same data:

| backend | `t_gpu` per volume | split (GPU/CPU) | volumes on the NONDETERMINISTIC path |
|---|---:|---|---:|
| CUDA | 0.51 s | 108 / 30 | 30 |
| WebGPU | 1.26 s | 78 / 60 | **60** |

**Metal will very likely also be slower than CUDA**, so expect it to land nearer the
WebGPU split - i.e. roughly twice as many volumes through the non-reproducible path as
the CUDA reference it is being compared with.

Two consequences, and the second is the nasty one:

1. Your Metal runs will vary run-to-run **more** than the CUDA reference does.
2. Because the two backends run **different splits**, a Step2 difference conflates the
   actual GPU-kernel difference with *differing amounts of ITK contamination*. It is
   not a clean backend delta in either direction.

This also means the comparison gets *less* trustworthy the slower your backend is -
exactly backwards from what you want while bringing a new backend up.

### Removing it entirely (recommended if you use Step2 at all)

`OMP_NUM_THREADS=1` makes the whole thing deterministic **with no rebuild**: with
`Nt = 1` the queue creates a single worker, that worker is the GPU thread, so every
volume takes the GPU path and no ITK CPU registration happens at all. Routing is then
fixed, DIFFPREP output is reproducible, and Step2's inputs are stable.

    OMP_NUM_THREADS=1 benchmark/scripts/drbuddi_isolated.sh run M1 Metal

Cost: it single-threads the rest of the pipeline, so the run is substantially slower.
For a correctness comparison that is the right trade. `-DTORTOISE_DETERMINISTIC_GPU=1`
achieves the same thing at build time while keeping other stages parallel.

**Both references shipped here were produced WITHOUT that flag**, so they carry the
variance described above. If you want an exact reference, ask for a regenerated pair -
it is ~25 minutes of GPU time on the Linux host.

### What to do about it

**Judge Metal against the CUDA-vs-CUDA floor, not against an absolute number.** This is
the same logic `CLAUDE.md` §5.4 applies to the end-to-end gate, which was demoted to a
smoke test for exactly this reason.

- Metal-vs-CUDA **at or below ~0.13 rel_rms / r ≳ 0.992** on the deformation fields:
  consistent with a correct backend.
- **Orders of magnitude larger**: a real bug.
- **Far smaller than the floor**: also suspect — it would suggest the comparison is not
  exercising what you think it is.

**The clean fix**, if you want an exact reference: rebuild with
`-DTORTOISE_DETERMINISTIC_GPU=1`. It routes every volume to the GPU path, making
DIFFPREP deterministic again. That switch was deliberately preserved through the
dynamic-queue rewrite for this purpose. Ask and it can be regenerated that way.

---

## The payload (2.8 GB, transferred out-of-band — see below)

| path | size | what |
|---|---:|---|
| `DRB_FIXTURE/` | 2.1 GB | frozen Step2 input. Copy to `benchmark/fast/DRB_FIXTURE` |
| `DRB_C1_ref/` | 383 MB | CUDA Step2 output — the 5 compared artefacts |
| `DRB_W1_ref/` | 384 MB | WebGPU Step2 output — same 5 |
| `delta.py`, `compare_outputs.py` | small | quantify a delta (see below) |

The 5 artefacts: `deformation_FINV.nii.gz`, `deformation_MINV.nii.gz`,
`blip_up_b0_corrected.nii`, `blip_down_b0_corrected.nii`, `b0_corrected_final.nii`.

`ap_TORTOISE_final.nii` (608 MB) was **excluded** — it is the pipeline's end-to-end
output, snapshotted into the fixture incidentally; Step2 never reads it. Verified by
running Step2 from the trimmed fixture before shipping.

### Use

```bash
cp -a DRB_FIXTURE  <repo>/benchmark/fast/DRB_FIXTURE
chmod -R a-w       <repo>/benchmark/fast/DRB_FIXTURE   # the script expects read-only

benchmark/scripts/drbuddi_isolated.sh run M1 Metal     # add a Metal case to the script
python3 delta.py benchmark/fast/DRB_M1/ap_temp_proc DRB_C1_ref Metal CUDA
```

Drive it through `drbuddi_isolated.sh`, not by hand. `run` deliberately deletes the
Step2 outputs from its working copy first, because the fixture contains CUDA-derived
copies of exactly the files being compared — a run that died early would otherwise
leave them in place and be scored as a pass.

---

## This is the SECONDARY gate. The primary one needs none of this.

`CLAUDE.md` §3.1: the per-kernel golden vectors are self-contained. `bin/webgpu_replay`
links no CUDA, and every record carries its own `out` tensors, which *are* the CUDA
reference. So on macOS:

```bash
bin/webgpu_replay --all <bundle>/records/fast        # 132
bin/webgpu_replay --all <bundle>/records/medium      # 118
bin/webgpu_replay --all <bundle>/records/slow        # 106
bin/webgpu_replay --all <bundle>/records/synthetic   #  25
```

grades Metal against exactly the reference and tolerances the Vulkan backend passes.
Build that bundle with `benchmark/scripts/make_reference_bundle.sh <out> [--tar]`.

**Prefer it, and if you are stalled, start here.** It is unaffected by everything
above - no DIFFPREP, no ITK CPU path, no scheduling, no fixture. Each record is a single
kernel invocation with its inputs and CUDA's recorded output, so a failure names the
kernel. The Step2 comparison can only tell you *that* the stage diverges, and right now
it cannot even do that cleanly.

Suggested order for bringing Metal up:

1. `bin/webgpu_probe` - confirm adapter selection before anything else.
2. `webgpu_replay --all records/synthetic` (25) - analytic references, exact answers,
   smallest volumes. Failures here are unambiguous.
3. `webgpu_replay --all records/fast` (132) - all 20 ported ops.
4. `records/medium` (118) and `records/slow` (106) - larger matrices, catches
   dimension- and pitch-dependent bugs.
5. Only then consider Step2, and only with `OMP_NUM_THREADS=1`.

Also note (§3.1): the Metal defaults are untested. Start with
`TORTOISE_WEBGPU_BACKEND=metal`, `TORTOISE_WEBGPU_ADAPTER_TYPE=integrated`,
`TORTOISE_WEBGPU_VENDOR_ID=0x106B`, and confirm with `bin/webgpu_probe` before anything
else — the default policy demands an NVIDIA *discrete* *Vulkan* adapter and will refuse
to run.

---

## Transport

Not via git: 10 of these files exceed GitHub's 100 MB per-file hard limit (largest
567 MB), and a 2.8 GB push would permanently inflate a 13 MB repository for every
clone — deleting the branch does not reclaim it. Options, best first:

1. **Direct copy** — `rsync -av <linux>:/home/chris/metal_drbuddi_bundle/ ./` . Minutes
   on a LAN, nothing persisted anywhere.
2. **GitHub Release assets** — up to 2 GB per asset, excluded from clones, deletable.
   Would need splitting into 2–3 assets.
3. **Git LFS** — works, but 2.8 GB exceeds the free 1 GB storage/bandwidth tier.
