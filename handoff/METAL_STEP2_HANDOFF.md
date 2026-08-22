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

**Prefer it.** It is unaffected by the contamination above, and it tells you *which*
kernel diverges. The Step2 comparison only tells you *that* the stage diverges.

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
