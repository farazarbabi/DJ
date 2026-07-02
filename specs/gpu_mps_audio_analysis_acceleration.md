# GPU (Apple-Silicon / MPS) Acceleration for the Analyze-Audio Step

Status: **spec / deferred** (not yet implemented)

## Context

The `analyze audio` step (`run_analysis()` in `src/dj_registry/adapters/local_analysis.py`)
is the pipeline bottleneck, dominated ~90% by Demucs `htdemucs` stem separation
(`src/dj_tagger/analyzers/vocal_stem.py`), which currently runs **CPU-only**.

CPU-side speedups already landed (they stay as the no-GPU fallback):
- **Lever A** — parallel-by-default analysis with torch/BLAS thread pinning
  (`_default_analysis_workers` / `_pool_worker_init` in `local_analysis.py`;
  CLI `-w` defaults to `0` = auto).
- **Lever B** — removed redundant librosa transforms, verified bit-identical
  (shared STFT/onset/chroma; dropped double MFCC; tonnetz from precomputed chroma).

Measured result of that work: **~1.7–1.9×**. It plateaus there because Demucs on
CPU is **memory-bandwidth-bound** — adding worker processes only makes them
contend for memory bandwidth (verified: thread pinning works, RAM ample, no
swapping). The installed torch is `2.9.0+cpu`, so no accelerator is used today.
Reducing Demucs work itself ("Tier 2": fewer slices / lighter model) is
**declined** because it changes cached feature values. The only remaining lever
to reach 5–10× is to move Demucs onto the GPU.

**Target device: Apple Silicon (Metal / MPS)**, written device-agnostically
(`cuda` → `mps` → `cpu`) so the same code runs on any machine.

**Facts confirmed while writing this spec:**
- `demucs.apply.apply_model` already accepts `device=`, `segment=`, `num_workers=`,
  `pool=` — GPU offload is a device move + passing `device`, not a rewrite.
- `htdemucs` `model.samplerate = 44100`; input is currently upsampled 22.05→44.1 kHz
  in `_load_stereo_for_demucs` (`vocal_stem.py:128`).
- CLAP (`laion_clap`, `src/dj_grouper/features/embeddings.py`) auto-uses the GPU
  when available — a free secondary win.

---

## Realistic speedup estimate (MPS)

Demucs is ~90% of per-track time, so overall speedup is Amdahl-bounded by the
remaining ~10% CPU work (librosa DSP/key). **Ceiling ≈ 10× even with an
infinitely fast Demucs**, unless that 10% is also parallelized/overlapped.

MPS Demucs speedup over the *same Mac's* CPU is chip-dependent and less
predictable than CUDA (partial MPS op coverage → some ops fall back to CPU):

| Demucs MPS speedup | New Demucs share | + 10% CPU | **Overall** | Typical chip |
|---|---|---|---|---|
| 2× | 45% | 55% | **~1.8×** | M1/M2 base |
| 3× | 30% | 40% | **~2.5×** | M2/M3 Pro |
| 5× | 18% | 28% | **~3.6×** | M-Max |
| 8× | 11% | 21% | **~4.7×** | M-Max/Ultra |

**Honest expectation on MPS: ~2.5–4.7× overall**, reaching ~5× only on high-end
M-Max/Ultra. Two ways to push toward the 5–10× goal:
1. **Pipeline overlap (Phase 2):** run the 10% CPU work (decode + librosa) on a
   CPU pool *concurrently* with MPS Demucs, so total ≈ `max(GPU_time, CPU_time)`
   instead of the sum. On a strong chip this can reach ~5–8×.
2. For reference, a **CUDA** GPU would hit 5–10× far more reliably
   (Demucs 15–40×); the code path is identical, only the device differs.

All numbers are estimates — **measure on the actual Mac** (benchmark step below)
and update this table before committing to a target.

---

## Design

### Phase 1 — Move Demucs to the device (captures most of the win)

Single accelerator owns one model; do **not** fan out N processes onto one GPU
(VRAM/queue contention). When an accelerator is present, the Demucs step runs in
**one process**; batching keeps the device busy.

1. **Device selection** — new helper in `vocal_stem.py`:
   `cuda` if `torch.cuda.is_available()`, else `mps` if
   `torch.backends.mps.is_available()`, else `cpu`. Cache it module-side next to
   `_DEMUCS_MODEL`.
2. **Model placement** — `_get_model()` (`vocal_stem.py:98`): `.to(device).eval()`.
3. **Batched inference** — replace the per-slice loop (`vocal_stem.py:268`, calls
   `_analyze_slice` 5×) with one `apply_model` call over a stacked
   `(n_slices, 2, N)` tensor moved to the device, then `.cpu()` the stems and
   compute the existing librosa metrics (`_stem_metrics`) unchanged. Per-slice
   results are numerically the same as 5 separate calls (no cross-slice coupling)
   — a batching change, not a feature change.
4. **Fallback** — if the device is `cpu`, keep exactly today's behavior (and the
   already-implemented Lever A CPU pool).

### Worker model in `run_analysis` (`local_analysis.py`)

- When an accelerator is selected, force the analysis pool to **1 worker** (the
  device is the bottleneck; multiple processes would fight over it). Extend
  `_default_analysis_workers()` to return `1` when the selected device is not
  `cpu`. The existing CPU multi-worker path stays for the no-GPU fallback.
- **Phase 2 (optional overlap):** a small CPU pool (decode + librosa +
  tensor prep) feeds a single MPS consumer that batches Demucs across tracks.
  This attacks the Amdahl 10% and is what lifts MPS toward the high end. Ship
  Phase 1 first, measure GPU utilization, add Phase 2 only if the device idles.

### MPS-specific gotchas (must-handle)

- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` (in-process before torch use) so Demucs ops
  MPS doesn't implement fall back to CPU instead of raising.
- **float32 only** — MPS has no float64; ensure tensors stay float32 (Demucs is
  float32 already; audit any `astype(np.float64)` feeding the device path).
- Unified memory = "VRAM" is system RAM; batch size is bounded by RAM, not a
  separate VRAM pool. Keep `apply_model(split=True)` so segments stay small.
- MPS is not fork-safe across processes → keep it to the single-process path.

### Files to change

- `src/dj_tagger/analyzers/vocal_stem.py` — device select, `_get_model().to(device)`,
  batched `apply_model(..., device=...)`, MPS fallback env + float32 guard.
- `src/dj_registry/adapters/local_analysis.py` — `_default_analysis_workers()`
  returns 1 when an accelerator is active; keep `_pool_worker_init` for CPU path.
- `src/dj_grouper/features/embeddings.py` — confirm/force CLAP onto the same
  device (laion_clap auto-detects; make it explicit for MPS).
- Docs (`README.md` / `docs/`) — install a Metal-enabled torch on the Mac
  (`pip install torch` on Apple Silicon already includes MPS; the current
  `2.9.0+cpu` build has none), and note the device auto-selection.

### Not doing (kept output-identical)

- No slice/model reduction, no dropping the 22→44.1 kHz upsample (all Tier 2,
  declined — they change feature values). GPU float differs only at ~1e-4, which
  the drift check below confirms is below the metrics' rounding.

---

## Consistency / drift (decision: verify, accept if sub-rounding)

GPU/MPS float math is not bit-identical to CPU (~1e-4 relative), but the stored
stem metrics are median-aggregated and rounded (e.g. `rms_db` 2 dp in
`result_to_dict`, `vocal_stem.py:346`). Raw layers are identity-keyed and never
re-derived, so CPU-cached tracks keep their values; only newly analyzed tracks
use the device.

**Drift check (gate before mixing device-new with CPU-cached tracks):** analyze
~10 representative tracks on CPU and on MPS, diff their `vocal_stem` /
`section_dsp` / `dsp` dicts, and confirm every field matches after rounding
(record the max observed drift). If within rounding → mix freely, no
re-analysis. If a field flips → re-analyze that layer on-device.

---

## Verification

1. **Device present:** `python -c "import torch; print(torch.backends.mps.is_available())"`
   → `True` on the Mac.
2. **Micro-benchmark:** time `analyze_vocal_stem` on one ~35 s track, `cpu` vs
   `mps`; record per-track and per-slice times and the measured Demucs multiplier.
   Update the estimate table with the real number.
3. **End-to-end:** `dj run --no-grouping --no-tags` on N cold-cache tracks; compare
   wall-clock and per-track vs the current CPU ~1.8× baseline. Confirm the MPS
   device is actually busy (Activity Monitor GPU history), not silently CPU-bound
   via op fallback.
4. **Drift check** as above.
5. **Regression:** `pytest -q` — the device path is guarded, so CPU-only test
   runs (currently 566 tests) must stay green.

## Rollout

1. Phase 1 (device move + batching + 1-worker gating), on the Mac.
2. Micro-benchmark → fill in the real MPS multiplier; decide if the target is met.
3. Drift check → confirm mixable with the existing CPU-analyzed library.
4. If the GPU idles / target unmet → add Phase 2 (CPU/GPU pipeline overlap).
