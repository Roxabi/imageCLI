# imageCLI — Stage-Axis Decomposition Audit
**Date:** 2026-05-20  
**Auditor:** Claude Sonnet 4.6 (read-only)  
**Reference framework:** lyra #1277 stage-axis decomposition

---

## Verdict Summary

imageCLI is **definitively on the target axis** with high N (8 engines, growing) and high M
(7+ concerns per 2-phase engine file). The cascade is **latent, not yet active**: recent
refactors (engine subpackage split, `helpers.py`, `_pulid/` sub-package, `nvfp4/` sub-package,
`validators.py` extraction) show the team is already fighting the pattern one file at a time
without naming the root cause. The 3-engine quantization family (`flux2-klein`,
`flux2-klein-fp8`, `flux2-klein-fp4`) is the hottest N×M zone: each adds a new row of 5 methods
(encode, generate, load_for_encode, start_generation_phase, generate_from_embeddings) that are
75–52% identical to the others. The next quant variant — e.g. NVFP4 + torchao for non-Blackwell,
or fp8 + LoRA path — would create a 4th copy of the same block. The PuLID sub-family (3 engines)
duplicates InsightFace + EVA-CLIP loading verbatim. The two daemons (socket-based `daemon.py`
and NATS `nats/adapter.py`) have divergent error-handling contracts: the socket daemon leaks
`str(exc)` to clients; the NATS adapter uses typed `WorkerError` — a two-tier protocol that will
require parallel maintenance as the engine surface grows.

---

## Section 1 — Axis of Decomposition

### 1.1 File map by axis

| File | LOC | Axis |
|---|---|---|
| `engines/flux2_klein.py` | 277 | Target (quanto FP8, FLUX.2-klein) |
| `engines/flux2_klein_fp8.py` | 270 | Target (torchao FP8, FLUX.2-klein) |
| `engines/flux2_klein_fp4.py` | 247 | Target (NVFP4, FLUX.2-klein) |
| `engines/pulid_flux2_klein.py` | 152 | Target (PuLID, FLUX.2-klein) |
| `engines/pulid_flux2_klein_fp4.py` | 282 | Target (PuLID + NVFP4, FLUX.2-klein) |
| `engines/flux1_dev.py` | 41 | Target (FLUX.1-dev GGUF) |
| `engines/flux1_schnell.py` | 86 | Target (FLUX.1-schnell GGUF) |
| `engines/pulid_flux1_dev.py` | 232 | Target (PuLID + FLUX.1-dev) |
| `engines/sd35.py` | 60 | Target (SD3.5 Turbo) |
| `engine/base.py` | 273 | Horizontal (ABC + shared generate/offload/compile) |
| `engine/helpers.py` | 267 | Horizontal (preflight, quantize, save_image, TwoPhaseMixin) |
| `engine/registry.py` | 89 | Horizontal (factory/dispatch) |
| `nats/adapter.py` | 357 | Horizontal (NATS transport + error mapping) |
| `nats/validators.py` | 211 | Horizontal (input validation) |
| `commands/` (6 files) | ~700 | CLI routing |

**Verdict: target axis.** Engine files are organized per integration target (model × quantization
variant). Horizontal concerns (quantization, phase management, GPU placement, LoRA fusion,
pivotal embeddings, image saving) accrete inside each target file.

### 1.2 Concerns accreted per 2-phase engine file

For the 3-variant FLUX.2-klein family (the highest-duplication zone):

| Concern | flux2_klein.py | flux2_klein_fp8.py | flux2_klein_fp4.py |
|---|---|---|---|
| LoRA load + fuse | ✓ (8 refs, :42–70) | ✓ (4 refs, :58–77) | — (pre-quantized, guarded) |
| Quant library import + apply | ✓ quanto FP8 (:33,77) | ✓ torchao FP8 (:42,86) | ✓ NVFP4 (:88–93) |
| Pivotal embedding application | ✓ (:74) | ✓ (:82) | ✓ (via guard, :65) |
| CPU offload / GPU placement | ✓ (:95,104–106,177,204–206) | ✓ (:95,121,185,211–213) | ✓ (:113–114,178–198) |
| `_execution_device` monkey-patch | 2× inline (load_all_on_gpu + start_generation_phase) | 1× method (fp8.py:101–115) | 1× method (fp4.py:95–107) |
| `encode_and_generate` loop | ✓ (:129–169) | ✓ (:129–177) | ✓ (:129–172) |
| `generate_from_embeddings` loop | ✓ (:229–277) | ✓ (:222–270) | ✓ (:204–247) |
| `encode_prompt` | ✓ (:180–194) | ✓ (:188–202) | ✓ (:181–190) |
| `load_for_encode` | ✓ (:173–178) | ✓ (:181–186) | ✓ (:176–179) |
| `start_generation_phase` | ✓ (:196–227) | ✓ (:204–220) | ✓ (:192–202) |
| Hardware requirements check | — | ImportError guard (:42–46) | `_check_requirements()` (:40–57) |

### 1.3 Quantitative diff: flux2_klein vs fp8 vs fp4

**Similarity ratios** (SequenceMatcher):
- `flux2_klein.py` vs `flux2_klein_fp8.py`: **75.0%** identical
- `flux2_klein_fp4.py` vs `flux2_klein_fp8.py`: **56.9%** identical
- `flux2_klein.py` vs `flux2_klein_fp4.py`: **52.7%** identical

**LoRA loading block** (`if self.loras:` … `_apply_pivotal_embeddings()`):
- `flux2_klein.py`: 33 lines, `flux2_klein_fp8.py`: 25 lines
- 26 identical lines, 14 differing lines — comments and LoRA‑count logic diverge, core fuse sequence is copied verbatim.

**2-phase loop bodies** (`encode_and_generate` + `generate_from_embeddings`):
- 76 lines are **word-for-word identical across all 3 files** (non-comment, non-blank).
  Sample: `"generator": generator,`, `"guidance_scale": guidance,`,
  `"prompt_embeds": embeddings["prompt_embeds"].to("cuda"),`, `return self._save_image(...)`.

**`start_generation_phase` / `gc` teardown** — identical in all 3:
```
self._pipe.text_encoder.to("cpu")
torch.cuda.empty_cache()
gc.collect()
```
(`flux2_klein.py:204–206`, `flux2_klein_fp8.py:211–213`, `flux2_klein_fp4.py:196–198`)

### 1.4 Diff: flux1_dev vs flux1_schnell

Both files are thin (41 and 86 LOC). GGUF loading sequence is identical:
`Flux1DevEngine._load` (:22–41) vs `Flux1SchnellEngine._load` (:26–48) — 12 shared lines,
difference is only GGUF repo/file constants and `_build_pipe_kwargs` override for guidance=0.0.
The FLUX.1 family has not yet grown batch infrastructure (no 2-phase), so the cascade is
**latent, not yet expressed** here.

### 1.5 Diff: pulid_flux2_klein vs pulid_flux1_dev

Both load InsightFace (AntelopeV2) and EVA-CLIP via identical 7-line blocks
(`pulid_flux2_klein.py:84–91`, `pulid_flux1_dev.py:89–96`). Divergence: `pulid_flux1_dev.py`
splits EVA-CLIP into trunk+head separately for intermediate layer extraction (63-line
`_extract_id_tokens` custom impl), while `pulid_flux2_klein.py` delegates to
`_pulid.extract_id_tokens`. The `_check_requirements` for FP4 variants is duplicated verbatim
between `flux2_klein_fp4.py:40–57` and `pulid_flux2_klein_fp4.py:54–76`.

**Verdict: target axis.** Every file is a target. Concerns are accreted.

---

## Section 2 — Cascade Symptoms

### 2.1 `str(exc)` leaks on bus-bound paths

**daemon_handlers.py** — socket wire protocol:
- `daemon_handlers.py:83`: `_send_json(conn, {"ok": False, "error": str(exc)})` — `_handle_blend`
- `daemon_handlers.py:154`: `_send_json(conn, {"ok": False, "error": str(exc)})` — `_handle_encode`
- `daemon_handlers.py:238`: `_send_json(conn, {"ok": False, "error": str(exc)})` — `_handle_job`

All 3 socket handlers send raw `str(exc)` over the wire. A GPU OOM, path error, or GGUF fetch
failure leaks full stack context to the client. This is a `str(exc)` leak on a bus-bound path —
the exact cascade symptom from lyra #1277.

**nats/adapter.py** — correctly typed via `_map_exception_to_error` → `WorkerError`. No str(exc)
leak on the NATS path. The dual-wire architecture means the same underlying generation failures
are handled inconsistently: typed on NATS, untyped on socket.

**nats/adapter.py:188**: `log.exception(f"Generation failed: {e}")` — logging use of `{e}` is
acceptable (not a bus-bound leak), but the f-string on a bus-bound log path is a minor code smell.

### 2.2 `except Exception` count

| File | Count | Sites |
|---|---|---|
| `nats/adapter.py` | 5 | :140, :148, :186, :254, :346 |
| `daemon_handlers.py` | 4 | :40(run_worker), :81, :152, :236 |
| `engine/helpers.py` | 1 | :127 (quantize_transformer fallback) |
| `engines/sd35.py` | 1 | :38 (T5 quantization fallback, intentional) |
| `daemon.py` | 1 | :115 (connection accept, intentional) |

Total: **12 broad-catch sites**. The `daemon_handlers.py` catches are the highest-risk because
they convert Exception → `str(exc)` and send to socket clients. The `nats/adapter.py` catches
properly delegate to `_map_exception_to_error`.

### 2.3 `__init_subclass__`, class-attr overrides, inheritance for sharing code

No `__init_subclass__` present. Class-attr overrides are the primary inheritance mechanism:
- All 9 engine classes override `name`, `description`, `model_id`, `vram_gb`, `capabilities`
  as class attributes on `ImageEngine`. This is the correct pattern for engine identity.
- `TwoPhaseMixin` in `helpers.py:132–164` — clean mixin, raises `NotImplementedError` for
  non-implementing engines.
- No code-sharing via inheritance (correct — sharing is via helper functions in `engine/helpers.py`).

### 2.4 Duplicated helpers across engine files

| Helper / Pattern | Files | Sites |
|---|---|---|
| `_set_execution_device()` | fp8.py, fp4.py, pulid_fp4.py | 3 separate implementations |
| `_execution_device_override` monkey-patch (inline) | flux2_klein.py:108–116, :215–222 | 2 inline copies in same file |
| InsightFace load block | pulid_flux2_klein.py:84–91, pulid_flux2_klein_fp4.py:151–158, pulid_flux1_dev.py:89–96 | 3 copies |
| EVA-CLIP load block | pulid_flux2_klein.py:94–100, pulid_flux2_klein_fp4.py:161–169, pulid_flux1_dev.py:99–107 | 3 copies |
| `start_generation_phase` gc teardown | flux2_klein.py:204–206, fp8.py:211–213, fp4.py:196–198 | 3 copies |
| `_check_requirements` | flux2_klein_fp4.py:40–57, pulid_flux2_klein_fp4.py:54–76 | 2 near-identical copies |
| LoRA fuse-unload sequence | flux2_klein.py:42–70, fp8.py:58–77 | 2 copies |
| seed/generator/pipe_kwargs pattern | flux2_klein.py (2×), fp8.py (2×), fp4.py (2×) | 6 copies |

**Active cascade site:** `_set_execution_device` / `_execution_device_override` — present in
`flux2_klein.py` (twice, inline), `flux2_klein_fp8.py` (extracted to method), `flux2_klein_fp4.py`
(extracted to method), `pulid_flux2_klein_fp4.py` (different strategy: anonymous subclass).
4 implementations, 3 strategies. A bug here generates a 4-way fix.

---

## Section 3 — Quantitative

### 3.1 Line counts and file size gate

| File | LOC | >300 gate? |
|---|---|---|
| `nats/adapter.py` | 357 | **YES — over 300** |
| `engine/base.py` | 273 | No |
| `engine/helpers.py` | 267 | No |
| `engines/pulid_flux2_klein_fp4.py` | 282 | No |
| `engines/flux2_klein.py` | 277 | No |
| `engines/flux2_klein_fp8.py` | 270 | No |
| `engines/flux2_klein_fp4.py` | 247 | No |
| `daemon_handlers.py` | 245 | No |
| `daemon.py` | 232 | No |
| `engines/pulid_flux1_dev.py` | 232 | No |
| `commands/_helpers.py` | 195 | No |
| `nats/validators.py` | 211 | No |

`nats/adapter.py` at 357 LOC is the only file over the 300-line gate (no exemption found in
`tools/file_exemptions.txt`). The 3 quant variant files are all within 247–277 LOC each, but
their **combined total** is 794 LOC for what is structurally ~250 LOC of unique logic +
~200 LOC × 3 of duplicated 2-phase infrastructure.

### 3.2 Git commit history — sibling-fix cascade evidence

The commit log shows no classic "fix offload in fp8, then fix offload in fp4 two days later"
pattern. This is consistent with **latent** (not yet active) cascade status.

What is visible is an **architecture-driven split cascade**: the team produced a sequence of
"split X into sub-modules" commits (`5e74db3`, `1f90969`, `3906159`, `bb06c2c`, `4effe2e`)
without addressing the root duplication. Each split moved a file under the 300-line gate but
left the concern-accreted structure intact. The commits show the process-level cascade
(tooling flags files, team files refactor issues, refactors happen file by file) described in
lyra #1277 stacked producer #2 and #3.

---

## Section 4 — DEBT Inventory

### 4.1 DEBT directory

No `artifacts/debt/` directory exists. `artifacts/` contains only `frames/`, `plans/`, `specs/`.
The project has no formal debt tracking structure.

### 4.2 Inline tags

| Tag | Count | Files |
|---|---|---|
| `TODO` | 1 | `engine/registry.py:61` — `# TODO(#72): remove legacy singular-kwargs branch after one release cycle.` |
| `FIXME` | 0 | — |
| `HACK` | 0 | — |
| `XXX` | 0 | — |
| `DEBT:` | 0 | — |

### 4.3 `noqa` annotations

7 sites total (see Section 2 grep). No unusual noqa patterns; all are legitimate type suppression
for untyped third-party libraries (torch, diffusers, insightface) or intentional re-exports.

**Verdict:** Debt is **untracked**. The single TODO (remove deprecated LoRA kwargs) has been
lingering since `7b62414` (deprecated) and `a555346` (reject at registry layer). No drain
pressure: no issue references, no milestone.

---

## Section 5 — Composition vs Inheritance for Capabilities

### 5.1 Quantization

**NOT once.** Quantization is implemented:
- `engine/helpers.py:105–129`: `quantize_transformer()` — shared helper for flux1_dev/schnell/sd35
  quanto path (fp8 on sm≥89, int8 on Ampere).
- `flux2_klein.py:77–87`: inline quanto FP8 for FLUX.2-klein (with QLinear contiguity patch).
- `flux2_klein_fp8.py:86`: torchao `quantize_()` — engine-specific.
- `flux2_klein_fp4.py:129`: `runtime_quantize_transformer_to_nvfp4()` from `nvfp4/` sub-package.
- `engines/sd35.py:34–36`: inline `qint8` quantization for T5 text encoder.

4 separate quantization strategies, 3 of which are per-engine. The `nvfp4/` sub-package
(`quantize.py`, `state_dict.py`) is the closest thing to an extracted quant strategy — but it
is scoped to one engine family only.

### 5.2 CPU offload / GPU placement

**NOT once.** Per-engine:
- `engine/base.py:162–175` `_finalize_load()`: generic VRAM check + `pipe.to("cuda")`.
- `flux2_klein.py:95`: `enable_model_cpu_offload()` in `_load()` only.
- `flux1_dev.py:39`, `flux1_schnell.py:46`: `enable_model_cpu_offload()` in `_load()`.
- `pulid_flux2_klein.py:59` in `_finalize_load()` override.
- 2-phase engines: manual `text_encoder.to("cpu"/"cuda")`, `transformer.to("cuda")`,
  `vae.to("cuda")` per-file (3 files × 2–3 placements each = ~12 scattered placement calls).

The `start_generation_phase` teardown (`text_encoder.to("cpu") + empty_cache + gc.collect`)
is verbatim in all 3 quant variant files. This is the prime candidate for a `PhaseManager`
or `GPUOrchestrator` abstraction.

### 5.3 Batch handling / 2-phase encode+generate

**NOT once.** `TwoPhaseMixin` in `helpers.py:132–164` declares the interface but provides no
implementation. The 5-method 2-phase implementation is duplicated 3 times (flux2_klein,
fp8, fp4). Any bug in `encode_prompt` return format or `generate_from_embeddings` kwargs
propagates to 3 files.

### 5.4 Image saving

**ONCE** — correctly extracted. `engine/helpers.py:205–228` `save_image()` is the single
implementation; `base.py:157–160` `_save_image()` delegates to it.

### 5.5 ABC enforcement

`ImageEngine` ABC (`base.py:61–64`) requires only `_load() -> None`. No abstract methods for
quantize, offload, or batch phases. `TwoPhaseMixin` defaults raise `NotImplementedError` but
are not abstract. An engine that forgets to implement `encode_prompt` will silently inherit the
`raise NotImplementedError` default — no static enforcement.

---

## Section 6 — Result[T,E] vs Raising

### 6.1 Engine boundaries

Engines raise, not return. `generate()` raises `RuntimeError`, `InsufficientResourcesError`,
`MemoryError`, `ValueError` — no typed Result wrapper. This is consistent with Python idiom
and not a cascade risk in itself.

### 6.2 NATS adapter error propagation

**Typed path (NATS):** `nats/validators.py:195–211` `_map_exception_to_error()` converts
`InsufficientResourcesError` → `"insufficient_resources"`, `MemoryError` → `"insufficient_resources"`,
`ValueError("Unknown engine")` → `"unknown_engine"`, fallthrough → `"generation_failed"`.
Structured `WorkerError` via `roxabi_contracts`. No `str(exc)` on the wire. **Clean.**

**Untyped path (socket daemon):**
- `daemon_handlers.py:83,154,238`: `{"ok": False, "error": str(exc)}` — raw exception string
  sent over Unix socket. 3 separate handlers, same pattern, no sanitization.
- `daemon_handlers.py:40–41`: `print(f"[imagecli daemon] worker error: {exc}", flush=True)` —
  bare `{exc}` in worker loop.

The two error contracts are divergent. The socket daemon is a legacy code path (used by
`imagecli batch --two-phase`), but it is still active. If the same OOM exception fires in both
paths, it is handled with typed `WorkerError` on NATS and raw `str(exc)` on the socket.

### 6.3 `str(exc)` in validators

`nats/validators.py:205–206`:
```python
if isinstance(exc, ValueError) and "Unknown engine" in str(exc):
    return "unknown_engine", str(exc).split(": ", 1)[-1]
```
String-matching on `str(exc)` to detect error type — brittle. A message change in
`engine/registry.py` silently breaks the mapping. This is the `adapter-magic-constants` DEBT slug.

---

## Section 7 — Cross-Repo Coupling with Lyra

### 7.1 Dependencies

`pyproject.toml:23`: `"roxabi-nats"` — direct dependency.
`pyproject.toml:49`: `roxabi-nats = { git = "...", subdirectory = "packages/roxabi-nats", tag = "roxabi-nats/v0.4.1" }`

`roxabi-contracts` is a **transitive dependency** (pulled in by roxabi-nats, pinned via
`uv.lock:1645–1659`). `nats/adapter.py:14–15` imports `WorkerError` and `ImageResponse` from
`roxabi_contracts` directly — so imageCLI is also a **direct consumer** of contracts despite
not listing it as a direct dep.

### 7.2 NATS subject layout

Defined in `nats/adapter.py:23–24`:
```python
SUBJECT = "lyra.image.generate.request"
HEARTBEAT_SUBJECT = "lyra.image.heartbeat"
```
Subject strings are hardcoded literals in the adapter. They are not imported from
`roxabi_contracts`. This is the `adapter-magic-constants` pattern — the subject namespace
`lyra.*` is semantically owned by lyra, but there is no import-time enforcement. A subject
rename in lyra requires manual sync here.

### 7.3 Pattern comparison with llmCLI and voiceCLI

- `llmCLI` and `voiceCLI` use the same `roxabi-nats` SDK + NATS adapter pattern.
- imageCLI uniquely has **two** wire protocols: NATS (primary satellite, `nats/adapter.py`) and
  Unix socket (legacy batch daemon, `daemon.py` + `daemon_handlers.py`). The socket protocol
  is not shared with llmCLI/voiceCLI. This is a structural oddity that creates the dual
  error-contract problem documented in Section 6.

---

## Section 8 — Findings and Recommendations

### Finding 1 — 2-Phase infrastructure is N×M: 5 methods × 3 files (LATENT)

**Evidence:** `flux2_klein.py:121–277`, `flux2_klein_fp8.py:117–270`,
`flux2_klein_fp4.py:119–247` — 5 methods each, 76 lines word-for-word identical.
75% file similarity between flux2_klein and fp8.

**Nature:** Latent. No bug has triggered a 3-way sibling fix yet. But the next quant variant
(e.g. quanto int8 for Ampere fallback, or NVFP4 + LoRA path) would add a 4th copy.

**Recommendation:** Extract `TwoPhaseProtocol` concrete base with:
- `encode_and_generate()` implemented once using `self._pipe` + `self._build_pipe_kwargs()`
- `generate_from_embeddings()` implemented once
- `start_generation_phase()` teardown (`text_encoder.to("cpu") + gc + empty_cache`) extracted
  to `_teardown_encoder_phase()`
- Engine subclasses provide only `_load_pipeline()` and `_set_execution_device()`
- Estimated LOC reduction: ~150 LOC (3 files × 50 LOC saved)

### Finding 2 — `_set_execution_device` / `_execution_device_override`: 4 implementations, 3 strategies (ACTIVE)

**Evidence:**
- `flux2_klein.py:108–116` (inline, load_all_on_gpu) + `flux2_klein.py:215–222` (inline, start_generation_phase) — 2 inline copies in one file
- `flux2_klein_fp8.py:101–115` — extracted `_set_execution_device()` with assert
- `flux2_klein_fp4.py:95–107` — extracted `_set_execution_device()` without assert
- `pulid_flux2_klein_fp4.py:79–94` — different strategy: anonymous subclass to isolate the patch

The anonymous-subclass approach in `pulid_flux2_klein_fp4.py` is strictly safer (scoped to
instance, not class), but was not backported to the 3 other engines. **This is an active cascade
risk**: a bug in the monkey-patch logic (e.g., wrong property fget chain) requires touching
4 sites.

**Recommendation:** Adopt the anonymous-subclass strategy once in `TwoPhaseProtocol` or
`helpers.py:_set_execution_device(pipe)`. Remove all per-engine copies.

### Finding 3 — `_check_requirements` duplicated between fp4 engines (LATENT)

**Evidence:** `flux2_klein_fp4.py:40–57` and `pulid_flux2_klein_fp4.py:54–76` are
near-identical (14-line diff, only error message strings differ). A new fp4 engine variant
would copy this block again.

**Recommendation:** Extract `_check_nvfp4_requirements()` to `engines/nvfp4/__init__.py`
and call it from both engines. This sub-package already exists (`nvfp4/quantize.py`,
`nvfp4/state_dict.py`).

### Finding 4 — Socket daemon leaks `str(exc)` to clients on 3 bus-bound paths (ACTIVE)

**Evidence:** `daemon_handlers.py:83,154,238` — identical `{"ok": False, "error": str(exc)}`
pattern in 3 separate exception handlers. An OOM, path access error, or GGUF download failure
is sent verbatim to the socket client.

**Nature:** Active for the socket daemon protocol. The NATS path correctly sanitizes.

**Recommendation:** Introduce `_map_exc_to_socket_error(exc) -> str` in `daemon_handlers.py`
that mirrors the NATS `_map_exception_to_error` logic. Replace all 3 `str(exc)` with calls
to this function. Alternatively, unify the two wire protocols by routing socket calls through
the same typed error mapping.

### Finding 5 — String-matching on `str(exc)` for exception type dispatch (ACTIVE)

**Evidence:** `nats/validators.py:205`: `"Unknown engine" in str(exc)` — detects engine type
by substring match on exception message.

**Nature:** Active. If `registry.py:50` message changes ("Unknown engine" → "Engine not found"),
the `unknown_engine` code path silently falls through to `generation_failed`, making all
unknown-engine errors retryable.

**Recommendation:** Introduce `UnknownEngineError(ValueError)` in `engine/registry.py` and
check `isinstance(exc, UnknownEngineError)` in `_map_exception_to_error`.

---

### Special Angle — PuLID Composition Path

**Current state:** `pulid_flux2_klein.py` overrides `generate()` to inject PuLID tokens
and wraps `super().generate()` in a patch/unpatch sandwich. `pulid_flux2_klein_fp4.py`
reimplements the same sandwich plus manual prompt encoding (because the NVFP4 engine
cannot use `enable_model_cpu_offload`, so it needs explicit Qwen3 encoder shuttling).
InsightFace + EVA-CLIP loading is duplicated 3× (Klein, Klein-FP4, FLUX1-dev).

**Clean composition path:** A `PuLIDMixin` that:
1. Holds `_pulid`, `_insightface`, `_eva_clip` state
2. Provides `_load_face_models(insightface_dir, pulid_path, eva_clip_model)` once
3. Provides `_extract_id_tokens(refs)` — delegates to `_pulid.extract_id_tokens` for Flux2
   or the custom 63-line implementation for Flux1 (the Flux1 path uses `forward_intermediates`
   which is architecturally different — not trivially unified, but the InsightFace block is)
4. Provides `_generate_with_pulid(prompt, refs, pulid_strength, **kwargs)` with the
   patch/unpatch pattern

`pulid_flux2_klein.py` and `pulid_flux2_klein_fp4.py` would then compose:
`PuLIDMixin + Flux2KleinEngine` vs `PuLIDMixin + Flux2KleinFP4Engine`.
The NVFP4 engine's manual encoder shuttling remains engine-specific (it is a VRAM constraint,
not a PuLID concern). Estimated LOC reduction: ~80 LOC across 3 PuLID files.

---

## Top 3 Actions

**1. Extract `TwoPhaseBase` with concrete encode+generate implementation (N×M hotspot)**
- Target: `engine/helpers.py` or new `engine/two_phase.py`
- Methods to implement once: `encode_and_generate`, `generate_from_embeddings`,
  `_teardown_encoder_phase` (the gc+empty_cache block)
- Engines provide: `_load_pipeline()`, optional `_set_execution_device()`
- Impact: removes ~150 LOC of duplication across 3 files, prevents 4th copy on next quant variant
- Risk: low — covered by existing tests once the interface is preserved

**2. Fix `str(exc)` leaks in socket daemon handlers + introduce typed UnknownEngineError**
- `daemon_handlers.py:83,154,238` → `_map_exc_to_socket_error(exc)`
- `engine/registry.py` → add `UnknownEngineError`
- `nats/validators.py:205` → `isinstance(exc, UnknownEngineError)` instead of string match
- Impact: removes 3 active `str(exc)` bus leaks, makes engine error routing safe to refactor
- Risk: very low — internal, no external contract change

**3. Extract `_check_nvfp4_requirements()` to `engines/nvfp4/__init__.py` + `PuLIDMixin`**
- Move duplicated hardware check to the sub-package that already owns NVFP4 logic
- Extract InsightFace + EVA-CLIP loading to `PuLIDMixin` (3 PuLID engines share it)
- Impact: removes 2 check-requirements copies + InsightFace/EVA-CLIP loading copies
- Risk: medium (PuLID mixin requires careful state management for trunk/head split in Flux1)
