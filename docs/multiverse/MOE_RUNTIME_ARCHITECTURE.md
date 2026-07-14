# Multiverse MoE Runtime Architecture

Status: implementation planning baseline

Base branch: `feature/turboquant-kv-cache`

Integration branch: `multiverse-integration`

## Objective

Build a general out-of-core MoE runtime on top of TurboQuant+ while preserving ordinary llama.cpp behavior, GGUF compatibility, TurboQuant KV compression, existing speculative decoding support, and multi-GPU execution.

The runtime should support a tiered expert path:

```text
SSD -> bounded RAM expert cache -> pinned RAM staging -> per-device VRAM cache -> compute
```

The implementation must remain opt-in. With all Multiverse features disabled, behavior should match the inherited TurboQuant+ branch.

## Sources and responsibilities

### TurboQuant+

TurboQuant+ is the integration trunk and retains ownership of:

- llama.cpp model loading and execution
- CUDA and multi-GPU backends
- TurboQuant KV-cache and weight formats
- DFlash and native speculative-decoding infrastructure
- server, CLI, and benchmark applications

### Fable

Port only the safe, isolated MoE transfer mechanisms:

- host registration for mmap-backed CPU expert weights
- asynchronous expert H2D transfer on a separate backend/stream
- three staging slots
- CUDA-event lifetime protection
- pointer restoration and allocation-fallback fixes

### Colibri and DS4

Reimplement storage ideas rather than merging model-specific engines:

- whole-expert disk records
- bounded complete-expert RAM cache
- positional/aligned reads
- asynchronous I/O workers
- LRU/LFRU heat tracking
- optional router lookahead
- explicit memory-budget planning

## Compatibility contract

1. Existing Unsloth and standard llama.cpp GGUFs remain valid.
2. Standard GGUF is the authoritative model format.
3. An optional `.moepack` sidecar may be generated for optimized expert reads.
4. TurboQuant KV compression does not require converting the target GGUF.
5. TurboQuant weight formats may require a separately quantized GGUF.
6. DFlash may require a matching draft GGUF.
7. Model architecture support remains separate from storage-tier support.

## Runtime state model

```text
DISK_ONLY
  -> READ_QUEUED
  -> READING
  -> RAM_READY
  -> H2D_QUEUED
  -> H2D_COPYING
  -> GPU_READY
  -> IN_USE
```

Every RAM and VRAM slot must carry a generation identifier. A delayed worker or CUDA completion event must never publish into a slot that has already been reused.

## Proposed modules

```text
src/moe/
  moe-catalog.h/.cpp
  moe-residency.h/.cpp
  moe-policy.h/.cpp
  moe-telemetry.h/.cpp
  moe-store.h
  moe-store-gguf.cpp
  moe-store-pack.cpp
  moe-cache-ram.h/.cpp
  moe-cache-gpu.h/.cpp
  moe-io.h/.cpp
```

## Feature flags

Initial flags should be explicit and conservative:

```text
--moe-tiering off|inspect|auto
--moe-store PATH
--moe-ram-gb N
--moe-vram-gb N
--moe-io-threads N
--moe-cache-policy lru|lfru
--moe-usage-file PATH
--moe-prefetch off|history|router
--spec-draft-n-max N
```

Environment variables inherited from the Fable port remain separately controllable during A/B testing:

```text
GGML_CUDA_REGISTER_HOST=1
GGML_SCHED_PREFETCH_EXPERTS=1
```

## Implementation sequence

### Phase 0: baseline

- Build current TurboQuant+ branch with CUDA.
- Record single- and dual-GPU performance.
- Test normal inference, TurboQuant KV, DFlash disabled, and DFlash enabled.
- Capture deterministic prompts and output hashes.

### Phase 1: Fable host registration

- Port host registration only.
- Verify identical output and no regression when disabled.
- Measure H2D bandwidth and prefill separately from decode.

### Phase 2: Fable asynchronous expert transfer

- Port the complete safe patch set, including three staging slots and lifetime fixes.
- Reconcile scheduler changes manually with TurboQuant speculative/MTP graph work.
- Stress repeated graph reuse and server concurrency.

### Phase 3: generic expert catalog

- Identify routed-expert tensors at model load.
- Record layer, expert, tensor role, quantization type, dimensions, shard, offset, and byte length.
- Add `--moe-tiering inspect` without changing execution.

### Phase 4: residency wrapper

- Introduce acquire/release APIs around already-resident experts.
- Keep all storage in ordinary GGUF/mmap memory.
- Prove the abstraction is output-identical before adding disk misses.

### Phase 5: `.moepack` tool

- Add an optional packer that stores gate/up/down blocks for one expert contiguously.
- Preserve source quantization bytes whenever possible.
- Align records for direct reads and include checksums plus source-model identity.

### Phase 6: SSD -> RAM cache

- Add bounded complete-expert RAM slots.
- Implement async reads, LRU/LFRU eviction, slot generations, and telemetry.
- Force a deliberately tiny cache on a known Qwen/Ornith model to exercise misses.

### Phase 7: RAM -> VRAM integration

- Connect RAM-ready expert slots to Fable's pinned asynchronous H2D path.
- Add per-device VRAM expert caches.
- Start with whole-expert, whole-layer placement; do not shard one expert across GPUs.

### Phase 8: phase-aware scheduling

- Prefill: batch-union unique experts and bulk transfer when advantageous.
- Decode: sparse whole-expert loading and caching.
- Select behavior from actual routed-expert density.

### Phase 9: adaptive speculation

- Keep DFlash and native MTP as independently benchmarked draft providers.
- Start DFlash at three draft tokens.
- Adapt depth from acceptance rate, unique experts, cache hit rates, SSD bytes per accepted token, visible I/O wait, and H2D wait.

### Phase 10: model-family expansion

1. Qwen/Ornith development fixture
2. DeepSeek adapter and shared-expert validation
3. GLM adapter after base model-architecture support is present

## Telemetry contract

Expose at least:

```text
moe_gpu_hits
moe_ram_hits
moe_disk_misses
moe_disk_bytes
moe_disk_service_ms
moe_visible_wait_ms
moe_h2d_ms
moe_prefetch_hits
moe_prefetch_waste
unique_experts_per_token
draft_tokens_proposed
draft_tokens_accepted
```

## Safety requirements

- No RAM slot eviction while an H2D copy references it.
- No VRAM slot reuse while a kernel references it.
- No stale I/O completion may publish into a newer slot generation.
- Feature-disabled behavior must remain equivalent to the inherited TurboQuant+ branch.
- Speculative or router prefetch may waste bandwidth but must never change model output.
- SSD failure or allocation pressure must degrade to a safe fallback or explicit error, never silent corruption.
- The canonical `feature/turboquant-kv-cache` branch remains untouched by Multiverse development.

## Initial hardware target

Primary validation system:

- Ryzen 7 5800X
- 64 GB DDR4
- 2 x RTX 3060 12 GB
- Ubuntu 24.04
- NVMe storage

Initial budget guidance:

```text
system + dense/runtime reserve: 14-18 GB RAM
warm expert RAM cache:         30-40 GB RAM
I/O and staging slabs:          4-6 GB RAM
VRAM expert tier:               5-8 GB per GPU, tuned experimentally
```

These are starting envelopes, not fixed defaults. The planner must derive safe values from loaded-model allocations and available device memory.
