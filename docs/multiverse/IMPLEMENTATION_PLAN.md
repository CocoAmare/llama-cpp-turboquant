# Multiverse MoE Tiering Implementation Plan

## Base and source repositories

Primary integration trunk:

- `CocoAmare/llama-cpp-turboquant`
- Base branch: `feature/turboquant-kv-cache`
- Integration branch: `multiverse/integration`

Reference sources:

- `TheTom/llama-cpp-turboquant` — TurboQuant+, DFlash, MTP, current llama.cpp runtime
- `thecodacus/llama.cpp` — pinned host memory and asynchronous MoE expert H2D prefetch
- `JustVugg/colibri` — SSD expert indexing, bounded RAM cache, asynchronous reads, usage-aware residency
- `antirez/ds4` — whole-expert cache budgeting and capacity-mode design patterns

## Non-negotiable invariants

1. All new behavior is opt-in.
2. Disabling Multiverse features restores the existing TurboQuant runtime path.
3. Standard GGUF files remain supported.
4. Existing Unsloth GGUFs are the first compatibility target.
5. TurboQuant KV compression and DFlash remain independently usable.
6. Speculative or predictive prefetch may waste work but may never change model output.
7. RAM and VRAM cache slots may not be reused while referenced by I/O, copies, or kernels.
8. Model-family logic is isolated from the generic residency manager.
9. Qwen/Ornith is the first implementation target, DeepSeek second, GLM third.
10. No performance claim is accepted without an A/B run against the unchanged base branch.

## Target architecture

```text
standard GGUF
    |
    +-- dense/shared/router tensors -> normal llama.cpp loading
    |
    +-- routed expert catalog
            |
            +-- optional .moepack sidecar
                    |
                    v
              SSD expert store
                    |
              asynchronous reads
                    v
             bounded RAM cache
                    |
              pinned host slots
                    |
            Fable async H2D stream
                    v
            per-device VRAM cache
                    |
                    v
                 compute
                    |
            TurboQuant KV cache
                    |
            DFlash or native MTP
```

## Delivery sequence

### Phase 0 — Baseline capture

- Build `feature/turboquant-kv-cache` unchanged with CUDA.
- Record hardware topology, CUDA runtime, compiler, and build flags.
- Benchmark ordinary inference, TurboQuant KV, single GPU, dual GPU, and DFlash-disabled paths.
- Save deterministic prompts and output hashes where sampling permits.

Exit criteria:

- Reproducible build on the dual RTX 3060 machine.
- Baseline benchmark files committed or attached to a tracking issue.

### Phase 1 — Fable host registration

Port only the host-memory registration behavior first.

Initial control:

```text
GGML_CUDA_REGISTER_HOST=1
```

Requirements:

- Standard GGUF compatibility.
- Safe failure when registration limits are reached.
- No attempt to pin an entire out-of-core model.
- Later SSD mode pins only bounded cache and staging regions.

Exit criteria:

- Token-identical output with registration off and on.
- Measured H2D bandwidth improvement or a documented no-regression result.

### Phase 2 — Fable asynchronous expert H2D

Port the complete safe patch series, including:

- secondary transfer backend/stream;
- CUDA event synchronization;
- three staging slots;
- fallback allocation behavior;
- pointer restoration for graph reuse;
- buffer lifetime fixes.

Initial control:

```text
GGML_SCHED_PREFETCH_EXPERTS=1
```

Exit criteria:

- Repeated prompt and server-slot reuse passes without corruption.
- Prefill benchmark compared against Phase 0 and Phase 1.
- DFlash and TurboQuant still initialize correctly.

### Phase 3 — Generic expert catalog and residency API

Introduce a model-independent identity for each routed expert:

```cpp
struct moe_expert_key {
    uint32_t layer;
    uint32_t expert;
};
```

Residency states:

```text
DISK_ONLY
READ_QUEUED
READING
RAM_READY
H2D_QUEUED
H2D_COPYING
GPU_READY
IN_USE
```

The first implementation wraps already-resident tensors without changing execution.

Exit criteria:

- Catalog inspection accurately reports Qwen/Ornith expert tensors and dimensions.
- Tiering disabled remains byte-for-byte equivalent in behavior.

### Phase 4 — Optional expert packer

Add a sidecar format rather than replacing GGUF:

```text
model.gguf
model.moepack
model.moepack.index
```

Each record contains one complete routed expert with its gate, up, and down quantized blocks and required scale data. Records are aligned, checksummed, and tied to the source GGUF hash.

Exit criteria:

- Repacked expert math matches the source GGUF path.
- Missing or stale sidecars fall back safely to the standard GGUF path.

### Phase 5 — SSD to RAM tier

Implement:

- positional reads or direct-I/O-compatible aligned reads;
- bounded whole-expert RAM cache;
- asynchronous worker pool;
- generation IDs to prevent stale worker publication;
- LRU first, then optional LFRU/heat scoring;
- explicit memory budgets and safety headroom;
- telemetry for hits, misses, bytes, service time, and visible wait.

Initial controls:

```text
--moe-tiering auto
--moe-store PATH
--moe-ram-gb N
--moe-io-threads N
--moe-cache-policy lru
```

Exit criteria:

- Correct inference with an artificially tiny RAM cache that forces SSD misses.
- Single-GPU stability before dual-GPU work begins.

### Phase 6 — RAM to VRAM integration

Connect RAM-ready experts to the Fable transfer path.

Rules:

- RAM slots stay pinned and referenced until the CUDA copy event completes.
- VRAM slots stay referenced until dependent kernels complete.
- Initial dual-GPU strategy assigns whole layers or whole experts to one device.
- No cross-device expert sharding in the first release.

Exit criteria:

- Forced cache churn on one GPU and two GPUs.
- No use-after-free, stale generation, or slot reuse failures.

### Phase 7 — Phase-aware scheduling

Prefill:

- form the union of experts selected across the batch;
- load each unique expert once;
- use bulk or whole-layer transfer when density warrants it.

Decode:

- load only top-k selected experts;
- favor whole-expert RAM/VRAM cache entries.

Exit criteria:

- Scheduler selects sparse or bulk mode based on measured expert density.
- Separate prefill and decode telemetry is available.

### Phase 8 — Predictive expert prefetch

Start with output-neutral hints:

1. persistent expert heat;
2. previous-token same-layer reuse;
3. cross-layer co-occurrence;
4. optional router lookahead.

Prediction must never modify router choices.

Exit criteria:

- Prefetch usefulness and waste are separately measured.
- Prediction can be disabled without altering cache correctness.

### Phase 9 — Adaptive DFlash and MTP

Keep DFlash and native MTP as separate draft modes initially.

Start DFlash at a maximum draft depth of three tokens. Adapt only at safe request or token boundaries using:

- acceptance rate;
- accepted tokens per target forward;
- unique expert union;
- GPU/RAM hit rates;
- SSD bytes per accepted token;
- visible SSD wait;
- H2D wait;
- end-to-end tokens per second.

The controller may reduce depth to one or temporarily disable speculation during cold-cache pressure.

Exit criteria:

- Normal decode, native MTP, and DFlash depths 1/3/5 are benchmarked independently.
- Adaptive mode beats or matches the best fixed safe configuration on representative workloads.

## Model-family order

### Qwen/Ornith

Development and correctness fixture. It already runs on the target machine and can be forced into artificial cache pressure.

### DeepSeek

Second adapter to expose assumptions around shared experts, router layout, MLA, and expert tensor organization.

### GLM

Third adapter. GLM requires both model-architecture support and the generic tiering layer; Colibri is a design reference, not code to merge wholesale.

## Initial telemetry

At minimum expose:

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

## First implementation milestone

The first code milestone is deliberately limited to:

```text
TurboQuant canonical base
+ Fable host registration
+ Fable safe three-slot asynchronous expert H2D
+ existing Unsloth Qwen/Ornith GGUF
+ no SSD streaming yet
```

Only after this milestone is stable does SSD expert tiering begin.
