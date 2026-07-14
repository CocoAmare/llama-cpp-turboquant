# Phase 0 Baseline Checklist

## Purpose

Capture a reproducible performance and correctness baseline before porting Fable or adding SSD expert tiering.

Base branch:

```text
feature/turboquant-kv-cache
```

Integration branch:

```text
multiverse/integration
```

## Target machine

- Ubuntu 24.04.4 LTS
- Ryzen 7 5800X
- 64 GB DDR4-3200
- 2 × RTX 3060 12 GB
- CUDA build

## Capture environment

```bash
uname -a
lsb_release -a
nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,memory.total --format=csv
cmake --version
gcc --version
g++ --version
```

## Clean CUDA build

```bash
git switch feature/turboquant-kv-cache
git pull --ff-only
rm -rf build
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

## Model inventory

Record exact path, filename, SHA-256, architecture, quant, and context setting for:

- [ ] Ornith 35B MoE
- [ ] Qwen 35B-A3B MoE
- [ ] one dense control model
- [ ] one small fast correctness model

```bash
sha256sum /path/to/model.gguf
```

## Baseline matrix

- [ ] ordinary KV cache
- [ ] TurboQuant `f16/turbo4`
- [ ] TurboQuant `q8_0/turbo4`
- [ ] TurboQuant `q8_0/turbo3`
- [ ] single GPU
- [ ] dual GPU
- [ ] prompt processing benchmark
- [ ] token generation benchmark
- [ ] `llama-server` repeated-request smoke test
- [ ] DFlash disabled
- [ ] DFlash startup smoke test when a matching draft model is available

For every run record:

- complete command line;
- model hash;
- prompt tokens/second;
- generation tokens/second;
- peak RAM;
- peak VRAM per GPU;
- output or deterministic output hash;
- stderr/log path.

## Correctness prompt set

Use temperature zero where supported and include:

- short factual response;
- structured JSON response;
- code generation;
- long-context retrieval;
- multi-turn server request;
- repeated identical request for graph and slot reuse.

## Suggested output tree

```text
benchmarks/multiverse/baseline/
├── hardware.txt
├── models.tsv
├── commands.sh
├── results.tsv
├── correctness-prompts.jsonl
└── logs/
```

## Exit criteria

- [ ] clean CUDA build succeeds;
- [ ] TurboQuant modes initialize and generate coherent output;
- [ ] single- and dual-GPU commands are reproducible;
- [ ] baseline files and commands are preserved;
- [ ] no Fable or SSD-tier code has been added yet.
