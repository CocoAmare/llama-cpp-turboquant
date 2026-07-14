#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  MODEL=/path/to/model.gguf ./scripts/multiverse-baseline.sh

Optional environment variables:
  BUILD_DIR=build-multiverse
  OUT_DIR=benchmarks/multiverse-baseline
  N_GPU_LAYERS=99
  N_CPU_MOE=0
  PROMPT_TOKENS=2048
  GEN_TOKENS=128
  BATCH_SIZE=2048
  UBATCH_SIZE=2048
  REPEATS=5
  CUDA_ARCHS="86"

The script builds the current branch with CUDA, records machine metadata, and runs
baseline llama-bench passes with Multiverse/Fable environment variables disabled.
It does not enable DFlash because that requires a matching draft model.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

: "${MODEL:?Set MODEL to an existing GGUF file}"

if [[ ! -f "$MODEL" ]]; then
    echo "error: MODEL does not exist: $MODEL" >&2
    exit 2
fi

BUILD_DIR="${BUILD_DIR:-build-multiverse}"
OUT_DIR="${OUT_DIR:-benchmarks/multiverse-baseline}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
N_CPU_MOE="${N_CPU_MOE:-0}"
PROMPT_TOKENS="${PROMPT_TOKENS:-2048}"
GEN_TOKENS="${GEN_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-2048}"
REPEATS="${REPEATS:-5}"
CUDA_ARCHS="${CUDA_ARCHS:-86}"

mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$OUT_DIR/$STAMP"
mkdir -p "$RUN_DIR"

{
    echo "timestamp_utc=$STAMP"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "git_branch=$(git branch --show-current)"
    echo "model=$MODEL"
    echo "model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
    echo "n_gpu_layers=$N_GPU_LAYERS"
    echo "n_cpu_moe=$N_CPU_MOE"
    echo "prompt_tokens=$PROMPT_TOKENS"
    echo "gen_tokens=$GEN_TOKENS"
    echo "batch_size=$BATCH_SIZE"
    echo "ubatch_size=$UBATCH_SIZE"
    echo "repeats=$REPEATS"
    echo "cuda_archs=$CUDA_ARCHS"
} > "$RUN_DIR/run.env"

uname -a > "$RUN_DIR/uname.txt"
(lscpu || true) > "$RUN_DIR/lscpu.txt" 2>&1
(free -h || true) > "$RUN_DIR/memory.txt" 2>&1
(nvidia-smi -q || true) > "$RUN_DIR/nvidia-smi-q.txt" 2>&1
(nvidia-smi topo -m || true) > "$RUN_DIR/nvidia-topology.txt" 2>&1
(lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS || true) > "$RUN_DIR/storage.txt" 2>&1

cmake -S . -B "$BUILD_DIR" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"$(nproc)"

BENCH="$BUILD_DIR/bin/llama-bench"
if [[ ! -x "$BENCH" ]]; then
    echo "error: llama-bench was not produced at $BENCH" >&2
    exit 3
fi

COMMON=(
    -m "$MODEL"
    -ngl "$N_GPU_LAYERS"
    -p "$PROMPT_TOKENS"
    -n "$GEN_TOKENS"
    -r "$REPEATS"
    -b "$BATCH_SIZE"
    -ub "$UBATCH_SIZE"
)

if [[ "$N_CPU_MOE" != "0" ]]; then
    COMMON+=( -ncmoe "$N_CPU_MOE" )
fi

# Explicitly disable the Fable-derived paths for the inherited baseline.
env \
    GGML_CUDA_REGISTER_HOST=0 \
    GGML_SCHED_PREFETCH_EXPERTS=0 \
    "$BENCH" "${COMMON[@]}" 2>&1 | tee "$RUN_DIR/llama-bench-baseline.txt"

# Capture a lightweight machine-readable summary when llama-bench supports JSON.
if "$BENCH" --help 2>&1 | grep -q -- '--output json'; then
    env \
        GGML_CUDA_REGISTER_HOST=0 \
        GGML_SCHED_PREFETCH_EXPERTS=0 \
        "$BENCH" "${COMMON[@]}" --output json \
        > "$RUN_DIR/llama-bench-baseline.json" 2> "$RUN_DIR/llama-bench-baseline-json.stderr" || true
fi

printf '\nBaseline captured in: %s\n' "$RUN_DIR"
