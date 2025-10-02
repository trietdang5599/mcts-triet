#!/usr/bin/env bash
set -eu

# Enable pipefail when supported (POSIX shells may not implement it)
if (set -o pipefail 2>/dev/null); then
    :
else
    printf '[WARN] pipefail not supported; continuing without it.\n' >&2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
    printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
    cat <<'USAGE'
Usage: run_method_negotiation.sh [options]

Runs the full Craigslist Bargains negotiation pipeline:
  1. Supervised fine-tuning (SFT)
  2. MCTS-guided improvement
  3. Metric evaluation on generated rollouts

Options:
  --model_name NAME          Base model to fine-tune (default: Qwen/Qwen2.5-0.5B-Instruct)
  --data_dir PATH            Dataset root (default: dataset/craigslist_bargains)
  --run_name NAME            Name for output folder (default: run-<timestamp>)
  --sft-output PATH          Override SFT output directory
  --mcts-output PATH         Override MCTS output directory
  --eval-output PATH         Override evaluation output directory
  --cuda-devices IDS         Comma-separated GPU ids to expose (default: 0)
  --require-cuda             Fail fast if CUDA is unavailable
  --num_samples N            Dialogues to generate during MCTS (default: 128)
  --rollouts N               Rollouts per search step for MCTS (default: 24)
  --prompt_turns N           Prompt turns passed to generator (default: 2)
  --eval_samples N           Limit test evaluation examples (optional)
  --sft-extra "..."          Additional flags forwarded to train_craigslist_sft.py
  --mcts-extra "..."         Additional flags forwarded to mcts_negotiation_training.py
  --eval-extra "..."         Additional flags forwarded to evaluate_negotiation_metrics.py
  -h, --help                 Show this message and exit

Examples:
  bash run/run_method_negotiation.sh \
      --model_name distilgpt2 \
      --cuda-devices 0 \
      --num_samples 256 --rollouts 32 \
      --mcts-extra "--epochs 1 --eval_samples 512"
USAGE
}

trap 'log "Pipeline failed." >&2' ERR

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
DATA_DIR="${PROJECT_ROOT}/dataset/craigslist_bargains"
RUN_NAME=""
SFT_OUTPUT=""
MCTS_OUTPUT=""
EVAL_OUTPUT=""
CUDA_DEVICES="0"
REQUIRE_CUDA=false
NUM_SAMPLES=128
ROLLOUTS=24
PROMPT_TURNS=2
EVAL_SAMPLES=""
SFT_EXTRA=""
MCTS_EXTRA=""
EVAL_EXTRA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            MODEL_NAME="$2"; shift 2 ;;
        --data_dir)
            DATA_DIR="$2"; shift 2 ;;
        --run_name)
            RUN_NAME="$2"; shift 2 ;;
        --sft-output)
            SFT_OUTPUT="$2"; shift 2 ;;
        --mcts-output)
            MCTS_OUTPUT="$2"; shift 2 ;;
        --eval-output)
            EVAL_OUTPUT="$2"; shift 2 ;;
        --cuda-devices)
            CUDA_DEVICES="$2"; shift 2 ;;
        --require-cuda)
            REQUIRE_CUDA=true; shift 1 ;;
        --num_samples)
            NUM_SAMPLES="$2"; shift 2 ;;
        --rollouts)
            ROLLOUTS="$2"; shift 2 ;;
        --prompt_turns)
            PROMPT_TURNS="$2"; shift 2 ;;
        --eval_samples)
            EVAL_SAMPLES="$2"; shift 2 ;;
        --sft-extra)
            SFT_EXTRA+=" $2"; shift 2 ;;
        --mcts-extra)
            MCTS_EXTRA+=" $2"; shift 2 ;;
        --eval-extra)
            EVAL_EXTRA+=" $2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage
            exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
log "CUDA_VISIBLE_DEVICES set to ${CUDA_VISIBLE_DEVICES}"

CUDA_REQUIRED_FLAG="0"
if ${REQUIRE_CUDA}; then
    CUDA_REQUIRED_FLAG="1"
fi

if ! RUNNER_REQUIRE_CUDA="${CUDA_REQUIRED_FLAG}" python - <<'PY'
import os
import sys
try:
    import torch
except Exception as exc:  # torch not installed or broken
    print(f"CUDA check failed to import torch: {exc}", file=sys.stderr)
    raise SystemExit(1)

require = os.environ.get("RUNNER_REQUIRE_CUDA") == "1"
available = torch.cuda.is_available()
print(f"CUDA available: {available}")
if require and not available:
    raise SystemExit(1)
PY
then
    echo "[ERROR] CUDA required but not available." >&2
    exit 1
fi

if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="run-$(date +'%Y%m%d-%H%M%S')"
fi

RUN_ROOT="${PROJECT_ROOT}/outputs/craigslist-pipeline/${RUN_NAME}"
SFT_OUTPUT="${SFT_OUTPUT:-${RUN_ROOT}/craigslist-sft}"
MCTS_OUTPUT="${MCTS_OUTPUT:-${RUN_ROOT}/craigslist-mcts}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${RUN_ROOT}/evaluation}"

mkdir -p "${SFT_OUTPUT}" "${MCTS_OUTPUT}" "${EVAL_OUTPUT}"

cd "${PROJECT_ROOT}"

log "Pipeline outputs will be written to ${RUN_ROOT}"

# Convert extra argument strings to arrays for safe expansion
SFT_ARGS=()
if [[ -n "${SFT_EXTRA// }" ]]; then
    # shellcheck disable=SC2206
    SFT_ARGS=(${SFT_EXTRA})
fi

MCTS_ARGS=()
if [[ -n "${MCTS_EXTRA// }" ]]; then
    # shellcheck disable=SC2206
    MCTS_ARGS=(${MCTS_EXTRA})
fi

EVAL_ARGS=()
if [[ -n "${EVAL_EXTRA// }" ]]; then
    # shellcheck disable=SC2206
    EVAL_ARGS=(${EVAL_EXTRA})
fi

log "Step 1/3: Supervised fine-tuning -> ${SFT_OUTPUT}"
python examples/train_craigslist_sft.py \
    --data_dir "${DATA_DIR}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${SFT_OUTPUT}" \
    "${SFT_ARGS[@]}"

log "Step 2/3: MCTS-guided improvement -> ${MCTS_OUTPUT}"
MCTS_CMD=(
    python examples/mcts_negotiation_training.py
    --data_dir "${DATA_DIR}"
    --model_path "${SFT_OUTPUT}"
    --output_dir "${MCTS_OUTPUT}"
    --num_samples "${NUM_SAMPLES}"
    --rollouts "${ROLLOUTS}"
    --prompt_turns "${PROMPT_TURNS}"
)
if [[ -n "${EVAL_SAMPLES}" ]]; then
    MCTS_CMD+=(--eval_samples "${EVAL_SAMPLES}")
fi
MCTS_CMD+=("${MCTS_ARGS[@]}")
"${MCTS_CMD[@]}"

log "Step 3/3: Evaluating rollouts -> ${EVAL_OUTPUT}"
EVAL_CMD=(
    python examples/evaluate_negotiation_metrics.py
    --input "${MCTS_OUTPUT}/mcts_dialogues.jsonl"
    --output_dir "${EVAL_OUTPUT}"
)
EVAL_CMD+=("${EVAL_ARGS[@]}")
"${EVAL_CMD[@]}"

trap - ERR
log "Pipeline complete. Key artifacts:"
log "  SFT checkpoint   : ${SFT_OUTPUT}"
log "  MCTS outputs     : ${MCTS_OUTPUT}"
log "  Evaluation metrics: ${EVAL_OUTPUT}/metrics.json"
