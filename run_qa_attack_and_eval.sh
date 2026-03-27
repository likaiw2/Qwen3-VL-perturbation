#!/bin/bash
# =============================================================================
# QA-Level PGD Attack + Evaluation Pipeline
#
# Uses attack_nuscenes_qa.py (full model, vision_only=False) to maximize
# CE loss against GT answers, then evaluates with eval_attack.py.
# Both attack and eval use the 4B model (full model required for QA attack).
#
# Logs, CSV stats, and eval results are saved to logs/<mmdd_hhmmss>/.
#
# Usage:
#   bash run_qa_attack_and_eval.sh
#   bash run_qa_attack_and_eval.sh --output_dir /data/likai/nuscene_tasks/0321_qa
#   bash run_qa_attack_and_eval.sh --num_gpus 4
# =============================================================================

set -e  # exit on error

# ── Defaults (override via CLI) ──────────────────────────────────────────────
INPUT_DIR="data/QA_Scenes_500"
OUTPUT_DIR="/data/likai/nuscene_tasks/0327_iter200"
ORIGINAL_DIR="data/QA_Scenes_500"
RANDNOISE_DIR="data/_ROOT_NuScenes/QA_Scenes_500_randomnoise"
QUESTIONS="data/_ROOT_NuScenes/data/questions/NuScenes_val_questions.json"
MODEL="Qwen/Qwen3-VL-4B-Instruct"   # 4B: full model fits in VRAM for QA attack
NUM_GPUS=2
EPS="0.03137"   # 8/255
ALPHA="0.00392" # 1/255
ITER=200
FPS="1.0"
LOG_DIR=""  # auto-created if empty

# ── Parse optional overrides ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dir)      INPUT_DIR="$2";      shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --original_dir)   ORIGINAL_DIR="$2";   shift 2 ;;
        --randnoise_dir)  RANDNOISE_DIR="$2";  shift 2 ;;
        --questions)      QUESTIONS="$2";      shift 2 ;;
        --model)          MODEL="$2";          shift 2 ;;
        --num_gpus)       NUM_GPUS="$2";       shift 2 ;;
        --eps)            EPS="$2";            shift 2 ;;
        --alpha)          ALPHA="$2";          shift 2 ;;
        --iter)           ITER="$2";           shift 2 ;;
        --fps)            FPS="$2";            shift 2 ;;
        --log_dir)        LOG_DIR="$2";        shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Create timestamped log directory ─────────────────────────────────────────
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="logs/$(date +%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

EVAL_OUTPUT="$LOG_DIR/eval_qa_attack.json"

# Build GPU list for eval (cuda:0 cuda:1 ... cuda:N-1)
EVAL_GPUS=""
for ((i=0; i<NUM_GPUS; i++)); do
    EVAL_GPUS="$EVAL_GPUS cuda:$i"
done

echo "============================================================"
echo "  QA-Level PGD Attack + Evaluation Pipeline"
echo "============================================================"
echo "  Input dir:     $INPUT_DIR"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Model:         $MODEL"
echo "  Num GPUs:      $NUM_GPUS"
echo "  eps:           $EPS  alpha: $ALPHA  iter: $ITER  fps: $FPS"
echo "  Log dir:       $LOG_DIR"
echo "  Eval output:   $EVAL_OUTPUT"
echo "============================================================"
echo ""

# ── Step 1: QA-Level PGD Attack (parallel, with resume) ─────────────────────
echo ">>> Step 1: Running QA-level PGD attack (${NUM_GPUS} GPUs) ..."
python attack_nuscenes_qa.py \
    --input_dir  "$INPUT_DIR"  \
    --output_dir "$OUTPUT_DIR" \
    --questions  "$QUESTIONS"  \
    --model      "$MODEL"      \
    --num_gpus   "$NUM_GPUS"   \
    --eps        "$EPS"        \
    --alpha      "$ALPHA"      \
    --iter       "$ITER"       \
    --fps        "$FPS"        \
    --log_dir    "$LOG_DIR"    \
    --parallel   \
    2>&1 | tee "$LOG_DIR/qa_attack_main.log"

echo ""
echo ">>> Step 1 complete: adversarial videos saved to $OUTPUT_DIR"
echo ""

# ── Step 2: Evaluation ───────────────────────────────────────────────────────
echo ">>> Step 2: Running evaluation (${NUM_GPUS} GPUs) ..."
python eval_attack.py \
    --original_dir  "$ORIGINAL_DIR"  \
    --pgd_dir       "$OUTPUT_DIR"    \
    --randnoise_dir "$RANDNOISE_DIR" \
    --questions     "$QUESTIONS"     \
    --model         "$MODEL"         \
    --gpus          $EVAL_GPUS       \
    --fps           "$FPS"           \
    --output        "$EVAL_OUTPUT"   \
    --resume        \
    2>&1 | tee "$LOG_DIR/eval.log"

echo ""
echo ">>> Step 2 complete: evaluation results saved to $EVAL_OUTPUT"
echo "============================================================"
echo "  All done! Logs saved to: $LOG_DIR/"
echo "============================================================"

