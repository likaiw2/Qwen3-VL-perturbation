#!/bin/bash
# =============================================================================
# Run PGD attack on NuScenes videos (dual-GPU parallel), then evaluate.
#
# Logs, CSV stats, and eval results are saved to logs/<mmdd_hhmmss>/.
#
# Usage:
#   bash run_attack_and_eval.sh
#   bash run_attack_and_eval.sh --output_dir /data/likai/nuscene_tasks/0321
#   bash run_attack_and_eval.sh --num_gpus 4
# =============================================================================

set -e  # exit on error

# ── Defaults (override via CLI) ──────────────────────────────────────────────
INPUT_DIR="data/QA_Scenes_500"
OUTPUT_DIR="/data/likai/nuscene_tasks/0320"
ORIGINAL_DIR="data/QA_Scenes_500"
RANDNOISE_DIR="data/_ROOT_NuScenes/QA_Scenes_500_randomnoise"
QUESTIONS="data/_ROOT_NuScenes/data/questions/NuScenes_val_questions.json"
MODEL="Qwen/Qwen3-VL-8B-Instruct"
NUM_GPUS=2
LOG_DIR=""  # auto-created if empty

# ── Parse optional overrides ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dir)      INPUT_DIR="$2";      shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";      shift 2 ;;
        --original_dir)   ORIGINAL_DIR="$2";    shift 2 ;;
        --randnoise_dir)  RANDNOISE_DIR="$2";   shift 2 ;;
        --questions)      QUESTIONS="$2";       shift 2 ;;
        --model)          MODEL="$2";           shift 2 ;;
        --num_gpus)       NUM_GPUS="$2";        shift 2 ;;
        --log_dir)        LOG_DIR="$2";         shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Create timestamped log directory if not specified
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="logs/$(date +%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

EVAL_OUTPUT="$LOG_DIR/eval_attack.json"

echo "============================================================"
echo "  PGD Attack + Evaluation Pipeline"
echo "============================================================"
echo "  Input dir:     $INPUT_DIR"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Model:         $MODEL"
echo "  Num GPUs:      $NUM_GPUS"
echo "  Log dir:       $LOG_DIR"
echo "  Eval output:   $EVAL_OUTPUT"
echo "============================================================"
echo ""

# ── Step 1: PGD Attack (parallel, with resume) ──────────────────────────────
echo ">>> Step 1: Running PGD attack (parallel=${NUM_GPUS} GPUs) ..."
python attack_nuscenes.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --num_gpus "$NUM_GPUS" \
    --log_dir "$LOG_DIR" \
    --parallel \
    2>&1 | tee "$LOG_DIR/attack_main.log"

echo ""
echo ">>> Step 1 complete: adversarial videos saved to $OUTPUT_DIR"
echo ""

# ── Step 2: Evaluate ────────────────────────────────────────────────────────
echo ">>> Step 2: Running evaluation ..."
python eval_attack.py \
    --original_dir "$ORIGINAL_DIR" \
    --pgd_dir "$OUTPUT_DIR" \
    --randnoise_dir "$RANDNOISE_DIR" \
    --questions "$QUESTIONS" \
    --model "$MODEL" \
    --gpus cuda:0 cuda:1 \
    --output "$EVAL_OUTPUT" \
    --resume \
    2>&1 | tee "$LOG_DIR/eval.log"

echo ""
echo ">>> Step 2 complete: evaluation results saved to $EVAL_OUTPUT"
echo "============================================================"
echo "  All done! Logs saved to: $LOG_DIR/"
echo "============================================================"

