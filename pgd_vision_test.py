"""
PGD Vision-Feature Attack Test

Randomly selects one video from QA_Scenes_500, runs a FEATURE-LEVEL PGD attack
(minimize cosine similarity of vision encoder outputs, vision_only style).
Saves a checkpoint every `--ckpt_every` iterations, then evaluates each checkpoint:
  - cosine similarity (attack metric)
  - CE loss vs GT answer  (measures QA degradation)
  - generated answer      (qualitative)

Usage:
    python pgd_vision_test.py --device cuda:0
    python pgd_vision_test.py --device cuda:0 --num_iter 50 --ckpt_every 5 --seed 42
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

import torch

from qwen_pgd_attack import Qwen3VLPGD


# ---------------------------------------------------------------------------
# Tee logger: writes every print() to both terminal AND a log file
# ---------------------------------------------------------------------------
class TeeLogger:
    def __init__(self, filepath):
        self._file = open(filepath, "w", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def fileno(self):          # needed by some libraries
        return self._stdout.fileno()

    def close(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._file.close()

    @staticmethod
    def start(log_dir: str, filename: str = "run.log") -> "TeeLogger":
        os.makedirs(log_dir, exist_ok=True)
        tee = TeeLogger(os.path.join(log_dir, filename))
        sys.stdout = tee
        sys.stderr = tee
        return tee

try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PGD vision-feature attack test with checkpoints")
    p.add_argument("--input_dir",  default="data/QA_Scenes_500")
    p.add_argument("--questions",  default="data/_ROOT_NuScenes/data/questions/NuScenes_val_questions.json")
    p.add_argument("--model",      default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--eps",        type=float, default=8 / 255)
    p.add_argument("--alpha",      type=float, default=1 / 255)
    p.add_argument("--num_iter",   type=int,   default=100)
    p.add_argument("--ckpt_every", type=int,   default=10)
    p.add_argument("--fps",        type=float, default=1.0)
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--log_dir",    default=None,
                   help="Log directory. Auto-created as logs/MMDD_HHMMSS if not set.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_questions(path, tokens):
    with open(path) as f:
        data = json.load(f)
    grouped = defaultdict(list)
    for q in data["questions"]:
        if q["sample_token"] in tokens:
            grouped[q["sample_token"]].append(q)
    return dict(grouped)


# ---------------------------------------------------------------------------
# Evaluation helpers  (require full model)
# ---------------------------------------------------------------------------
def compute_ce_loss(attacker, inputs_full, labels, pixel_values_videos, video_grid_thw):
    """CE loss between model output and GT answer tokens."""
    with torch.no_grad():
        outputs = attacker.model(
            input_ids=inputs_full["input_ids"],
            attention_mask=inputs_full["attention_mask"],
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            labels=labels,
        )
    return outputs.loss.item()


def generate_answer(attacker, inputs_full, pixel_values_videos, video_grid_thw, prompt_len):
    """Generate an answer given the (potentially perturbed) pixel values."""
    input_ids_p = inputs_full["input_ids"][:, :prompt_len]
    attn_mask_p = inputs_full["attention_mask"][:, :prompt_len]
    with torch.no_grad():
        out_ids = attacker.model.generate(
            input_ids=input_ids_p,
            attention_mask=attn_mask_p,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            max_new_tokens=64,
            do_sample=False,
        )
    generated = out_ids[:, prompt_len:]
    return attacker.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def build_qa_inputs(attacker, video_path, question, gt_answer, fps, device):
    """Build full-sequence inputs (prompt + answer) for CE-loss computation."""
    messages_prompt = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": fps},
        {"type": "text", "text": question},
    ]}]
    messages_full = messages_prompt + [
        {"role": "assistant", "content": [{"type": "text", "text": gt_answer}]}
    ]
    prompt_text = attacker.processor.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True)
    full_text = attacker.processor.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False)

    if QWEN_VL_UTILS_AVAILABLE:
        image_inputs, video_inputs = process_vision_info(messages_prompt)
    else:
        image_inputs, video_inputs = None, [video_path]

    inputs_prompt = attacker.processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    prompt_len = inputs_prompt["input_ids"].shape[1]

    inputs_full = attacker.processor(
        text=[full_text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    inputs_full = {k: v.to(device) for k, v in inputs_full.items()}

    labels = inputs_full["input_ids"].clone()
    labels[:, :prompt_len] = -100
    return inputs_full, prompt_len, labels




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # --- 0. Set up logging (tee to file + terminal) ---
    log_dir = args.log_dir or os.path.join("logs", datetime.now().strftime("%m%d_%H%M%S"))
    tee = TeeLogger.start(log_dir, "pgd_vision_test.log")
    print(f"Logging to: {os.path.join(log_dir, 'pgd_vision_test.log')}\n")

    # --- 1. Pick random video + QA pair ---
    tokens = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    token_to_qa = load_questions(args.questions, set(tokens))
    valid_tokens = [t for t in tokens if t in token_to_qa]
    token = random.choice(valid_tokens)
    qa = random.choice(token_to_qa[token])
    video_path = os.path.join(args.input_dir, token, "CAM_FRONT.mp4")
    question, gt_answer = qa["question"], qa["answer"]

    print(f"{'='*70}")
    print(f"Token:    {token}")
    print(f"Video:    {video_path}")
    print(f"Question: {question}")
    print(f"GT:       {gt_answer}")
    print(f"Attack:   VISION-FEATURE (minimize cos_sim)")
    print(f"PGD:      eps={args.eps:.4f}  alpha={args.alpha:.4f}  "
          f"iters={args.num_iter}  ckpt_every={args.ckpt_every}")
    print(f"{'='*70}\n")

    # --- 2. Load FULL model (needed for CE loss + generation eval) ---
    print("Loading model (full, vision_only=False for eval)...")
    attacker = Qwen3VLPGD(
        model_path=args.model,
        device=args.device,
        dtype=torch.float32,
        vision_only=False,
    )

    # --- 3. Preprocess video for feature-level attack ---
    inputs_vis = attacker.preprocess_video(video_path, "Describe this video.", fps=args.fps)
    pixel_values_clean = inputs_vis["pixel_values_videos"].to(attacker.dtype)
    video_grid_thw = inputs_vis["video_grid_thw"]

    with torch.no_grad():
        features_clean = attacker.get_visual_features(
            pixel_values_clean, video_grid_thw, is_video=True)

    # --- 4. Build QA inputs for CE loss / generation ---
    inputs_full, prompt_len, labels = build_qa_inputs(
        attacker, video_path, question, gt_answer, args.fps, args.device)
    print(f"prompt_len={prompt_len}, "
          f"answer_len={inputs_full['input_ids'].shape[1] - prompt_len}\n")

    # --- 5. Feature-level PGD: gradient DESCENT to minimize cos_sim ---
    checkpoints = {0: pixel_values_clean.clone().cpu()}

    pixel_values_adv = pixel_values_clean.clone().detach()
    best_adv = pixel_values_adv.clone()
    best_cos_sim = 1.0

    print("Running feature-level PGD (minimize cosine similarity)...")
    t0 = time.time()

    for i in range(args.num_iter):
        pixel_values_adv = pixel_values_adv.detach().requires_grad_(True)

        features_adv = attacker.get_visual_features(
            pixel_values_adv, video_grid_thw, is_video=True)
        loss = attacker.cosine_similarity_loss(features_adv, features_clean)
        loss.backward()

        grad = pixel_values_adv.grad.detach()
        current_cos_sim = loss.item()

        if current_cos_sim < best_cos_sim:
            best_cos_sim = current_cos_sim
            best_adv = pixel_values_adv.detach().clone()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Iter {i+1:3d}/{args.num_iter}: cos_sim = {current_cos_sim:.6f}")

        if (i + 1) % args.ckpt_every == 0:
            checkpoints[i + 1] = pixel_values_adv.detach().clone().cpu()

        # PGD step: DESCENT (subtract) to minimize cos_sim
        with torch.no_grad():
            N, D = pixel_values_adv.shape
            pv_adv = pixel_values_adv.view(N, 3, -1)
            grad_v = grad.view(N, 3, -1)
            pv_clean = pixel_values_clean.view(N, 3, -1)
            std  = attacker.image_std.view(1, 3, 1)
            mean = attacker.image_mean.view(1, 3, 1)

            pv_adv  = pv_adv - (args.alpha / std) * grad_v.sign()
            clean_01 = pv_clean * std + mean
            adv_01   = pv_adv   * std + mean
            pert = torch.clamp(adv_01 - clean_01, -args.eps, args.eps)
            adv_01 = torch.clamp(clean_01 + pert, 0.0, 1.0)
            pixel_values_adv = ((adv_01 - mean) / std).view(N, D)

    elapsed = time.time() - t0
    checkpoints["best"] = best_adv.clone().cpu()
    print(f"\nPGD done in {elapsed:.1f}s.  Best cos_sim = {best_cos_sim:.6f}\n")

    # --- 6. Evaluate every checkpoint ---
    print(f"{'='*70}")
    print(f"{'Checkpoint':<12} {'cos_sim':>9} {'CE Loss':>9}  Generated Answer")
    print(f"{'-'*70}")

    for ckpt_key in sorted([k for k in checkpoints if isinstance(k, int)]) + ["best"]:
        pv = checkpoints[ckpt_key].to(args.device).to(attacker.dtype)

        with torch.no_grad():
            feat_adv = attacker.get_visual_features(pv, video_grid_thw, is_video=True)
        cos_sim = attacker.cosine_similarity_loss(feat_adv, features_clean).item()

        ce = compute_ce_loss(attacker, inputs_full, labels, pv, video_grid_thw)
        ans = generate_answer(attacker, inputs_full, pv, video_grid_thw, prompt_len)

        tag = f"iter_{ckpt_key}" if isinstance(ckpt_key, int) else "best"
        marker = " ← clean" if ckpt_key == 0 else (" ★ best" if ckpt_key == "best" else "")
        print(f"{tag:<12} {cos_sim:>9.4f} {ce:>9.4f}  {ans[:48]}{marker}")

    print(f"{'-'*70}")
    print(f"GT answer: {gt_answer}")
    print(f"{'='*70}")
    tee.close()


if __name__ == "__main__":
    main()
