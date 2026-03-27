"""
PGD Vision Attack Test

Randomly selects one video from QA_Scenes_500, runs PGD attack with
checkpoints every 10 iterations (total 100 iters → 10 checkpoints + best),
then evaluates each checkpoint: generate answer & compute CE loss vs GT.

Usage:
    python pgd_vision_test.py --device cuda:0
    python pgd_vision_test.py --device cuda:0 --num_iter 50 --ckpt_every 5
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
from attack_nuscenes import save_adversarial_video_direct
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

    def fileno(self):
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


def parse_args():
    p = argparse.ArgumentParser(description="PGD QA-level attack test with checkpoints")
    p.add_argument("--input_dir", default="data/QA_Scenes_500")
    p.add_argument("--questions", default="data/_ROOT_NuScenes/data/questions/NuScenes_val_questions.json")
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--alpha", type=float, default=1 / 255)
    p.add_argument("--num_iter", type=int, default=200)
    p.add_argument("--ckpt_every", type=int, default=10, help="Save checkpoint every N iters")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_dir", default=None,
                   help="Log directory. Auto-created as logs/MMDD_HHMMSS if not set.")
    p.add_argument("--output_dir", default="test_results",
                   help="Base directory to save per-run results (videos, JSON). "
                        "A timestamped subdirectory is created automatically.")
    p.add_argument("--lossless", action="store_true",
                   help="Save checkpoint videos with lossless FFV1 codec (.mkv) "
                        "instead of H.264 (.mp4). Prevents compression from "
                        "destroying the adversarial perturbation.")
    return p.parse_args()


def load_questions(path, tokens):
    with open(path) as f:
        data = json.load(f)
    grouped = defaultdict(list)
    for q in data["questions"]:
        if q["sample_token"] in tokens:
            grouped[q["sample_token"]].append(q)
    return dict(grouped)


def build_inputs(processor, video_path, question, answer, fps, device):
    """Build prompt-only and full (prompt+answer) inputs for a QA pair."""
    messages_prompt = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": fps},
        {"type": "text", "text": question},
    ]}]
    messages_full = messages_prompt + [
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]

    prompt_text = processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)

    if QWEN_VL_UTILS_AVAILABLE:
        image_inputs, video_inputs = process_vision_info(messages_prompt)
    else:
        image_inputs, video_inputs = None, [video_path]

    inputs_prompt = processor(text=[prompt_text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    prompt_len = inputs_prompt["input_ids"].shape[1]

    inputs_full = processor(text=[full_text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    inputs_full = {k: v.to(device) for k, v in inputs_full.items()}

    labels = inputs_full["input_ids"].clone()
    labels[:, :prompt_len] = -100

    return inputs_full, prompt_len, labels


def compute_ce_loss(model, inputs_full, labels, pixel_values_videos, video_grid_thw):
    """Compute CE loss for given pixel_values."""
    with torch.no_grad():
        outputs = model(
            input_ids=inputs_full["input_ids"],
            attention_mask=inputs_full["attention_mask"],
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            labels=labels,
        )
    return outputs.loss.item()


def generate_answer(model, processor, inputs_full, pixel_values_videos, video_grid_thw, prompt_len):
    """Generate answer text given pixel_values (replacing the video tokens)."""
    # We only feed the prompt portion of input_ids for generation
    input_ids_prompt = inputs_full["input_ids"][:, :prompt_len]
    attention_mask_prompt = inputs_full["attention_mask"][:, :prompt_len]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids_prompt,
            attention_mask=attention_mask_prompt,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            max_new_tokens=64,
            do_sample=False,
        )
    generated_ids = output_ids[:, prompt_len:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def run_pgd_once(
    run_idx, result_dir, vid_ext, codec_label,
    model, processor, attacker, args,
    inputs_full, prompt_len, labels,
    pixel_values_clean, video_grid_thw,
    token, video_path, question, gt_answer,
):
    """Run one full PGD attack + evaluation pass. Returns list of checkpoint result dicts."""
    run_label = f"Run {run_idx}"
    run_dir = os.path.join(result_dir, f"run{run_idx}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  {run_label}")
    print(f"{'#'*70}")

    # --- PGD attack ---
    checkpoints = {0: pixel_values_clean.clone().cpu()}
    pixel_values_adv = pixel_values_clean.clone().detach()
    best_adv = pixel_values_adv.clone()
    best_loss = -float("inf")

    # --- Weight fingerprint: verify model params never change during PGD ---
    def param_fingerprint(m):
        """Sum of all parameter values as a cheap integrity check."""
        return sum(p.data.float().sum().item() for p in m.parameters())

    fp_before = param_fingerprint(model)
    print(f"[weight check] param fingerprint BEFORE attack: {fp_before:.6f}")

    print("Running PGD attack...")
    t0 = time.time()

    for i in range(args.num_iter):
        pixel_values_adv = pixel_values_adv.detach().requires_grad_(True)

        outputs = model(
            input_ids=inputs_full["input_ids"],
            attention_mask=inputs_full["attention_mask"],
            pixel_values_videos=pixel_values_adv,
            video_grid_thw=video_grid_thw,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        grad = pixel_values_adv.grad.detach().clone()
        current_loss = loss.item()

        del outputs, loss
        torch.cuda.empty_cache()

        # Check that no model parameter accumulated a gradient this iter
        if i == 0:
            param_with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
            if param_with_grad:
                print(f"[weight check] WARNING: {len(param_with_grad)} model params have "
                      f".grad after iter 0 (e.g. {param_with_grad[0]}). "
                      f"Weights may drift if an optimizer step is accidentally called.")
            else:
                print(f"[weight check] iter 0: no model param has .grad ✓")

        if current_loss > best_loss:
            best_loss = current_loss
            best_adv = pixel_values_adv.detach().clone()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Iter {i+1:3d}/{args.num_iter}: CE_loss = {current_loss:.6f}")

        if (i + 1) % args.ckpt_every == 0:
            checkpoints[i + 1] = pixel_values_adv.detach().clone().cpu()

        with torch.no_grad():
            N, D = pixel_values_adv.shape
            pv_adv = pixel_values_adv.view(N, 3, -1)
            grad_v = grad.view(N, 3, -1)
            pv_clean = pixel_values_clean.view(N, 3, -1)
            std = attacker.image_std.view(1, 3, 1)
            mean = attacker.image_mean.view(1, 3, 1)

            pv_adv = pv_adv + (args.alpha / std) * grad_v.sign()
            clean_01 = pv_clean * std + mean
            adv_01 = pv_adv * std + mean
            pert = torch.clamp(adv_01 - clean_01, -args.eps, args.eps)
            adv_01 = torch.clamp(clean_01 + pert, 0.0, 1.0)
            pixel_values_adv = ((adv_01 - mean) / std).view(N, D)

    elapsed = time.time() - t0
    checkpoints["best"] = best_adv.clone().cpu()

    fp_after = param_fingerprint(model)
    print(f"[weight check] param fingerprint AFTER  attack: {fp_after:.6f}")
    if fp_before == fp_after:
        print(f"[weight check] ✓ model weights unchanged (fingerprints match)")
    else:
        print(f"[weight check] ✗ WARNING: fingerprint changed by {fp_after - fp_before:.6f} "
              f"— model weights were modified during PGD!")

    print(f"\nPGD done in {elapsed:.1f}s. Best CE loss = {best_loss:.6f}")

    # --- Evaluate each checkpoint ---
    print(f"\n{'='*70}")
    print(f"{'Checkpoint':<12} {'CE Loss':>10} {'Generated Answer'}")
    print(f"{'-'*70}")

    import shutil as _shutil
    clean_video_dest = os.path.join(run_dir, f"clean{vid_ext}")
    _shutil.copy2(video_path, clean_video_dest)

    model_mean = attacker.image_mean.cpu()
    model_std = attacker.image_std.cpu()
    ckpt_order = sorted([k for k in checkpoints if isinstance(k, int)]) + ["best"]
    ckpt_results = []

    for ckpt_key in ckpt_order:
        pv = checkpoints[ckpt_key].to(args.device).to(attacker.dtype)
        ce = compute_ce_loss(model, inputs_full, labels, pv, video_grid_thw)
        ans = generate_answer(model, processor, inputs_full, pv, video_grid_thw, prompt_len)

        tag = f"iter_{ckpt_key}" if isinstance(ckpt_key, int) else "best"
        marker = " ← clean" if ckpt_key == 0 else (" ★ best" if ckpt_key == "best" else "")
        print(f"{tag:<12} {ce:>10.4f}   {ans[:60]}{marker}")

        video_out_path = os.path.join(run_dir, f"{tag}{vid_ext}")
        try:
            save_adversarial_video_direct(
                pixel_values_clean=pixel_values_clean.cpu(),
                pixel_values_adv=checkpoints[ckpt_key],
                video_grid_thw=video_grid_thw.cpu(),
                eps=args.eps,
                original_video_path=video_path,
                output_path=video_out_path,
                model_mean=model_mean,
                model_std=model_std,
                lossless=args.lossless,
                sample_fps=args.fps,
            )
        except Exception as e:
            print(f"  [Warning] Could not save video for {tag}: {e}")
            video_out_path = None

        ckpt_results.append({
            "tag": tag,
            "iter": ckpt_key if isinstance(ckpt_key, int) else "best",
            "ce_loss": ce,
            "answer": ans,
            "video_file": os.path.basename(video_out_path) if video_out_path else None,
        })

    print(f"{'-'*70}")
    print(f"GT answer: {gt_answer}")
    print(f"{'='*70}")

    run_summary = {
        "run": run_idx,
        "token": token,
        "video_path": video_path,
        "question": question,
        "gt_answer": gt_answer,
        "pgd_params": {
            "eps": args.eps, "alpha": args.alpha,
            "num_iter": args.num_iter, "ckpt_every": args.ckpt_every,
            "fps": args.fps, "lossless": args.lossless,
        },
        "checkpoints": ckpt_results,
    }
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(run_summary, jf, indent=2, ensure_ascii=False)
    print(f"Results saved to: {run_dir}/  [{codec_label}]")
    return ckpt_results


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # --- 0. Set up logging ---
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_dir = args.log_dir or os.path.join("logs", timestamp)
    tee = TeeLogger.start(log_dir, "pgd_qa_test.log")
    print(f"Logging to: {os.path.join(log_dir, 'pgd_qa_test.log')}\n")

    # --- 0b. Set up results output directory ---
    result_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    print(f"Results will be saved to: {result_dir}\n")

    # --- 1. Randomly pick a video + QA pair ---
    tokens = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    token_to_qa = load_questions(args.questions, set(tokens))
    token = '0186cbb6a991468c9d23226afeb62b15'
    target_qa = 'There is a moving thing to the back of me; what is it?'
    qa_list = token_to_qa.get(token, [])
    qa = next((q for q in qa_list if q["question"] == target_qa), None)
    if qa is None:
        raise ValueError(f"Question not found for token {token!r}: {target_qa!r}")
    video_path = os.path.join(args.input_dir, token, "CAM_FRONT.mp4")
    question = qa["question"]
    gt_answer = qa["answer"]

    print(f"{'='*70}")
    print(f"Token:    {token}")
    print(f"Video:    {video_path}")
    print(f"Question: {question}")
    print(f"GT:       {gt_answer}")
    print(f"PGD:      eps={args.eps:.4f}  alpha={args.alpha:.4f}  "
          f"iters={args.num_iter}  ckpt_every={args.ckpt_every}")
    print(f"{'='*70}\n")

    # --- 1b. Force deterministic CUDA execution ---
    # Without this, CUDA parallel reductions run in non-deterministic thread
    # order, so two identical runs produce slightly different float results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # cuBLAS also needs this env var for fully deterministic GEMM:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # --- 2. Load model (full, vision_only=False) ---
    print("Loading model...")
    attacker = Qwen3VLPGD(
        model_path=args.model,
        device=args.device,
        dtype=torch.bfloat16,   # bfloat16 halves VRAM vs float32
        vision_only=False,
    )
    model = attacker.model
    # Enable gradient checkpointing: recompute activations during backward
    # instead of storing them all, trading compute for VRAM.
    model.gradient_checkpointing_enable()
    processor = attacker.processor

    # --- 3. Build inputs ---
    inputs_full, prompt_len, labels = build_inputs(
        processor, video_path, question, gt_answer, args.fps, args.device
    )
    pixel_values_clean = inputs_full["pixel_values_videos"].to(attacker.dtype)
    video_grid_thw = inputs_full["video_grid_thw"]

    answer_len = inputs_full["input_ids"].shape[1] - prompt_len
    print(f"prompt_len={prompt_len}, answer_len={answer_len}, "
          f"total={inputs_full['input_ids'].shape[1]}\n")

    vid_ext = ".mkv" if args.lossless else ".mp4"
    codec_label = "FFV1/lossless" if args.lossless else "H.264/lossy"

    # --- 4 & 5. Run PGD attack twice and compare ---
    shared_kwargs = dict(
        result_dir=result_dir, vid_ext=vid_ext, codec_label=codec_label,
        model=model, processor=processor, attacker=attacker, args=args,
        inputs_full=inputs_full, prompt_len=prompt_len, labels=labels,
        pixel_values_clean=pixel_values_clean, video_grid_thw=video_grid_thw,
        token=token, video_path=video_path, question=question, gt_answer=gt_answer,
    )
    results1 = run_pgd_once(run_idx=1, **shared_kwargs)
    # results2 = run_pgd_once(run_idx=2, **shared_kwargs)

    # # --- 6. Side-by-side comparison ---
    # print(f"\n{'='*70}")
    # print("COMPARISON  (Run 1  vs  Run 2)")
    # print(f"{'='*70}")
    # print(f"{'Checkpoint':<12} {'CE1':>8} {'CE2':>8}  {'Same answer?':<14}  Answer1  /  Answer2")
    # print(f"{'-'*70}")
    # for r1, r2 in zip(results1, results2):
    #     same = "✓ same" if r1["answer"] == r2["answer"] else "✗ diff"
    #     ce_diff = r2["ce_loss"] - r1["ce_loss"]
    #     ans_str = (f"{r1['answer'][:30]}  /  {r2['answer'][:30]}"
    #                if r1["answer"] != r2["answer"] else r1["answer"][:60])
    #     print(f"{r1['tag']:<12} {r1['ce_loss']:>8.4f} {r2['ce_loss']:>8.4f} "
    #           f"({ce_diff:+.4f})  {same:<14}  {ans_str}")
    # print(f"{'='*70}")
    # print(f"GT answer: {gt_answer}")

    # # Save combined comparison JSON
    # comparison = {
    #     "token": token, "question": question, "gt_answer": gt_answer,
    #     "run1": results1, "run2": results2,
    # }
    # with open(os.path.join(result_dir, "comparison.json"), "w", encoding="utf-8") as jf:
    #     json.dump(comparison, jf, indent=2, ensure_ascii=False)
    # print(f"\nComparison saved to: {result_dir}/comparison.json")

    tee.close()


if __name__ == "__main__":
    main()
