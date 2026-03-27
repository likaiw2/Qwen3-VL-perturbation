"""
eval_saved_videos.py

Re-runs inference on every saved checkpoint video inside a test_results/<run>/
directory, then compares the re-computed CE loss and generated answer against
the values that were recorded during the PGD attack.

Purpose: detect whether H.264 encoding degrades the adversarial perturbation.

Usage:
    python eval_saved_videos.py --result_dir test_results/0323_231846
    python eval_saved_videos.py --result_dir test_results/0323_231846 --device cuda:1
"""

import argparse
import json
import os
import sys

import torch
from qwen_pgd_attack import Qwen3VLPGD

try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False


# ── helpers (mirrors pgd_qa_test.py) ────────────────────────────────────────

def build_inputs(processor, video_path, question, answer, fps, device):
    messages_prompt = [{"role": "user", "content": [
        {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": fps},
        {"type": "text", "text": question},
    ]}]
    messages_full = messages_prompt + [
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]
    prompt_text = processor.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False)

    if QWEN_VL_UTILS_AVAILABLE:
        image_inputs, video_inputs = process_vision_info(messages_prompt)
    else:
        image_inputs, video_inputs = None, [video_path]

    inputs_prompt = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        return_tensors="pt")
    prompt_len = inputs_prompt["input_ids"].shape[1]

    inputs_full = processor(
        text=[full_text], images=image_inputs, videos=video_inputs,
        return_tensors="pt")
    inputs_full = {k: v.to(device) for k, v in inputs_full.items()}

    labels = inputs_full["input_ids"].clone()
    labels[:, :prompt_len] = -100
    return inputs_full, prompt_len, labels


def compute_ce_loss(model, inputs_full, labels):
    with torch.no_grad():
        outputs = model(
            input_ids=inputs_full["input_ids"],
            attention_mask=inputs_full["attention_mask"],
            pixel_values_videos=inputs_full["pixel_values_videos"],
            video_grid_thw=inputs_full["video_grid_thw"],
            labels=labels,
        )
    return outputs.loss.item()


def generate_answer(model, processor, inputs_full, prompt_len):
    input_ids_p = inputs_full["input_ids"][:, :prompt_len]
    attn_p = inputs_full["attention_mask"][:, :prompt_len]
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids_p,
            attention_mask=attn_p,
            pixel_values_videos=inputs_full["pixel_values_videos"],
            video_grid_thw=inputs_full["video_grid_thw"],
            max_new_tokens=64,
            do_sample=False,
        )
    return processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)[0].strip()


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", default="test_results/0323_231846")
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()

    json_path = os.path.join(args.result_dir, "results.json")
    if not os.path.exists(json_path):
        sys.exit(f"results.json not found in {args.result_dir}")

    with open(json_path) as f:
        meta = json.load(f)

    question   = meta["question"]
    gt_answer  = meta["gt_answer"]
    fps        = meta["pgd_params"]["fps"]
    checkpoints = meta["checkpoints"]

    print(f"Question : {question}")
    print(f"GT Answer: {gt_answer}")
    print(f"FPS used : {fps}")
    print(f"Result dir: {args.result_dir}\n")

    # Load model once
    print("Loading model...")
    attacker = Qwen3VLPGD(
        model_path=args.model,
        device=args.device,
        dtype=torch.float32,
        vision_only=False,
    )
    model     = attacker.model
    processor = attacker.processor
    print("Model loaded.\n")

    W = 18   # column width for answers
    HDR = (f"{'Tag':<12} | {'Orig Loss':>9} {'Re Loss':>9} {'ΔLoss':>8} | "
           f"{'Orig Answer':<50} | {'Re Answer':<50}")
    print(HDR)
    print("-" * len(HDR))

    comparison = []
    for ckpt in checkpoints:
        tag        = ckpt["tag"]
        orig_loss  = ckpt["ce_loss"]
        orig_ans   = ckpt["answer"]
        video_file = ckpt.get("video_file")

        if not video_file:
            print(f"{tag:<12} | [no video file recorded, skipped]")
            continue

        video_path = os.path.join(args.result_dir, video_file)
        if not os.path.exists(video_path):
            print(f"{tag:<12} | [file not found: {video_path}]")
            continue

        try:
            inputs_full, prompt_len, labels = build_inputs(
                processor, video_path, question, gt_answer, fps, args.device)
            re_loss = compute_ce_loss(model, inputs_full, labels)
            re_ans  = generate_answer(model, processor, inputs_full, prompt_len)
        except Exception as e:
            print(f"{tag:<12} | [ERROR: {e}]")
            continue

        delta     = re_loss - orig_loss
        same_ans  = (orig_ans.strip() == re_ans.strip())
        row = dict(tag=tag, orig_loss=orig_loss, re_loss=re_loss, delta=delta,
                   orig_ans=orig_ans, re_ans=re_ans, same_ans=same_ans)
        comparison.append(row)

        print(f"{tag:<12} | {orig_loss:>9.4f} {re_loss:>9.4f} {delta:>+8.4f} | "
              f"{orig_ans[:50]:<50} | {re_ans[:50]:<50}")

    # Save comparison JSON
    out_path = os.path.join(args.result_dir, "video_eval_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"question": question, "gt_answer": gt_answer,
                   "comparison": comparison}, f, indent=2, ensure_ascii=False)

    print(f"\nComparison saved to: {out_path}")

    # Summary
    if comparison:
        avg_delta   = sum(r["delta"] for r in comparison) / len(comparison)
        same_count  = sum(1 for r in comparison if r["same_ans"])
        print(f"\n{'='*60}")
        print(f"Avg loss delta (re - orig): {avg_delta:+.4f}")
        print(f"Exact-match answers: {same_count}/{len(comparison)}")
        print(f"  → Positive Δ means re-computed loss is HIGHER than recorded")
        print(f"  → Negative Δ means H.264 encoding degraded the attack effect")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

