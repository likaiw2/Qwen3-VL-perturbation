"""
Evaluate adversarial attack effectiveness on NuScenes QA.

For each token in QA_Scenes_500, runs Qwen3-VL inference on:
  - original video
  - PGD adversarial video
  - random noise video

Saves results as JSON with fields: token_id, question, gt, original, randnoise, pgd

Single GPU:
    python eval_attack.py --model Qwen/Qwen3-VL-4B-Instruct --device cuda:0

Dual GPU:
    python eval_attack.py --model Qwen/Qwen3-VL-4B-Instruct --gpus cuda:0 cuda:1
"""

import argparse
import json
import os
import time
import multiprocessing as mp

import cv2
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False


def load_model(model_path: str, device: str):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def get_video_nframes(video_path: str, fps: float) -> int:
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total_frames / native_fps
    return max(1, round(duration * fps))


def infer_video(model, processor, video_path: str, question: str, device: str, fps: float = 1.0) -> str:
    nframes = get_video_nframes(video_path, fps)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "max_pixels": 360 * 420, "nframes": nframes},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if QWEN_VL_UTILS_AVAILABLE:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
    else:
        inputs = processor(text=[prompt], videos=[video_path], return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def run_worker(rank: int, world_size: int, tokens: list, token2qa: dict, args, output_path: str):
    """Worker function: processes tokens[rank::world_size] on one GPU."""
    device = args.gpus[rank]
    shard = tokens[rank::world_size]

    print(f"[GPU {rank} / {device}] Loading model, {len(shard)} tokens assigned")
    model, processor = load_model(args.model, device)

    # Resume support per shard
    results = []
    done_tokens = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_tokens = {r["token_id"] for r in results}
        print(f"[GPU {rank}] Resuming: {len(done_tokens)} tokens already done")

    video_name = f"{args.camera}.mp4"
    total = len(shard)
    times = []  # elapsed seconds per processed token

    for i, token in enumerate(shard):
        if token in done_tokens:
            print(f"[GPU{rank}|{device}] {i+1}/{total} {token} (skip)", flush=True)
            continue

        t0 = time.time()
        eta_str = ""
        if times:
            avg = sum(times) / len(times)
            left = total - i - 1
            eta_sec = avg * left
            h, m, s = int(eta_sec // 3600), int(eta_sec % 3600 // 60), int(eta_sec % 60)
            speed = 1 / avg
            eta_str = f"  ETA {h:02d}:{m:02d}:{s:02d}  ({speed:.2f} tok/s)"
        print(f"[GPU{rank}|{device}] {i+1}/{total} {token}{eta_str}", flush=True)

        qa_list = token2qa.get(token)
        if not qa_list:
            print(f"[GPU{rank}|{device}] WARN: No QA for token {token}, skipping")
            continue

        orig_video = os.path.join(args.original_dir, token, video_name)
        pgd_video  = os.path.join(args.pgd_dir,      token, video_name)
        rand_video = os.path.join(args.randnoise_dir, token, video_name)

        if not os.path.exists(orig_video):
            print(f"[GPU{rank}|{device}] WARN: Original video not found: {orig_video}")
            continue

        for qa in qa_list:
            question = qa["question"]
            gt = qa["answer"]

            entry = {
                "token_id": token,
                "question": question,
                "gt": gt,
                "original": None,
                "randnoise": None,
                "pgd": None,
            }

            try:
                entry["original"] = infer_video(model, processor, orig_video, question, device, args.fps)
            except Exception as e:
                print(f"[GPU{rank}|{device}] ERROR original {token}: {e}")

            if os.path.exists(pgd_video):
                try:
                    entry["pgd"] = infer_video(model, processor, pgd_video, question, device, args.fps)
                except Exception as e:
                    print(f"[GPU{rank}|{device}] ERROR pgd {token}: {e}")
            else:
                print(f"[GPU{rank}|{device}] WARN: PGD video not found: {pgd_video}")

            if os.path.exists(rand_video):
                try:
                    entry["randnoise"] = infer_video(model, processor, rand_video, question, device, args.fps)
                except Exception as e:
                    print(f"[GPU{rank}|{device}] ERROR randnoise {token}: {e}")
            else:
                print(f"[GPU{rank}|{device}] WARN: Randnoise video not found: {rand_video}")

            results.append(entry)

        elapsed = time.time() - t0
        times.append(elapsed)

        # Save after each token
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[GPU {rank}] Done. {len(results)} entries -> {output_path}")


def print_stats(results):
    def acc(field):
        valid = [r for r in results if r.get(field) is not None and r.get("gt") is not None]
        if not valid:
            return 0.0, 0
        correct = sum(r[field].strip().lower() == r["gt"].strip().lower() for r in valid)
        return correct / len(valid) * 100, len(valid)

    for field in ("original", "randnoise", "pgd"):
        a, n = acc(field)
        print(f"Accuracy  {field:<10}: {a:.1f}%  (n={n})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir",  type=str, default="data/_ROOT_NuScenes/QA_Scenes_500")
    parser.add_argument("--pgd_dir",       type=str, default="data/_ROOT_NuScenes/QA_Scenes_500_PGD")
    parser.add_argument("--randnoise_dir", type=str, default="data/_ROOT_NuScenes/QA_Scenes_500_randomnoise")
    parser.add_argument("--questions",     type=str, default="data/_ROOT_NuScenes/data/questions/NuScenes_val_questions.json")
    parser.add_argument("--model",         type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device",        type=str, default=None, help="Single GPU, e.g. cuda:0")
    parser.add_argument("--gpus",          type=str, nargs="+", default=None, help="e.g. --gpus cuda:0 cuda:1")
    parser.add_argument("--fps",           type=float, default=1.0)
    parser.add_argument("--camera",        type=str, default="CAM_FRONT")
    parser.add_argument("--output",        type=str, default="results/eval_attack.json")
    parser.add_argument("--resume",        action="store_true")
    args = parser.parse_args()

    # Resolve GPU list
    if args.gpus is None:
        args.gpus = [args.device or "cuda:0"]

    world_size = len(args.gpus)

    # Load questions
    with open(args.questions) as f:
        questions_data = json.load(f)
    token2qa = {}
    for item in questions_data["questions"]:
        token2qa.setdefault(item["sample_token"], []).append(item)

    tokens = sorted([
        d for d in os.listdir(args.original_dir)
        if os.path.isdir(os.path.join(args.original_dir, d))
    ])
    print(f"Found {len(tokens)} tokens, using {world_size} GPU(s): {args.gpus}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if world_size == 1:
        run_worker(0, 1, tokens, token2qa, args, args.output)
    else:
        stem, ext = os.path.splitext(args.output)
        shard_paths = [f"{stem}.rank{r}{ext}" for r in range(world_size)]

        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(world_size):
            p = ctx.Process(
                target=run_worker,
                args=(rank, world_size, tokens, token2qa, args, shard_paths[rank]),
                daemon=False,
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        # Merge shard files
        all_results = []
        for path in shard_paths:
            if os.path.exists(path):
                with open(path) as f:
                    all_results.extend(json.load(f))
            else:
                print(f"[WARN] Shard not found: {path}")

        all_results.sort(key=lambda r: r["token_id"])

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nMerged {len(all_results)} entries -> {args.output}")

    # Print stats
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print_stats(results)


if __name__ == "__main__":
    main()
