"""
Batch QA-level PGD Attack on NuScenes Videos

Unlike attack_nuscenes.py (feature-level, vision_only), this script performs
end-to-end QA attack: it maximizes the Cross-Entropy loss for the GT answer
so the VLM produces incorrect responses.

Requires the full model (vision_only=False) → more VRAM per GPU.

Usage:
    # Single GPU
    python attack_nuscenes_qa.py --device cuda:0

    # Multi-GPU parallel (auto-spawn workers)
    python attack_nuscenes_qa.py --parallel --num_gpus 2
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()


from attack_nuscenes import (
    get_qa_scenes_videos,
    save_adversarial_video_direct,
    save_video_with_uniform_perturbation,
)
from qwen_pgd_attack import Qwen3VLPGD


def load_questions(questions_path: str, tokens: set = None):
    """Load questions JSON and group by sample_token.

    Returns:
        dict: {sample_token: [{"question": ..., "answer": ..., ...}, ...]}
    """
    with open(questions_path) as f:
        data = json.load(f)

    grouped = defaultdict(list)
    for q in data["questions"]:
        tok = q["sample_token"]
        if tokens is not None and tok not in tokens:
            continue
        grouped[tok].append(q)
    return dict(grouped)


def parse_args():
    p = argparse.ArgumentParser(description="Batch QA-level PGD attack on NuScenes")
    p.add_argument("--input_dir", type=str, default="data/QA_Scenes_500",
                    help="Input directory with structure <token>/<camera>.mp4")
    p.add_argument("--output_dir", type=str, default="/data/likai/nuscene_tasks/0327_iter200",
                    help="Output directory for attacked videos")
    p.add_argument("--questions", type=str,
                    default="data/_ROOT_NuScenes/data/questions/NuScenes_val_questions.json",
                    help="Path to NuScenes_val_questions.json")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--alpha", type=float, default=1 / 255)
    p.add_argument("--iter", type=int, default=200)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--num_gpus", type=int, default=2)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--parallel", action="store_true",
                    help="Auto-launch one worker per GPU")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--save_pt", action="store_true")
    return p.parse_args()


def parallel_main(args):
    """Spawn one subprocess per GPU and merge results."""
    num_gpus = args.num_gpus if args.num_gpus > 1 else 2

    if args.log_dir is None:
        log_dir = os.path.join("logs", datetime.now().strftime("%m%d_%H%M%S"))
    else:
        log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory: {log_dir}")

    videos = get_qa_scenes_videos(args.input_dir)
    remaining = [
        v for v in videos
        if not os.path.exists(os.path.join(args.output_dir, v["token"], f"{v['camera']}.mp4"))
    ]
    print(f"=== Parallel mode: {len(videos)} total, {len(remaining)} remaining → {num_gpus} workers ===")
    if not remaining:
        print("All videos already processed!")
        return

    processes, log_files = [], []
    for gid in range(num_gpus):
        cmd = [sys.executable, __file__]
        for name, val in vars(args).items():
            if name == "parallel":
                continue
            key = f"--{name}"
            if name == "num_gpus":
                cmd += [key, str(num_gpus)]
            elif name == "gpu_id":
                cmd += [key, str(gid)]
            elif name == "device":
                cmd += [key, f"cuda:{gid}"]
            elif name == "log_dir":
                cmd += [key, log_dir]
            elif isinstance(val, bool):
                if val:
                    cmd.append(key)
            elif val is not None:
                cmd += [key, str(val)]

        log_path = os.path.join(log_dir, f"qa_attack_gpu{gid}.log")
        lf = open(log_path, "w")
        log_files.append(lf)
        n_assigned = len(remaining[gid::num_gpus])
        print(f"  Worker {gid}: cuda:{gid} (~{n_assigned} videos) → {log_path}")
        processes.append(subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT))

    exit_codes = [p.wait() for p in processes]
    for lf in log_files:
        lf.close()
    print(f"\n=== All workers finished. Exit codes: {exit_codes} ===")

    # Merge CSVs
    merged, fields = [], None
    for gid in range(num_gpus):
        csv_file = os.path.join(log_dir, f"qa_attack_stats_gpu{gid}.csv")
        if os.path.exists(csv_file):
            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                if fields is None:
                    fields = reader.fieldnames
                merged.extend(list(reader))
    if merged and fields:
        merged_path = os.path.join(log_dir, "qa_attack_stats.csv")
        with open(merged_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(merged)
        print(f"Merged CSV ({len(merged)} rows) → {merged_path}")

    print(f"\nLogs saved to: {log_dir}/")


def worker_main(args):
    """Single-GPU worker: attack videos assigned to this gpu_id."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect videos
    videos = get_qa_scenes_videos(args.input_dir)
    print(f"Found {len(videos)} videos in {args.input_dir}")

    # Load questions and group by token
    available_tokens = {v["token"] for v in videos}
    token_to_qa = load_questions(args.questions, tokens=available_tokens)
    print(f"Loaded questions for {len(token_to_qa)} tokens "
          f"(total {sum(len(v) for v in token_to_qa.values())} QA pairs)")

    # Filter videos: must have at least one QA pair
    videos = [v for v in videos if v["token"] in token_to_qa]

    # Multi-GPU split: distribute UNFINISHED videos by round-robin
    if args.num_gpus > 1:
        remaining = [
            v for v in videos
            if not os.path.exists(
                os.path.join(args.output_dir, v["token"], f"{v['camera']}.mp4")
            )
        ]
        videos = remaining[args.gpu_id :: args.num_gpus]
        print(f"GPU {args.gpu_id}/{args.num_gpus}: {len(remaining)} unfinished → assigned {len(videos)}")

    if not videos:
        print("No videos to process!")
        return

    # Load model (full model, vision_only=False for QA attack)
    print(f"\nLoading model: {args.model} (full model for QA attack)")
    attacker = Qwen3VLPGD(
        model_path=args.model,
        device=args.device,
        dtype=torch.float32,
        vision_only=False,
    )

    results_summary = []

    for video_info in tqdm(videos, desc=f"QA-Attack [GPU {args.gpu_id}]"):
        token = video_info["token"]
        camera = video_info["camera"]
        video_path = video_info["path"]
        label = f"{token}/{camera}"

        out_dir = os.path.join(args.output_dir, token)
        os.makedirs(out_dir, exist_ok=True)
        output_video_path = os.path.join(out_dir, f"{camera}.mp4")

        # Skip if already processed
        if os.path.exists(output_video_path):
            print(f"Skipping {label} (exists)")
            continue

        # Pick the first QA pair for this token as the attack target
        qa_list = token_to_qa[token]
        qa = qa_list[0]
        question = qa["question"]
        answer = qa["answer"]

        try:
            result = attacker.attack_video_qa(
                video_path=video_path,
                question=question,
                answer=answer,
                eps=args.eps,
                alpha=args.alpha,
                num_iter=args.iter,
                fps=args.fps,
                verbose=False,
            )

            # Save adversarial video
            save_adversarial_video_direct(
                pixel_values_clean=result["pixel_values_clean"],
                pixel_values_adv=result["pixel_values_adv"],
                video_grid_thw=result["video_grid_thw"],
                eps=args.eps,
                original_video_path=video_path,
                output_path=output_video_path,
                model_std=attacker.image_std.cpu(),
            )

            if args.save_pt:
                torch.save(result, os.path.join(out_dir, f"{camera}.pt"))

            row = {
                "token_id": token,
                "camera": camera,
                "question": question,
                "gt_answer": answer,
                "initial_loss": result["initial_loss"],
                "final_loss": result["final_loss"],
                "loss_increase": result["final_loss"] - result["initial_loss"],
                "elapsed_time": result.get("elapsed_time", 0),
                "perturbation_l_inf": result.get("perturbation_l_inf", 0),
                "perturbation_l2": result.get("perturbation_l2", 0),
                "num_qa_pairs": len(qa_list),
            }
            results_summary.append(row)

            print(f"  {label}: CE {result['initial_loss']:.4f} → {result['final_loss']:.4f} "
                  f"(+{result['final_loss'] - result['initial_loss']:.4f}) | "
                  f"Q: {question[:50]}  A: {answer}")

        except Exception as e:
            print(f"Error {label}: {e}")
            import traceback; traceback.print_exc()
            continue

    # Print summary
    print(f"\n{'='*60}")
    print(f"QA Attack Summary")
    print(f"{'='*60}")
    print(f"Processed: {len(results_summary)} videos")
    if results_summary:
        avg_init = np.mean([r["initial_loss"] for r in results_summary])
        avg_final = np.mean([r["final_loss"] for r in results_summary])
        avg_increase = np.mean([r["loss_increase"] for r in results_summary])
        print(f"Avg initial CE loss: {avg_init:.4f}")
        print(f"Avg final CE loss:   {avg_final:.4f}")
        print(f"Avg loss increase:   {avg_increase:.4f}")
    print(f"Output: {args.output_dir}")

    # Save CSV
    if results_summary:
        if args.log_dir:
            csv_dir = args.log_dir
        else:
            csv_dir = os.path.join("logs", datetime.now().strftime("%m%d_%H%M%S"))
        os.makedirs(csv_dir, exist_ok=True)

        csv_name = (f"qa_attack_stats_gpu{args.gpu_id}.csv"
                     if args.num_gpus > 1 else "qa_attack_stats.csv")
        csv_path = os.path.join(csv_dir, csv_name)

        csv_fields = [
            "token_id", "camera", "question", "gt_answer",
            "initial_loss", "final_loss", "loss_increase",
            "elapsed_time", "perturbation_l_inf", "perturbation_l2",
            "num_qa_pairs",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(results_summary)
        print(f"CSV saved to: {csv_path}")


def main():
    args = parse_args()
    if args.parallel:
        parallel_main(args)
    else:
        worker_main(args)


if __name__ == "__main__":
    main()

