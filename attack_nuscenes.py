"""
Batch PGD Attack on NuScenes Videos

This script attacks all videos in the NuScenes dataset and saves
the adversarial pixel_values to data/NuScenes_Attack/

Usage:
    python attack_nuscenes.py --model Qwen/Qwen3-VL-4B-Instruct
"""

import argparse
import os
import glob
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from qwen_pgd_attack import Qwen3VLPGD


def get_all_videos(data_dir: str) -> list:
    """Get all video paths from NuScenes directory."""
    videos = []
    nuscenes_dir = os.path.join(data_dir, "NuScenes")

    if not os.path.exists(nuscenes_dir):
        print(f"Error: NuScenes directory not found: {nuscenes_dir}")
        return videos

    # Find all mp4 files
    for scene_dir in sorted(os.listdir(nuscenes_dir)):
        scene_path = os.path.join(nuscenes_dir, scene_dir)
        if not os.path.isdir(scene_path):
            continue

        for token_dir in os.listdir(scene_path):
            token_path = os.path.join(scene_path, token_dir)
            if not os.path.isdir(token_path):
                continue

            for video_file in os.listdir(token_path):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(token_path, video_file)
                    videos.append({
                        'path': video_path,
                        'scene': scene_dir,
                        'token': token_dir,
                        'camera': video_file.replace('.mp4', '')
                    })

    return videos


def get_qa_scenes_videos(input_dir: str) -> list:
    """Get all video paths from a flat QA_Scenes directory (token/CAM_FRONT.mp4)."""
    videos = []
    if not os.path.exists(input_dir):
        print(f"Error: directory not found: {input_dir}")
        return videos

    for token_dir in sorted(os.listdir(input_dir)):
        token_path = os.path.join(input_dir, token_dir)
        if not os.path.isdir(token_path):
            continue
        for video_file in os.listdir(token_path):
            if video_file.endswith('.mp4'):
                videos.append({
                    'path': os.path.join(token_path, video_file),
                    'token': token_dir,
                    'camera': video_file.replace('.mp4', ''),
                })

    return videos


def create_output_dir(output_base: str, scene: str, token: str) -> str:
    """Create output directory structure matching input."""
    output_dir = os.path.join(output_base, scene, token)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_adversarial_video_direct(
    pixel_values_clean: torch.Tensor,
    pixel_values_adv: torch.Tensor,
    video_grid_thw: torch.Tensor,
    original_video_path: str,
    output_path: str,
    eps: float = 8/255,
    model_std: torch.Tensor = None,
):
    import subprocess
    import tempfile
    import shutil

    # pixel_values shape: [N_patches, D] where D = 3 * temporal_patch_size * 14 * 14
    # Qwen3-VL Conv3d patch embed: channel dim (3) is first in the D dimension.
    perturbation = pixel_values_adv - pixel_values_clean  # [N, D], normalized space
    N, D = perturbation.shape

    # Denormalize: reshape to [N, 3, D/3] so std broadcasts over channel dim only.
    # model_std should come from the same attacker instance to stay in sync (Fix 4).
    if model_std is None:
        model_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    model_std = model_std.to(device=perturbation.device, dtype=perturbation.dtype).view(3)
    p = perturbation.view(N, 3, -1) * model_std.view(1, 3, 1)  # [N, 3, D/3], in [0,1] scale

    # Mean over spatial+temporal dims → per-patch per-channel perturbation [N, 3]
    patch_perturbations = p.mean(dim=2).cpu().numpy()  # [N, 3] RGB

    t, grid_h, grid_w = video_grid_thw[0].tolist()
    t, grid_h, grid_w = int(t), int(grid_h), int(grid_w)

    temp_dir = tempfile.mkdtemp()

    try:
        frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
        subprocess.run(['ffmpeg', '-y', '-i', original_video_path, '-vsync', '0', frame_pattern],
                       capture_output=True, check=True)

        frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))
        total_frames = len(frame_files)
        if total_frames == 0:
            return None

        temporal_patch_size = 2
        total_temporal_units = t * temporal_patch_size
        frames_per_unit = max(1, total_frames // total_temporal_units) if total_temporal_units > 0 else total_frames

        for frame_idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(frame_file)  # BGR
            if frame is None:
                continue

            height, width = frame.shape[:2]
            temporal_idx = min(frame_idx // frames_per_unit, total_temporal_units - 1)
            t_idx = temporal_idx // temporal_patch_size

            # Build per-pixel RGB noise from continuous per-patch perturbations
            noise_blocks = np.zeros((height, width, 3), dtype=np.float32)

            for h_idx in range(grid_h):
                for w_idx in range(grid_w):
                    patch_idx = t_idx * grid_h * grid_w + h_idx * grid_w + w_idx
                    p_val_rgb = patch_perturbations[patch_idx] if patch_idx < len(patch_perturbations) else np.zeros(3)
                    p_val_rgb = np.clip(p_val_rgb, -eps, eps)

                    y_start = int(h_idx * height / grid_h)
                    y_end   = int((h_idx + 1) * height / grid_h)
                    x_start = int(w_idx * width / grid_w)
                    x_end   = int((w_idx + 1) * width / grid_w)

                    # RGB → BGR for OpenCV, scale to pixel range [0,255]
                    noise_blocks[y_start:y_end, x_start:x_end] = p_val_rgb[::-1] * 255.0

            kernel_size = max(31, min(height, width) // max(grid_h, grid_w)) | 1
            noise_smooth = cv2.GaussianBlur(noise_blocks, (kernel_size, kernel_size), kernel_size / 3)

            frame_adv = np.clip(frame.astype(np.float32) + noise_smooth, 0, 255).astype(np.uint8)
            cv2.imwrite(frame_file, frame_adv)

        # Get original video FPS
        probe_result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'csv=p=0', original_video_path
        ], capture_output=True, text=True)
        fps_str = probe_result.stdout.strip()
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str) if fps_str else 30.0

        subprocess.run([
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'fast', output_path
        ], capture_output=True, check=True)

        return output_path

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return None
    except Exception as e:
        print(f"Error processing video: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def save_video_with_uniform_perturbation(
    original_video_path: str,
    output_path: str,
    eps: float = 8/255,
):
    """
    Fallback: Save video with uniform random perturbation using ffmpeg.
    """
    import subprocess
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    rng = np.random.RandomState(42)

    try:
        # Extract frames
        frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
        subprocess.run([
            'ffmpeg', '-y', '-i', original_video_path,
            '-vsync', '0', frame_pattern
        ], capture_output=True, check=True)

        frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))

        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is None:
                continue

            frame_float = frame.astype(np.float32)
            noise = rng.uniform(-eps * 255, eps * 255, frame_float.shape).astype(np.float32)
            frame_adv = np.clip(frame_float + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(frame_file, frame_adv)

        # Get FPS
        probe_result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'csv=p=0', original_video_path
        ], capture_output=True, text=True)

        fps_str = probe_result.stdout.strip()
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str) if fps_str else 30.0

        subprocess.run([
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'fast', output_path
        ], capture_output=True, check=True)

        return output_path

    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def save_pixel_level_adversarial_video(
    result: dict,
    original_video_path: str,
    output_path: str,
):
    """
    Save adversarial video from pixel-level attack results.

    This function takes the perturbed frames from pixel-level attack
    and reconstructs the full video by interpolating perturbations
    to non-sampled frames.

    Args:
        result: Dictionary from attack_video_pixel_level()
        original_video_path: Path to original video
        output_path: Path to save adversarial video
    """
    import subprocess
    import tempfile
    import shutil

    adv_frames = result['adv_frames']  # [T_sampled, H, W, C] RGB uint8
    clean_frames = result['clean_frames']
    sampled_indices = result['sampled_indices']
    total_frames = result['total_frames']
    original_fps = result.get('original_fps', 30.0)

    # Compute per-frame perturbation
    perturbation = adv_frames.astype(np.float32) - clean_frames.astype(np.float32)

    temp_dir = tempfile.mkdtemp()

    try:
        # Extract all original frames
        frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
        subprocess.run([
            'ffmpeg', '-y', '-i', original_video_path,
            '-vsync', '0', frame_pattern
        ], capture_output=True, check=True)

        frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))

        # Create a mapping from sampled indices to perturbations
        # For non-sampled frames, interpolate from nearest sampled frames
        for frame_idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(frame_file)
            if frame is None:
                continue

            # Find the closest sampled frame(s)
            if frame_idx in sampled_indices:
                # Exact match - use the perturbation directly
                sample_idx = sampled_indices.index(frame_idx)
                noise = perturbation[sample_idx]
            else:
                # Interpolate between nearest sampled frames
                # Find surrounding sampled indices
                prev_idx = None
                next_idx = None
                for i, si in enumerate(sampled_indices):
                    if si <= frame_idx:
                        prev_idx = i
                    if si >= frame_idx and next_idx is None:
                        next_idx = i
                        break

                if prev_idx is None:
                    prev_idx = 0
                if next_idx is None:
                    next_idx = len(sampled_indices) - 1

                if prev_idx == next_idx:
                    noise = perturbation[prev_idx]
                else:
                    # Linear interpolation
                    prev_frame_idx = sampled_indices[prev_idx]
                    next_frame_idx = sampled_indices[next_idx]
                    t = (frame_idx - prev_frame_idx) / max(1, next_frame_idx - prev_frame_idx)
                    noise = (1 - t) * perturbation[prev_idx] + t * perturbation[next_idx]

            # Resize noise if needed (frame sizes might differ)
            if noise.shape[:2] != frame.shape[:2]:
                noise = cv2.resize(noise, (frame.shape[1], frame.shape[0]))

            # Apply perturbation (convert RGB noise to BGR for cv2)
            noise_bgr = noise[:, :, ::-1]  # RGB to BGR
            frame_float = frame.astype(np.float32)
            frame_adv = np.clip(frame_float + noise_bgr, 0, 255).astype(np.uint8)

            cv2.imwrite(frame_file, frame_adv)

        # Encode back to video
        subprocess.run([
            'ffmpeg', '-y', '-framerate', str(original_fps),
            '-i', frame_pattern,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'fast', output_path
        ], capture_output=True, check=True)

        return output_path

    except Exception as e:
        print(f"Error saving pixel-level video: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_attack(attacker, video_path, args):
    """Run attack on a single video and return result dict."""
    if args.attack_type == "pixel":
        result = attacker.attack_video_pixel_level(
            video_path=video_path,
            eps=args.eps,
            alpha=args.alpha,
            num_iter=args.iter,
            fps=args.fps,
            verbose=False,
        )
    else:
        result = attacker.attack_video(
            video_path=video_path,
            eps=args.eps,
            alpha=args.alpha,
            num_iter=args.iter,
            fps=args.fps,
            verbose=False,
        )
    return result


def save_result(result, args, video_path, output_video_path, attacker):
    """Save adversarial video to output_video_path."""
    if args.attack_type == "pixel":
        save_pixel_level_adversarial_video(
            result=result,
            original_video_path=video_path,
            output_path=output_video_path,
        )
    else:
        save_adversarial_video_direct(
            pixel_values_clean=result['pixel_values_clean'],
            pixel_values_adv=result['pixel_values_adv'],
            video_grid_thw=result['video_grid_thw'],
            eps=args.eps,
            original_video_path=video_path,
            output_path=output_video_path,
            model_std=attacker.image_std.cpu(),
        )


def main():
    parser = argparse.ArgumentParser(description="Batch attack NuScenes videos")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory containing NuScenes subdirectory (original mode)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Flat input directory with structure <token>/<camera>.mp4 "
                             "(e.g. data/QA_Scenes_500). Takes precedence over --data_dir.")
    parser.add_argument("--output_dir", type=str, default="data/NuScenes_Attack_Patch",
                        help="Output directory for attacked videos")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model path")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Device to run on")
    parser.add_argument("--eps", type=float, default=8/255,
                        help="Perturbation budget")
    parser.add_argument("--alpha", type=float, default=1/255,
                        help="Step size")
    parser.add_argument("--iter", type=int, default=100,
                        help="Number of iterations")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Video sampling FPS")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start processing from this video index (legacy; prefer --gpu_id/--num_gpus)")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Maximum number of videos to process (applied after --start_index)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Total number of parallel GPU workers")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Index of this GPU worker (0-based). Combined with --num_gpus, "
                             "this worker takes every num_gpus-th UNFINISHED video starting at gpu_id.")
    parser.add_argument("--save_pt", action="store_true",
                        help="Also save .pt files with pixel_values")
    parser.add_argument("--attack_type", type=str, default="patch",
                        choices=["patch", "pixel"],
                        help="Attack type: 'patch' (faster) or 'pixel' (smoother)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect videos
    if args.input_dir is not None:
        print(f"Scanning for videos in {args.input_dir} (flat mode)...")
        videos = get_qa_scenes_videos(args.input_dir)
        flat_mode = True
    else:
        print(f"Scanning for videos in {args.data_dir}/NuScenes...")
        videos = get_all_videos(args.data_dir)
        flat_mode = False
    print(f"Found {len(videos)} videos total")

    # Legacy index slicing
    if args.start_index > 0:
        videos = videos[args.start_index:]
        print(f"Starting from index {args.start_index}, {len(videos)} videos remaining")

    if args.max_videos:
        videos = videos[:args.max_videos]
        print(f"Capped at {args.max_videos} videos")

    # Multi-GPU split: filter to unfinished videos first, then distribute by round-robin
    if args.num_gpus > 1:
        def output_path_for(video_info):
            if flat_mode:
                return os.path.join(args.output_dir, video_info['token'],
                                    f"{video_info['camera']}.mp4")
            else:
                return os.path.join(args.output_dir, video_info['scene'],
                                    video_info['token'], f"{video_info['camera']}.mp4")

        remaining = [v for v in videos if not os.path.exists(output_path_for(v))]
        videos = remaining[args.gpu_id::args.num_gpus]
        print(f"GPU {args.gpu_id}/{args.num_gpus}: {len(remaining)} unfinished → assigned {len(videos)}")

    if len(videos) == 0:
        print("No videos found!")
        return

    # Initialize attacker
    print(f"\nLoading model: {args.model}")
    attacker = Qwen3VLPGD(
        model_path=args.model,
        device=args.device,
        dtype=torch.float32,
    )

    results_summary = []

    for video_info in tqdm(videos, desc="Attacking videos"):
        video_path = video_info['path']
        token = video_info['token']
        camera = video_info['camera']

        # Determine output path
        if flat_mode:
            out_dir = os.path.join(args.output_dir, token)
            os.makedirs(out_dir, exist_ok=True)
            output_video_path = os.path.join(out_dir, f"{camera}.mp4")
            output_pt_path = os.path.join(out_dir, f"{camera}.pt")
            label = f"{token}/{camera}"
        else:
            scene = video_info['scene']
            output_scene_dir = create_output_dir(args.output_dir, scene, token)
            output_video_path = os.path.join(output_scene_dir, f"{camera}.mp4")
            output_pt_path = os.path.join(output_scene_dir, f"{camera}.pt")
            label = f"{scene}/{token}/{camera}"

        # Skip if already processed
        if os.path.exists(output_video_path) and not args.save_pt:
            print(f"Skipping {label} (already exists)")
            continue

        try:
            result = run_attack(attacker, video_path, args)
            save_result(result, args, video_path, output_video_path, attacker)

            if args.save_pt:
                torch.save(result, output_pt_path)

            results_summary.append({
                'video': video_path,
                'output': output_video_path,
                'initial_cos_sim': result['initial_cos_sim'],
                'final_cos_sim': result['final_cos_sim'],
                'reduction': result['initial_cos_sim'] - result['final_cos_sim'],
            })

            print(f"  {label}: cos_sim {result['initial_cos_sim']:.4f} -> {result['final_cos_sim']:.4f}")

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            import traceback; traceback.print_exc()
            continue

    # Print summary
    print(f"\n{'='*60}")
    print(f"Attack Summary")
    print(f"{'='*60}")
    print(f"Total videos processed: {len(results_summary)}")

    if results_summary:
        avg_initial = np.mean([r['initial_cos_sim'] for r in results_summary])
        avg_final = np.mean([r['final_cos_sim'] for r in results_summary])
        avg_reduction = np.mean([r['reduction'] for r in results_summary])

        print(f"Average initial cos_sim: {avg_initial:.4f}")
        print(f"Average final cos_sim: {avg_final:.4f}")
        print(f"Average reduction: {avg_reduction:.4f}")

    print(f"\nOutput saved to: {args.output_dir}")

    # Save summary to file
    summary_path = os.path.join(args.output_dir, "attack_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Attack Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Attack type: {args.attack_type}\n")
        f.write(f"eps: {args.eps}\n")
        f.write(f"alpha: {args.alpha}\n")
        f.write(f"iterations: {args.iter}\n")
        f.write(f"Total videos: {len(results_summary)}\n\n")

        for r in results_summary:
            f.write(f"{r['video']}: {r['initial_cos_sim']:.4f} -> {r['final_cos_sim']:.4f} (reduction: {r['reduction']:.4f})\n")

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
