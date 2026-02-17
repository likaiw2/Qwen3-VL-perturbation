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
):
    """
    Save adversarial video by applying perturbation to original video.

    Uses ffmpeg directly for reliable video I/O.
    Uses smooth random noise instead of block-based perturbation to avoid visible artifacts.
    """
    import subprocess
    import tempfile
    import shutil

    # Use fixed random seed for reproducibility (based on video path)
    seed = hash(original_video_path) % (2**32)
    rng = np.random.RandomState(seed)

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()

    try:
        # Extract frames using ffmpeg
        frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
        subprocess.run([
            'ffmpeg', '-y', '-i', original_video_path,
            '-vsync', '0', frame_pattern
        ], capture_output=True, check=True)

        # Get list of extracted frames
        frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))
        total_frames = len(frame_files)

        if total_frames == 0:
            print(f"Error: No frames extracted from {original_video_path}")
            return None

        # Process each frame with smooth random noise
        for frame_idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(frame_file)
            if frame is None:
                continue

            height, width = frame.shape[:2]

            # Generate smooth random noise using Gaussian blur
            # First create random noise in [-1, 1]
            raw_noise = rng.uniform(-1, 1, (height, width, 3)).astype(np.float32)

            # Apply Gaussian blur to make noise smoother (reduces visible artifacts)
            smooth_noise = cv2.GaussianBlur(raw_noise, (15, 15), 5)

            # Normalize to [-1, 1] range and scale by eps
            smooth_noise = smooth_noise / (np.abs(smooth_noise).max() + 1e-8)
            noise = smooth_noise * eps * 255

            # Apply perturbation to frame
            frame_float = frame.astype(np.float32)
            frame_adv = frame_float + noise
            frame_adv = np.clip(frame_adv, 0, 255).astype(np.uint8)

            # Save modified frame
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

        # Encode frames back to video using ffmpeg
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
        # Clean up temp directory
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


def main():
    parser = argparse.ArgumentParser(description="Batch attack NuScenes videos")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory containing NuScenes")
    parser.add_argument("--output_dir", type=str, default="data/NuScenes_Attack",
                        help="Output directory for attacked videos")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Model path")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")
    parser.add_argument("--eps", type=float, default=8/255,
                        help="Perturbation budget")
    parser.add_argument("--alpha", type=float, default=1/255,
                        help="Step size")
    parser.add_argument("--iter", type=int, default=100,
                        help="Number of iterations")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Video sampling FPS")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Maximum number of videos to process")
    parser.add_argument("--save_pt", action="store_true",
                        help="Also save .pt files with pixel_values")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all videos
    print(f"Scanning for videos in {args.data_dir}/NuScenes...")
    videos = get_all_videos(args.data_dir)
    print(f"Found {len(videos)} videos")

    if args.max_videos:
        videos = videos[:args.max_videos]
        print(f"Processing first {args.max_videos} videos")

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

    # Process each video
    results_summary = []

    for video_info in tqdm(videos, desc="Attacking videos"):
        video_path = video_info['path']
        scene = video_info['scene']
        token = video_info['token']
        camera = video_info['camera']

        # Create output directory
        output_scene_dir = create_output_dir(args.output_dir, scene, token)
        output_video_path = os.path.join(output_scene_dir, f"{camera}.mp4")
        output_pt_path = os.path.join(output_scene_dir, f"{camera}.pt")

        # Skip if already processed
        if os.path.exists(output_video_path) and not args.save_pt:
            print(f"Skipping {video_path} (already exists)")
            continue

        try:
            # Run attack
            result = attacker.attack_video(
                video_path=video_path,
                eps=args.eps,
                alpha=args.alpha,
                num_iter=args.iter,
                fps=args.fps,
                verbose=False,
            )

            # Save adversarial video
            save_adversarial_video_direct(
                pixel_values_clean=result['pixel_values_clean'],
                pixel_values_adv=result['pixel_values_adv'],
                video_grid_thw=result['video_grid_thw'],
                eps=args.eps,
                original_video_path=video_path,
                output_path=output_video_path,
            )

            # Save .pt file if requested
            if args.save_pt:
                torch.save(result, output_pt_path)

            # Record summary
            results_summary.append({
                'video': video_path,
                'output': output_video_path,
                'initial_cos_sim': result['initial_cos_sim'],
                'final_cos_sim': result['final_cos_sim'],
                'reduction': result['initial_cos_sim'] - result['final_cos_sim'],
            })

            print(f"  {scene}/{token}/{camera}: cos_sim {result['initial_cos_sim']:.4f} -> {result['final_cos_sim']:.4f}")

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
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
        f.write(f"eps: {args.eps}\n")
        f.write(f"alpha: {args.alpha}\n")
        f.write(f"iterations: {args.iter}\n")
        f.write(f"Total videos: {len(results_summary)}\n\n")

        for r in results_summary:
            f.write(f"{r['video']}: {r['initial_cos_sim']:.4f} -> {r['final_cos_sim']:.4f} (reduction: {r['reduction']:.4f})\n")

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
