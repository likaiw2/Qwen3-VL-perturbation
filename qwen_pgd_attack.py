"""
PGD White-Box Adversarial Attack on Qwen3-VL (Feature-Level)

This script implements a Projected Gradient Descent (PGD) attack on the Qwen3-VL
vision-language model. The attack adds perturbations to input images/videos to maximize
the cosine similarity loss (make visual features dissimilar from the original).

Usage:
    # Image attack
    python qwen_pgd_attack.py --image test.jpg --output adv_result.pt

    # Video attack
    python qwen_pgd_attack.py --video test.mp4 --output adv_result.pt
"""

import argparse
import os
import time
from typing import Dict, Optional, Tuple, List, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Try to import video processing utilities
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    print("Warning: qwen_vl_utils not available. Video processing may be limited.")


class Qwen3VLPGD:
    """
    PGD adversarial attack on Qwen3-VL vision encoder.

    The attack perturbs input images to minimize cosine similarity
    between clean and adversarial visual features at the token level.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the attack.

        Args:
            model_path: HuggingFace model path or local path
            device: Device to run on (e.g., "cuda:0", "cpu")
            dtype: Data type for computations (float32 recommended for stable gradients)
        """
        self.device = device
        self.dtype = dtype
        self.model_path = model_path

        # Load model and processor
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # Freeze model parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # --- FIX 2: 动态获取归一化参数，对齐 CLIP ---
        # 重塑为 [3, 1, 1] 支持后续的基础广播
        if hasattr(self.processor, "image_processor"):
            mean = self.processor.image_processor.image_mean
            std = self.processor.image_processor.image_std
            self.image_mean = torch.tensor(mean, device=device, dtype=dtype).view(-1, 1, 1)
            self.image_std = torch.tensor(std, device=device, dtype=dtype).view(-1, 1, 1)
        else:
            # 回退到标准 CLIP 参数
            self.image_mean = torch.tensor([0.4814, 0.4578, 0.4082], device=device, dtype=dtype).view(-1, 1, 1)
            self.image_std = torch.tensor([0.2686, 0.2613, 0.2757], device=device, dtype=dtype).view(-1, 1, 1)

        print(f"Model loaded successfully. Device: {device}, dtype: {dtype}")

    def get_visual_features(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        is_video: bool = False,
    ) -> torch.Tensor:
        """
        Extract visual features from the vision encoder.
        """
        if is_video:
            video_embeds, _ = self.model.model.get_video_features(
                pixel_values.to(self.dtype),
                grid_thw
            )
            features = torch.cat(video_embeds, dim=0)
        else:
            image_embeds, _ = self.model.model.get_image_features(
                pixel_values.to(self.dtype),
                grid_thw
            )
            features = torch.cat(image_embeds, dim=0)
        return features

    def cosine_similarity_loss(
        self,
        feat_adv: torch.Tensor,
        feat_clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between adversarial and clean features.

        --- FIX 1: Token-Level 级别的余弦相似度 ---
        在特征维度 (dim=-1) 上计算每个 Token 的相似度，再求均值，防止模型作弊。
        """
        cos_sim = F.cosine_similarity(feat_adv, feat_clean, dim=-1)
        return cos_sim.mean()

    def preprocess_image(
        self,
        image: Image.Image,
        text: str = "Describe this image.",
    ) -> Dict[str, torch.Tensor]:
        """Preprocess an image for the model."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text}
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def preprocess_video(
        self,
        video_path: str,
        text: str = "Describe this video.",
        max_frames: int = 32,
        fps: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Preprocess a video for the model."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": fps},
                    {"type": "text", "text": text}
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if QWEN_VL_UTILS_AVAILABLE:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
        else:
            inputs = self.processor(text=[prompt], videos=[video_path], return_tensors="pt")
            
        return {k: v.to(self.device) for k, v in inputs.items()}

    def attack_video_pixel_level(
        self,
        video_path: str,
        eps: float = 8/255,
        alpha: float = 1/255,
        num_iter: int = 100,
        text: str = "Describe this video.",
        fps: float = 1.0,
        verbose: bool = True,
    ) -> Dict:
        """Execute pixel-level PGD attack on a video."""
        start_time = time.time()

        inputs = self.preprocess_video(video_path, text, fps=fps)
        video_grid_thw = inputs['video_grid_thw']

        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        total_frames = len(frames)
        if total_frames == 0:
            raise ValueError(f"No frames loaded from {video_path}")

        sample_interval = max(1, int(original_fps / fps))
        sampled_indices = list(range(0, total_frames, sample_interval))
        sampled_frames = [frames[i] for i in sampled_indices]

        t, grid_h, grid_w = video_grid_thw[0].tolist()
        t, grid_h, grid_w = int(t), int(grid_h), int(grid_w)

        num_model_frames = t * 2
        if len(sampled_frames) > num_model_frames:
            sampled_frames = sampled_frames[:num_model_frames]
        elif len(sampled_frames) < num_model_frames:
            while len(sampled_frames) < num_model_frames:
                sampled_frames.append(sampled_frames[-1])

        frames_tensor = torch.stack([
            torch.from_numpy(f).float() / 255.0 for f in sampled_frames
        ]).to(self.device).permute(0, 3, 1, 2)

        frames_clean = frames_tensor.clone()
        patch_size = 14
        target_h, target_w = grid_h * patch_size, grid_w * patch_size

        frames_adv = frames_tensor.clone().detach()
        best_adv = frames_adv.clone()
        best_cos_sim = 1.0
        cos_sim_history = []

        vision_model = self.model.model.visual
        patch_embed = vision_model.patch_embed
        
        # Make mean/std 4D to broadcast with [T, C, H, W]
        mean_4d = self.image_mean.view(1, 3, 1, 1)
        std_4d = self.image_std.view(1, 3, 1, 1)

        # Fix 2: construct cu_seqlens for variable-length attention
        # video_grid_thw[:,0]=t, [:,1]=h, [:,2]=w (all in patch units)
        # Each temporal patch group has h*w tokens; there are t groups per video.
        seq_lens = video_grid_thw[:, 1] * video_grid_thw[:, 2]
        cu_seqlens = F.pad(
            torch.repeat_interleave(seq_lens, video_grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32),
            (1, 0), value=0
        )

        # Pre-compute rotary_pos_emb once (same for every iteration)
        rotary_pos_emb = vision_model.rot_pos_emb(video_grid_thw)

        # Fix 1: compute features_clean via the SAME manual path used for features_adv
        # This ensures initial cos_sim == 1.0 and the attack target is consistent.
        with torch.no_grad():
            frames_5d_c = ((F.interpolate(frames_clean, size=(target_h, target_w),
                                          mode='bilinear', align_corners=False)
                            - mean_4d) / std_4d).permute(1, 0, 2, 3).unsqueeze(0)
            h_c = patch_embed(frames_5d_c)
            if h_c.dim() == 5:
                h_c = h_c.view(1, -1, h_c.shape[-1]).squeeze(0)
            elif h_c.dim() == 3:
                h_c = h_c.squeeze(0)
            for blk in vision_model.blocks:
                h_c = blk(h_c, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
            features_clean = vision_model.merger(h_c)

        for i in range(num_iter):
            frames_adv = frames_adv.detach().requires_grad_(True)

            frames_resized = F.interpolate(
                frames_adv, size=(target_h, target_w), mode='bilinear', align_corners=False
            )

            # Apply accurate normalization
            frames_norm = (frames_resized - mean_4d) / std_4d

            T = frames_norm.shape[0]
            frames_5d = frames_norm.permute(1, 0, 2, 3).unsqueeze(0)

            hidden_states = patch_embed(frames_5d)
            if hidden_states.dim() == 5:
                B, Tp, Hp, Wp, D = hidden_states.shape
                hidden_states = hidden_states.view(B, -1, D).squeeze(0)
            elif hidden_states.dim() == 3:
                hidden_states = hidden_states.squeeze(0)

            for blk in vision_model.blocks:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
            features_adv = vision_model.merger(hidden_states)

            loss = self.cosine_similarity_loss(features_adv, features_clean)
            loss.backward()

            grad = frames_adv.grad.detach()

            # Fix 3: track best BEFORE update so best_adv corresponds to best_cos_sim
            current_cos_sim = loss.item()
            cos_sim_history.append(current_cos_sim)

            if current_cos_sim < best_cos_sim:
                best_cos_sim = current_cos_sim
                best_adv = frames_adv.detach().clone()

            if verbose and (i % 10 == 0 or i == num_iter - 1):
                print(f"  Iter {i:3d}: cos_sim = {current_cos_sim:.6f}")

            with torch.no_grad():
                frames_adv = frames_adv - alpha * grad.sign()
                perturbation = frames_adv - frames_clean
                perturbation = torch.clamp(perturbation, -eps, eps)
                frames_adv = frames_clean + perturbation
                frames_adv = torch.clamp(frames_adv, 0.0, 1.0)

        adv_frames_np = (best_adv.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        clean_frames_np = (frames_clean.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        perturbation = (best_adv - frames_clean).detach()
        elapsed_time = time.time() - start_time

        return {
            'adv_frames': adv_frames_np,
            'clean_frames': clean_frames_np,
            'sampled_indices': sampled_indices,
            'total_frames': total_frames,
            'original_fps': original_fps,
            'video_grid_thw': video_grid_thw.cpu(),
            'initial_cos_sim': 1.0,
            'final_cos_sim': best_cos_sim,
            'cos_sim_history': cos_sim_history,
            'eps': eps,
            'alpha': alpha,
            'num_iter': num_iter,
            'elapsed_time': elapsed_time,
            'video_path': video_path,
            'perturbation_l_inf': perturbation.abs().max().item(),
            'perturbation_l2': perturbation.norm(2).item(),
            'is_video': True,
            'attack_type': 'pixel_level',
        }

    def attack_video(
        self,
        video_path: str,
        eps: float = 8/255,
        alpha: float = 1/255,
        num_iter: int = 100,
        text: str = "Describe this video.",
        fps: float = 1.0,
        verbose: bool = True,
    ) -> Dict:
        """Execute PGD patch-level attack on a video."""
        start_time = time.time()
        inputs = self.preprocess_video(video_path, text, fps=fps)

        pixel_values_clean = inputs['pixel_values_videos'].to(self.dtype)
        video_grid_thw = inputs['video_grid_thw']

        with torch.no_grad():
            features_clean = self.get_visual_features(pixel_values_clean, video_grid_thw, is_video=True)

        pixel_values_adv = pixel_values_clean.clone().detach()
        best_adv = pixel_values_adv.clone()
        best_cos_sim = 1.0
        cos_sim_history = []

        if verbose: print(f"\nStarting PGD attack on video...")

        for i in range(num_iter):
            pixel_values_adv = pixel_values_adv.detach().requires_grad_(True)
            features_adv = self.get_visual_features(pixel_values_adv, video_grid_thw, is_video=True)
            loss = self.cosine_similarity_loss(features_adv, features_clean)
            loss.backward()
            grad = pixel_values_adv.grad.detach()

            # Fix 3: track best BEFORE update so best_adv corresponds to best_cos_sim
            current_cos_sim = loss.item()
            cos_sim_history.append(current_cos_sim)

            if current_cos_sim < best_cos_sim:
                best_cos_sim = current_cos_sim
                best_adv = pixel_values_adv.detach().clone()

            if verbose and (i % 10 == 0 or i == num_iter - 1):
                print(f"  Iter {i:3d}: cos_sim = {current_cos_sim:.6f}")

            with torch.no_grad():
                # 严谨的物理空间投影
                N, D = pixel_values_adv.shape
                pv_adv_view = pixel_values_adv.view(N, 3, -1)
                grad_view = grad.view(N, 3, -1)
                pv_clean_view = pixel_values_clean.view(N, 3, -1)
                std_view = self.image_std.view(1, 3, 1)
                mean_view = self.image_mean.view(1, 3, 1)

                pv_adv_view = pv_adv_view - (alpha / std_view) * grad_view.sign()
                clean_01 = pv_clean_view * std_view + mean_view
                adv_01 = pv_adv_view * std_view + mean_view
                perturbation = adv_01 - clean_01
                perturbation = torch.clamp(perturbation, -eps, eps)
                adv_01_projected = torch.clamp(clean_01 + perturbation, 0.0, 1.0)
                pixel_values_adv = ((adv_01_projected - mean_view) / std_view).view(N, D)

        with torch.no_grad():
            N, D = pixel_values_clean.shape
            std_view = self.image_std.view(1, 3, 1)
            mean_view = self.image_mean.view(1, 3, 1)
            clean_01_final = pixel_values_clean.view(N, 3, -1) * std_view + mean_view
            best_adv_01 = best_adv.view(N, 3, -1) * std_view + mean_view
            perturbation_final = (best_adv_01 - clean_01_final).view(N, D)

        elapsed_time = time.time() - start_time

        return {
            'pixel_values_adv': best_adv.detach().cpu(),
            'pixel_values_clean': pixel_values_clean.detach().cpu(),
            'perturbation': perturbation_final.detach().cpu(),
            'video_grid_thw': video_grid_thw.cpu(),
            'initial_cos_sim': 1.0,
            'final_cos_sim': best_cos_sim,
            'cos_sim_history': cos_sim_history,
            'eps': eps,
            'alpha': alpha,
            'num_iter': num_iter,
            'elapsed_time': elapsed_time,
            'video_path': video_path,
            'perturbation_l_inf': perturbation_final.abs().max().item(),
            'perturbation_l2': perturbation_final.norm(2).item(),
            'is_video': True,
        }

    def attack(
        self,
        image_path: str,
        eps: float = 8/255,
        alpha: float = 1/255,
        num_iter: int = 100,
        text: str = "Describe this image.",
        verbose: bool = True,
    ) -> Dict:
        """Execute PGD patch-level attack on an image."""
        start_time = time.time()
        image_pil = Image.open(image_path).convert('RGB')
        inputs = self.preprocess_image(image_pil, text)

        pixel_values_clean = inputs['pixel_values'].to(self.dtype)
        image_grid_thw = inputs['image_grid_thw']

        with torch.no_grad():
            features_clean = self.get_visual_features(pixel_values_clean, image_grid_thw, is_video=False)

        pixel_values_adv = pixel_values_clean.clone().detach()
        best_adv = pixel_values_adv.clone()
        best_cos_sim = 1.0
        cos_sim_history = []

        if verbose: print(f"\nStarting PGD attack on image...")

        for i in range(num_iter):
            pixel_values_adv = pixel_values_adv.detach().requires_grad_(True)
            features_adv = self.get_visual_features(pixel_values_adv, image_grid_thw, is_video=False)
            loss = self.cosine_similarity_loss(features_adv, features_clean)
            loss.backward()
            grad = pixel_values_adv.grad.detach()

            # Fix 3: track best BEFORE update so best_adv corresponds to best_cos_sim
            current_cos_sim = loss.item()
            cos_sim_history.append(current_cos_sim)

            if current_cos_sim < best_cos_sim:
                best_cos_sim = current_cos_sim
                best_adv = pixel_values_adv.detach().clone()

            if verbose and (i % 10 == 0 or i == num_iter - 1):
                print(f"  Iter {i:3d}: cos_sim = {current_cos_sim:.6f}")

            with torch.no_grad():
                # 严谨的物理空间投影
                N, D = pixel_values_adv.shape
                pv_adv_view = pixel_values_adv.view(N, 3, -1)
                grad_view = grad.view(N, 3, -1)
                pv_clean_view = pixel_values_clean.view(N, 3, -1)
                std_view = self.image_std.view(1, 3, 1)
                mean_view = self.image_mean.view(1, 3, 1)

                pv_adv_view = pv_adv_view - (alpha / std_view) * grad_view.sign()
                clean_01 = pv_clean_view * std_view + mean_view
                adv_01 = pv_adv_view * std_view + mean_view
                perturbation = adv_01 - clean_01
                perturbation = torch.clamp(perturbation, -eps, eps)
                adv_01_projected = torch.clamp(clean_01 + perturbation, 0.0, 1.0)
                pixel_values_adv = ((adv_01_projected - mean_view) / std_view).view(N, D)

        with torch.no_grad():
            N, D = pixel_values_clean.shape
            std_view = self.image_std.view(1, 3, 1)
            mean_view = self.image_mean.view(1, 3, 1)
            
            clean_01_final = pixel_values_clean.view(N, 3, -1) * std_view + mean_view
            best_adv_01 = best_adv.view(N, 3, -1) * std_view + mean_view
            perturbation_final = (best_adv_01 - clean_01_final).view(N, D)

        elapsed_time = time.time() - start_time

        return {
            'pixel_values_adv': best_adv.detach().cpu(),
            'pixel_values_clean': pixel_values_clean.detach().cpu(),
            'perturbation': perturbation_final.detach().cpu(),
            'image_grid_thw': image_grid_thw.cpu(),
            'initial_cos_sim': 1.0,
            'final_cos_sim': best_cos_sim,
            'cos_sim_history': cos_sim_history,
            'eps': eps,
            'alpha': alpha,
            'num_iter': num_iter,
            'elapsed_time': elapsed_time,
            'image_path': image_path,
            'perturbation_l_inf': perturbation_final.abs().max().item(),
            'perturbation_l2': perturbation_final.norm(2).item(),
        }

    def save_result(self, result: Dict, save_path: str) -> None:
        torch.save(result, save_path)
        print(f"Result saved to {save_path}")

    def load_result(self, load_path: str) -> Dict:
        return torch.load(load_path)


def parse_args():
    parser = argparse.ArgumentParser(description="PGD attack on Qwen3-VL vision encoder")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to input image")
    input_group.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default="adv_result.pt", help="Path to save attack result")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Model path")
    parser.add_argument("--eps", type=float, default=8/255, help="Maximum perturbation")
    parser.add_argument("--alpha", type=float, default=1/255, help="Step size")
    parser.add_argument("--iter", type=int, default=100, help="Number of PGD iterations")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run on")
    parser.add_argument("--text", type=str, default=None, help="Text prompt")
    parser.add_argument("--fps", type=float, default=1.0, help="Video FPS sampling")
    return parser.parse_args()


def main():
    args = parse_args()
    is_video = args.video is not None
    input_path = args.video if is_video else args.image

    if not os.path.exists(input_path):
        print(f"Error: Input not found: {input_path}")
        return

    if args.text is None:
        args.text = "Describe this video." if is_video else "Describe this image."

    attacker = Qwen3VLPGD(model_path=args.model, device=args.device, dtype=torch.float32)

    if is_video:
        result = attacker.attack_video(
            video_path=input_path, eps=args.eps, alpha=args.alpha,
            num_iter=args.iter, text=args.text, fps=args.fps, verbose=True,
        )
    else:
        result = attacker.attack(
            image_path=input_path, eps=args.eps, alpha=args.alpha,
            num_iter=args.iter, text=args.text, verbose=True,
        )

    attacker.save_result(result, args.output)


if __name__ == "__main__":
    main()