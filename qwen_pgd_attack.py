"""
PGD White-Box Adversarial Attack on Qwen3-VL

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
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
import time

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
    between clean and adversarial visual features.
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

        # Image normalization parameters (from Qwen3VL image processor)
        self.image_mean = 0.5
        self.image_std = 0.5

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

        print(f"Model loaded successfully. Device: {device}, dtype: {dtype}")

    def get_visual_features(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        is_video: bool = False,
    ) -> torch.Tensor:
        """
        Extract visual features from the vision encoder.

        Args:
            pixel_values: Preprocessed pixel values, shape (seq_len, patch_dim)
            grid_thw: Grid information, shape (num_images/videos, 3)
            is_video: Whether the input is video

        Returns:
            Visual features tensor, shape (num_tokens, embed_dim)
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

        We return the similarity value; the attack will minimize this.

        Args:
            feat_adv: Adversarial features
            feat_clean: Clean features

        Returns:
            Cosine similarity (scalar)
        """
        feat_adv_flat = feat_adv.flatten()
        feat_clean_flat = feat_clean.flatten()

        cos_sim = F.cosine_similarity(
            feat_adv_flat.unsqueeze(0),
            feat_clean_flat.unsqueeze(0)
        )
        return cos_sim.mean()

    def preprocess_image(
        self,
        image: Image.Image,
        text: str = "Describe this image.",
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess an image for the model.

        Args:
            image: PIL Image
            text: Text prompt

        Returns:
            Dictionary containing model inputs
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt"
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def preprocess_video(
        self,
        video_path: str,
        text: str = "Describe this video.",
        max_frames: int = 32,
        fps: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a video for the model.

        Args:
            video_path: Path to video file
            text: Text prompt
            max_frames: Maximum number of frames to sample
            fps: Frames per second to sample

        Returns:
            Dictionary containing model inputs
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": fps,
                    },
                    {"type": "text", "text": text}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Use qwen_vl_utils for video processing if available
        if QWEN_VL_UTILS_AVAILABLE:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[prompt],
                videos=[video_path],
                return_tensors="pt"
            )

        return {k: v.to(self.device) for k, v in inputs.items()}

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
        """
        Execute PGD attack on a video.

        Args:
            video_path: Path to input video
            eps: Maximum perturbation in [0, 1] space
            alpha: Step size in [0, 1] space
            num_iter: Number of PGD iterations
            text: Text prompt for the model
            fps: Frames per second to sample
            verbose: Whether to print progress

        Returns:
            Dictionary containing attack results
        """
        start_time = time.time()

        # Preprocess video
        inputs = self.preprocess_video(video_path, text, fps=fps)

        pixel_values_clean = inputs['pixel_values_videos'].to(self.dtype)
        video_grid_thw = inputs['video_grid_thw']

        if verbose:
            print(f"Video loaded: {video_path}")
            print(f"pixel_values_videos shape: {pixel_values_clean.shape}")
            print(f"video_grid_thw: {video_grid_thw}")

        # Get clean features (no gradient needed)
        with torch.no_grad():
            features_clean = self.get_visual_features(
                pixel_values_clean,
                video_grid_thw,
                is_video=True
            )
            if verbose:
                print(f"Clean features shape: {features_clean.shape}")

        # Convert eps and alpha to normalized space
        eps_norm = eps / self.image_std
        alpha_norm = alpha / self.image_std

        if verbose:
            print(f"\nAttack parameters:")
            print(f"  eps (original): {eps:.6f}")
            print(f"  eps (normalized): {eps_norm:.6f}")
            print(f"  alpha (original): {alpha:.6f}")
            print(f"  alpha (normalized): {alpha_norm:.6f}")
            print(f"  iterations: {num_iter}")

        # Initialize adversarial sample
        pixel_values_adv = pixel_values_clean.clone().detach()

        # Track best result
        best_adv = pixel_values_adv.clone()
        best_cos_sim = 1.0

        # Logging
        cos_sim_history = []

        if verbose:
            print(f"\nStarting PGD attack on video...")

        # PGD iterations
        for i in range(num_iter):
            # Enable gradient for adversarial sample
            pixel_values_adv = pixel_values_adv.detach().requires_grad_(True)

            # Forward pass
            features_adv = self.get_visual_features(
                pixel_values_adv,
                video_grid_thw,
                is_video=True
            )

            # Compute cosine similarity loss
            loss = self.cosine_similarity_loss(features_adv, features_clean)

            # Backward pass
            loss.backward()

            # Get gradient
            grad = pixel_values_adv.grad.detach()

            # PGD update (gradient descent to minimize similarity)
            with torch.no_grad():
                # Update using sign of gradient
                pixel_values_adv = pixel_values_adv - alpha_norm * grad.sign()

                # Project to epsilon ball
                perturbation = pixel_values_adv - pixel_values_clean
                perturbation = torch.clamp(perturbation, -eps_norm, eps_norm)
                pixel_values_adv = pixel_values_clean + perturbation

                # Project to valid range [-1, 1]
                pixel_values_adv = torch.clamp(pixel_values_adv, -1.0, 1.0)

            # Track progress
            current_cos_sim = loss.item()
            cos_sim_history.append(current_cos_sim)

            # Update best result
            if current_cos_sim < best_cos_sim:
                best_cos_sim = current_cos_sim
                best_adv = pixel_values_adv.clone()

            if verbose and (i % 10 == 0 or i == num_iter - 1):
                print(f"  Iter {i:3d}: cos_sim = {current_cos_sim:.6f}")

        # Final evaluation
        with torch.no_grad():
            features_final = self.get_visual_features(
                best_adv, video_grid_thw, is_video=True
            )
            final_cos_sim = self.cosine_similarity_loss(
                features_final, features_clean
            ).item()

        # Calculate perturbation statistics
        perturbation = (best_adv - pixel_values_clean).detach()
        perturbation_denorm = perturbation * self.image_std

        elapsed_time = time.time() - start_time

        result = {
            'pixel_values_adv': best_adv.detach().cpu(),
            'pixel_values_clean': pixel_values_clean.detach().cpu(),
            'perturbation': perturbation.detach().cpu(),
            'video_grid_thw': video_grid_thw.cpu(),
            'initial_cos_sim': 1.0,
            'final_cos_sim': final_cos_sim,
            'best_cos_sim': best_cos_sim,
            'cos_sim_history': cos_sim_history,
            'eps': eps,
            'alpha': alpha,
            'num_iter': num_iter,
            'elapsed_time': elapsed_time,
            'video_path': video_path,
            'perturbation_l_inf': perturbation_denorm.abs().max().item(),
            'perturbation_l2': perturbation_denorm.norm(2).item(),
            'is_video': True,
        }

        if verbose:
            print(f"\n=== Video Attack Results ===")
            print(f"Initial cos_sim: {result['initial_cos_sim']:.6f}")
            print(f"Final cos_sim: {result['final_cos_sim']:.6f}")
            print(f"Similarity reduction: {result['initial_cos_sim'] - result['final_cos_sim']:.6f}")
            print(f"Perturbation L_inf: {result['perturbation_l_inf']:.6f}")
            print(f"Perturbation L_2: {result['perturbation_l2']:.6f}")
            print(f"Time elapsed: {elapsed_time:.2f}s")

        return result

    def attack(
        self,
        image_path: str,
        eps: float = 8/255,
        alpha: float = 1/255,
        num_iter: int = 100,
        text: str = "Describe this image.",
        verbose: bool = True,
    ) -> Dict:
        """
        Execute PGD attack on an image.

        Args:
            image_path: Path to input image
            eps: Maximum perturbation in [0, 1] space
            alpha: Step size in [0, 1] space
            num_iter: Number of PGD iterations
            text: Text prompt for the model
            verbose: Whether to print progress

        Returns:
            Dictionary containing attack results
        """
        start_time = time.time()

        # Load and preprocess image
        image_pil = Image.open(image_path).convert('RGB')
        inputs = self.preprocess_image(image_pil, text)

        pixel_values_clean = inputs['pixel_values'].to(self.dtype)
        image_grid_thw = inputs['image_grid_thw']

        if verbose:
            print(f"Image loaded: {image_path}")
            print(f"pixel_values shape: {pixel_values_clean.shape}")
            print(f"image_grid_thw: {image_grid_thw}")

        # Get clean features (no gradient needed)
        with torch.no_grad():
            features_clean = self.get_visual_features(
                pixel_values_clean,
                image_grid_thw,
                is_video=False
            )
            if verbose:
                print(f"Clean features shape: {features_clean.shape}")

        # Convert eps and alpha to normalized space
        # pixel_values are in [-1, 1], original eps is in [0, 1] space
        eps_norm = eps / self.image_std
        alpha_norm = alpha / self.image_std

        if verbose:
            print(f"\nAttack parameters:")
            print(f"  eps (original): {eps:.6f}")
            print(f"  eps (normalized): {eps_norm:.6f}")
            print(f"  alpha (original): {alpha:.6f}")
            print(f"  alpha (normalized): {alpha_norm:.6f}")
            print(f"  iterations: {num_iter}")

        # Initialize adversarial sample
        pixel_values_adv = pixel_values_clean.clone().detach()

        # Track best result
        best_adv = pixel_values_adv.clone()
        best_cos_sim = 1.0

        # Logging
        cos_sim_history = []

        if verbose:
            print(f"\nStarting PGD attack...")

        # PGD iterations
        for i in range(num_iter):
            # Enable gradient for adversarial sample
            pixel_values_adv = pixel_values_adv.detach().requires_grad_(True)

            # Forward pass
            features_adv = self.get_visual_features(
                pixel_values_adv,
                image_grid_thw,
                is_video=False
            )

            # Compute cosine similarity loss
            loss = self.cosine_similarity_loss(features_adv, features_clean)

            # Backward pass
            loss.backward()

            # Get gradient
            grad = pixel_values_adv.grad.detach()

            # PGD update (gradient descent to minimize similarity)
            with torch.no_grad():
                # Update using sign of gradient
                pixel_values_adv = pixel_values_adv - alpha_norm * grad.sign()

                # Project to epsilon ball
                perturbation = pixel_values_adv - pixel_values_clean
                perturbation = torch.clamp(perturbation, -eps_norm, eps_norm)
                pixel_values_adv = pixel_values_clean + perturbation

                # Project to valid range [-1, 1]
                pixel_values_adv = torch.clamp(pixel_values_adv, -1.0, 1.0)

            # Track progress
            current_cos_sim = loss.item()
            cos_sim_history.append(current_cos_sim)

            # Update best result
            if current_cos_sim < best_cos_sim:
                best_cos_sim = current_cos_sim
                best_adv = pixel_values_adv.clone()

            if verbose and (i % 10 == 0 or i == num_iter - 1):
                print(f"  Iter {i:3d}: cos_sim = {current_cos_sim:.6f}")

        # Final evaluation
        with torch.no_grad():
            features_final = self.get_visual_features(
                best_adv, image_grid_thw, is_video=False
            )
            final_cos_sim = self.cosine_similarity_loss(
                features_final, features_clean
            ).item()

        # Calculate perturbation statistics
        perturbation = (best_adv - pixel_values_clean).detach()
        perturbation_denorm = perturbation * self.image_std  # Back to [0, 1] scale

        elapsed_time = time.time() - start_time

        result = {
            'pixel_values_adv': best_adv.detach().cpu(),
            'pixel_values_clean': pixel_values_clean.detach().cpu(),
            'perturbation': perturbation.detach().cpu(),
            'image_grid_thw': image_grid_thw.cpu(),
            'initial_cos_sim': 1.0,
            'final_cos_sim': final_cos_sim,
            'best_cos_sim': best_cos_sim,
            'cos_sim_history': cos_sim_history,
            'eps': eps,
            'alpha': alpha,
            'num_iter': num_iter,
            'elapsed_time': elapsed_time,
            'image_path': image_path,
            'perturbation_l_inf': perturbation_denorm.abs().max().item(),
            'perturbation_l2': perturbation_denorm.norm(2).item(),
        }

        if verbose:
            print(f"\n=== Attack Results ===")
            print(f"Initial cos_sim: {result['initial_cos_sim']:.6f}")
            print(f"Final cos_sim: {result['final_cos_sim']:.6f}")
            print(f"Similarity reduction: {result['initial_cos_sim'] - result['final_cos_sim']:.6f}")
            print(f"Perturbation L_inf: {result['perturbation_l_inf']:.6f}")
            print(f"Perturbation L_2: {result['perturbation_l2']:.6f}")
            print(f"Time elapsed: {elapsed_time:.2f}s")

        return result

    def save_result(
        self,
        result: Dict,
        save_path: str,
    ) -> None:
        """
        Save attack result to file.

        Args:
            result: Attack result dictionary
            save_path: Path to save the result
        """
        torch.save(result, save_path)
        print(f"Result saved to {save_path}")

    def load_result(self, load_path: str) -> Dict:
        """
        Load attack result from file.

        Args:
            load_path: Path to load from

        Returns:
            Attack result dictionary
        """
        result = torch.load(load_path)
        print(f"Result loaded from {load_path}")
        return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="PGD attack on Qwen3-VL vision encoder"
    )
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to input image"
    )
    input_group.add_argument(
        "--video",
        type=str,
        help="Path to input video"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="adv_result.pt",
        help="Path to save attack result"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=8/255,
        help="Maximum perturbation (L_inf norm in [0,1] space)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1/255,
        help="Step size (in [0,1] space)"
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=100,
        help="Number of PGD iterations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text prompt (default: 'Describe this image/video.')"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second for video sampling"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine input type
    is_video = args.video is not None
    input_path = args.video if is_video else args.image

    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: Input not found: {input_path}")
        return

    # Set default text prompt
    if args.text is None:
        args.text = "Describe this video." if is_video else "Describe this image."

    # Initialize attacker
    attacker = Qwen3VLPGD(
        model_path=args.model,
        device=args.device,
        dtype=torch.float32,
    )

    # Run attack
    if is_video:
        result = attacker.attack_video(
            video_path=input_path,
            eps=args.eps,
            alpha=args.alpha,
            num_iter=args.iter,
            text=args.text,
            fps=args.fps,
            verbose=True,
        )
    else:
        result = attacker.attack(
            image_path=input_path,
            eps=args.eps,
            alpha=args.alpha,
            num_iter=args.iter,
            text=args.text,
            verbose=True,
        )

    # Save result
    attacker.save_result(result, args.output)


if __name__ == "__main__":
    main()
