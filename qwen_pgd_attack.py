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
        vision_only: bool = False,
    ):
        """
        Initialize the PGD attack on Qwen3-VL.

        Args:
            model_path: HuggingFace model path or local path to the Qwen3-VL model
            device: Device to run on (e.g., "cuda:0", "cpu"). Determines where tensors are placed.
            dtype: Data type for computations (float32 recommended for stable gradients)
            vision_only: If True, only load the vision encoder and discard the LLM
                         to save GPU memory. Sufficient for PGD attack which only
                         needs visual features.
        """
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.vision_only = vision_only

        # Load processor from HuggingFace
        print(f"Loading model from {model_path} (vision_only={vision_only})...")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,  # Allow custom code from processor repo
        )

        if vision_only:
            # Vision-only mode: load model to CPU first, strip LLM, then move
            # vision encoder to target device. This avoids GPU OOM during loading.
            # Model structure:
            #   self.model (Qwen3VLForConditionalGeneration)
            #     ├── model (Qwen3VLModel)
            #     │   ├── visual (Qwen3VLVisionModel)  ← keep
            #     │   └── language_model (Qwen3VLTextModel)  ← delete
            #     └── lm_head (nn.Linear)  ← delete
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                device_map="cpu",       # Load entirely to CPU first
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            del self.model.lm_head
            del self.model.model.language_model
            import gc; gc.collect()
            # Move the remaining vision encoder to target device
            self.model = self.model.to(device)
            torch.cuda.empty_cache()
            print("Vision-only mode: LLM and lm_head removed to save VRAM.")
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                device_map=device,      # Place full model on target device
                torch_dtype=dtype,
                trust_remote_code=True,
            )

        # Freeze model parameters - we only compute gradients w.r.t. input, not model weights
        self.model.eval()  # Set model to evaluation mode (disable dropout, batch norm, etc.)
        for param in self.model.parameters():
            param.requires_grad = False  # Disable gradient computation for model parameters

        # --- FIX 2: Dynamically obtain normalization parameters to align with CLIP ---
        # Reshape to [3, 1, 1] to support broadcasting in subsequent operations
        # These parameters are used to normalize pixel values before feeding to vision encoder
        if hasattr(self.processor, "image_processor"):
            # Try to get normalization parameters from the processor's image processor
            mean = self.processor.image_processor.image_mean
            std = self.processor.image_processor.image_std
            # Convert to tensors and reshape for broadcasting: [3] -> [3, 1, 1]
            self.image_mean = torch.tensor(mean, device=device, dtype=dtype).view(-1, 1, 1)
            self.image_std = torch.tensor(std, device=device, dtype=dtype).view(-1, 1, 1)
        else:
            # Fallback to standard CLIP normalization parameters if processor doesn't have them
            # These are the standard ImageNet normalization values used by CLIP
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

        This method calls the model's vision encoder to get feature representations
        of the input image or video. These features are used to compute the loss
        for the adversarial attack.

        Args:
            pixel_values: Preprocessed pixel values (normalized) of shape [N, D] or similar
            grid_thw: Grid dimensions [T, H, W] indicating temporal, height, width in patch units
            is_video: Whether the input is a video (True) or image (False)

        Returns:
            Concatenated visual features from the vision encoder
        """
        if is_video:
            # For video input, call the video feature extraction method
            video_embeds, _ = self.model.model.get_video_features(
                pixel_values.to(self.dtype),
                grid_thw
            )
            # Concatenate embeddings from all temporal frames
            features = torch.cat(video_embeds, dim=0)
        else:
            # For image input, call the image feature extraction method
            image_embeds, _ = self.model.model.get_image_features(
                pixel_values.to(self.dtype),
                grid_thw
            )
            # Concatenate embeddings from all spatial patches
            features = torch.cat(image_embeds, dim=0)
        return features

    def cosine_similarity_loss(
        self,
        feat_adv: torch.Tensor,
        feat_clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between adversarial and clean features.

        --- FIX 1: Token-Level cosine similarity ---
        Compute similarity for each token on the feature dimension (dim=-1), then average.
        This prevents the model from "cheating" by only perturbing a few tokens.

        The goal of the attack is to MINIMIZE this similarity (make adversarial features
        as different as possible from clean features). A lower cosine similarity means
        the attack is more successful.

        Args:
            feat_adv: Adversarial visual features from the vision encoder
            feat_clean: Clean (original) visual features from the vision encoder

        Returns:
            Mean cosine similarity across all tokens. Range: [-1, 1]
            - 1.0 means identical features (attack failed)
            - -1.0 means opposite features (attack very successful)
            - 0.0 means orthogonal features
        """
        # Compute cosine similarity for each token along the feature dimension
        cos_sim = F.cosine_similarity(feat_adv, feat_clean, dim=-1)
        # Return the mean similarity across all tokens
        return cos_sim.mean()

    def preprocess_image(
        self,
        image: Image.Image,
        text: str = "Describe this image.",
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess an image for the model.

        This method converts a PIL image and text prompt into the format expected
        by the Qwen3-VL model. It applies the chat template and tokenizes the input.

        Args:
            image: PIL Image object (RGB)
            text: Text prompt to accompany the image

        Returns:
            Dictionary containing:
            - pixel_values: Normalized pixel values ready for vision encoder
            - image_grid_thw: Grid dimensions [T, H, W] in patch units
            - input_ids: Tokenized text prompt
            - attention_mask: Attention mask for text tokens
            - Other model-specific inputs
        """
        # Create message structure following the chat template format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # Image content
                    {"type": "text", "text": text}      # Text prompt
                ]
            }
        ]
        # Apply the model's chat template to format the prompt
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Process the image and text through the processor
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        # Move all tensors to the specified device
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

        This method converts a video file and text prompt into the format expected
        by the Qwen3-VL model. It handles frame sampling and applies the chat template.

        Args:
            video_path: Path to the video file
            text: Text prompt to accompany the video
            max_frames: Maximum number of frames to extract (currently unused, kept for compatibility)
            fps: Frames per second for sampling. Lower fps means fewer frames are sampled.
                 For example, fps=1.0 means sample 1 frame per second.

        Returns:
            Dictionary containing:
            - pixel_values_videos: Normalized pixel values of sampled frames
            - video_grid_thw: Grid dimensions [T, H, W] in patch units
            - input_ids: Tokenized text prompt
            - attention_mask: Attention mask for text tokens
            - Other model-specific inputs
        """
        # Create message structure with video content
        messages = [
            {
                "role": "user",
                "content": [
                    # Video content with max resolution and fps sampling
                    {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": fps},
                    {"type": "text", "text": text}  # Text prompt
                ]
            }
        ]
        # Apply the model's chat template to format the prompt
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process video using qwen_vl_utils if available, otherwise use direct processing
        if QWEN_VL_UTILS_AVAILABLE:
            # Extract vision information (images and videos) from messages
            image_inputs, video_inputs = process_vision_info(messages)
            # Process with both image and video inputs
            inputs = self.processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
        else:
            # Fallback: process video directly without qwen_vl_utils
            inputs = self.processor(text=[prompt], videos=[video_path], return_tensors="pt")

        # Move all tensors to the specified device
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
        """
        Execute pixel-level PGD attack on a video.

        This method performs a Projected Gradient Descent attack directly on pixel values.
        It loads the video, samples frames, and iteratively perturbs them to minimize
        the cosine similarity between clean and adversarial visual features.

        Args:
            video_path: Path to the input video file
            eps: Maximum perturbation magnitude (L-infinity norm) in [0, 1] range
            alpha: Step size for each PGD iteration
            num_iter: Number of PGD iterations
            text: Text prompt describing the video
            fps: Frames per second for sampling
            verbose: Whether to print progress information

        Returns:
            Dictionary containing:
            - adv_frames: Adversarial frames as numpy arrays [T, H, W, 3]
            - clean_frames: Original clean frames as numpy arrays
            - sampled_indices: Indices of sampled frames from original video
            - total_frames: Total number of frames in original video
            - original_fps: Original video FPS
            - video_grid_thw: Grid dimensions in patch units
            - initial_cos_sim: Initial cosine similarity (always 1.0)
            - final_cos_sim: Final cosine similarity after attack
            - cos_sim_history: List of cosine similarities at each iteration
            - perturbation_l_inf: Maximum perturbation magnitude
            - perturbation_l2: L2 norm of perturbation
            - Other attack parameters (eps, alpha, num_iter, elapsed_time, etc.)
        """
        start_time = time.time()

        # Preprocess video to get grid dimensions and other metadata
        inputs = self.preprocess_video(video_path, text, fps=fps)
        video_grid_thw = inputs['video_grid_thw']

        # Load all frames from the video file
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 FPS if not available
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV format) to RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        # Validate that frames were loaded
        total_frames = len(frames)
        if total_frames == 0:
            raise ValueError(f"No frames loaded from {video_path}")

        # Sample frames according to the specified fps
        # sample_interval determines how many frames to skip
        sample_interval = max(1, int(original_fps / fps))
        sampled_indices = list(range(0, total_frames, sample_interval))
        sampled_frames = [frames[i] for i in sampled_indices]

        # Get grid dimensions from the preprocessed video
        # t: number of temporal patches, grid_h/grid_w: spatial patch grid dimensions
        t, grid_h, grid_w = video_grid_thw[0].tolist()
        t, grid_h, grid_w = int(t), int(grid_h), int(grid_w)

        # Adjust sampled frames to match model's expected frame count
        # The model expects t*2 frames (temporal dimension doubled)
        num_model_frames = t * 2
        if len(sampled_frames) > num_model_frames:
            # Truncate if too many frames
            sampled_frames = sampled_frames[:num_model_frames]
        elif len(sampled_frames) < num_model_frames:
            # Pad with last frame if too few frames
            while len(sampled_frames) < num_model_frames:
                sampled_frames.append(sampled_frames[-1])

        # Convert frames to tensor: [T, H, W, 3] -> [T, 3, H, W] with values in [0, 1]
        frames_tensor = torch.stack([
            torch.from_numpy(f).float() / 255.0 for f in sampled_frames
        ]).to(self.device).permute(0, 3, 1, 2)

        # Keep a copy of clean frames for reference
        frames_clean = frames_tensor.clone()

        # Patch size is 14 pixels (standard for vision transformers)
        patch_size = 14
        # Target resolution for the vision encoder
        target_h, target_w = grid_h * patch_size, grid_w * patch_size

        # Initialize adversarial frames
        frames_adv = frames_tensor.clone().detach()
        best_adv = frames_adv.clone()
        best_cos_sim = 1.0  # Start with maximum similarity (no perturbation)
        cos_sim_history = []

        # Get the vision model components
        vision_model = self.model.model.visual
        patch_embed = vision_model.patch_embed

        # Reshape normalization parameters for broadcasting with [T, C, H, W] tensors
        # From [3, 1, 1] to [1, 3, 1, 1] for proper broadcasting
        mean_4d = self.image_mean.view(1, 3, 1, 1)
        std_4d = self.image_std.view(1, 3, 1, 1)

        # Fix 2: Construct cu_seqlens for variable-length attention
        # cu_seqlens (cumulative sequence lengths) is needed for efficient attention computation
        # when sequences have different lengths.
        # video_grid_thw[:,0]=t (temporal), [:,1]=h (height), [:,2]=w (width) in patch units
        # Each temporal patch group has h*w tokens; there are t groups per video.
        seq_lens = video_grid_thw[:, 1] * video_grid_thw[:, 2]  # Number of spatial tokens per frame
        cu_seqlens = F.pad(
            torch.repeat_interleave(seq_lens, video_grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32),
            (1, 0), value=0
        )

        # Pre-compute rotary positional embeddings once (same for every iteration)
        # This avoids recomputing the same embeddings in each iteration
        rotary_pos_emb = vision_model.rot_pos_emb(video_grid_thw)

        # Fix 1: Compute features_clean via the SAME manual path used for features_adv
        # This ensures initial cos_sim == 1.0 and the attack target is consistent.
        # We manually go through the vision encoder to have full control over the computation.
        with torch.no_grad():
            # Resize frames to target resolution and normalize
            frames_5d_c = ((F.interpolate(frames_clean, size=(target_h, target_w),
                                          mode='bilinear', align_corners=False)
                            - mean_4d) / std_4d).permute(1, 0, 2, 3).unsqueeze(0)
            # Embed patches: convert image patches to embeddings
            h_c = patch_embed(frames_5d_c)
            # Handle different output shapes from patch_embed
            if h_c.dim() == 5:
                # If output is 5D, reshape to 3D: [B, T*H*W, D]
                h_c = h_c.view(1, -1, h_c.shape[-1]).squeeze(0)
            elif h_c.dim() == 3:
                # If output is already 3D, just remove batch dimension
                h_c = h_c.squeeze(0)
            # Pass through transformer blocks
            for blk in vision_model.blocks:
                h_c = blk(h_c, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
            # Merge tokens to get final features
            features_clean = vision_model.merger(h_c)

        # PGD attack loop: iteratively perturb frames to minimize cosine similarity
        for i in range(num_iter):
            # Enable gradient computation for adversarial frames
            frames_adv = frames_adv.detach().requires_grad_(True)

            # Resize frames to target resolution for the vision encoder
            frames_resized = F.interpolate(
                frames_adv, size=(target_h, target_w), mode='bilinear', align_corners=False
            )

            # Apply normalization using the model's normalization parameters
            frames_norm = (frames_resized - mean_4d) / std_4d

            # Reshape for vision encoder: [T, C, H, W] -> [1, C, T, H, W]
            frames_5d = frames_norm.permute(1, 0, 2, 3).unsqueeze(0)

            # Embed patches to get token embeddings
            hidden_states = patch_embed(frames_5d)
            # Handle different output shapes from patch_embed
            if hidden_states.dim() == 5:
                # If output is 5D, reshape to 3D: [B, T*H*W, D]
                hidden_states = hidden_states.view(1, -1, hidden_states.shape[-1]).squeeze(0)
            elif hidden_states.dim() == 3:
                # If output is already 3D, just remove batch dimension
                hidden_states = hidden_states.squeeze(0)

            # Pass through transformer blocks with variable-length attention
            for blk in vision_model.blocks:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
            # Merge tokens to get final visual features
            features_adv = vision_model.merger(hidden_states)

            # Compute loss: cosine similarity between adversarial and clean features
            loss = self.cosine_similarity_loss(features_adv, features_clean)
            # Backpropagate to compute gradients w.r.t. input frames
            loss.backward()

            # Get gradients w.r.t. adversarial frames
            grad = frames_adv.grad.detach()

            # Fix 3: Track best BEFORE update so best_adv corresponds to best_cos_sim
            # This ensures we save the frames that actually achieved the best loss
            current_cos_sim = loss.item()
            cos_sim_history.append(current_cos_sim)

            # Update best adversarial frames if current loss is better (lower)
            if current_cos_sim < best_cos_sim:
                best_cos_sim = current_cos_sim
                best_adv = frames_adv.detach().clone()

            # Print progress
            if verbose and (i % 10 == 0 or i == num_iter - 1):
                print(f"  Iter {i:3d}: cos_sim = {current_cos_sim:.6f}")

            # PGD update step: move in the direction of the gradient sign
            with torch.no_grad():
                # Step in the direction of increasing loss (decreasing similarity)
                frames_adv = frames_adv - alpha * grad.sign()
                # Compute perturbation in pixel space
                perturbation = frames_adv - frames_clean
                # Clip perturbation to stay within epsilon ball
                perturbation = torch.clamp(perturbation, -eps, eps)
                # Reconstruct adversarial frames
                frames_adv = frames_clean + perturbation
                # Clip to valid pixel range [0, 1]
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
            'final_loss': best_cos_sim,  # loss == cos_sim in this attack
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
        """
        Execute PGD patch-level attack on a video.

        This method performs a Projected Gradient Descent attack on the preprocessed
        pixel values (which are already in normalized space). Unlike attack_video_pixel_level,
        this operates on the model's input space directly.

        Args:
            video_path: Path to the input video file
            eps: Maximum perturbation magnitude in normalized space
            alpha: Step size for each PGD iteration
            num_iter: Number of PGD iterations
            text: Text prompt describing the video
            fps: Frames per second for sampling
            verbose: Whether to print progress information

        Returns:
            Dictionary containing attack results with keys:
            - pixel_values_adv: Adversarial pixel values in normalized space
            - pixel_values_clean: Clean pixel values in normalized space
            - perturbation: Perturbation in normalized space
            - video_grid_thw: Grid dimensions in patch units
            - initial_cos_sim: Initial cosine similarity (1.0)
            - final_cos_sim: Final cosine similarity after attack
            - cos_sim_history: Cosine similarity at each iteration
            - perturbation_l_inf: L-infinity norm of perturbation
            - perturbation_l2: L2 norm of perturbation
            - Other attack parameters (eps, alpha, num_iter, elapsed_time, etc.)
        """
        start_time = time.time()
        # Preprocess video to get normalized pixel values and grid dimensions
        inputs = self.preprocess_video(video_path, text, fps=fps)

        # Extract clean pixel values (already normalized by the processor)
        pixel_values_clean = inputs['pixel_values_videos'].to(self.dtype)
        video_grid_thw = inputs['video_grid_thw']

        # Compute clean features without gradients
        with torch.no_grad():
            features_clean = self.get_visual_features(pixel_values_clean, video_grid_thw, is_video=True)

        # Initialize adversarial pixel values
        pixel_values_adv = pixel_values_clean.clone().detach()
        best_adv = pixel_values_adv.clone()
        best_cos_sim = 1.0  # Start with maximum similarity
        cos_sim_history = []

        if verbose:
            print(f"\nStarting PGD attack on video...")

        # PGD attack loop on normalized pixel values
        for i in range(num_iter):
            # Enable gradient computation for adversarial pixel values
            pixel_values_adv = pixel_values_adv.detach().requires_grad_(True)
            # Extract features from adversarial pixel values
            features_adv = self.get_visual_features(pixel_values_adv, video_grid_thw, is_video=True)
            # Compute loss: cosine similarity between adversarial and clean features
            loss = self.cosine_similarity_loss(features_adv, features_clean)
            # Backpropagate to compute gradients
            loss.backward()
            # Get gradients w.r.t. pixel values
            grad = pixel_values_adv.grad.detach()

            # Fix 3: Track best adversarial example BEFORE update
            current_cos_sim = loss.item()
            cos_sim_history.append(current_cos_sim)

            # Update best adversarial example if current one is better
            if current_cos_sim < best_cos_sim:
                best_cos_sim = current_cos_sim
                best_adv = pixel_values_adv.detach().clone()

            # Print progress
            if verbose and (i % 10 == 0 or i == num_iter - 1):
                print(f"  Iter {i:3d}: cos_sim = {current_cos_sim:.6f}")

            # Update adversarial pixel values with proper projection to pixel space
            with torch.no_grad():
                # Rigorous projection in physical pixel space [0, 1]
                # This ensures perturbations are properly constrained in pixel space
                N, D = pixel_values_adv.shape
                # Reshape to [N, 3, -1] for channel-wise operations
                pv_adv_view = pixel_values_adv.view(N, 3, -1)
                grad_view = grad.view(N, 3, -1)
                pv_clean_view = pixel_values_clean.view(N, 3, -1)
                std_view = self.image_std.view(1, 3, 1)
                mean_view = self.image_mean.view(1, 3, 1)

                # Step 1: Gradient step in normalized space
                # Scale step size by std to account for normalization
                pv_adv_view = pv_adv_view - (alpha / std_view) * grad_view.sign()
                # Step 2: Convert to pixel space [0, 1]
                clean_01 = pv_clean_view * std_view + mean_view
                adv_01 = pv_adv_view * std_view + mean_view
                # Step 3: Compute perturbation in pixel space and clip to epsilon ball
                perturbation = adv_01 - clean_01
                perturbation = torch.clamp(perturbation, -eps, eps)
                # Step 4: Reconstruct and clip to valid pixel range
                adv_01_projected = torch.clamp(clean_01 + perturbation, 0.0, 1.0)
                # Step 5: Convert back to normalized space
                pixel_values_adv = ((adv_01_projected - mean_view) / std_view).view(N, D)

        # Compute final perturbation in pixel space for reporting
        with torch.no_grad():
            N, D = pixel_values_clean.shape
            std_view = self.image_std.view(1, 3, 1)
            mean_view = self.image_mean.view(1, 3, 1)
            # Convert clean and adversarial pixel values to pixel space [0, 1]
            clean_01_final = pixel_values_clean.view(N, 3, -1) * std_view + mean_view
            best_adv_01 = best_adv.view(N, 3, -1) * std_view + mean_view
            # Compute perturbation in pixel space
            perturbation_final = (best_adv_01 - clean_01_final).view(N, D)

        elapsed_time = time.time() - start_time

        # Return comprehensive attack results
        return {
            'pixel_values_adv': best_adv.detach().cpu(),  # Best adversarial pixel values (normalized)
            'pixel_values_clean': pixel_values_clean.detach().cpu(),  # Clean pixel values (normalized)
            'perturbation': perturbation_final.detach().cpu(),  # Perturbation in pixel space [0, 1]
            'video_grid_thw': video_grid_thw.cpu(),  # Grid dimensions in patch units
            'initial_cos_sim': 1.0,  # Initial cosine similarity (always 1.0)
            'final_cos_sim': best_cos_sim,  # Final cosine similarity after attack
            'final_loss': best_cos_sim,  # loss == cos_sim in this attack
            'cos_sim_history': cos_sim_history,  # Cosine similarity at each iteration
            'eps': eps,  # Maximum perturbation magnitude used
            'alpha': alpha,  # Step size used
            'num_iter': num_iter,  # Number of iterations performed
            'elapsed_time': elapsed_time,  # Total attack time in seconds
            'video_path': video_path,  # Path to input video
            'perturbation_l_inf': perturbation_final.abs().max().item(),  # L-infinity norm
            'perturbation_l2': perturbation_final.norm(2).item(),  # L2 norm
            'is_video': True,  # Indicates this is a video attack
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
        """
        Execute PGD patch-level attack on an image.

        This method performs a Projected Gradient Descent attack on an image to minimize
        the cosine similarity between clean and adversarial visual features.

        Args:
            image_path: Path to the input image file
            eps: Maximum perturbation magnitude in pixel space [0, 1]
            alpha: Step size for each PGD iteration
            num_iter: Number of PGD iterations
            text: Text prompt describing the image
            verbose: Whether to print progress information

        Returns:
            Dictionary containing attack results with keys:
            - pixel_values_adv: Adversarial pixel values (normalized)
            - pixel_values_clean: Clean pixel values (normalized)
            - perturbation: Perturbation in pixel space
            - image_grid_thw: Grid dimensions in patch units
            - initial_cos_sim: Initial cosine similarity (1.0)
            - final_cos_sim: Final cosine similarity after attack
            - cos_sim_history: Cosine similarity at each iteration
            - perturbation_l_inf: L-infinity norm of perturbation
            - perturbation_l2: L2 norm of perturbation
            - Other attack parameters (eps, alpha, num_iter, elapsed_time, etc.)
        """
        start_time = time.time()
        # Load image and convert to RGB
        image_pil = Image.open(image_path).convert('RGB')
        # Preprocess image to get normalized pixel values
        inputs = self.preprocess_image(image_pil, text)

        # Extract clean pixel values (already normalized by the processor)
        pixel_values_clean = inputs['pixel_values'].to(self.dtype)
        # Get grid dimensions for the image
        image_grid_thw = inputs['image_grid_thw']

        # Compute clean features without gradients
        with torch.no_grad():
            features_clean = self.get_visual_features(pixel_values_clean, image_grid_thw, is_video=False)

        # Initialize adversarial pixel values
        pixel_values_adv = pixel_values_clean.clone().detach()
        best_adv = pixel_values_adv.clone()
        best_cos_sim = 1.0  # Start with maximum similarity
        cos_sim_history = []

        if verbose:
            print(f"\nStarting PGD attack on image...")

        # PGD attack loop on normalized pixel values
        for i in range(num_iter):
            # Enable gradient computation for adversarial pixel values
            pixel_values_adv = pixel_values_adv.detach().requires_grad_(True)
            # Extract features from adversarial pixel values
            features_adv = self.get_visual_features(pixel_values_adv, image_grid_thw, is_video=False)
            # Compute loss: cosine similarity between adversarial and clean features
            loss = self.cosine_similarity_loss(features_adv, features_clean)
            # Backpropagate to compute gradients
            loss.backward()
            # Get gradients w.r.t. pixel values
            grad = pixel_values_adv.grad.detach()

            # Track best adversarial example BEFORE update
            current_cos_sim = loss.item()
            cos_sim_history.append(current_cos_sim)

            # Update best adversarial example if current one is better
            if current_cos_sim < best_cos_sim:
                best_cos_sim = current_cos_sim
                best_adv = pixel_values_adv.detach().clone()

            # Print progress
            if verbose and (i % 10 == 0 or i == num_iter - 1):
                print(f"  Iter {i:3d}: cos_sim = {current_cos_sim:.6f}")

            # Update adversarial pixel values with proper projection to pixel space
            with torch.no_grad():
                # Rigorous projection in physical pixel space [0, 1]
                # This ensures perturbations are properly constrained in pixel space
                N, D = pixel_values_adv.shape
                # Reshape to [N, 3, -1] for channel-wise operations
                pv_adv_view = pixel_values_adv.view(N, 3, -1)
                grad_view = grad.view(N, 3, -1)
                pv_clean_view = pixel_values_clean.view(N, 3, -1)
                std_view = self.image_std.view(1, 3, 1)
                mean_view = self.image_mean.view(1, 3, 1)

                # Step 1: Gradient step in normalized space
                # Scale step size by std to account for normalization
                pv_adv_view = pv_adv_view - (alpha / std_view) * grad_view.sign()
                # Step 2: Convert to pixel space [0, 1]
                clean_01 = pv_clean_view * std_view + mean_view
                adv_01 = pv_adv_view * std_view + mean_view
                # Step 3: Compute perturbation in pixel space and clip to epsilon ball
                perturbation = adv_01 - clean_01
                perturbation = torch.clamp(perturbation, -eps, eps)
                # Step 4: Reconstruct and clip to valid pixel range
                adv_01_projected = torch.clamp(clean_01 + perturbation, 0.0, 1.0)
                # Step 5: Convert back to normalized space
                pixel_values_adv = ((adv_01_projected - mean_view) / std_view).view(N, D)

        # Compute final perturbation in pixel space for reporting
        with torch.no_grad():
            N, D = pixel_values_clean.shape
            std_view = self.image_std.view(1, 3, 1)
            mean_view = self.image_mean.view(1, 3, 1)

            # Convert clean and adversarial pixel values to pixel space [0, 1]
            clean_01_final = pixel_values_clean.view(N, 3, -1) * std_view + mean_view
            best_adv_01 = best_adv.view(N, 3, -1) * std_view + mean_view
            # Compute perturbation in pixel space
            perturbation_final = (best_adv_01 - clean_01_final).view(N, D)

        elapsed_time = time.time() - start_time

        # Return comprehensive attack results
        return {
            'pixel_values_adv': best_adv.detach().cpu(),  # Best adversarial pixel values (normalized)
            'pixel_values_clean': pixel_values_clean.detach().cpu(),  # Clean pixel values (normalized)
            'perturbation': perturbation_final.detach().cpu(),  # Perturbation in pixel space [0, 1]
            'image_grid_thw': image_grid_thw.cpu(),  # Grid dimensions in patch units
            'initial_cos_sim': 1.0,  # Initial cosine similarity (always 1.0)
            'final_cos_sim': best_cos_sim,  # Final cosine similarity after attack
            'final_loss': best_cos_sim,  # loss == cos_sim in this attack
            'cos_sim_history': cos_sim_history,  # Cosine similarity at each iteration
            'eps': eps,  # Maximum perturbation magnitude used
            'alpha': alpha,  # Step size used
            'num_iter': num_iter,  # Number of iterations performed
            'elapsed_time': elapsed_time,  # Total attack time in seconds
            'image_path': image_path,  # Path to input image
            'perturbation_l_inf': perturbation_final.abs().max().item(),  # L-infinity norm
            'perturbation_l2': perturbation_final.norm(2).item(),  # L2 norm
        }

    def save_result(self, result: Dict, save_path: str) -> None:
        """
        Save attack result to a file.

        Args:
            result: Dictionary containing attack results
            save_path: Path where to save the result file
        """
        torch.save(result, save_path)
        print(f"Result saved to {save_path}")

    def load_result(self, load_path: str) -> Dict:
        """
        Load attack result from a file.

        Args:
            load_path: Path to the result file

        Returns:
            Dictionary containing attack results
        """
        return torch.load(load_path)


def parse_args():
    """
    Parse command-line arguments for the PGD attack script.

    Returns:
        Parsed arguments with the following attributes:
        - image: Path to input image (mutually exclusive with --video)
        - video: Path to input video (mutually exclusive with --image)
        - output: Path to save attack result (default: adv_result.pt)
        - model: HuggingFace model path (default: Qwen/Qwen3-VL-8B-Instruct)
        - eps: Maximum perturbation magnitude (default: 8/255)
        - alpha: Step size for PGD iterations (default: 1/255)
        - iter: Number of PGD iterations (default: 100)
        - device: Device to run on (default: cuda:1)
        - text: Text prompt for the model (default: auto-generated)
        - fps: Video frame sampling rate (default: 1.0)
    """
    parser = argparse.ArgumentParser(description="PGD attack on Qwen3-VL vision encoder")

    # Input: either image or video (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to input image")
    input_group.add_argument("--video", type=str, help="Path to input video")

    # Output and model configuration
    parser.add_argument("--output", type=str, default="adv_result.pt",
                        help="Path to save attack result")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model path (HuggingFace or local)")

    # Attack parameters
    parser.add_argument("--eps", type=float, default=8/255,
                        help="Maximum perturbation magnitude (L-infinity norm)")
    parser.add_argument("--alpha", type=float, default=1/255,
                        help="Step size for each PGD iteration")
    parser.add_argument("--iter", type=int, default=100,
                        help="Number of PGD iterations")

    # Hardware and input configuration
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Device to run on (e.g., cuda:0, cpu)")
    parser.add_argument("--text", type=str, default=None,
                        help="Text prompt for the model (auto-generated if not provided)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Video frame sampling rate (frames per second)")
    parser.add_argument("--vision-only", action="store_true",
                        help="Only load vision encoder, skip LLM to save VRAM")

    return parser.parse_args()


def main():
    """
    Main function to execute the PGD attack.

    This function:
    1. Parses command-line arguments
    2. Validates input file existence
    3. Initializes the Qwen3VLPGD attacker
    4. Executes the appropriate attack (image or video)
    5. Saves the attack results
    """
    # Parse command-line arguments
    args = parse_args()

    # Determine if input is video or image
    is_video = args.video is not None
    input_path = args.video if is_video else args.image

    # Validate that input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input not found: {input_path}")
        return

    # Set default text prompt if not provided
    if args.text is None:
        args.text = "Describe this video." if is_video else "Describe this image."

    # Initialize the attacker with specified model and device
    attacker = Qwen3VLPGD(
        model_path=args.model, device=args.device, dtype=torch.float32,
        vision_only=args.vision_only,
    )

    # Execute attack based on input type
    if is_video:
        # Perform patch-level PGD attack on video
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
        # Perform patch-level PGD attack on image
        result = attacker.attack(
            image_path=input_path,
            eps=args.eps,
            alpha=args.alpha,
            num_iter=args.iter,
            text=args.text,
            verbose=True,
        )

    # Save attack results to file
    attacker.save_result(result, args.output)


if __name__ == "__main__":
    main()