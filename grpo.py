"""
Online GRPO Training for CUDA Kernel Optimization
==================================================

Architecture: Generate kernels → Benchmark with KernelBench → Train with GRPO → Repeat

This implements true online RL where the model improves between generation cycles,
not supervised learning on a fixed dataset.

Custom GRPO implementation without TRL dependency.
"""

import modal
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import subprocess
import tempfile
import os
import re
import numpy as np

# ============================================================================
# MODAL SETUP
# ============================================================================

app = modal.App("cuda-kernel-online-grpo")

# Use official NVIDIA CUDA image with development tools
# This matches KernelBench's approach and ensures proper CUDA environment
cuda_version = "12.4.0"  # Compatible with most GPUs
flavor = "devel"  # Includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Build image with all dependencies
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git",
        "gcc-10",
        "g++-10",
        "build-essential",
        "cmake",
        "ninja-build",

    )
    .run_commands(
        # Set gcc-10 and g++-10 as default
        "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100",
        "update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100",
    )
    .pip_install(
        "torch==2.7.0",
        "transformers==4.51.1",
        "vllm==0.9.1",
        "datasets==3.5.1",
        "peft==0.15.0",
        "accelerate==1.5.1",
        "wandb==0.17.6",
        "ninja",
        "litellm",
        "openai",
        "requests",
        "func-timeout",
    )
    .run_commands(
        "git clone https://github.com/ScalingIntelligence/KernelBench /root/KernelBench",
        "cd /root/KernelBench && pip install -e ."
    )
)

# Persistent volumes
checkpoint_vol = modal.Volume.from_name("grpo-checkpoints", create_if_missing=True)
results_vol = modal.Volume.from_name("grpo-results", create_if_missing=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Online GRPO training configuration"""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # Online RL parameters
    num_training_iterations: int = 50  # How many generate-train cycles
    num_generations_per_iter: int = 5  # Completions per prompt per iteration (reduced to prevent OOM)
    problems_per_iter: int = 6  # How many problems to sample each iteration (30 total samples)
    
    # GRPO parameters
    learning_rate: float = 1e-6
    grpo_epsilon: float = 0.2  # Clipping parameter
    grpo_beta: float = 0.1  # KL penalty coefficient (increased from 0.01 to prevent divergence)
    
    # Generation parameters
    max_prompt_length: int = 1024
    max_completion_length: int = 2048
    temperature: float = 0.7
    
    # Training parameters
    mini_epochs: int = 1  # Reduced to 1 for memory efficiency
    batch_size: int = 1
    gradient_accumulation_steps: int = 4  # Reduced from 8
    max_grad_norm: float = 1.0
    
    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0  # CHANGED: Disabled dropout to prevent NaN with bfloat16
    
    # Checkpointing
    save_every_n_iters: int = 10
    load_checkpoint: int = 10  # Load checkpoint from this iteration, or None
    # Reward weights
    reward_compilation: float = 0.2
    reward_correctness: float = 0.5
    reward_speedup: float = 0.3
    speedup_target: float = 2.0  # Normalize speedups to this value



# ============================================================================
# CUSTOM GRPO IMPLEMENTATION
# ============================================================================

class GRPOTrainer:
    """
    Custom Group Relative Policy Optimization Trainer
    
    GRPO is a variant of PPO designed for language models that:
    1. Groups multiple completions per prompt
    2. Computes advantages relative to the group mean
    3. Uses clipped objective to prevent large updates
    
    Memory optimizations:
    - Reference model stored on CPU, moved to GPU only when needed
    - Batch processing for reference log probs
    - Aggressive gradient checkpointing
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer,
        config: TrainingConfig,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        vocab_size: int  # Added for proper token validation
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.vocab_size = vocab_size  # Store vocab size for token validation
        
        # Move reference model to CPU to save GPU memory
        print("Moving reference model to CPU for memory efficiency...")
        self.ref_model = self.ref_model.to('cpu')
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Enable gradient checkpointing for policy model
        #if hasattr(self.model, 'gradient_checkpointing_enable'):
        #    self.model.gradient_checkpointing_enable()
        #    print("Enabled gradient checkpointing for policy model")
    #
    def compute_log_probs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute log probabilities for generated tokens with batching

        NOTE: Assumes model is in correct mode (train/eval) - doesn't change it

        Args:
            model: Language model
            input_ids: Full sequence (prompt + completion)
            attention_mask: Attention mask for full sequence
            labels: Labels with -100 for prompt tokens
            batch_size: Process in mini-batches to save memory

        Returns:
            log_probs: Log probabilities of generated tokens (with gradients if model.training)
        """
        all_log_probs = []
        num_samples = input_ids.shape[0]

        # Determine if this is the policy model (needs gradients) or ref model
        use_autocast = not model.training

        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_input_ids = input_ids[i:end_idx]
            batch_attention_mask = attention_mask[i:end_idx]
            batch_labels = labels[i:end_idx]

            # Validate inputs before forward pass to prevent NaN
            if torch.isnan(batch_input_ids.float()).any():
                raise RuntimeError("NaN in input_ids before forward pass!")
            if (batch_input_ids < 0).any() or (batch_input_ids >= self.vocab_size).any():
                max_id = batch_input_ids.max().item()
                min_id = batch_input_ids.min().item()
                print(f"ERROR: Token IDs out of range [{min_id}, {max_id}], vocab_size={self.vocab_size}")
                # Clamp to valid range
                batch_input_ids = torch.clamp(batch_input_ids, 0, self.vocab_size - 1)
                print(f"  Clamped to valid range")

            # CRITICAL FIX: Only use autocast for eval mode (reference model)
            # For training mode (policy model), disable autocast to prevent NaN
            if use_autocast:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                    )
                    logits = outputs.logits
            else:
                # Training mode: NO autocast - compute in model's native dtype (bfloat16)
                # This prevents NaN issues that can occur with bfloat16 + LoRA + autocast
                with torch.cuda.amp.autocast(enabled=False):
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                    )
                    logits = outputs.logits

            # Check for NaN in logits immediately after forward pass
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"ERROR: NaN/Inf detected in logits after forward pass!")
                print(f"  Model training mode: {model.training}")
                print(f"  Logits dtype: {logits.dtype}")
                print(f"  Logits stats: min={logits.min()}, max={logits.max()}")
                print(f"  NaN count: {torch.isnan(logits).sum().item()}")
                print(f"  Inf count: {torch.isinf(logits).sum().item()}")

                # Try to identify which token positions have NaN
                nan_positions = torch.isnan(logits).any(dim=-1)
                print(f"  Sequences with NaN: {nan_positions.sum(dim=-1)}")

                raise RuntimeError("NaN/Inf in model output - training cannot continue")

            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_labels[..., 1:].contiguous()

            # Cast to float32 for numerical stability in log_softmax
            shift_logits_fp32 = shift_logits.float()

            # Compute log probabilities in float32
            log_probs = F.log_softmax(shift_logits_fp32, dim=-1)

            # Check for NaN after log_softmax
            if torch.isnan(log_probs).any():
                print("ERROR: NaN appeared in log_softmax computation!")
                raise RuntimeError("NaN in log_softmax")

            # IMPORTANT: Replace -100 with 0 before gather to avoid out-of-bounds
            shift_labels_safe = shift_labels.clone()
            shift_labels_safe[shift_labels == -100] = 0

            # Clamp to valid range using actual vocab size
            shift_labels_safe = torch.clamp(shift_labels_safe, 0, self.vocab_size - 1)

            # Gather log probs for actual tokens
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=shift_labels_safe.unsqueeze(-1)
            ).squeeze(-1)

            # Mask out prompt tokens (where labels == -100)
            mask = (shift_labels != -100).float()
            token_log_probs = token_log_probs * mask

            # CRITICAL FIX: Normalize by sequence length to prevent explosion
            # Sum log probs and divide by number of tokens (average per token)
            num_tokens = mask.sum(dim=-1)  # Count non-masked tokens
            sequence_log_probs = token_log_probs.sum(dim=-1) / (num_tokens + 1e-8)

            # For training mode, keep on GPU with gradients
            # For eval mode, move to CPU
            if model.training:
                all_log_probs.append(sequence_log_probs)
            else:
                all_log_probs.append(sequence_log_probs.detach())

            # Clear intermediate tensors
            del outputs, logits, shift_logits, shift_labels_safe, mask
            if i < num_samples - batch_size:
                torch.cuda.empty_cache()

        # Concatenate results
        if len(all_log_probs) == 1:
            return all_log_probs[0]
        else:
            return torch.cat(all_log_probs, dim=0)
    
    def compute_ref_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Compute reference log probs with memory-efficient batching
        
        Temporarily moves ref model to GPU, processes in batches,
        then moves back to CPU to save memory.
        """
        all_log_probs = []
        num_samples = input_ids.shape[0]
        
        # Check available GPU memory before moving ref model
        if torch.cuda.is_available():
            free_mem = torch.cuda.get_device_properties(self.device).total_memory
            allocated = torch.cuda.memory_allocated(self.device)
            free_mem_gb = (free_mem - allocated) / 1024**3
            print(f"  Available GPU memory: {free_mem_gb:.2f}GB")

            if free_mem_gb < 10.0:
                print(f"  WARNING: Low GPU memory ({free_mem_gb:.2f}GB). Clearing cache...")
                torch.cuda.empty_cache()
                import gc
                gc.collect()

        # Temporarily move ref model to GPU
        print("  Moving reference model to GPU for log prob computation...")
        try:
            self.ref_model = self.ref_model.to(self.device)
        except RuntimeError as e:
            print(f"  ERROR: Failed to move ref model to GPU: {e}")
            print(f"  This likely means GPU is out of memory")
            raise RuntimeError(f"OOM when moving reference model to GPU. Reduce batch size.") from e

        try:
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    end_idx = min(i + batch_size, num_samples)
                    batch_input_ids = input_ids[i:end_idx]
                    batch_attention_mask = attention_mask[i:end_idx]
                    batch_labels = labels[i:end_idx]

                    # CRITICAL: Use same precision as policy model for consistency
                    # Policy model disables autocast to prevent NaN, so ref model should too
                    try:
                        with torch.cuda.amp.autocast(enabled=False):
                            outputs = self.ref_model(
                                input_ids=batch_input_ids,
                                attention_mask=batch_attention_mask,
                            )
                            logits = outputs.logits
                    except RuntimeError as e:
                        error_msg = str(e)
                        if "out of memory" in error_msg.lower():
                            print(f"  ERROR: GPU OOM during reference model forward pass!")
                            print(f"  Batch: {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")
                            print(f"  Batch size: {end_idx - i}")
                            print(f"  Sequence length: {batch_input_ids.shape[1]}")
                            raise RuntimeError(f"OOM in reference model. Reduce problems_per_iter or num_generations_per_iter") from e
                        else:
                            print(f"  ERROR: CUDA error in reference model forward pass: {error_msg}")
                            raise
                    
                    # Shift logits and labels for next-token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch_labels[..., 1:].contiguous()
                    
                    # CRITICAL: Cast to float32 for numerical stability
                    # bfloat16 can cause NaN in log_softmax due to overflow/underflow
                    shift_logits_fp32 = shift_logits.float()
                    
                    # Check for NaN/Inf in logits BEFORE log_softmax
                    if torch.isnan(shift_logits_fp32).any():
                        print("ERROR: NaN detected in logits before log_softmax!")
                        print(f"  NaN count: {torch.isnan(shift_logits_fp32).sum().item()}")
                        print(f"  Logits stats: min={shift_logits_fp32.min()}, max={shift_logits_fp32.max()}")
                    if torch.isinf(shift_logits_fp32).any():
                        print("ERROR: Inf detected in logits before log_softmax!")
                        print(f"  Inf count: {torch.isinf(shift_logits_fp32).sum().item()}")
                        print(f"  Logits stats: min={shift_logits_fp32.min()}, max={shift_logits_fp32.max()}")
                    
                    # Compute log probabilities in float32 for stability
                    log_probs = F.log_softmax(shift_logits_fp32, dim=-1)
                    
                    # Check for NaN after log_softmax
                    if torch.isnan(log_probs).any():
                        print("ERROR: NaN in log_probs after log_softmax!")
                        print(f"  This suggests numerical instability in the forward pass")
                    
                    # IMPORTANT: Replace -100 with 0 before gather to avoid out-of-bounds
                    # The mask will zero these out anyway
                    shift_labels_safe = shift_labels.clone()
                    shift_labels_safe[shift_labels == -100] = 0
                    
                    # Clamp to valid range using actual vocab size
                    shift_labels_safe = torch.clamp(shift_labels_safe, 0, self.vocab_size - 1)
                    
                    # Gather log probs for actual tokens
                    token_log_probs = torch.gather(
                        log_probs,
                        dim=-1,
                        index=shift_labels_safe.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    # Mask out prompt tokens and invalid tokens
                    mask = (shift_labels != -100).float()
                    token_log_probs = token_log_probs * mask

                    # CRITICAL FIX: Normalize by sequence length (same as policy model)
                    # Average per token to prevent explosion with different sequence lengths
                    num_tokens = mask.sum(dim=-1)  # Count non-masked tokens
                    sequence_log_probs = token_log_probs.sum(dim=-1) / (num_tokens + 1e-8)
                    
                    # Move to CPU immediately to free GPU memory
                    all_log_probs.append(sequence_log_probs.detach().cpu())
                    
                    # Clear cache after each batch
                    del outputs, logits, shift_logits, log_probs, token_log_probs, shift_labels_safe
                    torch.cuda.empty_cache()
        
        finally:
            # Always move ref model back to CPU
            print("  Moving reference model back to CPU...")
            self.ref_model = self.ref_model.to('cpu')
            torch.cuda.empty_cache()
        
        # Concatenate and move back to GPU
        result = torch.cat(all_log_probs, dim=0).to(self.device)
        return result
    
    def compute_advantages_robust(
        self,
        rewards: torch.Tensor,
        num_generations: int,
        epsilon: float = 1e-4  # Increased from 1e-8 for stability
    ) -> torch.Tensor:
        """
        Compute group-relative advantages with robust handling of edge cases

        Improvements:
        1. Larger epsilon for numerical stability
        2. Add small noise if all rewards identical
        3. Clip extreme advantages
        """
        # Reshape to (num_prompts, num_generations)
        num_prompts = len(rewards) // num_generations
        rewards_grouped = rewards.view(num_prompts, num_generations)

        # Compute group mean and std
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True)

        # Check for zero variance (all rewards identical in a group)
        zero_std_mask = group_std < epsilon

        if zero_std_mask.any():
            print(f"  Warning: {zero_std_mask.sum().item()} groups have identical rewards")
            print(f"  Adding small noise for numerical stability...")

            # Add small random noise to rewards where std is zero
            noise = torch.randn_like(rewards_grouped) * 0.01
            rewards_grouped = rewards_grouped + noise * zero_std_mask.float()

            # Recompute statistics
            group_mean = rewards_grouped.mean(dim=1, keepdim=True)
            group_std = rewards_grouped.std(dim=1, keepdim=True)

        # Add epsilon for stability
        group_std = group_std + epsilon

        # Normalize advantages
        advantages = (rewards_grouped - group_mean) / group_std

        # Clip extreme advantages to prevent instability
        advantages = torch.clamp(advantages, -10.0, 10.0)

        # Flatten back
        advantages = advantages.view(-1)

        return advantages

    
    def compute_grpo_loss(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss with clipping and KL penalty
        
        Args:
            policy_log_probs: Log probs from current policy
            ref_log_probs: Log probs from reference policy
            advantages: Normalized advantages
        
        Returns:
            loss: Scalar loss
            stats: Dictionary of statistics
        """
        
        # Debug: Check for NaN/Inf in inputs
        if torch.isnan(policy_log_probs).any():
            print("ERROR: NaN in policy_log_probs!")
            print(f"  policy_log_probs stats: min={policy_log_probs.min()}, max={policy_log_probs.max()}")
        if torch.isnan(ref_log_probs).any():
            print("ERROR: NaN in ref_log_probs!")
            print(f"  ref_log_probs stats: min={ref_log_probs.min()}, max={ref_log_probs.max()}")
        if torch.isnan(advantages).any():
            print("ERROR: NaN in advantages!")
            print(f"  advantages stats: min={advantages.min()}, max={advantages.max()}")

        # Compute probability ratio with clipping to prevent overflow
        log_ratio = policy_log_probs - ref_log_probs

        # CRITICAL FIX: Much tighter clipping to prevent loss explosion
        # With per-token averaging, log_ratio should be close to 0 in healthy training
        # exp(5) = 148, exp(-5) = 0.0067 - this is already a very large policy change
        # Original [-20, 20] allowed exp(20) = 485M which caused the loss explosion
        log_ratio_clipped = torch.clamp(log_ratio, -5.0, 5.0)

        # Debug: Check if clipping was needed
        if (log_ratio.abs() > 5.0).any():
            print(f"WARNING: Extreme log_ratio detected (clipped to [-5, 5])!")
            print(f"  log_ratio stats: min={log_ratio.min():.2f}, max={log_ratio.max():.2f}")
            print(f"  Number of extreme values: {(log_ratio.abs() > 5.0).sum().item()}")
            print(f"  policy_log_probs: min={policy_log_probs.min():.2f}, max={policy_log_probs.max():.2f}")
            print(f"  ref_log_probs: min={ref_log_probs.min():.2f}, max={ref_log_probs.max():.2f}")
            print(f"  This suggests policy and reference models are diverging!")

        ratio = torch.exp(log_ratio_clipped)

        # Additional safety: Check ratio range
        if ratio.max() > 100 or ratio.min() < 0.01:
            print(f"WARNING: Extreme ratio values detected!")
            print(f"  ratio stats: min={ratio.min():.4f}, max={ratio.max():.4f}, mean={ratio.mean():.4f}")
            print(f"  This indicates policy has changed significantly from reference")

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.grpo_epsilon,
            1.0 + self.config.grpo_epsilon
        )
        
        # Policy loss
        policy_loss_unclipped = -advantages * ratio
        policy_loss_clipped = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

        # KL divergence penalty (approximate) using clipped values
        # KL(p||q) ≈ ratio - 1 - log(ratio) where ratio = p/q
        kl_div = (ratio - 1.0 - log_ratio_clipped).mean()

        # Additional safety: clip KL divergence to prevent explosion
        # Typical KL values should be < 10 in well-behaved training
        kl_div = torch.clamp(kl_div, -10.0, 10.0)
        
        # Total loss
        loss = policy_loss + self.config.grpo_beta * kl_div

        # CRITICAL: Sanity check for extreme loss values
        if loss.abs() > 100.0:
            print(f"❌ ERROR: Extreme loss detected: {loss.item():.2f}")
            print(f"  policy_loss: {policy_loss.item():.2f}")
            print(f"  kl_div: {kl_div.item():.2f}")
            print(f"  ratio range: [{ratio.min():.4f}, {ratio.max():.4f}]")
            print(f"  advantages range: [{advantages.min():.4f}, {advantages.max():.4f}]")
            print(f"  This should not happen with the fixes applied!")
            # Don't raise error - let training continue with clipped loss
            loss = torch.clamp(loss, -100.0, 100.0)
            print(f"  Clamped loss to: {loss.item():.2f}")

        # Statistics (detach for logging to avoid grad issues)
        with torch.no_grad():
            # Handle single-element tensors for std
            ratio_std = ratio.std().item() if ratio.numel() > 1 else 0.0
            advantages_std = advantages.std().item() if advantages.numel() > 1 else 0.0
            
            stats = {
                "policy_loss": policy_loss.item(),
                "kl_div": kl_div.item(),
                "total_loss": loss.item(),
                "ratio_mean": ratio.mean().item(),
                "ratio_std": ratio_std,
                "advantages_mean": advantages.mean().item(),
                "advantages_std": advantages_std,
            }
        
        return loss, stats
    
    def train_step(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: List[float],
    ) -> Dict[str, float]:
        """
        Single training step on a batch of data with memory optimizations
        
        Args:
            prompts: List of prompts
            completions: List of completions
            rewards: List of rewards
        
        Returns:
            stats: Training statistics
        """
        # CRITICAL: Ensure model is in training mode
        self.model.train()
        
        # Verify model has trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("Model has no trainable parameters! Check LoRA configuration.")
        
        # Clear cache before starting
        torch.cuda.empty_cache()
        
        # Tokenize prompts and full sequences
        prompt_encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
            return_tensors="pt"
        ).to(self.device)
        
        full_texts = [p + c for p, c in zip(prompts, completions)]
        full_encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length + self.config.max_completion_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Validate token IDs are within vocabulary
        if full_encodings["input_ids"].max() >= self.vocab_size:
            max_token = full_encodings["input_ids"].max().item()
            num_bad = (full_encodings["input_ids"] >= self.vocab_size).sum().item()
            print(f"WARNING: Token IDs out of range! Max token: {max_token}, Vocab size: {self.vocab_size}")
            print(f"  Clamping {num_bad} tokens to valid range")
            # Clamp to valid range
            full_encodings["input_ids"] = torch.clamp(full_encodings["input_ids"], 0, self.vocab_size - 1)
        
        # Create labels (mask prompt tokens with -100)
        labels = full_encodings["input_ids"].clone()
        prompt_lengths = prompt_encodings["attention_mask"].sum(dim=1)
        
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100
        
        # Validate labels
        valid_label_mask = (labels >= 0) & (labels < self.vocab_size)
        invalid_labels = (~valid_label_mask & (labels != -100)).sum().item()
        if invalid_labels > 0:
            print(f"WARNING: Found {invalid_labels} invalid label values (not -100 and not in vocab range)")
            # Replace invalid labels with -100
            labels[~valid_label_mask & (labels != -100)] = -100
        
        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        # Debug: Check reward distribution
        print(f"Debug: Reward statistics")
        print(f"  Mean: {rewards_tensor.mean().item():.4f}")
        print(f"  Std: {rewards_tensor.std().item():.4f}")
        print(f"  Min: {rewards_tensor.min().item():.4f}")
        print(f"  Max: {rewards_tensor.max().item():.4f}")
        print(f"  Unique values: {len(rewards_tensor.unique())}")
        
        # Check for NaN in rewards
        if torch.isnan(rewards_tensor).any():
            raise ValueError("NaN detected in rewards!")
        
        # Compute advantages with robust method
        advantages = self.compute_advantages_robust(
            rewards_tensor,
            self.config.num_generations_per_iter
        )
        
        # Check advantages
        print(f"Debug: Advantage statistics")
        print(f"  Mean: {advantages.mean().item():.4f}")
        print(f"  Std: {advantages.std().item():.4f}")
        print(f"  Min: {advantages.min().item():.4f}")
        print(f"  Max: {advantages.max().item():.4f}")
        
        if torch.isnan(advantages).any():
            raise ValueError("NaN detected in advantages!")
        
        # Compute reference log probs (memory-efficient with CPU offloading)
        print(f"Computing reference log probs...")

        # CRITICAL: Aggressive memory cleanup before moving ref model to GPU
        # This prevents OOM/segfaults with large batches
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.synchronize()

        ref_log_probs = self.compute_ref_log_probs(
            full_encodings["input_ids"],
            full_encodings["attention_mask"],
            labels,
            batch_size=1  # Reduced to 1 to save memory
        )

        # Debug: Check ref log probs range
        print(f"Reference log probs stats:")
        print(f"  min={ref_log_probs.min():.2f}, max={ref_log_probs.max():.2f}")
        print(f"  mean={ref_log_probs.mean():.2f}, std={ref_log_probs.std():.2f}")

        # Clear cache after reference computation
        torch.cuda.empty_cache()
        
        # Training loop with gradient accumulation
        total_stats = {
            "policy_loss": 0.0,
            "kl_div": 0.0,
            "total_loss": 0.0,
            "ratio_mean": 0.0,
            "ratio_std": 0.0,
        }
        
        num_batches = max(1, len(prompts) // self.config.batch_size)
        self.optimizer.zero_grad()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(prompts))
            
            # Get batch
            batch_input_ids = full_encodings["input_ids"][start_idx:end_idx]
            batch_attention_mask = full_encodings["attention_mask"][start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            batch_advantages = advantages[start_idx:end_idx]
            batch_ref_log_probs = ref_log_probs[start_idx:end_idx]
            
            # Compute policy log probs
            policy_log_probs = self.compute_log_probs(
                self.model,
                batch_input_ids,
                batch_attention_mask,
                batch_labels,
                batch_size=1  # Process one at a time for policy model
            )

            # Debug: Check policy log probs
            if batch_idx == 0:  # Only print for first batch to avoid spam
                print(f"Policy log probs stats (batch {batch_idx}):")
                print(f"  min={policy_log_probs.min():.2f}, max={policy_log_probs.max():.2f}")
                print(f"  mean={policy_log_probs.mean():.2f}, std={policy_log_probs.std():.2f}")
                print(f"  Compare to ref: diff_mean={(policy_log_probs - batch_ref_log_probs).mean():.2f}")

            # Debug: Check if gradients are enabled
            if not policy_log_probs.requires_grad:
                print(f"ERROR: policy_log_probs does not require grad!")
                print(f"  Model training mode: {self.model.training}")
                print(f"  Batch size: {end_idx - start_idx}")
                print(f"  Input IDs shape: {batch_input_ids.shape}")
                print(f"  Input IDs device: {batch_input_ids.device}")
                print(f"  Input IDs requires_grad: {batch_input_ids.requires_grad}")
                
                # Check model parameters
                trainable_params = sum(1 for p in self.model.parameters() if p.requires_grad)
                total_params = sum(1 for p in self.model.parameters())
                print(f"  Trainable params: {trainable_params}/{total_params}")
                
                # Check if base model has gradient checkpointing enabled
                if hasattr(self.model, 'is_gradient_checkpointing'):
                    print(f"  Gradient checkpointing: {self.model.is_gradient_checkpointing}")
                
                # Try to trace back through computation
                print(f"  policy_log_probs grad_fn: {policy_log_probs.grad_fn}")
                print(f"  policy_log_probs is_leaf: {policy_log_probs.is_leaf}")
                
                raise RuntimeError(
                    "Policy log probs do not require grad. "
                    "This suggests gradients were detached somewhere in the computation. "
                    "Check that gradient checkpointing is disabled and model is in training mode."
                )
            
            # Compute loss
            loss, stats = self.compute_grpo_loss(
                policy_log_probs,
                batch_ref_log_probs,
                batch_advantages
            )
            
            # Check loss has gradients
            if not loss.requires_grad:
                print(f"WARNING: loss does not require grad!")
                print(f"  policy_log_probs.requires_grad: {policy_log_probs.requires_grad}")
                print(f"  batch_ref_log_probs.requires_grad: {batch_ref_log_probs.requires_grad}")
                print(f"  batch_advantages.requires_grad: {batch_advantages.requires_grad}")
                raise RuntimeError("Loss does not require grad. Cannot backpropagate.")
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            try:
                loss.backward()
            except RuntimeError as e:
                print(f"Error during backward pass: {e}")
                print(f"Loss value: {loss.item()}")
                print(f"Loss requires_grad: {loss.requires_grad}")
                print(f"Batch size: {end_idx - start_idx}")
                raise
            
            # Accumulate stats
            for key in total_stats:
                total_stats[key] += stats.get(key, 0.0)
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Clear cache after optimizer step
                torch.cuda.empty_cache()
        
        # Final optimizer step if there are remaining gradients
        if num_batches % self.config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Average stats
        for key in total_stats:
            total_stats[key] /= max(1, num_batches)
        
        # Clear cache at end
        torch.cuda.empty_cache()
        
        return total_stats
    
    def train(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: List[float],
    ) -> Dict[str, List[float]]:
        """
        Train for multiple epochs on the same batch
        
        Args:
            prompts: List of prompts
            completions: List of completions  
            rewards: List of rewards
        
        Returns:
            history: Training history
        """
        history = {
            "policy_loss": [],
            "kl_div": [],
            "total_loss": [],
        }
        
        for epoch in range(self.config.mini_epochs):
            stats = self.train_step(prompts, completions, rewards)
            
            print(f"  Epoch {epoch + 1}/{self.config.mini_epochs}")
            print(f"    Loss: {stats['total_loss']:.4f}")
            print(f"    Policy Loss: {stats['policy_loss']:.4f}")
            print(f"    KL Div: {stats['kl_div']:.4f}")
            
            for key in history:
                if key in stats:
                    history[key].append(stats[key])
        
        return history

# ============================================================================
# KERNELBENCH INTEGRATION
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result from benchmarking a single kernel"""
    compiles: bool
    correct: bool
    speedup: float
    error_message: str = ""
    execution_time: float = 0.0

def get_kernelbench_problems(level: int = 1) -> List[str]:
    """
    Load problems from KernelBench
    
    Returns list of dicts with:
    - problem_id: Unique identifier
    - reference_code: PyTorch implementation
    """
    import sys
    sys.path.insert(0, '/root/KernelBench')
    from src.dataset import construct_kernelbench_dataset
    problems = construct_kernelbench_dataset(level=level)

    return problems

@app.function(
    image=image,
    gpu="L40S",
    timeout=600,
)
def benchmark_kernel(
    kernel_code: str,
    reference_code: str,
    problem_id: str
) -> BenchmarkResult:
    """
    Benchmark a single generated kernel against reference

    Executes in isolated environment with proper error handling
    Uses KernelBench's eval_kernel_against_ref API

    CRITICAL: This function must be extremely robust since it runs in Modal workers.
    Any uncaught exception during initialization causes Modal to retry 8 times then abort.
    """
    import torch
    import sys
    import traceback

    # Default result in case of early failure
    result = BenchmarkResult(
        compiles=False,
        correct=False,
        speedup=0.0,
        error_message="Initialization failed"
    )

    try:
        # Set up CUDA environment for compilation
        try:
            setup_cuda_environment()
        except Exception as setup_error:
            result.error_message = f"CUDA setup failed: {str(setup_error)[:200]}"
            print(f"Problem {problem_id} CUDA setup error: {setup_error}")
            return result

        # Import KernelBench
        try:
            sys.path.insert(0, '/root/KernelBench')
            from src.eval import eval_kernel_against_ref, get_torch_dtype_from_string
            from src.utils import set_gpu_arch
        except Exception as import_error:
            result.error_message = f"KernelBench import failed: {str(import_error)[:200]}"
            print(f"Problem {problem_id} import error: {import_error}")
            return result

        # Initialize GPU
        try:
            set_gpu_arch(["Ada"])
            device = torch.device("cuda:0")
            # Test CUDA is accessible
            _ = torch.zeros(1, device=device)
        except Exception as gpu_error:
            result.error_message = f"GPU initialization failed: {str(gpu_error)[:200]}"
            print(f"Problem {problem_id} GPU error: {gpu_error}")
            return result

    except Exception as init_error:
        # Catch-all for any initialization error
        result.error_message = f"Benchmark initialization failed: {str(init_error)[:200]}"
        print(f"Problem {problem_id} CRITICAL initialization error: {init_error}")
        print(traceback.format_exc())
        return result

    # Main benchmark execution (GPU and imports already initialized above)
    try:
        # Use KernelBench's proper evaluation API
        # Capture verbose output to get full error messages
        import io
        import contextlib
        
        # Redirect stderr to capture compilation errors
        stderr_capture = io.StringIO()
        from func_timeout import FunctionTimedOut,func_timeout
        try:
            with contextlib.redirect_stderr(stderr_capture):
                eval_result = func_timeout(
                    timeout=240,
                    func= eval_kernel_against_ref,
                    kwargs = {
                    "original_model_src":reference_code,
                    "custom_model_src":kernel_code,
                    "measure_performance":True,
                    "verbose":False,
                    "num_correct_trials":3,
                    "num_perf_trials":5,
                    "device":device,
                    "backend":"cuda",
                    "precision":get_torch_dtype_from_string("fp32")
                    }
                )
        except FunctionTimedOut:
            result.compiles = False
            result.correct = False
            result.error_message = "Evaluation timeout (>4 minutes)"
            print(f"Problem {problem_id} TIMEOUT in KernelBench evaluation")
            return result
        except Exception as eval_error:
            # CRITICAL: Catch segfaults and runtime errors during kernel execution
            # This is EXPECTED in early training - model generates buggy kernels
            stderr_output = stderr_capture.getvalue()
            error_str = str(eval_error)

            # Check if it's a kernel execution error
            if "segmentation fault" in error_str.lower() or "segmentation fault" in stderr_output.lower():
                result.compiles = True  # It compiled but crashed at runtime
                result.correct = False
                result.error_message = "Kernel segfault during execution (buggy generated code)"
                print(f"Problem {problem_id} SEGFAULT during execution (this is normal in early training)")
            elif "cuda" in error_str.lower() or "cuda" in stderr_output.lower():
                result.compiles = True  # Likely compiled but CUDA runtime error
                result.correct = False
                result.error_message = f"CUDA runtime error: {error_str[:200]}"
                print(f"Problem {problem_id} CUDA error: {error_str[:100]}")
            else:
                # Other error during evaluation
                result.compiles = False
                result.correct = False
                result.error_message = f"Evaluation error: {error_str[:200]}"
                print(f"Problem {problem_id} evaluation error: {error_str[:100]}")

            return result
        # Get captured stderr
        stderr_output = stderr_capture.getvalue()
        if stderr_output:
            print(f"Compilation output for problem {problem_id}:")
            print(stderr_output[:1000])  # Print first 1000 chars

        # Map KernelBench result to our BenchmarkResult
        result.compiles = eval_result.compiled
        result.correct = eval_result.correctness

        if eval_result.correctness and eval_result.runtime is not None:
            # Calculate speedup from baseline and runtime
            from scripts.generate_baseline_time import measure_program_time

            baseline_result = measure_program_time(
                ref_arch_name="Reference",
                ref_arch_src=reference_code,
                num_trials=10,
                use_torch_compile=False,
                device=device
            )

            baseline_time = baseline_result.get("mean", None)
            if baseline_time and baseline_time > 0:
                result.speedup = baseline_time / eval_result.runtime
                result.execution_time = eval_result.runtime
            else:
                result.speedup = 0.0
                result.execution_time = eval_result.runtime
        else:
            result.speedup = 0.0

        # Extract error message from metadata if available
        if hasattr(eval_result, 'metadata') and eval_result.metadata:
            if 'cuda_error' in eval_result.metadata:
                result.error_message = eval_result.metadata['cuda_error']
            elif 'other_error' in eval_result.metadata:
                result.error_message = eval_result.metadata['other_error']
            elif 'correctness_error' in eval_result.metadata:
                result.error_message = eval_result.metadata['correctness_error']
        
        # Add stderr output to error message if compilation failed
        if not result.compiles and stderr_output:
            result.error_message = (result.error_message or "Compilation failed") + f"\n{stderr_output[:500]}"

        if not result.compiles:
            result.error_message = result.error_message or "Kernel compilation failed"
            print(f"Problem {problem_id} failed to compile: {result.error_message[:200]}")
        elif not result.correct:
            result.error_message = result.error_message or "Kernel correctness check failed"
            print(f"Problem {problem_id} incorrect: {result.error_message[:200]}")
        else:
            print(f"Problem {problem_id} SUCCESS: speedup={result.speedup:.2f}x")

        return result

    except Exception as e:
        error_trace = traceback.format_exc()
        result.error_message = f"Unexpected error: {str(e)}\n{error_trace[:500]}"
        print(f"Problem {problem_id} exception: {result.error_message}")
        return result

def compute_fast_p(results: List[BenchmarkResult], p: float) -> float:
    """
    Compute fast_p metric: fraction of samples that are correct AND have speedup > p

    This is the official KernelBench metric.
    """
    total = len(results)
    if total == 0:
        return 0.0

    # Count samples that are both correct and faster than threshold
    fast_and_correct = sum(
        1 for r in results
        if r.correct and r.speedup > p
    )

    return fast_and_correct / total

def compute_reward(result: BenchmarkResult, completion: str, config: TrainingConfig) -> float:
    """
    Compute scalar reward with multiple reward levels to provide training signal
    even when kernels don't compile or crash during execution

    Reward structure (cumulative):
    - Level 0: Valid output format (0.05)
    - Level 1: Contains code block (0.10)
    - Level 2: Code has CUDA keywords (0.15)
    - Level 3: Compiles successfully (0.20)
    - Level 4: Passes correctness tests (0.50)
    - Level 5: Performance speedup (0.30 * normalized_speedup)

    Special cases:
    - Segfault/runtime crash: Gets compilation reward but penalty for crash (-0.10)
    - This teaches model that syntax is good but logic is bad

    Maximum possible reward: 1.30
    """
    reward = 0.0
    
    # Level 0: Non-empty output (basic sanity check)
    if completion and len(completion.strip()) > 10:
        reward += 0.05
    else:
        # Penalty for empty/trivial output
        return -0.1
    
    # Level 1: Contains code block
    if "```" in completion:
        reward += 0.10
    
    # Level 2: Check for CUDA-specific keywords (code quality proxy)
    cuda_keywords = ["__global__", "__device__", "<<<", ">>>", "threadIdx", "blockIdx", "cudaMalloc", "__shared__"]
    cuda_keyword_count = sum(1 for kw in cuda_keywords if kw in completion)
    if cuda_keyword_count > 0:
        # Reward proportional to CUDA keyword presence (up to 0.15)
        reward += min(0.15, cuda_keyword_count * 0.03)
    
    # Level 3: Compilation success
    if result.compiles:
        reward += config.reward_compilation

        # Check if it compiled but crashed at runtime (segfault/CUDA error)
        if not result.correct and result.error_message:
            if "segfault" in result.error_message.lower() or "cuda runtime error" in result.error_message.lower():
                # Kernel compiled but crashed - penalize but less than compilation failure
                # This provides gradient: syntax ok (0.20) but logic bad (-0.10) = net 0.10
                reward -= 0.10
    else:
        # Compilation failure - larger penalty to guide away from invalid syntax
        reward -= 0.05

    # Level 4: Correctness
    if result.correct:
        reward += config.reward_correctness

        # Level 5: Speedup bonus (only if correct)
        if result.speedup > 0:
            normalized_speedup = min(result.speedup / config.speedup_target, 2.0)  # Cap at 2x target
            reward += config.reward_speedup * normalized_speedup

    return reward

def extract_last_code(output_string: str, code_language_types: list[str]) -> str | None:
    """
    Extract last code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Find all matches of code blocks
    code_matches = re.finditer(r"```(.*?)```", trimmed, re.DOTALL)
    
    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)
    if matches_list:
        last_match = matches_list[-1]
        code = last_match.group(1).strip()

        # Remove language type headers
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()

        return code
    
    return None

def setup_cuda_environment():
    """
    Verify CUDA environment (should be pre-configured in NVIDIA image)
    
    The NVIDIA CUDA base image comes with CUDA properly installed and configured.
    This function just verifies everything is in place.
    """
    import glob
    
    # CUDA_HOME should already be set by the NVIDIA image
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    
    print(f"CUDA environment verification:")
    print(f"  CUDA_HOME: {cuda_home}")
    
    # Verify critical components exist
    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if os.path.exists(nvcc_path):
        print(f"  ✓ nvcc found at {nvcc_path}")
        # Get nvcc version
        try:
            nvcc_version = subprocess.check_output([nvcc_path, "--version"], text=True)
            version_lines = [l for l in nvcc_version.split('\n') if 'release' in l.lower()]
            if version_lines:
                version_line = version_lines[0].strip()
                print(f"  ✓ {version_line}")
        except:
            pass
    else:
        print(f"  ✗ WARNING: nvcc not found at {nvcc_path}")
        # Set CUDA_HOME if not found
        print(f"  Attempting to locate CUDA...")
        cuda_paths = ["/usr/local/cuda", "/usr/local/cuda-12.4", "/usr/local/cuda-12.1"]
        for path in cuda_paths:
            if os.path.exists(os.path.join(path, "bin", "nvcc")):
                cuda_home = path
                os.environ["CUDA_HOME"] = cuda_home
                print(f"  ✓ Found CUDA at {cuda_home}")
                break
    
    # Ensure CUDA_HOME is set
    if "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = cuda_home
    
    # Verify headers
    cuda_runtime_h = os.path.join(cuda_home, "include", "cuda_runtime.h")
    if os.path.exists(cuda_runtime_h):
        print(f"  ✓ CUDA headers found")
    else:
        print(f"  ✗ WARNING: CUDA headers not found")
    
    # Check build tools
    try:
        g_version = subprocess.check_output(["g++", "--version"], text=True)
        g_first_line = g_version.split('\n')[0]
        print(f"  ✓ g++ found: {g_first_line}")
    except:
        print(f"  ✗ WARNING: g++ not found")
    
    try:
        ninja_version = subprocess.check_output(["ninja", "--version"], text=True).strip()
        print(f"  ✓ ninja found: version {ninja_version}")
    except:
        print(f"  ✗ WARNING: ninja not found")
    
    # Ensure PATH includes CUDA
    cuda_bin = os.path.join(cuda_home, "bin")
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
    
    # Set compilation flags for L40S (Ada architecture)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ["MAX_JOBS"] = "4"
    
    print(f"  ✓ Environment configured for CUDA compilation")
    
    return cuda_home

# ============================================================================
# PROMPT ENGINEERING
# ============================================================================

def create_kernel_optimization_prompt(reference_code: str, iteration: int) -> str:
    """
    Create prompt for kernel optimization
    
    Includes clear instructions and constraints for CUDA generation
    """
    import sys

    sys.path.insert(0, '/root/KernelBench')
    from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
    prompt = prompt_generate_custom_cuda_from_prompt_template(
        ref_arch_src=reference_code,
    )
    return prompt

def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def print_gpu_memory_summary(device_id: int = 1):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3
        print(f"GPU {device_id} Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_allocated:.2f}GB")

def test_cuda_compilation():
    """
    Test that CUDA compilation works before starting training
    
    Returns True if compilation works, False otherwise
    This is a best-effort test - failure doesn't necessarily mean training will fail
    """
    print("\nTesting CUDA compilation environment...")
    
    try:
        import torch
        import subprocess
        from torch.utils.cpp_extension import load_inline
        
        # Simple test CUDA kernel
        cuda_source = """
        __global__ void add_kernel(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """
        
        cpp_source = """
        torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
            auto c = torch::zeros_like(a);
            int n = a.numel();
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            add_kernel<<<blocks, threads>>>(
                a.data_ptr<float>(),
                b.data_ptr<float>(),
                c.data_ptr<float>(),
                n
            );
            return c;
        }
        """
        
        print("  Attempting to compile test kernel...")
        print("  (This may take 30-60 seconds on first run...)")
        
        module = load_inline(
            name='test_add',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['add_cuda'],
            verbose=False,  # Reduce noise
            with_cuda=True,
            extra_cuda_cflags=['-O2'],
        )
        
        # Test execution
        a = torch.randn(100, device='cuda')
        b = torch.randn(100, device='cuda')
        c = module.add_cuda(a, b)
        
        # Verify result
        expected = a + b
        if torch.allclose(c, expected, rtol=1e-5):
            print("  ✓ Test kernel compiled and executed successfully!")
            print("  ✓ CUDA compilation environment is working properly")
            return True
        else:
            print("  ✗ Test kernel executed but results incorrect")
            print("  ⚠️  This may indicate numerical issues")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Ninja build failed with exit code: {e.returncode}")
        print(f"  ⚠️  This test uses torch.utils.cpp_extension")
        print(f"  ⚠️  KernelBench may use a different compilation method")
        print(f"  ➜  Training will continue - actual compilations may still work")
        return False
        
    except RuntimeError as e:
        error_str = str(e)
        print(f"  ✗ Test compilation failed: {error_str[:200]}")
        if "ninja" in error_str.lower():
            print(f"  ⚠️  Ninja build system issue detected")
            print(f"  ⚠️  This may not affect KernelBench compilation")
        print(f"  ➜  Training will continue - watch for compilation errors")
        return False
        
    except Exception as e:
        print(f"  ✗ Unexpected error: {type(e).__name__}: {str(e)[:200]}")
        print(f"  ➜  Training will continue")
        import traceback
        print(f"  Debug info: {traceback.format_exc()[:500]}")
        return False

# ============================================================================
# ONLINE RL TRAINING LOOP
# ============================================================================

@app.function(
    gpu="L40S:2",  # 2 GPUs: one for generation (vLLM), one for training
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={
        "/checkpoints": checkpoint_vol,
        "/results": results_vol
    },
    image=image
)
def train_online_grpo():
    """
    Main online GRPO training loop
    
    For each iteration:
    1. Sample problems from KernelBench
    2. Generate multiple kernel candidates with current model
    3. Benchmark all candidates in parallel
    4. Compute rewards from benchmark results
    5. Update model with GRPO on this batch
    6. Save checkpoint and metrics
    7. Repeat with improved model
    """
    # Set CUDA memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Enable synchronous CUDA execution for better error messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Set up CUDA environment
    setup_cuda_environment()
    
    # Test CUDA compilation before starting
    test_passed = test_cuda_compilation()
    if test_passed:
        print("\n" + "="*80)
        print("✓ CUDA compilation test PASSED")
        print("✓ Environment is properly configured")
        print("✓ Ready to start training")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("⚠️  CUDA compilation test did not pass")
        print("")
        print("This is not necessarily a problem. The test uses torch.utils.cpp_extension")
        print("which can be finicky, but KernelBench uses its own compilation pipeline.")
        print("")
        print("Training will proceed. Watch Phase 2 (Benchmarking) to see if kernels")
        print("compile successfully - that's the real test!")
        print("="*80 + "\n")
    
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    import copy
    
    config = TrainingConfig()
    
    # Initialize wandb
    wandb.init(
        project="cuda-kernel-grpo",
        config=config.__dict__,
        name=f"online_grpo_custom_{config.model_name.split('/')[-1]}"
    )
    
    # Load base model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 1},  # Put on GPU 1 (GPU 0 for vLLM)
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    # CRITICAL: Get actual vocab size from model config
    # tokenizer.vocab_size may not include all special tokens
    actual_vocab_size = model.config.vocab_size
    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"Model config vocab_size: {actual_vocab_size}")
    print(f"Tokenizer length: {len(tokenizer)}")

    # Check embedding layer for NaN
    print("\nChecking embedding layer for NaN...")
    if hasattr(model, 'get_input_embeddings'):
        embed_layer = model.get_input_embeddings()
        if hasattr(embed_layer, 'weight'):
            embed_weight = embed_layer.weight
            if torch.isnan(embed_weight).any():
                print(f"ERROR: Found NaN in embedding layer!")
                raise RuntimeError("Embedding layer contains NaN")
            print(f"✓ Embedding layer OK: shape={embed_weight.shape}, dtype={embed_weight.dtype}")
            print(f"  Embedding stats: min={embed_weight.min():.4f}, max={embed_weight.max():.4f}")
    
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA for efficient training
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Important for gradient flow
        inference_mode=False,  # Must be False for training
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load checkpoint if resuming training
    if config.load_checkpoint is not None:
        checkpoint_iter = config.load_checkpoint
        checkpoint_path = f"/checkpoints/iteration_{checkpoint_iter}"
        if os.path.exists(checkpoint_path):
            print(f"\n{'='*60}")
            print(f"LOADING CHECKPOINT from iteration {checkpoint_iter}")
            print(f"{'='*60}\n")
            # Use safetensors for loading (PEFT saves in safetensors format by default)
            from safetensors.torch import load_file
            adapter_file = f"{checkpoint_path}/adapter_model.safetensors"
            if os.path.exists(adapter_file):
                state_dict = load_file(adapter_file)
                model.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded adapter weights from {adapter_file}")
            else:
                print(f"⚠️  No adapter weights found at {adapter_file}, will train from scratch")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
    # After applying LoRA and before training, add this test:
    print("\nChecking for NaN in model parameters...")
    nan_params = []
    extreme_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
        # Check for extreme values that could cause NaN
        if torch.isinf(param).any():
            extreme_params.append((name, "inf"))
        elif param.abs().max() > 1e4:
            extreme_params.append((name, f"large:{param.abs().max().item():.2e}"))

    if nan_params:
        print(f"ERROR: Found NaN in {len(nan_params)} parameters:")
        for name in nan_params[:10]:  # Show first 10
            print(f"  - {name}")
        raise RuntimeError("Model has NaN parameters before training!")

    if extreme_params:
        print(f"WARNING: Found {len(extreme_params)} parameters with extreme values:")
        for name, val in extreme_params[:5]:
            print(f"  - {name}: {val}")
        # Reinitialize LoRA adapters with smaller scale
        print("  Re-initializing LoRA adapters with smaller scale...")
        for name, module in model.named_modules():
            if 'lora_A' in name and hasattr(module, 'weight'):
                if isinstance(module.weight, torch.Tensor):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif 'lora_B' in name and hasattr(module, 'weight'):
                if isinstance(module.weight, torch.Tensor):
                    torch.nn.init.zeros_(module.weight)
    else:
        print("✓ All model parameters are valid (no NaN, no extreme values)")
        # CRITICAL: Disable gradient checkpointing for LoRA training
    # It causes NaN with mixed precision
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        print("✓ Gradient checkpointing DISABLED for GRPO training (prevents NaN)")
    elif hasattr(model, 'is_gradient_checkpointing'):
        print(f"WARNING: Cannot disable gradient checkpointing. Current state: {model.is_gradient_checkpointing}")
    
    # Verify gradient checkpointing is actually disabled
    if hasattr(model.base_model, 'model'):
        base = model.base_model.model
        if hasattr(base, 'gradient_checkpointing'):
            if base.gradient_checkpointing:
                print("ERROR: Gradient checkpointing is still enabled on base model!")
                base.gradient_checkpointing = False
                print("  Force-disabled gradient checkpointing")
    
    # Verify LoRA is configured correctly
    print(f"LoRA configuration:")
    print(f"  inference_mode: {peft_config.inference_mode}")
    print(f"  r: {peft_config.r}")
    print(f"  alpha: {peft_config.lora_alpha}")
    
    ## Gradient checkpointing can interfere with RL training
    ## Disable it for now to ensure proper gradient flow
    #if hasattr(model, 'gradient_checkpointing_disable'):
    #    model.gradient_checkpointing_disable()
    #    print("Gradient checkpointing disabled for GRPO training")
        
    
    # Verify model has trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Set model to training mode explicitly
    model.train()
    print(f"Model training mode: {model.training}")
    
    print_gpu_memory_summary(1)
    
    # Create reference model on CPU (memory efficient)
    print("Creating reference model on CPU...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},  # Keep on CPU
        trust_remote_code=True
    )
    ref_peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    ref_model = get_peft_model(ref_model, ref_peft_config)
    
    # Copy weights from policy model to reference model
    print("Copying weights to reference model...")
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    print(f"Reference model on: CPU (will be moved to GPU only during inference)")
    print(f"Policy model on: GPU 1")
    print_gpu_memory_summary(1)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Create GRPO trainer
    device = torch.device("cuda:1")
    grpo_trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
        optimizer=optimizer,
        device=device,
        vocab_size=actual_vocab_size  # Pass vocab size for token validation
    )
    # TEST: Verify model can do forward pass without NaN
    print("\nTesting forward pass for NaN detection...")
    test_input = tokenizer("Hello world", return_tensors="pt").to(device)
    model.eval()  # Temporarily set to eval for test
    with torch.no_grad():
        test_output = model(**test_input)
        test_logits = test_output.logits
        
        if torch.isnan(test_logits).any():
            print("❌ ERROR: NaN in logits during test forward pass!")
            print("  This indicates model weights or configuration issue")
            raise RuntimeError("Model produces NaN in forward pass")
        elif torch.isinf(test_logits).any():
            print("⚠️  WARNING: Inf in logits during test forward pass")
            print(f"  Logits range: [{test_logits.min():.2f}, {test_logits.max():.2f}]")
        else:
            print(f"✓ Test forward pass OK. Logits range: [{test_logits.min():.2f}, {test_logits.max():.2f}]")
    
    model.train()  # Back to training mode
    del test_input, test_output, test_logits
    torch.cuda.empty_cache()
    # Start vLLM server on GPU 0 for fast generation
    print("Starting vLLM server...")
    import time
    import requests

    vllm_env = os.environ.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
    vllm_stdout_file = open("/tmp/vllm_stdout.log", "w")
    vllm_stderr_file = open("/tmp/vllm_stderr.log", "w")

    vllm_process = subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.model_name,
        "--gpu-memory-utilization", "0.7",
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--max-num-seqs", "4",
        "--max-num-batched-tokens", "8192",
        "--disable-log-requests",
        "--enforce-eager",
    ],
    env=vllm_env,
    stdout=vllm_stdout_file,
    stderr=vllm_stderr_file,
    text=True
    )

    # Wait for vLLM to be ready with health check
    print("Waiting for vLLM server to be ready...")
    max_wait = 180  # 3 minutes
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < max_wait:
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                server_ready = True
                print("vLLM server is ready!")
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass

        # Check if process died
        if vllm_process.poll() is not None:
            print(f"vLLM process died! Exit code: {vllm_process.returncode}")
            print(f"Check logs at /tmp/vllm_stdout.log and /tmp/vllm_stderr.log")
            raise RuntimeError("vLLM server failed to start")

        time.sleep(2)

    if not server_ready:
        vllm_process.terminate()
        print(f"vLLM server failed to start within {max_wait}s")
        raise RuntimeError("vLLM server startup timeout")
    
    # Load KernelBench problems
    print("Loading KernelBench problems...")
    curr_level_dataset = get_kernelbench_problems(level=1)
    
    # Training loop
    for iteration in range(config.num_training_iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{config.num_training_iterations}")
        print(f"{'='*80}\n")
        
        # Sample problems for this iteration
        problem_indices = np.random.choice(
            curr_level_dataset,
            size=min(config.problems_per_iter, len(curr_level_dataset)),
            replace=False
        ).tolist()

        # Phase 1: GENERATION
        print("Phase 1: Generating kernel candidates...")
        all_completions = []
        all_prompts = []
        all_references = []
        all_problem_ids = []
        
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy",
            timeout=120.0,
            max_retries=2
        )

        for problem in problem_indices:
            problem_name = os.path.basename(problem)
            ref_arch_src = read_file(problem)
            problem_number = int(problem_name.split("_")[0])
            prompt = create_kernel_optimization_prompt(
                ref_arch_src,
                iteration
            )

            # Generate multiple candidates
            for gen_idx in range(config.num_generations_per_iter):
                print(f"Generating candidate {gen_idx + 1}/{config.num_generations_per_iter} for problem {problem_number}...")
                try:
                    response = client.completions.create(
                        model=config.model_name,
                        prompt=prompt,
                        max_tokens=config.max_completion_length,
                        temperature=config.temperature,
                        n=1
                    )
                    
                    completion = response.choices[0].text
                    code = extract_last_code(completion, ["python", "cpp"])
                    if code is None:
                        code = ""
                        print(f"Warning: No code extracted for problem {problem_number} (candidate {gen_idx + 1})")
                    else:
                        print(f"Generated code for problem {problem_number} (candidate {gen_idx + 1})\n")
                    
                    all_completions.append(completion)  # Store full completion for training
                    all_prompts.append(prompt)
                    all_references.append(ref_arch_src)
                    all_problem_ids.append(problem_number)
                    
                except Exception as e:
                    print(f"Generation error for problem {problem_number} (candidate {gen_idx + 1}): {e}")
                    # Add empty completion to maintain structure
                    all_completions.append("")
                    all_prompts.append(prompt)
                    all_references.append(ref_arch_src)
                    all_problem_ids.append(problem_number)
        
        print(f"Generated {len(all_completions)} kernel candidates")
        
        # Clear CUDA cache before benchmarking
        torch.cuda.empty_cache()
        
        # Phase 2: BENCHMARKING
        print("Phase 2: Benchmarking kernels in parallel...")
        
        # Aggressively clear CUDA cache before benchmarking
        print("Clearing CUDA memory before benchmarking...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print_gpu_memory_summary(1)
        
        # Extract code for benchmarking
        all_codes = [extract_last_code(comp, ["python", "cpp"]) or "" for comp in all_completions]

        # CRITICAL: Batch benchmark execution to prevent Modal worker overload
        # Processing all 30 benchmarks at once can cause worker initialization failures
        print(f"Benchmarking {len(all_codes)} kernels in batches...")
        benchmark_results = []
        batch_size = 10  # Process 10 benchmarks at a time

        for batch_idx in range(0, len(all_codes), batch_size):
            batch_end = min(batch_idx + batch_size, len(all_codes))
            batch_codes = all_codes[batch_idx:batch_end]
            batch_refs = all_references[batch_idx:batch_end]
            batch_ids = all_problem_ids[batch_idx:batch_end]

            print(f"  Batch {batch_idx//batch_size + 1}/{(len(all_codes) + batch_size - 1)//batch_size}: problems {batch_idx} to {batch_end-1}")

            try:
                batch_results = list(benchmark_kernel.starmap(
                    zip(batch_codes, batch_refs, batch_ids)
                ))
                benchmark_results.extend(batch_results)
                print(f"  Batch completed: {len(batch_results)} benchmarks")
            except Exception as batch_error:
                print(f"  ERROR: Batch failed: {batch_error}")
                print(f"  Creating fallback results for batch...")
                # Create fallback results for failed batch
                for code, ref, pid in zip(batch_codes, batch_refs, batch_ids):
                    fallback = BenchmarkResult(
                        compiles=False,
                        correct=False,
                        speedup=0.0,
                        error_message=f"Batch benchmark failed: {str(batch_error)[:200]}"
                    )
                    benchmark_results.append(fallback)
                print(f"  Added {len(batch_codes)} fallback results")

            # Small delay between batches to avoid overwhelming Modal
            import time
            time.sleep(2)
        
        # Clear cache after benchmarking
        torch.cuda.empty_cache()
        
        rewards = [
        compute_reward(result, completion, config) 
            for result, completion in zip(benchmark_results, all_completions)
        ]

        # Debug rewards
        print(f"Rewards statistics:")
        print(f"  Mean: {np.mean(rewards):.4f}")
        print(f"  Std: {np.std(rewards):.4f}")
        print(f"  Min: {np.min(rewards):.4f}, Max: {np.max(rewards):.4f}")
        print(f"  Unique rewards: {len(set(rewards))}")
        print(f"  Zero rewards: {sum(1 for r in rewards if abs(r) < 0.01)}/{len(rewards)}")
        print(f"  Negative rewards: {sum(1 for r in rewards if r < 0)}/{len(rewards)}")

        # Check for all-zero or all-identical rewards
        if len(set(rewards)) == 1:
            print("WARNING: All rewards are identical!")
            if all(r == 0 for r in rewards):
                print("  All rewards are ZERO - this is problematic!")
                print("  The model will receive no learning signal")
            print("  Adding small random noise to break ties...")
            # Add noise proportional to reward scale
            noise_scale = max(0.01, np.std(rewards) * 0.1)
            rewards = [r + np.random.normal(0, noise_scale) for r in rewards]
            print(f"  After noise - Std: {np.std(rewards):.4f}")

        # Log statistics
        num_compiled = sum(1 for r in benchmark_results if r.compiles)
        num_correct = sum(1 for r in benchmark_results if r.correct)
        speedups = [r.speedup for r in benchmark_results if r.correct]
        avg_speedup = np.mean(speedups) if speedups else 0.0

        # Compute fast_p metrics for different thresholds (KernelBench standard)
        fast_p_thresholds = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
        fast_p_scores = {
            f"fast_p_{p}": compute_fast_p(benchmark_results, p)
            for p in fast_p_thresholds
        }

        print(f"Benchmark Results:")
        print(f"  Compiled: {num_compiled}/{len(all_completions)} ({100*num_compiled/len(all_completions):.1f}%)")
        print(f"  Correct: {num_correct}/{len(all_completions)} ({100*num_correct/len(all_completions):.1f}%)")
        print(f"  Avg Speedup (correct only): {avg_speedup:.2f}x")
        print(f"  Avg Reward: {np.mean(rewards):.3f}")
        print(f"  Fast_p Metrics:")
        for p in fast_p_thresholds:
            print(f"    fast_p@{p}: {fast_p_scores[f'fast_p_{p}']:.3f}")
        
        # Warning if compilation rate is very low
        if num_compiled == 0:
            print("\n" + "!"*80)
            print("WARNING: No kernels compiled successfully in this iteration!")
            print("This is normal in early training, but check:")
            print("  1. Generated code format (should be in code blocks)")
            print("  2. CUDA compilation environment (run test_cuda_compilation)")
            print("  3. Model prompt (ensure it's asking for CUDA code)")
            print("!"*80 + "\n")
        elif num_compiled < len(all_completions) * 0.1:  # Less than 10% compile
            print(f"\nNote: Low compilation rate ({100*num_compiled/len(all_completions):.1f}%)")
            print("This may improve as the model learns from GRPO feedback.\n")

        wandb.log({
            "iteration": iteration,
            "compilation_rate": num_compiled / len(all_completions),
            "correctness_rate": num_correct / len(all_completions),
            "avg_speedup": avg_speedup,
            "avg_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            **fast_p_scores,
        })
        
        # Phase 3: GRPO TRAINING
        print("Phase 3: Training model with custom GRPO...")
        print_gpu_memory_summary(1)
        
        # Train with custom GRPO trainer
        training_history = grpo_trainer.train(
            prompts=all_prompts,
            completions=all_completions,
            rewards=rewards
        )
        
        print_gpu_memory_summary(1)
        
        # Log training stats
        wandb.log({
            "iteration": iteration,
            "train/policy_loss": np.mean(training_history["policy_loss"]),
            "train/kl_div": np.mean(training_history["kl_div"]),
            "train/total_loss": np.mean(training_history["total_loss"]),
        })
        
        # Update reference model to current policy periodically
        if (iteration + 1) % 5 == 0:
            print("Updating reference model...")
            # Move ref model to CPU if it's on GPU
            if next(ref_model.parameters()).device.type == 'cuda':
                ref_model = ref_model.to('cpu')
            
            # Copy state dict
            ref_model.load_state_dict(model.state_dict())
            ref_model.eval()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            print("Reference model updated and moved to CPU")
        
        # Phase 4: CHECKPOINTING
        if (iteration + 1) % config.save_every_n_iters == 0:
            checkpoint_path = f"/checkpoints/iteration_{iteration+1}"
            print(f"Saving checkpoint to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            checkpoint_vol.commit()
            
            # Save metrics
            metrics = {
                "iteration": iteration + 1,
                "compilation_rate": num_compiled / len(all_completions),
                "correctness_rate": num_correct / len(all_completions),
                "avg_speedup": avg_speedup,
                "avg_reward": np.mean(rewards),
                "speedup_distribution": speedups[:100],
                "fast_p_scores": fast_p_scores,
                "training_history": {
                    k: [float(v) for v in vals]
                    for k, vals in training_history.items()
                }
            }

            with open(f"/results/metrics_iter_{iteration+1}.json", "w") as f:
                json.dump(metrics, f, indent=2)
            results_vol.commit()
    
    # Final save
    print("\nTraining complete! Saving final model...")
    model.save_pretrained("/checkpoints/final_model")
    tokenizer.save_pretrained("/checkpoints/final_model")
    checkpoint_vol.commit()
    
    # Cleanup
    vllm_process.terminate()
    try:
        vllm_stdout_file.close()
        vllm_stderr_file.close()
    except:
        pass
    wandb.finish()

# ============================================================================
# INFERENCE ENDPOINT
# ============================================================================

@app.function(
    gpu="L40S",
    image=image,
    volumes={"/checkpoints": checkpoint_vol},
)
def optimize_kernel(
    pytorch_code: str,
    checkpoint_path: str = "/checkpoints/final_model"
) -> Dict[str, str]:
    """
    Inference endpoint: optimize a PyTorch kernel with trained model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Load trained model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # Generate optimized kernel
    prompt = create_kernel_optimization_prompt(pytorch_code, 0)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=768,
        temperature=0.7,
        do_sample=True
    )
    
    optimized_code = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:])
    
    return {
        "original_code": pytorch_code,
        "optimized_code": optimized_code,
        "prompt": prompt
    }

# ============================================================================
# ENTRY POINTS
# ============================================================================

@app.local_entrypoint()
def main(action: str = "train"):
    """
    Entry point for the application
    
    Usage:
        modal run cuda_kernel_online_grpo_custom.py --action train
        modal run cuda_kernel_online_grpo_custom.py --action optimize
    """
    if action == "train":
        print("Starting online GRPO training with custom implementation...")
        train_online_grpo.remote()
    
    elif action == "optimize":
        # Example optimization
        example_code = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return x + y

def get_inputs():
    x = torch.randn(1024, 1024).cuda()
    y = torch.randn(1024, 1024).cuda()
    return [x, y]
"""
        result = optimize_kernel.remote(example_code)
        print("Optimized kernel:")
        print(result['optimized_code'])
    
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: train, optimize")

if __name__ == "__main__":
    main()