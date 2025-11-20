"""
Online GRPO Training for CUDA Kernel Optimization - FIXED VERSION
==================================================================

Architecture: Generate kernels → Benchmark with KernelBench → Train with GRPO → Repeat

This implements true online RL where the model improves between generation cycles,
with fixes for numerical stability, memory management, and training efficiency.
"""

import modal
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import json
import subprocess
import tempfile
import os
import re
import numpy as np
import time
import traceback
import gc
from collections import defaultdict, deque
from torch.cuda.amp import GradScaler, autocast

# ============================================================================
# MODAL SETUP
# ============================================================================

app = modal.App("cuda-kernel-online-grpo-single-turn-reward-fixed")

# Use official NVIDIA CUDA image with development tools
cuda_version = "12.4.0"
flavor = "devel"
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
        "bitsandbytes",
        
    )
    .run_commands(
        "git clone https://github.com/ScalingIntelligence/KernelBench /root/KernelBench",
        "cd /root/KernelBench && pip install -e ."
    )
)

# Persistent volumes
checkpoint_vol = modal.Volume.from_name("grpov3-checkpoints", create_if_missing=True)
results_vol = modal.Volume.from_name("grpov3-results", create_if_missing=True)

# ============================================================================
# CONFIGURATION - IMPROVED
# ============================================================================

@dataclass
class TrainingConfig:
    """Optimized Online GRPO training configuration"""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    # Online RL parameters - IMPROVED
    num_training_iterations: int = 100
    num_generations_per_iter: int = 4
    problems_per_iter: int = 8  # Reduced for better quality
    
    # GRPO parameters - IMPROVED
    learning_rate: float = 1e-5  # More conservative
    grpo_epsilon: float = 0.2
    grpo_beta: float = 0.01  # KL penalty
    
    # Generation parameters
    max_prompt_length: int = 1024
    max_completion_length: int = 1536
    temperature: float = 0.8
    
    # Training parameters - IMPROVED
    mini_epochs: int = 1
    batch_size: int = 1  # Increased from 1
    gradient_accumulation_steps: int = 4 
    max_grad_norm: float = 1.0
    
    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05  # Small dropout for regularization
    
    # Checkpointing
    save_every_n_iters: int = 10
    load_checkpoint: Optional[int] = None
    
    # Reward weights - KernelBench paper structure
    # S = 0.3 · 1{correct} + (T_baseline / T_kernel) · 1{correct}
    reward_correct_base: float = 0.3  # Base reward for correctness
    reward_compilation_fail: float = -1.0  # Penalty for compilation failure
    reward_runtime_fail: float = -0.5  # Penalty for runtime errors
    reward_incorrect: float = -0.2  # Penalty for incorrect output
    
    # Memory and stability - NEW
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True  # Disabled for LoRA
    ref_model_update_freq: int = 3  # Update more frequently
    offload_ref_model: bool = True  # Move ref model to CPU

    # Problem tracking - NEW
    use_curriculum: bool = True
    problem_replay_buffer_size: int = 100
    min_problem_appearances: int = 3  # Each problem trained at least 3 times

    # Model quantization - NEW
    use_8bit: bool = False  # Use 8-bit quantization
# ============================================================================
# TRAINING STATE MANAGER - NEW
# ============================================================================

class TrainingStateManager:
    """Manages training state, problem selection, and curriculum learning"""
    
    def __init__(self, num_problems: int):
        self.problem_stats = defaultdict(lambda: {
            'appearances': 0,
            'compile_rate': 0.0,
            'correct_rate': 0.0,
            'avg_speedup': 0.0,
            'history': deque(maxlen=10)
        })
        self.iteration = 0
        self.all_problems = list(range(num_problems))
        self.problem_queue = deque(self.all_problems)
        np.random.shuffle(self.problem_queue)
        
    def select_problems(self, n: int, use_curriculum: bool = True) -> List[int]:
        """Select problems with curriculum learning and balanced coverage"""
        if use_curriculum and self.iteration < 20:
            # Early training: focus on easier problems
            candidates = [p for p in self.all_problems 
                         if self.problem_stats[p]['appearances'] < 2]
            if len(candidates) < n:
                candidates = self.all_problems
        else:
            # Later training: prioritize underperforming problems
            stats = [(p, self.problem_stats[p]['correct_rate']) 
                    for p in self.all_problems]
            stats.sort(key=lambda x: x[1])  # Sort by correctness rate
            candidates = [p for p, _ in stats[:len(stats)//2]]  # Bottom 50%
            
        # Ensure minimum coverage
        needed = [p for p in self.all_problems 
                 if self.problem_stats[p]['appearances'] < 3]
        
        selected = []
        # First add needed problems
        for p in needed[:n//2]:
            selected.append(p)
        
        # Then add from candidates
        remaining = n - len(selected)
        if remaining > 0:
            selected.extend(np.random.choice(
                candidates, 
                size=min(remaining, len(candidates)),
                replace=False
            ).tolist())
            
        return selected[:n]
    
    def update_stats(self, problem_id: int, result: 'BenchmarkResult'):
        """Update problem statistics"""
        stats = self.problem_stats[problem_id]
        stats['appearances'] += 1
        
        # Update rolling averages
        alpha = 0.3  # Exponential moving average factor
        stats['compile_rate'] = (1-alpha) * stats['compile_rate'] + alpha * float(result.compiles)
        stats['correct_rate'] = (1-alpha) * stats['correct_rate'] + alpha * float(result.correct)
        if result.correct and result.speedup > 0:
            stats['avg_speedup'] = (1-alpha) * stats['avg_speedup'] + alpha * result.speedup
            
        stats['history'].append({
            'iteration': self.iteration,
            'compiled': result.compiles,
            'correct': result.correct,
            'speedup': result.speedup
        })

# ============================================================================
# IMPROVED GRPO TRAINER
# ============================================================================

class GRPOTrainer:
    """
    Improved Group Relative Policy Optimization Trainer with better stability
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        tokenizer,
        config: TrainingConfig,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        vocab_size: int
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.vocab_size = vocab_size
        
        # Keep reference model on same GPU for consistency
        self.ref_model = self.ref_model.to(device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Statistics tracking
        self.training_stats = defaultdict(list)

    def compute_log_probs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 2
    ) -> torch.Tensor:
        """
        Compute log probabilities with CPU offloading support
        """
        all_log_probs = []
        num_samples = input_ids.shape[0]
        
        # Determine if this is ref model (might be on CPU)
        is_ref_model = not model.training
        model_device = next(model.parameters()).device
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            
            # Move batch to model device
            batch_input_ids = input_ids[i:end_idx].to(model_device)
            batch_attention_mask = attention_mask[i:end_idx].to(model_device)
            batch_labels = labels[i:end_idx].to(model_device)
            
            # Validate inputs
            batch_input_ids = torch.clamp(batch_input_ids, 0, self.vocab_size - 1)
            
            # Forward pass
            with torch.no_grad() if is_ref_model else torch.enable_grad():
                if self.config.use_mixed_precision and not is_ref_model:
                    with autocast(dtype=torch.float16):
                        outputs = model(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                        )
                        logits = outputs.logits
                else:
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                    )
                    logits = outputs.logits.float()
            
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_labels[..., 1:].contiguous()
            
            # Numerical stability: clamp logits before softmax
            shift_logits = torch.clamp(shift_logits, -100, 100)
            
            # Compute log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Safe gather with bounds checking
            shift_labels_safe = shift_labels.clone()
            shift_labels_safe[shift_labels == -100] = 0
            shift_labels_safe = torch.clamp(shift_labels_safe, 0, self.vocab_size - 1)
            
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=shift_labels_safe.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask and normalize
            mask = (shift_labels != -100).float()
            token_log_probs = token_log_probs * mask
            
            # Average log prob per token (not sum) for stability
            num_tokens = mask.sum(dim=-1).clamp(min=1)
            sequence_log_probs = token_log_probs.sum(dim=-1) / num_tokens
            
            # Move result back to training device
            all_log_probs.append(sequence_log_probs.to(self.device))
            
            # Clear cache after each batch
            if model_device.type == 'cuda':
                torch.cuda.empty_cache()
            
        return torch.cat(all_log_probs, dim=0)
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        num_generations: int
    ) -> torch.Tensor:
        """
        Compute advantages with improved normalization
        """
        num_prompts = len(rewards) // num_generations
        rewards_grouped = rewards.view(num_prompts, num_generations)
        
        # Robust standardization
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True).clamp(min=0.01)
        
        # Add small noise if all rewards identical
        if (group_std < 0.01).any():
            rewards_grouped = rewards_grouped + torch.randn_like(rewards_grouped) * 0.01
            group_mean = rewards_grouped.mean(dim=1, keepdim=True)
            group_std = rewards_grouped.std(dim=1, keepdim=True).clamp(min=0.01)
        
        advantages = (rewards_grouped - group_mean) / group_std
        advantages = torch.clamp(advantages, -5.0, 5.0)
        
        return advantages.view(-1)
    
    def compute_grpo_loss(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Improved GRPO loss computation with better clipping
        """
        # Compute log ratio with safety checks
        log_ratio = policy_log_probs - ref_log_probs
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)  # Prevent extreme values
        
        # Check for divergence
        if log_ratio.abs().mean() > 3.0:
            print(f"Warning: Large policy divergence detected: {log_ratio.abs().mean():.2f}")
        
        ratio = torch.exp(log_ratio)
        
        # PPO-style clipping
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.grpo_epsilon,
            1.0 + self.config.grpo_epsilon
        )
        
        # Policy loss
        policy_loss_unclipped = -advantages * ratio
        policy_loss_clipped = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()
        
        # KL penalty (more stable computation)
        with torch.no_grad():
            approx_kl = (ratio - 1.0 - log_ratio).mean()
        
        kl_penalty = self.config.grpo_beta * approx_kl
        
        # Total loss with gradient-friendly KL
        loss = policy_loss + kl_penalty
        
        # Clip final loss to prevent explosions
        loss = torch.clamp(loss, -10.0, 10.0)
        
        stats = {
            "policy_loss": policy_loss.item(),
            "kl_div": approx_kl.item(),
            "total_loss": loss.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_max": ratio.max().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item() if advantages.numel() > 1 else 0.0,
        }
        
        return loss, stats
    
    def train_step(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: List[float],
    ) -> Dict[str, float]:
        """
        Improved training step with better batching and stability
        """
        self.model.train()
        
        # Clear gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        # Tokenize inputs efficiently
        full_texts = [p + c for p, c in zip(prompts, completions)]
        
        # Batch tokenization
        prompt_encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
            return_tensors="pt"
        )
        
        full_encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length + self.config.max_completion_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = full_encodings["input_ids"].to(self.device)
        attention_mask = full_encodings["attention_mask"].to(self.device)
        
        # Create labels
        labels = input_ids.clone()
        prompt_lengths = prompt_encodings["attention_mask"].sum(dim=1)
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100
        
        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        # Compute advantages
        advantages = self.compute_advantages(
            rewards_tensor,
            self.config.num_generations_per_iter
        )
        
        # Compute reference log probs (no gradients needed)
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(
                self.ref_model,
                input_ids,
                attention_mask,
                labels,
                batch_size=self.config.batch_size
            )
            ref_log_probs = ref_log_probs.to(self.device) 
        # Training with gradient accumulation
        total_loss = 0.0
        num_batches = (len(prompts) + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(prompts))
            
            # Get batch
            batch_input_ids = input_ids[start_idx:end_idx]
            batch_attention_mask = attention_mask[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            batch_advantages = advantages[start_idx:end_idx]
            batch_ref_log_probs = ref_log_probs[start_idx:end_idx]
            
            # Compute policy log probs with gradients
            if self.config.use_mixed_precision:
                with autocast():
                    policy_log_probs = self.compute_log_probs(
                        self.model,
                        batch_input_ids,
                        batch_attention_mask,
                        batch_labels,
                        batch_size=self.config.batch_size
                    )
                    
                    # Compute loss
                    loss, stats = self.compute_grpo_loss(
                        policy_log_probs,
                        batch_ref_log_probs,
                        batch_advantages
                    )
                    
                    loss = loss / self.config.gradient_accumulation_steps
                    
                # Backward with mixed precision
                self.scaler.scale(loss).backward()
            else:
                policy_log_probs = self.compute_log_probs(
                    self.model,
                    batch_input_ids,
                    batch_attention_mask,
                    batch_labels,
                    batch_size=self.config.batch_size
                )
                
                loss, stats = self.compute_grpo_loss(
                    policy_log_probs,
                    batch_ref_log_probs,
                    batch_advantages
                )
                
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            torch.cuda.empty_cache()
            gc.collect()
        
        # Final gradient step if needed
        if num_batches % self.config.gradient_accumulation_steps != 0:
            if self.config.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        del input_ids, attention_mask, labels, ref_log_probs
        torch.cuda.empty_cache()
        gc.collect()
        return {"total_loss": total_loss / max(1, num_batches), **stats}

# ============================================================================
# IMPROVED BENCHMARKING
# ============================================================================

@dataclass
class BenchmarkResult:
    """Enhanced benchmark result with more detail"""
    compiles: bool
    correct: bool
    speedup: float
    error_message: str = ""
    execution_time: float = 0.0
    error_type: str = ""  # compilation, runtime, correctness
def compute_fast_p(results: List[BenchmarkResult], p: float, total_samples: int) -> float:
    """
    Compute fast_p metric: fraction of ALL attempted samples that are correct AND have speedup > p
    This is the official KernelBench metric.
    Args:
        results: List of benchmark results
        p: Speedup threshold
        total_samples: Total number of samples attempted (not just correct ones)
    """
    if total_samples == 0:
        return 0.0
    
    fast_and_correct = sum(
        1 for r in results
        if r.correct and r.speedup > p
    )
    
    return fast_and_correct / total_samples
def compute_reward(result: BenchmarkResult, completion: str, config: TrainingConfig) -> float:
    """
    Reward structure based on Kevin 32B Single Turn paper:
    S = 0.3 · 1{correct} + (T_baseline / T_kernel) · 1{correct}
    
    Where:
    - 0.3 is the base reward for correctness
    - T_baseline / T_kernel is the speedup (only added if correct)
    - Failures get negative rewards to discourage them
    """
    # Check for empty or too short completion
    if not completion or len(completion.strip()) < 50:
        return -1.5  # Extra penalty for no attempt
    
    # Compilation failure
    if not result.compiles:
        return -1.0
    
    # Runtime failures (segfault, CUDA errors)
    if result.error_type == "runtime":
        return -0.5
    
    # Incorrect output (compiled and ran, but wrong result)
    if not result.correct:
        return -0.2
    
    # Correct kernel: base reward (0.3) + speedup
    reward = 0.3
    
    if result.speedup > 0:
        # Add speedup component: T_baseline / T_kernel
        reward += result.speedup
    
    return reward

@app.function(
    image=image,
    gpu="L40S",
    timeout=300,
    max_containers=2,
    retries=1,
)
def benchmark_kernel(
    kernel_code: str,
    reference_code: str,
    problem_id: int
) -> BenchmarkResult:
    """
    Improved kernel benchmarking with better error handling
    """
    import torch
    import sys
    from func_timeout import func_timeout, FunctionTimedOut
    
    # Initialize result
    result = BenchmarkResult(
        compiles=False,
        correct=False,
        speedup=0.0,
        error_type="unknown"
    )
    
    try:
        # Setup environment with debugging enabled
        setup_cuda_environment()
        
        # Enable CUDA error checking
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        
        # Import KernelBench
        sys.path.insert(0, '/root/KernelBench')
        from src.eval import eval_kernel_against_ref
        from src.utils import set_gpu_arch
        
        # Configure for L40S
        set_gpu_arch(["Ada"])
        device = torch.device("cuda:0")
        
        # Clear CUDA cache before evaluation
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        
        # Evaluate with timeout and error isolation
        try:
            # Reset CUDA context to avoid stale errors
            torch.cuda.reset_peak_memory_stats(device)
            
            eval_result = func_timeout(
                timeout=240,
                func=eval_kernel_against_ref,
                kwargs={
                    "original_model_src": reference_code,
                    "custom_model_src": kernel_code,
                    "measure_performance": True,
                    "verbose": False,
                    "num_correct_trials": 3,
                    "num_perf_trials": 5,
                    "device": device,
                }
            )
            
            # Synchronize to catch any async errors
            torch.cuda.synchronize(device)
            
            # Process results
            result.compiles = eval_result.compiled
            result.correct = eval_result.correctness
            
            if result.compiles and not result.correct:
                result.error_type = "correctness"
            
            if result.correct and eval_result.runtime:
                from scripts.generate_baseline_time import measure_program_time
                baseline_result = measure_program_time(
                    ref_arch_name="Reference",
                    ref_arch_src=reference_code,
                    num_trials=5,
                    use_torch_compile=False,
                    device=device
                )
                
                baseline_time = baseline_result.get("mean", 1.0)
                result.speedup = baseline_time / eval_result.runtime if baseline_time > 0 else 0.0
                result.execution_time = eval_result.runtime
            
        except FunctionTimedOut:
            result.error_message = "Evaluation timeout"
            result.error_type = "timeout"
            
        except RuntimeError as e:
            error_str = str(e).lower()
            result.compiles = True  # Compiled but runtime error
            
            if "illegal memory access" in error_str:
                result.error_type = "runtime"
                result.error_message = "Illegal memory access (bounds violation)"
            elif "segmentation" in error_str or "segfault" in error_str:
                result.error_type = "runtime"
                result.error_message = "Segmentation fault"
            elif "cuda" in error_str:
                result.error_type = "runtime"
                result.error_message = f"CUDA error: {str(e)[:100]}"
            else:
                result.error_type = "runtime"
                result.error_message = str(e)[:200]
                
        except Exception as e:
            error_str = str(e).lower()
            if "compile" in error_str or "syntax" in error_str:
                result.error_type = "compilation"
                result.error_message = str(e)[:200]
            else:
                result.compiles = True  # Assume compiled if not explicit compile error
                result.error_type = "runtime"
                result.error_message = str(e)[:200]
                
    except Exception as e:
        result.error_message = f"Setup error: {str(e)[:200]}"
        result.error_type = "setup"
    
    finally:
        # Aggressive cleanup
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Reset device to clear any error state
            torch.cuda.device(0).__enter__()
        except:
            pass
        gc.collect()
    
    return result

# ============================================================================
# VLLM SERVER MANAGER - NEW
# ============================================================================

class VLLMServerManager:
    """Manages vLLM server lifecycle with auto-restart"""
    
    def __init__(self, model_name: str, gpu_id: int = 0, port: int = 8000):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.port = port
        self.process = None
        self.client = None
        self.start_time = None
        
    def start(self):
        """Start vLLM server with retries"""
        import subprocess
        from openai import OpenAI
        
        # Kill any existing process
        if self.process and self.process.poll() is None:
            self.process.terminate()
            time.sleep(2)
        
        # Start new server
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
        log_file = f"/tmp/vllm_{self.port}.log"
        with open(log_file, "w") as f:
            self.process = subprocess.Popen([
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.model_name,
                "--gpu-memory-utilization", "0.8",
                "--dtype", "bfloat16",
                "--max-model-len", "4096",
                "--port", str(self.port),
                "--tensor-parallel-size", "1",
                "--max-num-seqs", "8",  # Increased
                "--max-num-batched-tokens", "16384",  # Increased
                "--disable-log-requests",
                "--enforce-eager",
            ],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT
            )
        
        # Wait for server to be ready
        self.wait_for_ready()
        
        # Create client
        self.client = OpenAI(
            base_url=f"http://localhost:{self.port}/v1",
            api_key="dummy",
            timeout=60.0,
            max_retries=3
        )
        
        self.start_time = time.time()
        
    def wait_for_ready(self, timeout: int = 120):
        """Wait for server to be ready"""
        import requests
        
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"http://localhost:{self.port}/health", timeout=1)
                if resp.status_code == 200:
                    print(f"vLLM server ready on port {self.port}")
                    return
            except:
                pass
            
            # Check if process died
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f"vLLM process died with code {self.process.returncode}")
            
            time.sleep(2)
        
        raise RuntimeError(f"vLLM server failed to start within {timeout}s")
    
    def is_healthy(self) -> bool:
        """Check if server is still healthy"""
        import requests
        
        if not self.process or self.process.poll() is not None:
            return False
        
        try:
            resp = requests.get(f"http://localhost:{self.port}/health", timeout=2)
            return resp.status_code == 200
        except:
            return False
    
    def restart_if_needed(self):
        """Restart server if unhealthy"""
        if not self.is_healthy():
            print(f"vLLM server unhealthy, restarting...")
            self.start()
    
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate with auto-restart on failure"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.restart_if_needed()
                
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1
                )
                
                return response.choices[0].text
                
            except Exception as e:
                print(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.start()  # Force restart
                    time.sleep(5)
                else:
                    return ""  # Return empty on final failure
    
    def shutdown(self):
        """Clean shutdown"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)

# ============================================================================
# MAIN TRAINING LOOP - IMPROVED
# ============================================================================

@app.function(
    gpu="L40S:2",
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={
        "/checkpoints": checkpoint_vol,
        "/results": results_vol
    },
    image=image
)
def train_online_grpo():
    """
    Improved online GRPO training with aggressive memory management
    """
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    
    # Configuration
    config = TrainingConfig()
    
    # Initialize wandb
    run = wandb.init(
        project="cuda-kernel-grpo-fixed",
        config=config.__dict__,
        name=f"grpo_fixed_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Setup CUDA with memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    setup_cuda_environment()
    
    # Test CUDA compilation
    cuda_test_passed = test_cuda_compilation()
    print(f"CUDA compilation test: {'PASSED' if cuda_test_passed else 'WARNING - may still work'}")
    
    # Load model with 8-bit quantization
    print("Loading model with 8-bit quantization...")
    device = torch.device("cuda:1")
    
    # Quantization config
    if config.use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not config.use_8bit else None,
        device_map={"": device},
        trust_remote_code=True,
    )
    
    # Prepare for k-bit training if quantized
    if config.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    vocab_size = model.config.vocab_size
    print(f"Model vocab size: {vocab_size}")
    
    # Apply LoRA with smaller rank to save memory
    peft_config = LoraConfig(
        r=64,  # Reduced from 64
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Reduced targets
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # CRITICAL FIX: Ensure inputs require grads for gradient checkpointing + LoRA
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # Load checkpoint if specified
    start_iteration = 0
    if config.load_checkpoint is not None:
        checkpoint_path = f"/checkpoints/iteration_{config.load_checkpoint}"
        if os.path.exists(checkpoint_path):
            print(f"\n{'='*60}")
            print(f"LOADING CHECKPOINT from iteration {config.load_checkpoint}...")
            print(f"{'='*60}\n")
            # Use safetensors for loading (PEFT saves in safetensors format by default)
            from safetensors.torch import load_file
            model.load_state_dict(
                load_file(f"{checkpoint_path}/adapter_model.bin"),
                strict=False
            )
            start_iteration = config.load_checkpoint
    
    # Create reference model - OFFLOAD TO CPU
    print("Creating reference model on CPU...")
    ref_device = torch.device("cpu") if config.offload_ref_model else device
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not config.use_8bit else None,
        device_map={"": ref_device},
        trust_remote_code=True,
    )
    
    if config.use_8bit:
        ref_model = prepare_model_for_kbit_training(ref_model)
    
    ref_model = get_peft_model(ref_model, peft_config)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    
    print(f"Reference model on: {ref_device}")
    
    # Setup optimizer with memory-efficient settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
        foreach=False,  # More memory efficient
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_training_iterations,
        eta_min=config.learning_rate * 0.1
    )
    
    # Create GRPO trainer with updated ref_model device
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
        optimizer=optimizer,
        device=device,
        vocab_size=vocab_size
    )
    
    # Initialize vLLM server manager
    vllm_manager = VLLMServerManager(
        model_name=config.model_name,
        gpu_id=0,
        port=8000
    )
    vllm_manager.start()
    
    # Load problems and create state manager
    print("Loading KernelBench problems...")
    problems = get_kernelbench_problems(level=1)
    state_manager = TrainingStateManager(len(problems))
    
    # Training loop
    try:
        for iteration in range(start_iteration, config.num_training_iterations):
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}/{config.num_training_iterations}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
            print(f"{'='*80}\n")
            
            state_manager.iteration = iteration
            
            # Select problems with curriculum learning
            selected_problems = state_manager.select_problems(
                config.problems_per_iter,
                use_curriculum=config.use_curriculum
            )
            
            # Phase 1: Generation
            print(f"Generating for problems: {selected_problems}")
            all_prompts = []
            all_completions = []
            all_references = []
            all_problem_ids = []
            
            for problem_idx in selected_problems:
                problem = problems[problem_idx]
                problem_name = os.path.basename(problem)
                reference_code = read_file(problem)
                
                if not reference_code:
                    continue
                
                prompt = create_kernel_optimization_prompt(reference_code, iteration)
                
                # Generate multiple candidates
                for gen_idx in range(config.num_generations_per_iter):
                    completion = vllm_manager.generate(
                        prompt,
                        config.max_completion_length,
                        config.temperature
                    )
                    
                    if completion:
                        all_prompts.append(prompt)
                        all_completions.append(completion)
                        all_references.append(reference_code)
                        all_problem_ids.append(problem_idx)
            
            if not all_completions:
                print("No completions generated, skipping iteration")
                continue
            
            print(f"Generated {len(all_completions)} candidates")
            
            # Phase 2: Benchmarking
            print("Benchmarking kernels...")
            
            # Extract code blocks
            all_codes = []
            for comp in all_completions:
                code = extract_last_code(comp, ["python", "cpp"])
                all_codes.append(code if code else "")
            benchmark_results = []
            successful_benchmarks = 0
            failed_benchmarks = 0
            
            # Benchmark with better error isolation
            for idx, (code, ref, pid) in enumerate(zip(all_codes, all_references, all_problem_ids)):
                try:
                    print(f"  Benchmarking {idx+1}/{len(all_codes)}...")
                    
                    # Skip empty code
                    if not code or len(code.strip()) < 50:
                        print(f"  Skipping sample {idx+1} - empty/too short")
                        fallback = BenchmarkResult(
                            compiles=False,
                            correct=False,
                            speedup=0.0,
                            error_message="Empty or too short code",
                            error_type="invalid_input"
                        )
                        benchmark_results.append(fallback)
                        continue
                    
                    result = benchmark_kernel.remote(code, ref, pid)
                    benchmark_results.append(result)
                    
                    if result.compiles:
                        successful_benchmarks += 1
                    else:
                        failed_benchmarks += 1
                        
                    # Log error type distribution
                    if result.error_type == "runtime":
                        print(f"  Runtime error: {result.error_message[:100]}")
                        
                except Exception as e:
                    print(f"  ERROR on sample {idx+1}: {str(e)[:200]}")
                    failed_benchmarks += 1
                    
                    # Create fallback
                    fallback = BenchmarkResult(
                        compiles=False,
                        correct=False,
                        speedup=0.0,
                        error_message=f"Benchmark failed: {str(e)[:200]}",
                        error_type="modal_error"
                    )
                    benchmark_results.append(fallback)
                
                # Small delay to avoid overwhelming Modal
                if idx % 4 == 3:
                    time.sleep(1)
            
            print(f"Benchmark summary: {successful_benchmarks} compiled, {failed_benchmarks} failed")
                    
            
            # Update problem statistics
            for problem_id, result in zip(all_problem_ids, benchmark_results):
                state_manager.update_stats(problem_id, result)
            total_attempted = len(all_completions)
            # Compute fast_p metrics for different thresholds (KernelBench standard)
            fast_p_thresholds = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
            fast_p_scores = {
                f"fast_p_{p}": compute_fast_p(benchmark_results, p, total_attempted)
                for p in fast_p_thresholds
            }
            # Compute rewards
            rewards = [
                compute_reward(result, completion, config)
                for result, completion in zip(benchmark_results, all_completions)
            ]
            
            # Log metrics
            metrics = {
                "iteration": iteration + 1,
                "compilation_rate": sum(r.compiles for r in benchmark_results) / len(benchmark_results),
                "correctness_rate": sum(r.correct for r in benchmark_results) / len(benchmark_results),
                "avg_speedup": np.mean([r.speedup for r in benchmark_results if r.correct]) if any(r.correct for r in benchmark_results) else 0.0,
                "avg_reward": np.mean(rewards),
                "reward_std": np.std(rewards),
                "learning_rate": scheduler.get_last_lr()[0],
                **fast_p_scores
            }
            
            wandb.log(metrics)
            print(f"Metrics: {metrics}")
            
            # Phase 3: Training
            if any(r != 0 for r in rewards):  # Only train if we have signal
                print("Training on batch...")
                train_stats = trainer.train_step(all_prompts, all_completions, rewards)
                
                wandb.log({
                    "train/loss": train_stats["total_loss"],
                    "train/policy_loss": train_stats["policy_loss"],
                    "train/kl_div": train_stats["kl_div"],
                    "train/ratio_max": train_stats["ratio_max"],
                })
                del all_prompts, all_completions, all_codes
                torch.cuda.empty_cache()
                gc.collect()
                # Step scheduler
                scheduler.step()
            
            # Update reference model periodically
            if (iteration + 1) % config.ref_model_update_freq == 0:
                print("Updating reference model...")
                ref_model.load_state_dict(model.state_dict())
            
            # Checkpoint
            if (iteration + 1) % config.save_every_n_iters == 0:
                checkpoint_path = f"/checkpoints/iteration_{iteration + 1}"
                print(f"Saving checkpoint to {checkpoint_path}")
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                # Save training state - Convert deques to lists for JSON serialization
                state_file = f"{checkpoint_path}/training_state.json"
                with open(state_file, "w") as f:
                    # Convert problem_stats to JSON-serializable format
                    serializable_stats = {}
                    for problem_id, stats in state_manager.problem_stats.items():
                        serializable_stats[problem_id] = {
                            'appearances': stats['appearances'],
                            'compile_rate': stats['compile_rate'],
                            'correct_rate': stats['correct_rate'],
                            'avg_speedup': stats['avg_speedup'],
                            'history': list(stats['history'])  # Convert deque to list
                        }

                    json.dump({
                        "iteration": iteration + 1,
                        "problem_stats": serializable_stats,
                        "metrics": metrics
                    }, f, indent=2)
                
                checkpoint_vol.commit()
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
    finally:
        # Cleanup
        print("\nShutting down...")
        vllm_manager.shutdown()
        
        # Final save
        final_path = "/checkpoints/final_model"
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        checkpoint_vol.commit()
        
        wandb.finish()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_kernelbench_problems(level: int = 1) -> List[str]:
    """Load KernelBench problem paths"""
    import sys
    sys.path.insert(0, '/root/KernelBench')
    from src.dataset import construct_kernelbench_dataset
    return construct_kernelbench_dataset(level=level)

def read_file(file_path: str) -> str:
    """Safely read file contents"""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_last_code(output: str, languages: List[str]) -> Optional[str]:
    """Extract last code block from output"""
    import re
    
    matches = list(re.finditer(r"```(.*?)```", output.strip(), re.DOTALL))
    if not matches:
        return None
    
    code = matches[-1].group(1).strip()
    
    # Remove language headers
    for lang in languages:
        if code.startswith(lang):
            code = code[len(lang):].strip()
            break
    
    return code if len(code) > 20 else None

def setup_cuda_environment():
    """Setup and verify CUDA environment"""
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    
    if not os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
        # Try to find CUDA
        for path in ["/usr/local/cuda", "/usr/local/cuda-12.4"]:
            if os.path.exists(os.path.join(path, "bin", "nvcc")):
                cuda_home = path
                break
    
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"  # L40S architecture
    os.environ["MAX_JOBS"] = "4"
    
    return cuda_home

def test_cuda_compilation() -> bool:
    """Test CUDA compilation environment"""
    try:
        import torch
        from torch.utils.cpp_extension import load_inline
        
        # Simple test kernel
        cuda_src = """
        __global__ void test_kernel(float* x) {
            x[threadIdx.x] = threadIdx.x;
        }
        """
        
        cpp_src = """
        torch::Tensor test_cuda() {
            auto x = torch::zeros(32, torch::kCUDA);
            test_kernel<<<1, 32>>>(x.data_ptr<float>());
            return x;
        }
        """
        
        module = load_inline(
            name='test_cuda',
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=['test_cuda'],
            verbose=False,
            with_cuda=True
        )
        
        result = module.test_cuda()
        expected = torch.arange(32, dtype=torch.float32, device='cuda')
        
        return torch.allclose(result, expected)
        
    except Exception as e:
        print(f"CUDA test failed: {e}")
        return False

def create_kernel_optimization_prompt(reference_code: str, iteration: int) -> str:
    """Create optimization prompt using KernelBench format"""
    import sys
    sys.path.insert(0, '/root/KernelBench')
    from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
    return prompt_generate_custom_cuda_from_prompt_template(ref_arch_src=reference_code)

# ============================================================================
# ENTRY POINT
# ============================================================================

@app.local_entrypoint()
def main(action: str = "train"):
    """
    Entry point for the application
    
    Usage:
        modal run cuda_kernel_online_grpo_fixed.py --action train
    """
    if action == "train":
        print("Starting improved online GRPO training...")
        train_online_grpo.remote()
    else:
        print(f"Unknown action: {action}")

if __name__ == "__main__":
    main()